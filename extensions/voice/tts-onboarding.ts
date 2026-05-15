/**
 * First-run TTS onboarding helper.
 *
 * v7.0.0 ships the lightweight version: when a user enables TTS for the
 * first time (and onboarding hasn't been completed), the orchestrator
 * shows a single notify() with the smart-default recommendation and
 * tells the user how to either accept it (run /voice-speak-test) or
 * customize (run /voice-speak-models).
 *
 * Why not a multi-step picker overlay (the v7 plan's full vision):
 *   - The settings panel already exposes every knob with proper UX
 *   - A first-run popup that hijacks the editor on every initial enable
 *     is annoying for advanced users who already configured things via
 *     settings.json
 *   - The lightweight surface is honest: "here's the recommendation,
 *     here's where to change it" — and it composes with the rest of
 *     the v7 surface (Speak tab, /voice-speak-info)
 *
 * If field reports show users want a richer flow, we can swap this
 * notify-based version for a `ctx.ui.custom()` overlay in v7.1
 * without changing any other code path.
 */

import type { ExtensionCommandContext, ExtensionContext } from "@earendil-works/pi-coding-agent";
import type { VoiceConfig, VoiceSettingsScope } from "./config";
import type { DeviceProfile } from "./device";
import { recommendDefaultModel, isTtsModelInstalled, getTtsModel } from "./tts-local-models";

type NotifyContext = ExtensionContext | ExtensionCommandContext;

export interface OnboardTtsOpts {
	ctx: NotifyContext;
	config: VoiceConfig;
	device: DeviceProfile;
	cwd: string;
	saveConfig: (config: VoiceConfig, scope: VoiceSettingsScope, cwd: string) => void;
}

/**
 * Run the first-run TTS onboarding hint. Idempotent — only shows the
 * hint once per config (`config.onboarding.completed` controls the
 * generic onboarding flag; we co-opt a parallel `ttsOnboardingShown`
 * marker on the config to avoid spamming the hint on every enable).
 *
 * Returns true if the hint was shown this call.
 */
export function maybeShowTtsOnboarding(opts: OnboardTtsOpts): boolean {
	const { ctx, config, device, cwd, saveConfig } = opts;
	if (!ctx.hasUI) return false;
	if ((config as any).ttsOnboardingShown) return false;

	const recommendation = recommendDefaultModel(device.systemLocale ?? "en");
	let recModel;
	try { recModel = getTtsModel(recommendation.modelId); } catch { recModel = undefined; }
	const installed = isTtsModelInstalled(recommendation.modelId);

	const lines = [
		"TTS enabled — voice output for Pi.",
		"",
		`  ${recommendation.reason}`,
		"",
		recModel
			? `  Recommended:  ${recModel.name} (${recModel.size}, ${recModel.languages.join("/")})`
			: `  Recommended:  ${recommendation.modelId}`,
		`  Status:       ${installed ? "ready ✓" : `not installed — first speak downloads ${recModel?.size ?? "model"}`}`,
		"",
		"  Try it:       /voice-speak-test",
		"  Pick another: /voice-speak-models  (or /voice-settings → Speak tab)",
		"  Diagnose:     /voice-speak-info",
		"  Disable:      /voice-speak-toggle",
	];
	if (recommendation.fallback) {
		lines.push("");
		lines.push(`  Note: ${device.systemLocale ?? "your locale"} has no built-in voice — English fallback chosen.`);
	}

	ctx.ui.notify(lines.join("\n"), "info");

	// Mark the hint as shown (and persist) so subsequent enables are quiet.
	(config as any).ttsOnboardingShown = true;
	saveConfig(config, config.scope === "project" ? "project" : "global", cwd);
	return true;
}

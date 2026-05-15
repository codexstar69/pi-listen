/**
 * Rich first-run onboarding overlay for TTS — §9 of the v7.1 plan.
 *
 * Replaces the v7.0 notify-based hint with a focused overlay that
 * shows the recommendation, the install size, and three explicit
 * actions ([enter] try it / [m] pick another / [esc] skip). Per the
 * §9 event-ordering contract:
 *
 *   - All three actions mark `ttsOnboardingShown = true` BEFORE any
 *     async work so a failed install/cancel never prompts the user
 *     again.
 *   - `[enter]` returns `{ kind: "test" }` — the caller runs
 *     /voice-speak-test which auto-installs the recommended model.
 *   - `[m]` returns `{ kind: "pickModel" }` — caller opens settings
 *     panel on the Speak tab so the user can browse alternatives.
 *   - `[esc]` returns `{ kind: "skip" }` — silent dismissal.
 *
 * Visual design (no emoji per v7.1 hard constraint):
 *
 *   ┌─ pi-listen TTS ─────────────────────────────────────────────┐
 *   │                                                              │
 *   │   Voice output ready.                                        │
 *   │                                                              │
 *   │   ● Recommended:  Kitten Nano v0.2  (25 MB, en)              │
 *   │     Smallest English TTS — sub-real-time on M-series         │
 *   │                                                              │
 *   │   Status:         not installed — press [↵] to download      │
 *   │                                                              │
 *   │   [↵] Try it now  [m] Pick another  [esc] Skip               │
 *   └──────────────────────────────────────────────────────────────┘
 */

import { matchesKey, Key } from "@earendil-works/pi-tui";
import type { Theme } from "@earendil-works/pi-coding-agent";
import { recommendDefaultModel, isTtsModelInstalled, getTtsModel } from "./tts-local-models";
import { ICON } from "./ui-icons";
import { isPanelTooNarrow, visualWidth, padRightVisual } from "./ui-width";

export type OnboardingResult =
	| { kind: "test" }
	| { kind: "pickModel" }
	| { kind: "skip" };

export interface OnboardingOverlayDeps {
	/** Active locale (e.g. "en", "zh") for recommendation. */
	systemLocale: string | undefined;
	/** Optional Pi theme for color routing. */
	theme?: Theme;
}

/**
 * Pi `Component`-shaped class — pass into `ctx.ui.custom()` directly.
 * `done` is provided by the caller's factory closure to resolve the
 * outer promise.
 */
export class TtsOnboardingOverlay {
	private readonly deps: OnboardingOverlayDeps;
	private readonly done: (result: OnboardingResult) => void;
	private resolved = false;

	constructor(deps: OnboardingOverlayDeps, done: (result: OnboardingResult) => void) {
		this.deps = deps;
		this.done = done;
	}

	render(width: number): string[] {
		// v7.1 §10/§13: hard-block below 60 cols. The overlay is
		// pure-content; below that, we render a dim "resize" hint and
		// expose the same three actions as keys without the chrome.
		if (isPanelTooNarrow(width)) {
			return this.renderNarrow();
		}

		const w = Math.max(60, Math.min(width - 2, 80));
		const innerW = w - 4;
		const t = this.deps.theme;
		const dim = (s: string) => (t ? t.fg("dim", s) : s);
		const accent = (s: string) => (t ? t.fg("accent", s) : s);
		const bold = (s: string) => (t ? t.fg("accent", s) : s);
		const success = (s: string) => (t ? t.fg("success", s) : s);
		const warning = (s: string) => (t ? t.fg("warning", s) : s);

		const recommendation = recommendDefaultModel(this.deps.systemLocale ?? "en");
		let recModel;
		try { recModel = getTtsModel(recommendation.modelId); } catch { recModel = undefined; }
		const installed = recModel ? isTtsModelInstalled(recModel.id) : false;

		const lines: string[] = [];
		// v7.2: rounded corners for modal feel. Title sits inline on the
		// top edge with thin spacing so the box reads as a "sheet"
		// rather than a hard frame (HIG modal aesthetic).
		const top = `${ICON.boxRoundedTL}${ICON.boxH.repeat(2)} ${bold("pi-listen TTS")} ${ICON.boxH.repeat(Math.max(0, innerW - 16))}${ICON.boxRoundedTR}`;
		const bottom = `${ICON.boxRoundedBL}${ICON.boxH.repeat(innerW)}${ICON.boxRoundedBR}`;
		const hr = `${ICON.boxV}${" ".repeat(innerW)}${ICON.boxV}`;
		const row = (s: string): string => {
			// CJK-aware pad-right (Codex final-review nit): use visualWidth
			// not String.length so wide glyphs in Chinese/Japanese/Korean
			// recommendation copy align correctly.
			const w = visualWidth(s);
			if (w <= innerW) return `${ICON.boxV}${padRightVisual(s, innerW)}${ICON.boxV}`;
			return `${ICON.boxV}${s.slice(0, innerW)}${ICON.boxV}`;
		};

		lines.push(top);
		lines.push(hr);
		lines.push(row(`  ${accent("Voice output ready.")}`));
		lines.push(hr);

		if (recModel) {
			const langs = recModel.languages.length > 1 ? `${recModel.languages.length} langs` : recModel.languages[0];
			lines.push(row(`  ${success(ICON.bulletActive)} ${dim("Recommended:")}  ${accent(recModel.name)} ${dim(`(${recModel.size}, ${langs})`)}`));
			lines.push(row(`    ${dim(recModel.notes)}`));
		} else {
			lines.push(row(`  ${success(ICON.bulletActive)} ${dim("Recommended:")}  ${accent(recommendation.modelId)}`));
		}

		lines.push(hr);

		const statusLabel = installed
			? success("ready")
			: warning(`not installed ${ICON.middot} press ${accent("[↵]")} to download`);
		lines.push(row(`  ${dim("Status:")}        ${statusLabel}`));

		if (recommendation.fallback) {
			lines.push(hr);
			lines.push(row(`  ${dim(`Note: ${this.deps.systemLocale ?? "your locale"} has no native voice — English fallback chosen.`)}`));
		}

		lines.push(hr);
		const hint = `  ${accent("[↵]")} ${dim("Try it now")}   ${accent("[m]")} ${dim("Pick another")}   ${accent("[esc]")} ${dim("Skip")}`;
		lines.push(row(hint));
		lines.push(bottom);
		return lines;
	}

	private renderNarrow(): string[] {
		const t = this.deps.theme;
		const dim = (s: string) => (t ? t.fg("dim", s) : s);
		const accent = (s: string) => (t ? t.fg("accent", s) : s);
		return [
			` ${accent("pi-listen TTS")}`,
			` ${dim("Voice output ready. Resize to ≥60 cols for the full hint.")}`,
			` ${accent("[↵]")} ${dim("test")}   ${accent("[m]")} ${dim("pick")}   ${accent("[esc]")} ${dim("skip")}`,
		];
	}

	handleInput(data: string): void {
		if (this.resolved) return;
		if (matchesKey(data, Key.enter)) {
			this.resolve({ kind: "test" });
			return;
		}
		if (matchesKey(data, Key.escape)) {
			this.resolve({ kind: "skip" });
			return;
		}
		// Lowercase 'm' or 'M' picks model
		if (data === "m" || data === "M") {
			this.resolve({ kind: "pickModel" });
			return;
		}
	}

	invalidate(): void { /* render is uncached */ }

	private resolve(result: OnboardingResult): void {
		if (this.resolved) return;
		this.resolved = true;
		try { this.done(result); } catch { /* never fail closure */ }
	}
}

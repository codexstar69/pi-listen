import type { ExtensionCommandContext, ExtensionContext } from "@mariozechner/pi-coding-agent";
import type { VoiceConfig, VoiceSettingsScope } from "./config";

type VoiceUiContext = ExtensionContext | ExtensionCommandContext;

export interface OnboardingResult {
	config: VoiceConfig;
	selectedScope: VoiceSettingsScope;
	summaryLines: string[];
}

export interface FirstRunDecision {
	action: "start" | "later";
}

export function finalizeOnboardingConfig(
	config: VoiceConfig,
	options: { validated: boolean; source: "first-run" | "setup-command" },
): VoiceConfig {
	if (options.validated) {
		const timestamp = new Date().toISOString();
		return {
			...config,
			onboarding: {
				...config.onboarding,
				completed: true,
				schemaVersion: config.version,
				completedAt: timestamp,
				lastValidatedAt: timestamp,
				source: options.source,
				skippedAt: undefined,
			},
		};
	}

	return {
		...config,
		onboarding: {
			...config.onboarding,
			completed: false,
			schemaVersion: config.version,
			completedAt: undefined,
			lastValidatedAt: undefined,
			source: "repair",
			skippedAt: undefined,
		},
	};
}

export async function promptFirstRunOnboarding(ctx: VoiceUiContext): Promise<FirstRunDecision> {
	const choice = await ctx.ui.select("Set up pi-voice now?", [
		"Start voice setup",
		"Remind me later",
	]);

	return { action: choice === "Start voice setup" ? "start" : "later" };
}

export async function runVoiceOnboarding(
	ctx: VoiceUiContext,
	currentConfig: VoiceConfig,
): Promise<OnboardingResult | undefined> {
	const hasDeepgramKey = Boolean(process.env.DEEPGRAM_API_KEY || currentConfig.deepgramApiKey);

	// ─── Deepgram API key setup ──────────────────────────────
	if (!hasDeepgramKey) {
		const keyAction = await ctx.ui.select(
			"Deepgram API key not found. What would you like to do?",
			[
				"Paste API key now",
				"I'll set it up later (ask pi to help or export DEEPGRAM_API_KEY=...)",
			],
		);
		if (!keyAction) return undefined;

		if (keyAction.startsWith("Paste")) {
			ctx.ui.notify(
				[
					"Get your free Deepgram API key:",
					"  → https://dpgr.am/pi-voice",
					"  (Sign up → $200 free credits, no card needed)",
					"",
					"Paste your key below:",
				].join("\n"),
				"info",
			);
			const apiKey = await ctx.ui.input("DEEPGRAM_API_KEY");
			if (apiKey && apiKey.trim().length > 10) {
				const trimmedKey = apiKey.trim();
				const fs = await import("node:fs");
				const os = await import("node:os");
				const home = os.homedir();
				const envSecretsPath = `${home}/.env.secrets`;
				const zshrcPath = `${home}/.zshrc`;
				const exportLine = `export DEEPGRAM_API_KEY="${trimmedKey}"`;

				const targetFile = fs.existsSync(envSecretsPath) ? envSecretsPath : zshrcPath;
				const existing = fs.existsSync(targetFile) ? fs.readFileSync(targetFile, "utf-8") : "";

				if (existing.includes("DEEPGRAM_API_KEY")) {
					const updated = existing.replace(/^export DEEPGRAM_API_KEY=.*$/m, exportLine);
					fs.writeFileSync(targetFile, updated);
				} else {
					fs.appendFileSync(targetFile, `\n${exportLine}\n`);
				}

				process.env.DEEPGRAM_API_KEY = trimmedKey;

				ctx.ui.notify(
					`API key saved to ${targetFile}\nActive in this session. New terminals will pick it up automatically.`,
					"info",
				);
			} else if (apiKey !== undefined && apiKey !== null) {
				ctx.ui.notify(
					"Key looks too short — skipped. You can set it later:\n  export DEEPGRAM_API_KEY=\"your-key\"",
					"warning",
				);
			}
		} else {
			ctx.ui.notify(
				[
					"No problem! When you're ready:",
					"  1. Get a key → https://dpgr.am/pi-voice ($200 free credits)",
					"  2. Run: export DEEPGRAM_API_KEY=\"your-key\"",
					"  3. Or ask pi: \"help me set up my Deepgram API key\"",
				].join("\n"),
				"info",
			);
		}
	}

	// ─── Choose scope ────────────────────────────────────────
	const scopeChoice = await ctx.ui.select("Where should pi-voice settings be saved?", [
		"Global (all projects)",
		"Project only (this repo)",
	]);
	if (!scopeChoice) return undefined;
	const selectedScope: VoiceSettingsScope = scopeChoice.startsWith("Project") ? "project" : "global";

	const summaryLines = [
		"Backend: Deepgram (streaming)",
		`Scope: ${selectedScope}`,
		`API key: ${process.env.DEEPGRAM_API_KEY ? "configured" : "not yet set"}`,
	];

	const confirm = await ctx.ui.confirm("Confirm voice setup", summaryLines.join("\n"));
	if (!confirm) return undefined;

	return {
		selectedScope,
		summaryLines,
		config: {
			...currentConfig,
			scope: selectedScope,
			onboarding: {
				...currentConfig.onboarding,
				completed: false,
				schemaVersion: currentConfig.version,
				source: "first-run",
			},
		},
	};
}

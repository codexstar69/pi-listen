import type { ExtensionCommandContext, ExtensionContext } from "@mariozechner/pi-coding-agent";
import type { VoiceConfig, VoiceSettingsScope } from "./config";
import {
	getModelReadiness,
	recommendVoiceSetup,
	type BackendAvailability,
	type DiagnosticsPreference,
	type EnvironmentDiagnostics,
} from "./diagnostics";
import { buildProvisioningPlan } from "./install";

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

export function getModelOptions(backend: BackendAvailability, recommendedModel: string): string[] {
	return backend.models.map((model) => {
		const installed = backend.installed_models?.includes(model) ?? false;
		if (model === recommendedModel && installed) return `${model} (recommended, installed)`;
		if (model === recommendedModel) return `${model} (recommended)`;
		if (installed) return `${model} (installed)`;
		return model;
	});
}

export function buildSelectableBackends(
	targetMode: "api" | "local",
	diagnostics: EnvironmentDiagnostics,
): BackendAvailability[] {
	if (targetMode === "api") {
		return [
			diagnostics.backends.find((candidate) => candidate.name === "deepgram") ?? {
				name: "deepgram",
				available: diagnostics.hasDeepgramKey,
				type: "cloud",
				default_model: "nova-3",
				models: ["nova-3", "nova-2", "whisper-large", "whisper-medium", "whisper-small"],
				install: "Set DEEPGRAM_API_KEY",
			},
		];
	}

	const discoveredLocalBackends = diagnostics.backends.filter((candidate) => candidate.type === "local");
	const fallbackLocalBackends: BackendAvailability[] = [
		{
			name: "faster-whisper",
			available: false,
			type: "local",
			default_model: "small",
			models: ["tiny", "base", "small", "medium", "large-v3-turbo"],
			install: "python3 -m pip install faster-whisper",
		},
		{
			name: "moonshine",
			available: false,
			type: "local",
			default_model: "moonshine/base",
			models: ["moonshine/tiny", "moonshine/base"],
			install: "python3 -m pip install 'useful-moonshine[onnx]'",
		},
		{
			name: "whisper-cpp",
			available: false,
			type: "local",
			default_model: "small",
			models: ["tiny", "base", "small", "medium", "large"],
			install: "brew install whisper-cpp",
		},
		{
			name: "parakeet",
			available: false,
			type: "local",
			default_model: "nvidia/parakeet-tdt-0.6b-v2",
			models: ["nvidia/parakeet-tdt-0.6b-v2", "nvidia/parakeet-ctc-0.6b", "nvidia/parakeet-tdt-1.1b"],
			install: "python3 -m pip install 'nemo_toolkit[asr]'",
		},
	];
	const merged = new Map<string, BackendAvailability>();
	for (const backend of [...discoveredLocalBackends, ...fallbackLocalBackends]) {
		if (!merged.has(backend.name)) merged.set(backend.name, backend);
	}
	return [...merged.values()];
}

export async function runVoiceOnboarding(
	ctx: VoiceUiContext,
	currentConfig: VoiceConfig,
	diagnostics: EnvironmentDiagnostics,
): Promise<OnboardingResult | undefined> {
	const modeChoice = await ctx.ui.select("How do you want to use speech-to-text?", [
		"Recommended for me",
		"Cloud API (fastest setup)",
		"Local download (offline / private)",
	]);
	if (!modeChoice) return undefined;

	const preference = await askPreference(ctx, modeChoice);
	if (!preference) return undefined;

	const recommendation = recommendVoiceSetup(diagnostics, preference);
	const targetMode =
		modeChoice === "Recommended for me"
			? recommendation.mode
			: modeChoice.startsWith("Cloud API")
				? "api"
				: "local";

	let backend = recommendation.backend;
	let model = recommendation.model;

	if (targetMode === "api") {
		const cloudSelection = await chooseBackendAndModel(
			ctx,
			buildSelectableBackends("api", diagnostics),
			"Choose API provider/model",
		);
		if (!cloudSelection) return undefined;
		backend = cloudSelection.backend;
		model = cloudSelection.model;
	} else {
		const localSelection = await chooseBackendAndModel(
			ctx,
			buildSelectableBackends("local", diagnostics),
			"Choose local backend/model",
		);
		if (!localSelection) return undefined;
		backend = localSelection.backend;
		model = localSelection.model;
	}

	const scopeChoice = await ctx.ui.select("Where should pi-voice settings be saved?", [
		"Global (all projects)",
		"Project only (this repo)",
	]);
	if (!scopeChoice) return undefined;
	const selectedScope: VoiceSettingsScope = scopeChoice.startsWith("Project") ? "project" : "global";

	const draftConfig: VoiceConfig = {
		...currentConfig,
		mode: targetMode,
		backend,
		model,
		scope: selectedScope,
	};
	const selectedBackend = buildSelectableBackends(targetMode, diagnostics).find((candidate) => candidate.name === backend);
	const modelReadiness = getModelReadiness(selectedBackend, model);
	const provisioningPlan = buildProvisioningPlan(draftConfig, diagnostics);
	const fixableIssues = recommendation.fixableIssues.length > 0
		? recommendation.fixableIssues.map((issue) => `- ${issue}`)
		: ["- No immediate setup issues detected"];
	const recommendationLine =
		modeChoice === "Recommended for me"
			? `Recommendation: ${recommendation.reason}`
			: targetMode === "api"
				? "Selected path: Cloud API. We'll validate credentials and recording prerequisites."
				: modelReadiness === "installed"
					? "Selected path: Local model already installed and ready to configure."
					: "Selected path: Local setup. We'll validate dependencies and any required downloads.";
	const summaryLines = [
		`Mode: ${targetMode === "api" ? "Cloud API" : "Local download"}`,
		`Backend: ${backend}`,
		`Model: ${model}`,
		`Model status: ${modelReadiness}`,
		`Scope: ${selectedScope}`,
		"",
		recommendationLine,
		"",
		"Checks before voice is fully ready:",
		...fixableIssues,
		"",
		"Suggested commands:",
		...(provisioningPlan.commands.length > 0
			? provisioningPlan.commands.map((command) => `- ${command}`)
			: ["- No install commands required"]),
		...(provisioningPlan.manualSteps.length > 0
			? ["", "Manual steps:", ...provisioningPlan.manualSteps.map((step) => `- ${step}`)]
			: []),
	];

	const confirm = await ctx.ui.confirm("Confirm voice setup", summaryLines.join("\n"));
	if (!confirm) return undefined;

	return {
		selectedScope,
		summaryLines,
		config: {
			...currentConfig,
			mode: targetMode,
			backend,
			model,
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

async function askPreference(
	ctx: VoiceUiContext,
	modeChoice: string,
): Promise<DiagnosticsPreference | undefined> {
	if (modeChoice.startsWith("Cloud API")) return "speed";
	if (modeChoice.startsWith("Local download")) return "privacy";

	const choice = await ctx.ui.select("What matters most to you?", [
		"Balanced default",
		"Fastest setup",
		"Best privacy / offline use",
		"Best accuracy",
		"Lowest resource usage",
	]);
	if (!choice) return undefined;

	if (choice.startsWith("Fastest")) return "speed";
	if (choice.startsWith("Best privacy")) return "privacy";
	if (choice.startsWith("Best accuracy")) return "accuracy";
	if (choice.startsWith("Lowest resource")) return "low-resource";
	return "balanced";
}

async function chooseBackendAndModel(
	ctx: VoiceUiContext,
	backends: BackendAvailability[],
	title: string,
): Promise<{ backend: BackendAvailability["name"]; model: string } | undefined> {
	if (backends.length === 0) return undefined;

	const backendOptions = backends.map((backend) => {
		const installedCount = backend.installed_models?.length ?? 0;
		if (backend.type === "cloud") {
			return `${backend.name} — ${backend.available ? "API ready" : backend.install ?? "needs API key"}`;
		}
		if (installedCount > 0) {
			return `${backend.name} — ${installedCount} installed model${installedCount === 1 ? "" : "s"}`;
		}
		const suffix = backend.available ? "backend ready" : backend.install ?? "needs setup";
		return `${backend.name} — ${suffix}`;
	});
	const backendChoice = await ctx.ui.select(title, backendOptions);
	if (!backendChoice) return undefined;
	const backendIndex = backendOptions.indexOf(backendChoice);
	const selectedBackend = backends[Math.max(0, backendIndex)];

	const recommendedModel = selectedBackend.installed_models?.includes(selectedBackend.default_model)
		? selectedBackend.default_model
		: selectedBackend.installed_models?.[0] ?? selectedBackend.default_model;
	const modelChoice = await ctx.ui.select(
		`Choose model for ${selectedBackend.name}`,
		getModelOptions(selectedBackend, recommendedModel),
	);
	if (!modelChoice) return undefined;

	return {
		backend: selectedBackend.name,
		model: modelChoice
			.replace(" (recommended, installed)", "")
			.replace(" (recommended)", "")
			.replace(" (installed)", ""),
	};
}

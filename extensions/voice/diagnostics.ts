import { spawnSync } from "node:child_process";

export type DiagnosticsPreference = "balanced" | "speed" | "privacy" | "accuracy" | "low-resource";

export interface BackendAvailability {
	name: "faster-whisper" | "moonshine" | "whisper-cpp" | "deepgram" | "parakeet" | "parakeet-mlx";
	available: boolean;
	type: "local" | "cloud";
	default_model: string;
	models: string[];
	installed_models?: string[];
	install_detection?: string;
	install?: string | null;
}

export interface EnvironmentDiagnostics {
	hasPython: boolean;
	hasSox: boolean;
	hasHomebrew: boolean;
	hasDeepgramKey: boolean;
	backends: BackendAvailability[];
	issues: string[];
}

export interface VoiceRecommendation {
	mode: "local" | "api";
	backend: BackendAvailability["name"];
	model: string;
	reason: string;
	fixableIssues: string[];
}

export type ModelReadiness = "installed" | "download required" | "unknown" | "api";

function commandExists(command: string): boolean {
	return spawnSync("which", [command], { stdio: "pipe", timeout: 3000 }).status === 0;
}

export function scanEnvironment(transcribeScriptPath: string): EnvironmentDiagnostics {
	const hasPython = commandExists("python3");
	const hasSox = commandExists("rec");
	const hasHomebrew = commandExists("brew");
	const hasDeepgramKey = Boolean(process.env.DEEPGRAM_API_KEY);

	let backends: BackendAvailability[] = [];
	// Only try to list backends if the transcribe script actually exists
	const scriptExists = require("fs").existsSync(transcribeScriptPath);
	if (hasPython && scriptExists) {
		const result = spawnSync("python3", [transcribeScriptPath, "--list-backends"], {
			stdio: ["pipe", "pipe", "pipe"],
			timeout: 10000,
			encoding: "utf8",
		});
		try {
			backends = JSON.parse(result.stdout || "[]") as BackendAvailability[];
		} catch {
			backends = [];
		}
	}

	const issues: string[] = [];
	if (!hasSox) issues.push("Install SoX for microphone recording");
	// Only flag missing python/backends if user doesn't have Deepgram
	if (!hasDeepgramKey) {
		if (!hasPython) issues.push("python3 is required for local STT backends");
		if (!backends.some((backend) => backend.available)) {
			issues.push("No STT backend is currently installed or configured");
		}
		issues.push("Deepgram API key is not configured");
	}

	return {
		hasPython,
		hasSox,
		hasHomebrew,
		hasDeepgramKey,
		backends,
		issues,
	};
}

function getBackend(backends: BackendAvailability[], name: BackendAvailability["name"]): BackendAvailability | undefined {
	return backends.find((backend) => backend.name === name);
}

export function getModelReadiness(backend: BackendAvailability | undefined, model: string): ModelReadiness {
	if (!backend) return "unknown";
	if (backend.type === "cloud") return "api";
	if ((backend.installed_models ?? []).includes(model)) return "installed";
	const highConfidenceDetectors = new Set(["huggingface-cache", "whisper-cpp-model-paths"]);
	if (backend.available && highConfidenceDetectors.has(backend.install_detection ?? "")) {
		return "download required";
	}
	return "unknown";
}

function getPreferredInstalledModel(backend: BackendAvailability | undefined): string | undefined {
	if (!backend) return undefined;
	const installed = backend.installed_models ?? [];
	if (installed.includes(backend.default_model)) return backend.default_model;
	return installed[0];
}

const HIGH_CONFIDENCE_DETECTORS = new Set(["huggingface-cache", "whisper-cpp-model-paths"]);

function getDetectorConfidenceKey(backend: BackendAvailability): string {
	if (backend.install_detection) return backend.install_detection;
	if (backend.name === "faster-whisper") return "huggingface-cache";
	if (backend.name === "whisper-cpp") return "whisper-cpp-model-paths";
	return "";
}

function hasHighConfidenceInstalledModel(backend: BackendAvailability): boolean {
	return (backend.installed_models?.length ?? 0) > 0 && HIGH_CONFIDENCE_DETECTORS.has(getDetectorConfidenceKey(backend));
}

function hasHighConfidenceBackendAvailability(backend: BackendAvailability): boolean {
	return backend.available && HIGH_CONFIDENCE_DETECTORS.has(getDetectorConfidenceKey(backend));
}

function getPreferredLocalBackend(backends: BackendAvailability[]): BackendAvailability | undefined {
	const localBackends = backends.filter((backend) => backend.type === "local");
	const withHighConfidenceInstalledModel = localBackends.find((backend) => hasHighConfidenceInstalledModel(backend));
	if (withHighConfidenceInstalledModel) return withHighConfidenceInstalledModel;
	const highConfidenceAvailable = localBackends.find((backend) => hasHighConfidenceBackendAvailability(backend));
	if (highConfidenceAvailable) return highConfidenceAvailable;
	const withHeuristicInstalledModel = localBackends.find((backend) => (backend.installed_models?.length ?? 0) > 0);
	if (withHeuristicInstalledModel) return withHeuristicInstalledModel;
	return localBackends.find((backend) => backend.available);
}

export function recommendVoiceSetup(
	diagnostics: EnvironmentDiagnostics,
	preference: DiagnosticsPreference = "balanced",
): VoiceRecommendation {
	const fixableIssues: string[] = [];
	if (!diagnostics.hasSox) {
		fixableIssues.push("Install SoX for microphone recording");
	}
	if (!diagnostics.hasPython) {
		fixableIssues.push("Install python3 for local transcription backends");
	}

	const fasterWhisper = getBackend(diagnostics.backends, "faster-whisper");
	const deepgram = getBackend(diagnostics.backends, "deepgram");
	const preferredLocalBackend = getPreferredLocalBackend(diagnostics.backends);
	const anyLocalAvailable = diagnostics.backends.some((backend) => backend.type === "local" && backend.available);

	if (!anyLocalAvailable) {
		fixableIssues.push("Install a local STT backend such as faster-whisper");
	}

	const installedLocalModel = getPreferredInstalledModel(preferredLocalBackend);
	const installedModelIsHighConfidence = preferredLocalBackend ? hasHighConfidenceInstalledModel(preferredLocalBackend) : false;
	const localRecommendation = {
		mode: "local" as const,
		backend: preferredLocalBackend?.name ?? "faster-whisper",
		model: installedLocalModel ?? preferredLocalBackend?.default_model ?? fasterWhisper?.default_model ?? "small",
		reason: installedLocalModel && installedModelIsHighConfidence
			? `Recommended local default because ${installedLocalModel} is already installed and ready to configure.`
			: preference === "privacy"
				? "Best for privacy and offline use with a strong local default."
				: preference === "accuracy"
					? "Best balance of local quality and maturity."
					: "Recommended local default with good balance of quality and setup effort.",
		fixableIssues,
	};

	if (preference === "speed" && diagnostics.hasDeepgramKey && deepgram?.available) {
		return {
			mode: "api",
			backend: "deepgram",
			model: deepgram.default_model,
			reason: "The fastest path to a working setup because cloud transcription is already configured.",
			fixableIssues,
		};
	}

	if (preference === "privacy") {
		return {
			...localRecommendation,
			reason: "Best for privacy and offline use without sending audio to a cloud API.",
		};
	}

	if (preference === "balanced") {
		if (preferredLocalBackend?.available && (hasHighConfidenceInstalledModel(preferredLocalBackend) || hasHighConfidenceBackendAvailability(preferredLocalBackend) || !(diagnostics.hasDeepgramKey && deepgram?.available))) {
			return localRecommendation;
		}
		if (diagnostics.hasDeepgramKey && deepgram?.available) {
			return {
				mode: "api",
				backend: "deepgram",
				model: deepgram.default_model,
				reason: "Recommended because cloud transcription is already configured and ready to use.",
				fixableIssues,
			};
		}
	}

	if (preference === "low-resource") {
		const moonshine = getBackend(diagnostics.backends, "moonshine");
		if (moonshine?.available) {
			return {
				mode: "local",
				backend: "moonshine",
				model: moonshine.default_model,
				reason: "Lightweight local option with lower resource requirements.",
				fixableIssues,
			};
		}
	}

	return localRecommendation;
}

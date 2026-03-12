import type { VoiceConfig } from "./config";
import type { EnvironmentDiagnostics } from "./diagnostics";

export interface ProvisioningPlan {
	ready: boolean;
	summary: string;
	commands: string[];
	manualSteps: string[];
}

export function buildProvisioningPlan(config: VoiceConfig, diagnostics: EnvironmentDiagnostics): ProvisioningPlan {
	const commands: string[] = [];
	const manualSteps: string[] = [];

	if (!diagnostics.hasSox) {
		commands.push("brew install sox");
	}

	if (config.mode === "api") {
		if (config.backend === "deepgram" && !diagnostics.hasDeepgramKey) {
			manualSteps.push("Set DEEPGRAM_API_KEY before using Deepgram API mode");
		}
	} else {
		if (!diagnostics.hasPython) {
			manualSteps.push("Install python3 before enabling local STT backends");
		}

		const selectedBackend = diagnostics.backends.find((backend) => backend.name === config.backend);
		if (!selectedBackend?.available) {
			switch (config.backend) {
				case "faster-whisper":
					commands.push("python3 -m pip install faster-whisper");
					break;
				case "moonshine":
					commands.push("python3 -m pip install 'useful-moonshine[onnx]'");
					break;
				case "whisper-cpp":
					commands.push("brew install whisper-cpp");
					break;
				case "parakeet":
					commands.push("python3 -m pip install 'nemo_toolkit[asr]'");
					break;
				default:
					manualSteps.push(`Install the selected backend: ${config.backend}`);
			}
		} else if (!(selectedBackend.installed_models ?? []).includes(config.model)) {
			const highConfidenceDetectors = new Set(["huggingface-cache", "whisper-cpp-model-paths"]);
			if (highConfidenceDetectors.has(selectedBackend.install_detection ?? "")) {
				manualSteps.push(`Selected model ${config.model} is not installed yet and may need to be downloaded on first use`);
			} else {
				manualSteps.push(`Selected model ${config.model} could not be confirmed locally and may still need a download on first use`);
			}
		}
	}

	return {
		ready: commands.length === 0 && manualSteps.length === 0,
		summary: config.mode === "api" ? "Provisioning plan for api mode" : "Provisioning plan for local mode",
		commands,
		manualSteps,
	};
}

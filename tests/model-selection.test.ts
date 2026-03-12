import { describe, expect, test } from "bun:test";
import {
	getModelReadiness,
	recommendVoiceSetup,
	type BackendAvailability,
	type EnvironmentDiagnostics,
} from "../extensions/voice/diagnostics";
import { getModelOptions } from "../extensions/voice/onboarding";

function makeDiagnostics(backends: BackendAvailability[]): EnvironmentDiagnostics {
	return {
		hasPython: true,
		hasSox: true,
		hasHomebrew: true,
		hasDeepgramKey: false,
		backends,
		issues: [],
	};
}

describe("model-aware recommendation", () => {
	test("prefers an already-installed local model when available", () => {
		const diagnostics = makeDiagnostics([
			{
				name: "faster-whisper",
				available: true,
				type: "local",
				default_model: "small",
				models: ["small", "medium", "large-v3-turbo"],
				installed_models: ["medium"],
				install_detection: "huggingface-cache",
			},
		]);

		const recommendation = recommendVoiceSetup(diagnostics, "balanced");

		expect(recommendation.mode).toBe("local");
		expect(recommendation.backend).toBe("faster-whisper");
		expect(recommendation.model).toBe("medium");
		expect(recommendation.reason.toLowerCase()).toContain("already installed");
	});
});

describe("model option labeling", () => {
	test("labels installed and recommended models distinctly", () => {
		const backend: BackendAvailability = {
			name: "faster-whisper",
			available: true,
			type: "local",
			default_model: "small",
			models: ["small", "medium"],
			installed_models: ["small"],
			install_detection: "huggingface-cache",
		};

		const options = getModelOptions(backend, "small");

		expect(options).toEqual(["small (recommended, installed)", "medium"]);
	});
});

describe("model readiness", () => {
	test("reports installed when the selected local model is present", () => {
		const backend: BackendAvailability = {
			name: "faster-whisper",
			available: true,
			type: "local",
			default_model: "small",
			models: ["small", "medium"],
			installed_models: ["small"],
			install_detection: "huggingface-cache",
		};

		expect(getModelReadiness(backend, "small")).toBe("installed");
		expect(getModelReadiness(backend, "medium")).toBe("download required");
	});

	test("reports api readiness for cloud backends", () => {
		const backend: BackendAvailability = {
			name: "deepgram",
			available: true,
			type: "cloud",
			default_model: "nova-3",
			models: ["nova-3"],
			installed_models: [],
			install_detection: "api-key",
		};

		expect(getModelReadiness(backend, "nova-3")).toBe("api");
	});

	test("reports unknown when backend is heuristic-only and model presence cannot be confirmed", () => {
		const backend: BackendAvailability = {
			name: "parakeet",
			available: true,
			type: "local",
			default_model: "nvidia/parakeet-tdt-0.6b-v2",
			models: ["nvidia/parakeet-tdt-0.6b-v2"],
			installed_models: [],
			install_detection: "huggingface-or-nemo-cache",
		};

		expect(getModelReadiness(backend, "nvidia/parakeet-tdt-0.6b-v2")).toBe("unknown");
	});
});

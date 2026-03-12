import { describe, expect, test } from "bun:test";
import {
	recommendVoiceSetup,
	type BackendAvailability,
	type DiagnosticsPreference,
	type EnvironmentDiagnostics,
} from "../extensions/voice/diagnostics";

function makeBackends(overrides: Partial<Record<BackendAvailability["name"], Partial<BackendAvailability>>> = {}): BackendAvailability[] {
	const defaults: BackendAvailability[] = [
		{ name: "faster-whisper", available: false, type: "local", default_model: "small", models: ["small"] },
		{ name: "moonshine", available: false, type: "local", default_model: "moonshine/base", models: ["moonshine/base"] },
		{ name: "whisper-cpp", available: false, type: "local", default_model: "small", models: ["small"] },
		{ name: "deepgram", available: false, type: "cloud", default_model: "nova-3", models: ["nova-3"] },
		{ name: "parakeet", available: false, type: "local", default_model: "nvidia/parakeet-tdt-0.6b-v2", models: ["nvidia/parakeet-tdt-0.6b-v2"] },
	];

	return defaults.map((backend) => ({ ...backend, ...(overrides[backend.name] ?? {}) }));
}

function makeDiagnostics(args: Partial<EnvironmentDiagnostics> = {}): EnvironmentDiagnostics {
	return {
		hasPython: true,
		hasSox: true,
		hasHomebrew: true,
		hasDeepgramKey: false,
		backends: makeBackends(),
		issues: [],
		...args,
	};
}

describe("recommendVoiceSetup", () => {
	test("recommends local faster-whisper for privacy when available", () => {
		const diagnostics = makeDiagnostics({
			backends: makeBackends({
				"faster-whisper": { available: true },
			}),
		});

		const recommendation = recommendVoiceSetup(diagnostics, "privacy");

		expect(recommendation.mode).toBe("local");
		expect(recommendation.backend).toBe("faster-whisper");
		expect(recommendation.model).toBe("small");
		expect(recommendation.reason).toContain("privacy");
	});

	test("recommends api mode for speed when deepgram key is available and no local backend exists", () => {
		const diagnostics = makeDiagnostics({
			hasDeepgramKey: true,
			backends: makeBackends({
				deepgram: { available: true },
			}),
		});

		const recommendation = recommendVoiceSetup(diagnostics, "speed");

		expect(recommendation.mode).toBe("api");
		expect(recommendation.backend).toBe("deepgram");
		expect(recommendation.reason).toContain("fastest");
	});

	test("surfaces fixable issues when local mode would need installation", () => {
		const diagnostics = makeDiagnostics({
			hasSox: false,
			backends: makeBackends(),
		});

		const recommendation = recommendVoiceSetup(diagnostics, "privacy");

		expect(recommendation.mode).toBe("local");
		expect(recommendation.backend).toBe("faster-whisper");
		expect(recommendation.fixableIssues).toContain("Install SoX for microphone recording");
		expect(recommendation.fixableIssues).toContain("Install a local STT backend such as faster-whisper");
	});

	test("balanced preference favors existing local backend over cloud", () => {
		const diagnostics = makeDiagnostics({
			hasDeepgramKey: true,
			backends: makeBackends({
				"faster-whisper": { available: true },
				deepgram: { available: true },
			}),
		});

		const recommendation = recommendVoiceSetup(diagnostics, "balanced");

		expect(recommendation.mode).toBe("local");
		expect(recommendation.backend).toBe("faster-whisper");
	});

	test("balanced preference can prefer another local backend when it already has an installed model", () => {
		const diagnostics = makeDiagnostics({
			hasDeepgramKey: true,
			backends: makeBackends({
				"whisper-cpp": { available: true, installed_models: ["small"], install_detection: "whisper-cpp-model-paths" },
				deepgram: { available: true },
			}),
		});

		const recommendation = recommendVoiceSetup(diagnostics, "balanced");

		expect(recommendation.mode).toBe("local");
		expect(recommendation.backend).toBe("whisper-cpp");
		expect(recommendation.model).toBe("small");
		expect(recommendation.reason.toLowerCase()).toContain("already installed");
	});

		test("balanced preference does not over-trust heuristic installed-model signals", () => {
			const diagnostics = makeDiagnostics({
				hasDeepgramKey: true,
				backends: makeBackends({
					parakeet: { available: true, installed_models: ["nvidia/parakeet-tdt-0.6b-v2"], install_detection: "huggingface-or-nemo-cache" },
					deepgram: { available: true },
				}),
			});

			const recommendation = recommendVoiceSetup(diagnostics, "balanced");

			expect(recommendation.mode).toBe("api");
			expect(recommendation.backend).toBe("deepgram");
		});
});

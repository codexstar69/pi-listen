import { describe, expect, test } from "bun:test";
import { buildSelectableBackends, finalizeOnboardingConfig } from "../extensions/voice/onboarding";
import { DEFAULT_CONFIG } from "../extensions/voice/config";
import type { EnvironmentDiagnostics } from "../extensions/voice/diagnostics";

function makeDiagnostics(overrides: Partial<EnvironmentDiagnostics> = {}): EnvironmentDiagnostics {
	return {
		hasPython: true,
		hasSox: true,
		hasHomebrew: true,
		hasDeepgramKey: false,
		backends: [],
		issues: [],
		...overrides,
	};
}

describe("buildSelectableBackends", () => {
	test("provides installable local backend choices even when discovery returns none", () => {
		const diagnostics = makeDiagnostics({ backends: [] });

		const backends = buildSelectableBackends("local", diagnostics);

		expect(backends.map((backend) => backend.name)).toEqual([
			"faster-whisper",
			"moonshine",
			"whisper-cpp",
			"parakeet",
		]);
		expect(backends.every((backend) => backend.available === false)).toBe(true);
	});

	test("preserves discovered local backends when they exist", () => {
		const diagnostics = makeDiagnostics({
			backends: [
				{ name: "faster-whisper", available: true, type: "local", default_model: "small", models: ["small"] },
				{ name: "deepgram", available: true, type: "cloud", default_model: "nova-3", models: ["nova-3"] },
			],
		});

		const backends = buildSelectableBackends("local", diagnostics);

		expect(backends[0]?.name).toBe("faster-whisper");
		expect(backends[0]?.available).toBe(true);
	});

	test("keeps discovered backends first but still offers undiscovered fallback backends", () => {
		const diagnostics = makeDiagnostics({
			backends: [
				{ name: "faster-whisper", available: true, type: "local", default_model: "small", models: ["small"] },
			],
		});

		const backends = buildSelectableBackends("local", diagnostics);

		expect(backends.map((backend) => backend.name)).toEqual([
			"faster-whisper",
			"moonshine",
			"whisper-cpp",
			"parakeet",
		]);
	});
});

describe("finalizeOnboardingConfig", () => {
	test("marks onboarding complete only after validation succeeds", () => {
		const updated = finalizeOnboardingConfig(DEFAULT_CONFIG, { validated: true, source: "setup-command" });

		expect(updated.onboarding.completed).toBe(true);
		expect(updated.onboarding.source).toBe("setup-command");
		expect(updated.onboarding.completedAt).toBeString();
		expect(updated.onboarding.lastValidatedAt).toBeString();
	});

	test("keeps onboarding incomplete when validation is still pending", () => {
		const updated = finalizeOnboardingConfig(DEFAULT_CONFIG, { validated: false, source: "first-run" });

		expect(updated.onboarding.completed).toBe(false);
		expect(updated.onboarding.source).toBe("repair");
		expect(updated.onboarding.completedAt).toBeUndefined();
	});
});

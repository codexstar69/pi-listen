import { describe, expect, test } from "bun:test";
import { finalizeOnboardingConfig } from "../extensions/voice/onboarding";
import { DEFAULT_CONFIG } from "../extensions/voice/config";

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

	test("clears skippedAt when validation succeeds", () => {
		const configWithSkip = {
			...DEFAULT_CONFIG,
			onboarding: {
				...DEFAULT_CONFIG.onboarding,
				skippedAt: "2026-01-01T00:00:00.000Z",
			},
		};
		const updated = finalizeOnboardingConfig(configWithSkip, { validated: true, source: "setup-command" });

		expect(updated.onboarding.completed).toBe(true);
		expect(updated.onboarding.skippedAt).toBeUndefined();
	});

	test("preserves config fields that aren't onboarding-related", () => {
		const customConfig = {
			...DEFAULT_CONFIG,
			language: "es",
			deepgramApiKey: "test-key",
			scope: "project" as const,
		};
		const updated = finalizeOnboardingConfig(customConfig, { validated: true, source: "first-run" });

		expect(updated.language).toBe("es");
		expect(updated.deepgramApiKey).toBe("test-key");
		expect(updated.scope).toBe("project");
		expect(updated.onboarding.completed).toBe(true);
	});

	test("sets schemaVersion to match config version", () => {
		const updated = finalizeOnboardingConfig(DEFAULT_CONFIG, { validated: true, source: "first-run" });

		expect(updated.onboarding.schemaVersion).toBe(DEFAULT_CONFIG.version);
	});
});

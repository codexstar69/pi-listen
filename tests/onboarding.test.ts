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
});

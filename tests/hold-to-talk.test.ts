import { describe, expect, test } from "bun:test";
import {
	getKittyHoldTiming,
	shouldInsertSpaceOnKittyReleaseBeforeRecording,
} from "../extensions/voice/hold-to-talk";

describe("getKittyHoldTiming", () => {
	test("delays warmup briefly but keeps full hold threshold", () => {
		expect(getKittyHoldTiming({ heldMs: 0, intentDelayMs: 200, holdThresholdMs: 1200 })).toEqual({
			warmupDelayMs: 200,
			activationDelayMs: 1200,
		});
	});

	test("starts warmup immediately once intent delay has passed", () => {
		expect(getKittyHoldTiming({ heldMs: 350, intentDelayMs: 200, holdThresholdMs: 1200 })).toEqual({
			warmupDelayMs: 0,
			activationDelayMs: 850,
		});
	});
});

describe("shouldInsertSpaceOnKittyReleaseBeforeRecording", () => {
	test("preserves space for releases before recording starts", () => {
		expect(shouldInsertSpaceOnKittyReleaseBeforeRecording("idle", false)).toBe(true);
		expect(shouldInsertSpaceOnKittyReleaseBeforeRecording("warmup", false)).toBe(true);
	});

	test("does not insert a space once recording has started", () => {
		expect(shouldInsertSpaceOnKittyReleaseBeforeRecording("recording", true)).toBe(false);
	});
});

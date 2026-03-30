import { describe, expect, test } from "bun:test";
import {
	getKittyHoldTiming,
	stripInsertedSpaceOnWarmupStart,
	shouldInsertSpaceOnKittyReleaseBeforeRecording,
	shouldStartRecordingFromWarmup,
} from "../extensions/voice/hold-to-talk";

describe("getKittyHoldTiming", () => {
	test("uses the faster 500ms hold threshold", () => {
		expect(getKittyHoldTiming({ heldMs: 0, intentDelayMs: 200, holdThresholdMs: 500 })).toEqual({
			warmupDelayMs: 200,
			activationDelayMs: 500,
		});
	});

	test("starts warmup immediately once intent delay has passed", () => {
		expect(getKittyHoldTiming({ heldMs: 350, intentDelayMs: 200, holdThresholdMs: 500 })).toEqual({
			warmupDelayMs: 0,
			activationDelayMs: 150,
		});
	});
});

describe("shouldInsertSpaceOnKittyReleaseBeforeRecording", () => {
	test("preserves a space only before warmup has started", () => {
		expect(shouldInsertSpaceOnKittyReleaseBeforeRecording("idle", false)).toBe(true);
		expect(shouldInsertSpaceOnKittyReleaseBeforeRecording("warmup", false)).toBe(false);
	});

	test("does not insert a space once recording has started", () => {
		expect(shouldInsertSpaceOnKittyReleaseBeforeRecording("recording", true)).toBe(false);
	});
});

describe("stripInsertedSpaceOnWarmupStart", () => {
	test("removes the extra trailing space added by the initial press", () => {
		expect(stripInsertedSpaceOnWarmupStart({ textBeforePress: "hello", currentText: "hello " })).toBe("hello");
	});

	test("leaves unrelated editor changes untouched", () => {
		expect(stripInsertedSpaceOnWarmupStart({ textBeforePress: "hello", currentText: "hello world" })).toBe("hello world");
	});
});

describe("shouldStartRecordingFromWarmup", () => {
	test("blocks recording while a kitty warmup release is pending", () => {
		expect(shouldStartRecordingFromWarmup({ state: "warmup", releasePending: true })).toBe(false);
	});

	test("allows recording once warmup is active and no release is pending", () => {
		expect(shouldStartRecordingFromWarmup({ state: "warmup", releasePending: false })).toBe(true);
	});
});

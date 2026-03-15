import { describe, expect, test } from "bun:test";
import {
	initSherpa,
	isSherpaAvailable,
	getSherpaError,
	clearRecognizerCache,
} from "../extensions/voice/sherpa-engine";

// ─── Module loading ──────────────────────────────────────────────────────────

describe("sherpa-onnx-node Bun compatibility", () => {
	test("sherpa-onnx-node module loads via require()", () => {
		const sherpa = require("sherpa-onnx-node");
		expect(sherpa).toBeDefined();
		expect(typeof sherpa.version).toBe("string");
		expect(sherpa.version).toMatch(/^\d+\.\d+\.\d+$/);
	});

	test("sherpa-onnx-node exports OfflineRecognizer constructor", () => {
		const sherpa = require("sherpa-onnx-node");
		expect(typeof sherpa.OfflineRecognizer).toBe("function");
	});

	test("sherpa-onnx-node exports OnlineRecognizer constructor", () => {
		const sherpa = require("sherpa-onnx-node");
		expect(typeof sherpa.OnlineRecognizer).toBe("function");
	});

	test("sherpa-onnx-node loads via ESM dynamic import()", async () => {
		const sherpa = await import("sherpa-onnx-node");
		expect(sherpa).toBeDefined();
		expect(typeof sherpa.OfflineRecognizer).toBe("function");
	});
});

// ─── initSherpa ──────────────────────────────────────────────────────────────

describe("initSherpa", () => {
	test("initializes successfully", async () => {
		const result = await initSherpa();
		expect(result).toBe(true);
	});

	test("isSherpaAvailable returns true after init", async () => {
		await initSherpa();
		expect(isSherpaAvailable()).toBe(true);
	});

	test("getSherpaError returns null on success", async () => {
		await initSherpa();
		expect(getSherpaError()).toBeNull();
	});

	test("initSherpa is idempotent", async () => {
		const first = await initSherpa();
		const second = await initSherpa();
		expect(first).toBe(true);
		expect(second).toBe(true);
	});
});

// ─── Recognizer creation (no model files — error handling) ───────────────────

describe("recognizer error handling", () => {
	test("OfflineRecognizer rejects invalid config gracefully", async () => {
		await initSherpa();
		const sherpa = require("sherpa-onnx-node");

		expect(() => {
			new sherpa.OfflineRecognizer({
				featConfig: { sampleRate: 16000, featureDim: 80 },
				modelConfig: {
					whisper: {
						encoder: "/tmp/nonexistent-encoder.onnx",
						decoder: "/tmp/nonexistent-decoder.onnx",
					},
					tokens: "/tmp/nonexistent-tokens.txt",
					debug: 0,
				},
			});
		}).toThrow();
	});
});

// ─── Cache management ────────────────────────────────────────────────────────

describe("clearRecognizerCache", () => {
	test("clears without error when no recognizer cached", () => {
		expect(() => clearRecognizerCache()).not.toThrow();
	});
});

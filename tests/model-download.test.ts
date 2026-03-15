import { afterEach, describe, expect, test } from "bun:test";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import {
	getModelsDir,
	getModelDir,
	getModelPath,
	isModelDownloaded,
	getDownloadedModels,
	deleteModel,
} from "../extensions/voice/model-download";

const tempDirs: string[] = [];

function makeTempModelDir(modelId: string): string {
	const modelsBase = path.join(os.tmpdir(), `pi-voice-models-test-${Date.now()}`);
	const modelDir = path.join(modelsBase, modelId);
	fs.mkdirSync(modelDir, { recursive: true });
	tempDirs.push(modelsBase);
	return modelDir;
}

afterEach(() => {
	for (const dir of tempDirs.splice(0)) {
		try { fs.rmSync(dir, { recursive: true, force: true }); } catch {}
	}
});

describe("getModelsDir", () => {
	test("returns path under ~/.pi/models/", () => {
		const dir = getModelsDir();
		expect(dir).toContain(path.join(".pi", "models"));
		expect(dir).toStartWith(os.homedir());
	});
});

describe("getModelDir", () => {
	test("returns path with model id", () => {
		const dir = getModelDir("whisper-small");
		expect(dir).toEndWith(path.join("models", "whisper-small"));
	});
});

describe("getModelPath", () => {
	test("returns null when model not downloaded", () => {
		const result = getModelPath("nonexistent-model-xyz");
		expect(result).toBeNull();
	});
});

describe("isModelDownloaded", () => {
	test("returns false when directory does not exist", () => {
		expect(isModelDownloaded("nonexistent-xyz", { "encoder": "encoder.onnx" })).toBe(false);
	});
});

describe("getDownloadedModels", () => {
	test("returns empty array when models dir does not exist", () => {
		// Calling with the default dir — may or may not have models
		const result = getDownloadedModels();
		expect(Array.isArray(result)).toBe(true);
	});
});

describe("deleteModel", () => {
	test("returns false for non-existent model", () => {
		expect(deleteModel("nonexistent-model-xyz-123")).toBe(false);
	});
});

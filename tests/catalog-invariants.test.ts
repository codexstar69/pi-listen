import * as path from "node:path";
import { describe, expect, test } from "bun:test";
import { LOCAL_MODELS } from "../extensions/voice/local";
import { ULTIMATE_FALLBACK_MODEL_ID } from "../extensions/voice/engine-factory";
import { ggufFilenameFor } from "../extensions/voice/engine-whispercpp";

describe("catalog invariants (CI gate)", () => {
	test("LOCAL_MODELS is non-empty", () => {
		expect(LOCAL_MODELS.length).toBeGreaterThan(0);
	});

	test("ULTIMATE_FALLBACK_MODEL_ID exists in LOCAL_MODELS", () => {
		const m = LOCAL_MODELS.find(x => x.id === ULTIMATE_FALLBACK_MODEL_ID);
		expect(m).toBeDefined();
	});

	test("ultimate fallback is platform-agnostic (no `supports` constraint)", () => {
		const m = LOCAL_MODELS.find(x => x.id === ULTIMATE_FALLBACK_MODEL_ID);
		expect(m).toBeDefined();
		const hasConstraint = m!.supports !== undefined && m!.supports.length > 0;
		expect(hasConstraint).toBe(false);
	});

	test("ultimate fallback is a sherpa-onnx model (engine undefined or sherpa-onnx)", () => {
		const m = LOCAL_MODELS.find(x => x.id === ULTIMATE_FALLBACK_MODEL_ID);
		expect(m).toBeDefined();
		const isSherpa = m!.engine === undefined || m!.engine === "sherpa-onnx";
		expect(isSherpa).toBe(true);
	});

	test("each entry's id is unique", () => {
		const ids = LOCAL_MODELS.map(m => m.id);
		expect(new Set(ids).size).toBe(ids.length);
	});

	test("each entry has the legacy sherpaModel field set (placeholder OK for whisper-cpp entries)", () => {
		// LocalModelInfo currently requires `sherpaModel` for shape
		// compatibility with the legacy catalog. whisper-cpp entries stub
		// it with empty `files`/`downloadUrls`; the engine factory routes
		// via `model.engine` so the stub never reaches sherpa-onnx.
		for (const m of LOCAL_MODELS) {
			expect(m.sherpaModel).toBeDefined();
		}
	});
});

describe("Phase 2/3 GGUF catalog entries", () => {
	test("at least one whisper-cpp entry exists (Phase 2 shipped)", () => {
		const wc = LOCAL_MODELS.filter(m => m.engine === "whisper-cpp");
		expect(wc.length).toBeGreaterThan(0);
	});

	test("every whisper-cpp entry has format=gguf + whisperCppModel.files.model + whisperCppModel.downloadUrls.model", () => {
		const wc = LOCAL_MODELS.filter(m => m.engine === "whisper-cpp");
		for (const m of wc) {
			expect(m.format).toBe("gguf");
			expect(m.whisperCppModel).toBeDefined();
			expect(typeof m.whisperCppModel?.files.model).toBe("string");
			expect(m.whisperCppModel?.files.model.length ?? 0).toBeGreaterThan(0);
			expect(typeof m.whisperCppModel?.downloadUrls.model).toBe("string");
			expect(m.whisperCppModel?.downloadUrls.model.length ?? 0).toBeGreaterThan(0);
		}
	});

	test("every whisper-cpp filename passes the path-traversal guard (security CI gate)", () => {
		// THIS IS THE CI GATE for the security/round-7 VETO. A future
		// catalog edit introducing an unsafe filename — `..`, `%2F`,
		// `COM1.bin`, etc. — fails CI before it can ship.
		const wc = LOCAL_MODELS.filter(m => m.engine === "whisper-cpp");
		for (const m of wc) {
			expect(() => ggufFilenameFor(m)).not.toThrow();
		}
	});

	test("every whisper-cpp downloadUrl passes new URL() and basename matches files.model", () => {
		// Mirrors the on-disk write path: model-download.fileNameFromUrl
		// uses path.basename(new URL(url).pathname). On-disk filename MUST
		// equal the catalog's declared `files.model` so the engine finds
		// the file the downloader writes.
		const wc = LOCAL_MODELS.filter(m => m.engine === "whisper-cpp");
		for (const m of wc) {
			const url = m.whisperCppModel!.downloadUrls.model;
			expect(() => new URL(url)).not.toThrow();
			const urlBasename = path.basename(new URL(url).pathname);
			expect(urlBasename).toBe(m.whisperCppModel!.files.model);
		}
	});

	test("Phase-2 default for fresh M-series installs exists (whisper-base-en-metal)", () => {
		// engine-factory.DEFAULT_MODEL_PRIORITY references this id; if it
		// disappears, Apple Silicon installs silently fall to
		// `moonshine-v2-tiny` (still works but loses the Metal speedup).
		const m = LOCAL_MODELS.find(x => x.id === "whisper-base-en-metal");
		expect(m).toBeDefined();
		expect(m?.engine).toBe("whisper-cpp");
	});

	test("Phase-3 CUDA entry exists with requiresCUDA: true on linux x64", () => {
		const m = LOCAL_MODELS.find(x => x.id === "whisper-large-v3-turbo-cuda");
		expect(m).toBeDefined();
		expect(m?.engine).toBe("whisper-cpp");
		const supportsCuda = m?.supports?.some(s =>
			s.platform === "linux" && s.arch === "x64" && s.requiresCUDA === true,
		);
		expect(supportsCuda).toBe(true);
	});
});

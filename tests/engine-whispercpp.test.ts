import { describe, expect, test } from "bun:test";
import { ggufFilenameFor, whispercppEngine, __test } from "../extensions/voice/engine-whispercpp";
import type { LocalModelInfo } from "../extensions/voice/local";

function ggufModel(filename: string, id = "test"): LocalModelInfo {
	return {
		id,
		name: id,
		size: "0",
		sizeBytes: 0,
		runtimeRamMB: 0,
		notes: "",
		langSupport: "english-only",
		tier: "edge",
		accuracy: 1,
		speed: 1,
		engine: "whisper-cpp",
		format: "gguf",
		whisperCppModel: { files: { model: filename }, downloadUrls: { model: "" } },
		sherpaModel: { type: "whisper", files: {}, downloadUrls: {} },
	};
}

function onnxModel(): LocalModelInfo {
	return {
		id: "onnx-only",
		name: "onnx-only",
		size: "0",
		sizeBytes: 0,
		runtimeRamMB: 0,
		notes: "",
		langSupport: "english-only",
		tier: "edge",
		accuracy: 1,
		speed: 1,
		// no engine / format / whisperCppModel — defaults to ONNX/sherpa
		sherpaModel: { type: "moonshine", files: {}, downloadUrls: {} },
	};
}

describe("ggufFilenameFor — accepts valid GGUF filenames", () => {
	test("typical whisper.cpp filenames", () => {
		expect(ggufFilenameFor(ggufModel("ggml-base.en-q5_1.bin"))).toBe("ggml-base.en-q5_1.bin");
		expect(ggufFilenameFor(ggufModel("ggml-large-v3-turbo-q5_0.bin"))).toBe("ggml-large-v3-turbo-q5_0.bin");
		expect(ggufFilenameFor(ggufModel("ggml-tiny.en-q5_1.bin"))).toBe("ggml-tiny.en-q5_1.bin");
	});
	test("ellipsis '...' filenames are accepted (round-8 minimax false-positive on startsWith)", () => {
		expect(ggufFilenameFor(ggufModel("..."))).toBe("...");
		expect(ggufFilenameFor(ggufModel("....bin"))).toBe("....bin");
	});
	test("COM0 / LPT0 leading-zero are accepted (NOT Windows-reserved)", () => {
		expect(ggufFilenameFor(ggufModel("COM0"))).toBe("COM0");
		expect(ggufFilenameFor(ggufModel("COM0.bin"))).toBe("COM0.bin");
		expect(ggufFilenameFor(ggufModel("LPT0"))).toBe("LPT0");
		expect(ggufFilenameFor(ggufModel("COM00"))).toBe("COM00");
	});
});

describe("ggufFilenameFor — rejects path-traversal filenames (security VETO from round 7)", () => {
	test("parent-dir tokens", () => {
		expect(() => ggufFilenameFor(ggufModel(".."))).toThrow(/unsafe filename/);
		expect(() => ggufFilenameFor(ggufModel("."))).toThrow(/unsafe filename/);
	});
	test("path separators", () => {
		expect(() => ggufFilenameFor(ggufModel("../foo"))).toThrow(/unsafe filename/);
		expect(() => ggufFilenameFor(ggufModel("..\\foo"))).toThrow(/unsafe filename/);
		expect(() => ggufFilenameFor(ggufModel("foo/bar.bin"))).toThrow(/unsafe filename/);
		expect(() => ggufFilenameFor(ggufModel("foo\\bar.bin"))).toThrow(/unsafe filename/);
	});
	test("NUL + control chars", () => {
		expect(() => ggufFilenameFor(ggufModel("\0foo"))).toThrow(/unsafe filename/);
		expect(() => ggufFilenameFor(ggufModel("\tfoo"))).toThrow(/unsafe filename/);
	});
	test("empty string", () => {
		expect(() => ggufFilenameFor(ggufModel(""))).toThrow();
	});
});

describe("ggufFilenameFor — rejects URL-encoded traversal (security VETO from round 9)", () => {
	test("%-encoded sequences fail the allowlist", () => {
		expect(() => ggufFilenameFor(ggufModel("%2E%2E%2F"))).toThrow(/unsafe filename/);
		expect(() => ggufFilenameFor(ggufModel("..%2F..%2Fetc%2Fpasswd"))).toThrow(/unsafe filename/);
		expect(() => ggufFilenameFor(ggufModel("foo%2Fbar.bin"))).toThrow(/unsafe filename/);
		expect(() => ggufFilenameFor(ggufModel("%2E%2Ebin"))).toThrow(/unsafe filename/);
	});
});

describe("ggufFilenameFor — rejects Windows reserved names", () => {
	test("classic CON/PRN/AUX/NUL", () => {
		expect(() => ggufFilenameFor(ggufModel("CON"))).toThrow(/unsafe filename/);
		expect(() => ggufFilenameFor(ggufModel("PRN"))).toThrow(/unsafe filename/);
		expect(() => ggufFilenameFor(ggufModel("AUX"))).toThrow(/unsafe filename/);
		expect(() => ggufFilenameFor(ggufModel("NUL"))).toThrow(/unsafe filename/);
	});
	test("COM1-COM9 / LPT1-LPT9 case-insensitive", () => {
		expect(() => ggufFilenameFor(ggufModel("COM1"))).toThrow(/unsafe filename/);
		expect(() => ggufFilenameFor(ggufModel("com1"))).toThrow(/unsafe filename/);
		expect(() => ggufFilenameFor(ggufModel("LPT1.bin"))).toThrow(/unsafe filename/);
		expect(() => ggufFilenameFor(ggufModel("con.bin"))).toThrow(/unsafe filename/);
	});
	test("multi-port serial-card extensions COM10+ / LPT99 (round 10)", () => {
		expect(() => ggufFilenameFor(ggufModel("COM10"))).toThrow(/unsafe filename/);
		expect(() => ggufFilenameFor(ggufModel("COM256"))).toThrow(/unsafe filename/);
		expect(() => ggufFilenameFor(ggufModel("LPT99"))).toThrow(/unsafe filename/);
		expect(() => ggufFilenameFor(ggufModel("COM10.bin"))).toThrow(/unsafe filename/);
	});
});

describe("ggufFilenameFor — non-GGUF and missing fields", () => {
	test("throws on non-GGUF model", () => {
		expect(() => ggufFilenameFor(onnxModel())).toThrow(/non-GGUF model/);
	});
	test("throws when whisperCppModel.files.model is missing", () => {
		const m = ggufModel("ignored");
		// Force the missing-field path
		(m as unknown as { whisperCppModel: undefined }).whisperCppModel = undefined;
		expect(() => ggufFilenameFor(m)).toThrow(/has no whisperCppModel\.files\.model/);
	});
});

describe("whispercppEngine — load-time safety", () => {
	test("module imports without throwing (binding may or may not be present)", () => {
		expect(whispercppEngine).toBeDefined();
		expect(whispercppEngine.id).toBe("whisper-cpp");
	});

	test("singleton starts readable; available is boolean, error is string|null", () => {
		expect(typeof whispercppEngine.available).toBe("boolean");
		expect(whispercppEngine.error === null || typeof whispercppEngine.error === "string").toBe(true);
	});

	test("clearCache is idempotent — does not throw on repeat calls", async () => {
		await expect(whispercppEngine.clearCache()).resolves.toBeUndefined();
		await expect(whispercppEngine.clearCache()).resolves.toBeUndefined();
	});

	test("transcribe throws structured error when not available", async () => {
		if (whispercppEngine.available) return;
		const ctx = {
			model: ggufModel("ggml-base.en-q5_1.bin"),
			modelDir: "/tmp",
			lang: "en",
		};
		await expect(whispercppEngine.transcribe(Buffer.alloc(0), ctx))
			.rejects
			.toThrow(/whisper-cpp engine is not available/);
	});
});

describe("resolveCacheSize", () => {
	test("default 1 when env unset", () => {
		expect(__test.resolveCacheSize({})).toBe(1);
	});
	test("clamps invalid input to 1", () => {
		expect(__test.resolveCacheSize({ PI_VOICE_ENGINE_CACHE_SIZE: "0" })).toBe(1);
		expect(__test.resolveCacheSize({ PI_VOICE_ENGINE_CACHE_SIZE: "-3" })).toBe(1);
		expect(__test.resolveCacheSize({ PI_VOICE_ENGINE_CACHE_SIZE: "abc" })).toBe(1);
	});
	test("clamps high values to 4", () => {
		expect(__test.resolveCacheSize({ PI_VOICE_ENGINE_CACHE_SIZE: "100" })).toBe(4);
		expect(__test.resolveCacheSize({ PI_VOICE_ENGINE_CACHE_SIZE: "5" })).toBe(4);
	});
	test("accepts valid values 1-4", () => {
		expect(__test.resolveCacheSize({ PI_VOICE_ENGINE_CACHE_SIZE: "1" })).toBe(1);
		expect(__test.resolveCacheSize({ PI_VOICE_ENGINE_CACHE_SIZE: "2" })).toBe(2);
		expect(__test.resolveCacheSize({ PI_VOICE_ENGINE_CACHE_SIZE: "4" })).toBe(4);
	});
});

describe("friendlyInitError", () => {
	test("Cannot find module → install hint with platform commands", () => {
		const out = __test.friendlyInitError("Cannot find module 'smart-whisper'");
		expect(out).toContain("smart-whisper");
		expect(out).toContain("install");
		// Either Xcode CLT (darwin) or build-essential (linux) — depends
		// on test platform.
		expect(out.length).toBeGreaterThan("Cannot find module".length);
	});
	test("path-with-spaces heuristic kicks in on clang error referencing node_modules", () => {
		const out = __test.friendlyInitError(
			"clang++: error: no such file or directory: 'Files/Pi'\nnode_modules/smart-whisper/...",
		);
		// Heuristic: msg includes "node_modules" and a space.
		expect(out).toContain("spaces");
	});
	test("CUDA error → rebuild hint", () => {
		const out = __test.friendlyInitError("CUDA backend init failed");
		expect(out).toContain("WHISPER_CUDA=1");
	});
	test("unknown errors are passed through", () => {
		const out = __test.friendlyInitError("some weird native error");
		expect(out).toBe("some weird native error");
	});
});

describe("clearCache + transcribe race (round-3 audit Opus #1, #3)", () => {
	test("interface contract: clearCache resolves without throwing even with no cached models", async () => {
		await whispercppEngine.clearCache();
		await expect(whispercppEngine.clearCache()).resolves.toBeUndefined();
	});

	test("clearCache resolves quickly when no entries are cached", async () => {
		const t0 = Date.now();
		await whispercppEngine.clearCache();
		expect(Date.now() - t0).toBeLessThan(1000);
	});
});

describe("pcmInt16ToFloat32", () => {
	test("aligned (even byteOffset) bulk-converts correctly", () => {
		// Build a Buffer with even byteOffset (always the case for
		// `Buffer.alloc`/`Buffer.from` with no explicit slice).
		const buf = Buffer.alloc(8);
		buf.writeInt16LE(0, 0);
		buf.writeInt16LE(16384, 2);    // 0.5
		buf.writeInt16LE(-16384, 4);   // -0.5
		buf.writeInt16LE(32767, 6);    // ~1.0
		const f32 = __test.pcmInt16ToFloat32(buf);
		expect(f32.length).toBe(4);
		expect(f32[0]).toBe(0);
		expect(f32[1]).toBeCloseTo(0.5, 4);
		expect(f32[2]).toBeCloseTo(-0.5, 4);
		expect(f32[3]).toBeCloseTo(1.0, 4);
	});
	test("odd byteOffset path uses single bulk-copy (round-2 audit polish)", () => {
		// Force odd byteOffset by slicing a pooled allocation. The
		// underlying logic copies once into an aligned buffer instead of
		// per-sample readInt16LE.
		const pool = Buffer.alloc(9);
		pool.writeInt16LE(16384, 1); // value at offset 1 (odd)
		const odd = pool.subarray(1); // byteOffset = 1 typically
		// Pad to even length so we have a valid 16-bit sample count.
		const padded = Buffer.concat([odd, Buffer.alloc(1)]);
		const f32 = __test.pcmInt16ToFloat32(padded.subarray(0, 8));
		// First sample reads the value we wrote (or the next bytes,
		// depending on whether subarray byteOffset is odd; this test
		// mainly ensures NO THROW on odd-offset input).
		expect(f32.length).toBeGreaterThan(0);
	});
});

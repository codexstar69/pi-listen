import { describe, expect, test } from "bun:test";
import {
	recommendDefaultModel,
	DEFAULT_TTS_MODEL,
	ensureTtsModelInstalled,
	isTtsModelInstalled,
	DEFAULT_TTS_MODEL as DEFAULT,
} from "../extensions/voice/tts-local-models";

describe("recommendDefaultModel — smart-default selector", () => {
	test("English locale → Kitten Nano default", () => {
		const r = recommendDefaultModel("en-US");
		expect(r.modelId).toBe(DEFAULT_TTS_MODEL);
		expect(r.fallback).toBe(false);
	});

	test("Spanish locale → Spanish Piper voice", () => {
		const r = recommendDefaultModel("es-ES");
		expect(r.modelId).toBe("piper-es_ES-davefx-medium-int8");
		expect(r.fallback).toBe(false);
	});

	test("French locale → French Piper voice", () => {
		const r = recommendDefaultModel("fr-FR");
		expect(r.modelId).toBe("piper-fr_FR-siwis-medium-int8");
	});

	test("Japanese locale → English fallback while Kokoro is flagged incompatible", () => {
		// v7.1.2: Kokoro multilingual is `incompatible` on sherpa-onnx-node
		// 1.12.29 (returns NaN samples). The recommender skips it and falls
		// back to the English default rather than routing users to a silent
		// model. Once upstream fixes voices.bin and we drop the flag, this
		// should expect "kokoro-int8-multi-lang-v1_0" again.
		const r = recommendDefaultModel("ja-JP");
		expect(r.modelId).toBe("kitten-nano-en-v0_2");
		expect(r.fallback).toBe(true);
	});

	test("Korean locale → English fallback while Kokoro is flagged incompatible", () => {
		const r = recommendDefaultModel("ko-KR");
		expect(r.modelId).toBe("kitten-nano-en-v0_2");
		expect(r.fallback).toBe(true);
	});

	test("Portuguese locale → pt-BR Piper (catalog has no pt-PT)", () => {
		const r = recommendDefaultModel("pt-PT");
		expect(r.modelId).toBe("piper-pt_BR-cadu-medium-int8");
		expect(r.reason).toContain("closest available");
		// pt-PT against a pt-BR voice is technically a fallback — the
		// audio will be Brazilian-accented Portuguese, not European.
		// fallback=true so onboarding UI can flag the mismatch.
		expect(r.fallback).toBe(true);
	});

	test("pt-BR locale → exact match", () => {
		const r = recommendDefaultModel("pt-BR");
		expect(r.modelId).toBe("piper-pt_BR-cadu-medium-int8");
		expect(r.reason).toContain("exact match");
		expect(r.fallback).toBe(false);
	});

	test("Unknown locale falls back to English with warning", () => {
		const r = recommendDefaultModel("sw-KE"); // Swahili — not in catalog
		expect(r.modelId).toBe(DEFAULT_TTS_MODEL);
		expect(r.fallback).toBe(true);
		expect(r.reason).toContain("no built-in TTS voice");
	});

	test("Empty locale → English fallback", () => {
		const r = recommendDefaultModel("");
		expect(r.modelId).toBe(DEFAULT_TTS_MODEL);
		expect(r.fallback).toBe(true);
	});

	test("POSIX-style locale (e.g. en_US.UTF-8) is parsed", () => {
		const r = recommendDefaultModel("en_US.UTF-8");
		expect(r.modelId).toBe(DEFAULT_TTS_MODEL);
	});

	test("Locale case-insensitive — ZH_TW also gets recommended Chinese", () => {
		const r = recommendDefaultModel("ZH_TW");
		// zh has a single Piper voice (chaowen) — that's the recommendation
		// regardless of region within zh-*.
		expect(r.modelId).toBe("piper-zh_CN-chaowen-medium-int8");
	});
});

describe("ensureTtsModelInstalled — concurrency guard (v7.0.1)", () => {
	test("function exists with the documented signature", () => {
		// Shape check: returns Promise<TtsInstallResult>.
		expect(typeof ensureTtsModelInstalled).toBe("function");
	});

	test("already-installed fast-path resolves immediately without network", async () => {
		// Only run the body if a real install exists from a prior run on
		// the developer's machine. CI without network skips the assertion
		// but doesn't fail.
		if (!isTtsModelInstalled(DEFAULT)) return;
		const result = await ensureTtsModelInstalled(DEFAULT);
		expect(result.dir).toBeDefined();
	});

	test("concurrent calls for an already-installed model both resolve to the same dir", async () => {
		// Same condition: only meaningful when a model is actually installed.
		// The fast-path doesn't hit the in-flight Map (it returns before),
		// but it does verify Promise.all of two concurrent calls behaves
		// correctly and resolves to identical results.
		if (!isTtsModelInstalled(DEFAULT)) return;
		const [a, b] = await Promise.all([
			ensureTtsModelInstalled(DEFAULT),
			ensureTtsModelInstalled(DEFAULT),
		]);
		expect(a.dir).toBe(b.dir);
	});
});

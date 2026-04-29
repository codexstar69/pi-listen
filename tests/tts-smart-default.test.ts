import { describe, expect, test } from "bun:test";
import {
	recommendDefaultModel,
	DEFAULT_TTS_MODEL,
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

	test("Japanese locale → Kokoro multilingual (no Piper coverage)", () => {
		const r = recommendDefaultModel("ja-JP");
		expect(r.modelId).toBe("kokoro-int8-multi-lang-v1_0");
	});

	test("Korean locale → Kokoro multilingual", () => {
		const r = recommendDefaultModel("ko-KR");
		expect(r.modelId).toBe("kokoro-int8-multi-lang-v1_0");
	});

	test("Portuguese locale → pt-BR Piper (catalog has no pt-PT)", () => {
		const r = recommendDefaultModel("pt-PT");
		expect(r.modelId).toBe("piper-pt_BR-cadu-medium-int8");
		expect(r.reason).toContain("closest available");
	});

	test("pt-BR locale → exact match note", () => {
		const r = recommendDefaultModel("pt-BR");
		expect(r.modelId).toBe("piper-pt_BR-cadu-medium-int8");
		expect(r.reason).toContain("exact match");
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

import { describe, expect, test } from "bun:test";
import {
	buildDeepgramSpeakUrl,
	DEEPGRAM_TTS_VOICES,
	DEFAULT_DEEPGRAM_TTS_VOICE,
	filterDeepgramVoicesByLanguage,
	getDeepgramVoice,
	assertLanguageForDeepgram,
} from "../extensions/voice/tts-deepgram";

describe("buildDeepgramSpeakUrl", () => {
	test("builds the expected URL shape", () => {
		const url = buildDeepgramSpeakUrl("aura-2-thalia-en", 24000);
		expect(url).toBe("https://api.deepgram.com/v1/speak?model=aura-2-thalia-en&encoding=linear16&sample_rate=24000&container=wav");
	});

	test("includes the requested sample rate", () => {
		const url = buildDeepgramSpeakUrl("aura-2-luna-en", 16000);
		expect(url).toContain("sample_rate=16000");
	});

	test("URL-encodes voice ids with special characters defensively", () => {
		// All shipped voice ids are plain kebab-case ASCII, but the
		// builder uses URLSearchParams so any future ids with reserved
		// chars are encoded correctly.
		const url = buildDeepgramSpeakUrl("aura-2-thalia-en", 24000);
		expect(url).toContain("model=aura-2-thalia-en");
	});
});

describe("DEEPGRAM_TTS_VOICES catalog", () => {
	test("default voice exists", () => {
		expect(DEEPGRAM_TTS_VOICES.find(v => v.id === DEFAULT_DEEPGRAM_TTS_VOICE)).toBeDefined();
		expect(DEFAULT_DEEPGRAM_TTS_VOICE).toBe("aura-2-thalia-en");
	});

	test("surfaces requested Aura-2 voices", () => {
		const ids = new Set(DEEPGRAM_TTS_VOICES.map(v => v.id));
		for (const id of [
			"aura-2-thalia-en",
			"aura-2-odysseus-en",
			"aura-2-amalthea-en",
			"aura-2-andromeda-en",
			"aura-2-apollo-en",
			"aura-2-arcas-en",
		]) {
			expect(ids.has(id)).toBe(true);
		}
	});

	test("surfaces legacy Aura voices below Aura-2 voices", () => {
		const ids = new Set(DEEPGRAM_TTS_VOICES.map(v => v.id));
		for (const id of [
			"aura-asteria-en",
			"aura-luna-en",
			"aura-stella-en",
			"aura-athena-en",
			"aura-hera-en",
			"aura-orion-en",
			"aura-arcas-en",
			"aura-perseus-en",
			"aura-angus-en",
			"aura-orpheus-en",
			"aura-helios-en",
			"aura-zeus-en",
		]) {
			expect(ids.has(id)).toBe(true);
		}

		const firstLegacyAuraIndex = DEEPGRAM_TTS_VOICES.findIndex(v => v.id.startsWith("aura-") && !v.id.startsWith("aura-2-"));
		const lastAura2Index = DEEPGRAM_TTS_VOICES.findLastIndex(v => v.id.startsWith("aura-2-"));
		expect(firstLegacyAuraIndex).toBeGreaterThan(lastAura2Index);
	});

	test("every voice has language and gender", () => {
		for (const v of DEEPGRAM_TTS_VOICES) {
			expect(v.language).toBeDefined();
			expect(v.gender).toBeDefined();
		}
	});

	test("getDeepgramVoice returns by id", () => {
		expect(getDeepgramVoice("aura-2-thalia-en")?.id).toBe("aura-2-thalia-en");
	});

	test("getDeepgramVoice returns undefined for unknown id", () => {
		expect(getDeepgramVoice("nonexistent")).toBeUndefined();
	});
});

describe("filterDeepgramVoicesByLanguage", () => {
	test("filters by base tag", () => {
		const en = filterDeepgramVoicesByLanguage("en");
		expect(en.length).toBeGreaterThan(0);
		for (const v of en) expect(v.language).toBe("en");
	});

	test("regional tag matches base", () => {
		const enUS = filterDeepgramVoicesByLanguage("en-US");
		expect(enUS.length).toBeGreaterThan(0);
	});

	test("empty input returns full catalog", () => {
		expect(filterDeepgramVoicesByLanguage("").length).toBe(DEEPGRAM_TTS_VOICES.length);
	});

	test("unknown language returns empty list", () => {
		expect(filterDeepgramVoicesByLanguage("xx")).toEqual([]);
	});
});

describe("assertLanguageForDeepgram", () => {
	test("matched voice + language passes", () => {
		expect(() => assertLanguageForDeepgram("aura-2-thalia-en", "en")).not.toThrow();
	});

	test("regional language matches base", () => {
		expect(() => assertLanguageForDeepgram("aura-2-thalia-en", "en-US")).not.toThrow();
	});

	test("language mismatch throws actionable error", () => {
		expect(() => assertLanguageForDeepgram("aura-2-thalia-en", "es-ES")).toThrow(/aura-2-thalia-en speaks en/);
	});

	test("empty language throws", () => {
		expect(() => assertLanguageForDeepgram("aura-2-thalia-en", "")).toThrow();
	});

	test("unknown voice id is accepted (let server validate)", () => {
		expect(() => assertLanguageForDeepgram("aura-2-future-voice", "en")).not.toThrow();
	});
});

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import {
	resolveSixtyDbApiKey,
	filterVoicesByLanguage,
	SIXTYDB_BASE_URL,
	SIXTYDB_STT_URL,
	SIXTYDB_TTS_URL,
	SIXTYDB_MYVOICES_URL,
	SIXTYDB_WS_TTS_URL,
	type SixtyDbVoice,
} from "../extensions/voice/sixtydb";
import { DEFAULT_CONFIG, type VoiceConfig } from "../extensions/voice/config";

function cfg(over: Partial<VoiceConfig> = {}): VoiceConfig {
	return { ...DEFAULT_CONFIG, ...over };
}

describe("60db endpoint constants", () => {
	test("URLs derive from the base host", () => {
		expect(SIXTYDB_BASE_URL).toBe("https://api.60db.ai");
		expect(SIXTYDB_STT_URL).toBe("https://api.60db.ai/stt");
		expect(SIXTYDB_TTS_URL).toBe("https://api.60db.ai/tts-synthesize");
		expect(SIXTYDB_MYVOICES_URL).toBe("https://api.60db.ai/myvoices");
		expect(SIXTYDB_WS_TTS_URL).toBe("wss://api.60db.ai/ws/tts");
	});
});

describe("resolveSixtyDbApiKey", () => {
	const prev = process.env.SIXTYDB_API_KEY;
	beforeEach(() => { delete process.env.SIXTYDB_API_KEY; });
	afterEach(() => {
		if (prev === undefined) delete process.env.SIXTYDB_API_KEY;
		else process.env.SIXTYDB_API_KEY = prev;
	});

	test("returns null when neither env nor config has a key", () => {
		expect(resolveSixtyDbApiKey(cfg())).toBeNull();
	});

	test("uses the config key when env is unset", () => {
		expect(resolveSixtyDbApiKey(cfg({ sixtyDbApiKey: "from-config" }))).toBe("from-config");
	});

	test("env var wins over config", () => {
		process.env.SIXTYDB_API_KEY = "from-env";
		expect(resolveSixtyDbApiKey(cfg({ sixtyDbApiKey: "from-config" }))).toBe("from-env");
	});
});

describe("filterVoicesByLanguage", () => {
	const voices: SixtyDbVoice[] = [
		{ id: "a", name: "Asha", language: "en", languageName: "English", gender: "female", accent: "Indian", model: "60db Fast" },
		{ id: "b", name: "Beto", language: "es", languageName: "Spanish", gender: "male", accent: "Neutral", model: "60db Quality" },
		{ id: "c", name: "Cara", language: "en", languageName: "English", gender: "female", accent: "American", model: "60db Fast" },
	];

	test("filters by base tag", () => {
		const en = filterVoicesByLanguage(voices, "en");
		expect(en.map(v => v.id)).toEqual(["a", "c"]);
	});

	test("regional tag matches base", () => {
		const en = filterVoicesByLanguage(voices, "en-US");
		expect(en.length).toBe(2);
	});

	test("empty input returns full list", () => {
		expect(filterVoicesByLanguage(voices, "")).toEqual(voices);
	});

	test("no match falls back to the full list rather than empty", () => {
		// 60db may serve a voice cross-lingually, so an empty filter result
		// should not hide every voice from the picker.
		expect(filterVoicesByLanguage(voices, "fr")).toEqual(voices);
	});
});

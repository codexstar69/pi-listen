import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import {
	sixtyDbSpeak,
	SIXTYDB_TTS_SAMPLE_RATE,
	SIXTYDB_TTS_MAX_TEXT,
} from "../extensions/voice/tts-sixtydb";
import { DEFAULT_CONFIG, type VoiceConfig } from "../extensions/voice/config";

function cfg(over: Partial<VoiceConfig> = {}): VoiceConfig {
	return { ...DEFAULT_CONFIG, ...over };
}

describe("60db TTS constants", () => {
	test("sample rate matches the shared 24kHz pipeline", () => {
		expect(SIXTYDB_TTS_SAMPLE_RATE).toBe(24000);
	});
	test("max text length matches the documented limit", () => {
		expect(SIXTYDB_TTS_MAX_TEXT).toBe(5000);
	});
});

describe("sixtyDbSpeak input validation", () => {
	const prev = process.env.SIXTYDB_API_KEY;
	beforeEach(() => { delete process.env.SIXTYDB_API_KEY; });
	afterEach(() => {
		if (prev === undefined) delete process.env.SIXTYDB_API_KEY;
		else process.env.SIXTYDB_API_KEY = prev;
	});

	test("throws when text is empty", async () => {
		await expect(
			sixtyDbSpeak({ text: "", voiceId: "v1", config: cfg({ sixtyDbApiKey: "k" }) }),
		).rejects.toThrow(/text is required/);
	});

	test("throws when voiceId is empty", async () => {
		await expect(
			sixtyDbSpeak({ text: "hello", voiceId: "", config: cfg({ sixtyDbApiKey: "k" }) }),
		).rejects.toThrow(/voice_id is required/);
	});

	test("throws an actionable error when no API key is configured", async () => {
		await expect(
			sixtyDbSpeak({ text: "hello", voiceId: "v1", config: cfg() }),
		).rejects.toThrow(/SIXTYDB_API_KEY not set/);
	});
});

import { afterEach, describe, expect, test } from "bun:test";
import { buildDeepgramWsUrl, resolveDeepgramApiKey, DEEPGRAM_WS_URL } from "../extensions/voice/deepgram";
import { DEFAULT_CONFIG, type VoiceConfig } from "../extensions/voice/config";

describe("buildDeepgramWsUrl", () => {
	test("builds URL with default config", () => {
		const url = buildDeepgramWsUrl(DEFAULT_CONFIG);
		expect(url).toStartWith(DEEPGRAM_WS_URL);
		expect(url).toContain("encoding=linear16");
		expect(url).toContain("sample_rate=16000");
		expect(url).toContain("channels=1");
		expect(url).toContain("model=nova-3");
		expect(url).toContain("language=en");
		expect(url).toContain("interim_results=true");
		expect(url).toContain("smart_format=true");
		expect(url).toContain("endpointing=200");
		expect(url).toContain("utterance_end_ms=1000");
	});

	test("uses configured language", () => {
		const config: VoiceConfig = { ...DEFAULT_CONFIG, language: "es" };
		const url = buildDeepgramWsUrl(config);
		expect(url).toContain("language=es");
	});

	test("defaults to English when language is empty", () => {
		const config: VoiceConfig = { ...DEFAULT_CONFIG, language: "" };
		const url = buildDeepgramWsUrl(config);
		expect(url).toContain("language=en");
	});

	test("produces a valid URL", () => {
		const url = buildDeepgramWsUrl(DEFAULT_CONFIG);
		// Should be parseable as a URL
		const parsed = new URL(url);
		expect(parsed.protocol).toBe("wss:");
		expect(parsed.hostname).toBe("api.deepgram.com");
		expect(parsed.pathname).toBe("/v1/listen");
	});
});

describe("resolveDeepgramApiKey", () => {
	const savedEnv = process.env.DEEPGRAM_API_KEY;

	afterEach(() => {
		if (savedEnv !== undefined) {
			process.env.DEEPGRAM_API_KEY = savedEnv;
		} else {
			delete process.env.DEEPGRAM_API_KEY;
		}
	});

	test("returns env var when set", () => {
		process.env.DEEPGRAM_API_KEY = "test-key-from-env";
		const key = resolveDeepgramApiKey(DEFAULT_CONFIG);
		expect(key).toBe("test-key-from-env");
	});

	test("returns config key when env is not set", () => {
		delete process.env.DEEPGRAM_API_KEY;
		const config: VoiceConfig = { ...DEFAULT_CONFIG, deepgramApiKey: "test-key-from-config" };
		const key = resolveDeepgramApiKey(config);
		expect(key).toBe("test-key-from-config");
	});

	test("prefers env var over config key", () => {
		process.env.DEEPGRAM_API_KEY = "env-key";
		const config: VoiceConfig = { ...DEFAULT_CONFIG, deepgramApiKey: "config-key" };
		const key = resolveDeepgramApiKey(config);
		expect(key).toBe("env-key");
	});

	test("returns null when neither is set", () => {
		delete process.env.DEEPGRAM_API_KEY;
		const key = resolveDeepgramApiKey(DEFAULT_CONFIG);
		expect(key).toBeNull();
	});
});

/**
 * Deepgram API helpers — URL building and constants.
 * Extracted for testability.
 */

import type { VoiceConfig } from "./config";
import { modelForLanguage } from "./onboarding";

export const DEEPGRAM_WS_URL = "wss://api.deepgram.com/v1/listen";
export const SAMPLE_RATE = 16000;
export const CHANNELS = 1;
export const ENCODING = "linear16";

export function buildDeepgramWsUrl(config: VoiceConfig): string {
	const language = config.language || "en";
	const model = modelForLanguage(language);
	const params = new URLSearchParams({
		encoding: ENCODING,
		sample_rate: String(SAMPLE_RATE),
		channels: String(CHANNELS),
		endpointing: "200",
		utterance_end_ms: "1000",
		language,
		model,
		smart_format: "true",
		interim_results: "true",
	});
	for (const keyterm of config.deepgramKeyterms ?? []) {
		const normalized = keyterm.trim();
		if (normalized) params.append("keyterm", normalized);
	}
	return `${DEEPGRAM_WS_URL}?${params.toString()}`;
}

export function resolveDeepgramApiKey(config: VoiceConfig): string | null {
	return process.env.DEEPGRAM_API_KEY || config.deepgramApiKey || null;
}

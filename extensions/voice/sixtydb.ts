/**
 * 60db API helpers — base URLs, API-key resolution, and the account voice
 * catalog (`GET /myvoices`). Extracted for testability and to keep the
 * 60db-specific surface in one place, mirroring `./deepgram.ts`.
 *
 * Auth model (differs from Deepgram's `Token`):
 *   - REST endpoints:  `Authorization: Bearer <api-key>` header
 *   - WebSocket TTS:   `?apiKey=<api-key>` query parameter
 *
 * One `SIXTYDB_API_KEY` (env var, falling back to `config.sixtyDbApiKey`)
 * drives both STT (`POST /stt`) and TTS (`/tts-synthesize`, `/ws/tts`).
 */

import type { VoiceConfig } from "./config";

export const SIXTYDB_BASE_URL = "https://api.60db.ai";
export const SIXTYDB_STT_URL = `${SIXTYDB_BASE_URL}/stt`;
export const SIXTYDB_TTS_URL = `${SIXTYDB_BASE_URL}/tts-synthesize`;
export const SIXTYDB_MYVOICES_URL = `${SIXTYDB_BASE_URL}/myvoices`;
export const SIXTYDB_WS_TTS_URL = "wss://api.60db.ai/ws/tts";

/**
 * Resolve the 60db API key. Env var wins over persisted config so a user
 * can override per-shell without editing settings, exactly like
 * `resolveDeepgramApiKey`.
 */
export function resolveSixtyDbApiKey(config: VoiceConfig): string | null {
	return process.env.SIXTYDB_API_KEY || config.sixtyDbApiKey || null;
}

// ─── Account voice catalog (GET /myvoices) ──────────────────────────────────────

/**
 * A 60db voice as surfaced to the picker. Flattened from the API's nested
 * `labels` object so callers don't have to reach through `labels.*`.
 */
export interface SixtyDbVoice {
	id: string;          // voice_id (UUID)
	name: string;
	language: string;    // labels.language — ISO code ("en", "hi", …)
	languageName: string;
	gender: string;      // "male" | "female" | "" (unknown)
	accent: string;      // "American" | "British" | "Indian" | "Neutral" | ""
	model: string;       // "60db Fast" | "60db Quality"
}

interface MyVoicesResponse {
	success?: boolean;
	message?: string;
	data?: Array<{
		voice_id?: string;
		name?: string;
		model?: string;
		labels?: {
			language?: string;
			language_name?: string;
			gender?: string;
			accent?: string;
		};
	}>;
}

/**
 * In-memory cache. 60db voices are account-specific and change rarely
 * (a user clones a voice, then uses it for weeks), so a short TTL keeps the
 * settings panel snappy on repeat opens without holding a stale list across
 * sessions. Keyed by API key so switching keys doesn't serve another
 * account's voices.
 */
interface VoicesCacheEntry {
	voices: SixtyDbVoice[];
	fetchedAtMs: number;
	apiKey: string;
}
let _voicesCache: VoicesCacheEntry | null = null;

/** Cache lifetime (ms). 5 minutes — long enough for a settings session. */
const VOICES_CACHE_TTL_MS = 5 * 60_000;

/** Maximum response size for /myvoices — defends against a runaway body. */
const MYVOICES_MAX_BYTES = 5_000_000; // ~5 MB of JSON is already absurd

/** Drop the cached voice list — call after the user changes the API key. */
export function clearVoicesCache(): void {
	_voicesCache = null;
}

/**
 * Fetch the account's voices from `GET /myvoices`. Returns the flattened
 * list, served from cache when fresh.
 *
 * Errors are normalized to a small user-facing set, matching the Deepgram
 * helpers' style:
 *   - "SIXTYDB_API_KEY not set"
 *   - "60db /myvoices HTTP <status>: <body>"
 *   - AbortError (signal fired)
 *   - "60db network error: <msg>"
 */
export async function fetchMyVoices(
	config: VoiceConfig,
	opts: { signal?: AbortSignal; force?: boolean } = {},
): Promise<SixtyDbVoice[]> {
	const apiKey = resolveSixtyDbApiKey(config);
	if (!apiKey) {
		throw new Error("SIXTYDB_API_KEY not set. Run /voice-settings to configure it, or export SIXTYDB_API_KEY in your shell.");
	}

	const now = Date.now();
	if (
		!opts.force &&
		_voicesCache &&
		_voicesCache.apiKey === apiKey &&
		now - _voicesCache.fetchedAtMs < VOICES_CACHE_TTL_MS
	) {
		return _voicesCache.voices;
	}

	let response: Response;
	try {
		response = await fetch(SIXTYDB_MYVOICES_URL, {
			method: "GET",
			headers: {
				"Authorization": `Bearer ${apiKey}`,
				"Accept": "application/json",
			},
			signal: opts.signal,
		});
	} catch (err: any) {
		if (err?.name === "AbortError") throw err;
		throw new Error(`60db network error: ${err?.message ?? String(err)}`);
	}

	if (!response.ok) {
		let body = "";
		try { body = (await response.text()).slice(0, 300); } catch {}
		throw new Error(`60db /myvoices HTTP ${response.status}${body ? `: ${body}` : ""}`);
	}

	// Size-bound the JSON body before parsing.
	const raw = await readBoundedText(response, MYVOICES_MAX_BYTES);
	let json: MyVoicesResponse;
	try {
		json = JSON.parse(raw) as MyVoicesResponse;
	} catch (err: any) {
		throw new Error(`60db /myvoices returned non-JSON: ${err?.message ?? String(err)}`);
	}

	const voices = normalizeVoices(json);
	_voicesCache = { voices, fetchedAtMs: now, apiKey };
	return voices;
}

/** Flatten the API's nested voice objects into `SixtyDbVoice`. */
function normalizeVoices(json: MyVoicesResponse): SixtyDbVoice[] {
	if (!Array.isArray(json.data)) return [];
	const out: SixtyDbVoice[] = [];
	for (const v of json.data) {
		if (!v || typeof v.voice_id !== "string" || !v.voice_id) continue;
		const labels = v.labels ?? {};
		out.push({
			id: v.voice_id,
			name: typeof v.name === "string" && v.name ? v.name : v.voice_id,
			language: typeof labels.language === "string" ? labels.language : "",
			languageName: typeof labels.language_name === "string" ? labels.language_name : "",
			gender: typeof labels.gender === "string" ? labels.gender : "",
			accent: typeof labels.accent === "string" ? labels.accent : "",
			model: typeof v.model === "string" ? v.model : "",
		});
	}
	return out;
}

/** Look up a cached voice by id, if the catalog has been fetched. */
export function getCachedVoice(id: string): SixtyDbVoice | undefined {
	return _voicesCache?.voices.find(v => v.id === id);
}

/**
 * Filter voices by base language tag, mirroring
 * `filterDeepgramVoicesByLanguage`. Returns all voices when `lang` is empty
 * or no voice carries a language label.
 */
export function filterVoicesByLanguage(voices: SixtyDbVoice[], lang: string): SixtyDbVoice[] {
	const base = (lang.split("-")[0] ?? "").toLowerCase();
	if (!base) return voices;
	const matched = voices.filter(v => v.language.toLowerCase() === base);
	// If the account has no voice for this language, fall back to the full
	// list rather than showing an empty picker — the user may still want to
	// pick a voice that 60db serves cross-lingually.
	return matched.length > 0 ? matched : voices;
}

/**
 * Read a fetch response body into a string, aborting if it exceeds
 * `maxBytes`. Mirrors the bounded-read defense in `tts-deepgram.ts`.
 */
async function readBoundedText(response: Response, maxBytes: number): Promise<string> {
	if (!response.body) {
		const buf = new Uint8Array(await response.arrayBuffer());
		if (buf.byteLength > maxBytes) {
			throw new Error(`60db /myvoices response too large (${buf.byteLength} bytes, max ${maxBytes}).`);
		}
		return new TextDecoder().decode(buf);
	}
	const reader = response.body.getReader();
	const chunks: Uint8Array[] = [];
	let total = 0;
	try {
		while (true) {
			const { value, done } = await reader.read();
			if (done) break;
			if (!value) continue;
			total += value.byteLength;
			if (total > maxBytes) {
				try { await reader.cancel(`60db response exceeded ${maxBytes} bytes`); } catch {}
				throw new Error(`60db /myvoices response too large (>${maxBytes} bytes).`);
			}
			chunks.push(value);
		}
	} finally {
		try { reader.releaseLock(); } catch {}
	}
	const out = new Uint8Array(total);
	let offset = 0;
	for (const chunk of chunks) {
		out.set(chunk, offset);
		offset += chunk.byteLength;
	}
	return new TextDecoder().decode(out);
}

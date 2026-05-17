/**
 * Deepgram cloud TTS — REST `/v1/speak` client.
 *
 * Uses the same `DEEPGRAM_API_KEY` users configure for STT (resolved via
 * `resolveDeepgramApiKey` from `./deepgram.ts`). One API key drives both
 * sides of the voice loop.
 *
 * v6.0 ships REST only. WebSocket streaming (`wss://api.deepgram.com/v1/speak`)
 * for sub-200ms TTFB is gated behind the `tts.deepgramStreaming` config flag
 * and lives in a future v6.1.
 *
 * Endpoint contract (verified against
 * https://developers.deepgram.com/docs/text-to-speech-rest):
 *   POST https://api.deepgram.com/v1/speak
 *     ?model=<voice>
 *     &encoding=linear16
 *     &sample_rate=<hz>
 *     &container=wav
 *   Headers: Authorization: Token <api-key>
 *            Content-Type: application/json
 *   Body:    { "text": "..." }
 *   Returns: audio/wav bytes (or audio/mpeg if container=none for some voices)
 *
 * Concurrency contract: this module is stateless. Each call opens a fresh
 * fetch — there is no client cache. The caller (speak.ts) holds the abort
 * signal and any response-level cancellation.
 */

import type { VoiceConfig } from "./config";
import { resolveDeepgramApiKey } from "./deepgram";
import type { PlaybackStream } from "./tts-playback";

// ─── Catalog ──────────────────────────────────────────────────────────────────

/**
 * Subset of Deepgram Aura-2 voices surfaced in the picker. The full Aura
 * catalog is large and changes; we list the stable, well-known English
 * voices a coding agent would actually use. Users can override with any
 * voice id supported by Deepgram by setting `ttsDeepgramVoiceId` directly.
 *
 * Naming convention: `aura-2-<name>-<lang>`. The trailing language code
 * makes language ↔ voice matching a substring check.
 *
 * Sample rate is the recommended default per Deepgram docs — 24 kHz gives
 * good fidelity and matches the Kitten/Kokoro local pipeline so playback
 * code doesn't need backend-specific buffer handling.
 */
export const DEEPGRAM_TTS_VOICES = [
	{ id: "aura-2-thalia-en", name: "Thalia (en, female, clear)", language: "en", gender: "female" },
	{ id: "aura-2-andromeda-en", name: "Andromeda (en, female, expressive)", language: "en", gender: "female" },
	{ id: "aura-2-amalthea-en", name: "Amalthea (en, female, cheerful)", language: "en", gender: "female" },
	{ id: "aura-2-apollo-en", name: "Apollo (en, male, confident)", language: "en", gender: "male" },
	{ id: "aura-2-arcas-en", name: "Arcas (en, male, smooth)", language: "en", gender: "male" },
	{ id: "aura-2-odysseus-en", name: "Odysseus (en, male, professional)", language: "en", gender: "male" },
	{ id: "aura-2-asteria-en", name: "Asteria (en, female, knowledgeable)", language: "en", gender: "female" },
	{ id: "aura-2-athena-en", name: "Athena (en, female, professional)", language: "en", gender: "female" },
	{ id: "aura-2-helena-en", name: "Helena (en, female, friendly)", language: "en", gender: "female" },
	{ id: "aura-2-hera-en", name: "Hera (en, female, warm)", language: "en", gender: "female" },
	{ id: "aura-2-luna-en", name: "Luna (en, female, natural)", language: "en", gender: "female" },
	{ id: "aura-2-orion-en", name: "Orion (en, male, polite)", language: "en", gender: "male" },
	{ id: "aura-2-orpheus-en", name: "Orpheus (en, male, trustworthy)", language: "en", gender: "male" },
	{ id: "aura-2-zeus-en", name: "Zeus (en, male, deep)", language: "en", gender: "male" },
] as const;

/** Default Deepgram voice id — used if the user hasn't picked one. */
export const DEFAULT_DEEPGRAM_TTS_VOICE = "aura-2-thalia-en";

/** Sample rate negotiated with the REST endpoint (Hz). */
export const DEEPGRAM_TTS_SAMPLE_RATE = 24000;

/**
 * Maximum response size (bytes). At 24 kHz mono 16-bit PCM that is
 * ~26 minutes of audio — far more than any conceivable agent-response
 * synthesis. Defends against a misconfigured account or a Deepgram outage
 * returning a huge HTML error page; without this a pathological response
 * could OOM the agent process.
 */
const DEEPGRAM_TTS_MAX_BYTES = 75_000_000; // ~75 MB

// ─── REST client ──────────────────────────────────────────────────────────────

export interface DeepgramSpeakOpts {
	text: string;
	voiceId: string;
	config: VoiceConfig;
	signal?: AbortSignal;
	/**
	 * Override the sample rate (Hz). Defaults to DEEPGRAM_TTS_SAMPLE_RATE.
	 * 8000 / 16000 / 22050 / 24000 / 32000 / 48000 are all supported by Aura.
	 */
	sampleRate?: number;
}

export interface DeepgramSpeakResult {
	/** WAV bytes from the REST response — write to a temp file then play. */
	wav: Uint8Array;
	/** Effective sample rate for playback configuration. */
	sampleRate: number;
}

/**
 * Synthesize via Deepgram REST. Returns a complete WAV blob suitable for
 * `tts-playback.ts` to write to a temp file and spawn the player against.
 *
 * Errors are normalized to a small, user-facing set:
 *   - "DEEPGRAM_API_KEY not set" (resolveDeepgramApiKey returned null)
 *   - "Deepgram TTS HTTP <status>: <body>" (4xx/5xx response)
 *   - AbortError (signal fired)
 *   - "Deepgram TTS network error: <msg>" (fetch failure)
 *
 * Aborts are wired through `fetch(url, { signal })`. When the user hits
 * Escape mid-synthesis, fetch tears down the connection and rejects with
 * a DOMException whose name === "AbortError".
 */
export async function deepgramSpeak(opts: DeepgramSpeakOpts): Promise<DeepgramSpeakResult> {
	const { text, voiceId, config, signal } = opts;
	const sampleRate = opts.sampleRate ?? DEEPGRAM_TTS_SAMPLE_RATE;

	if (!text || typeof text !== "string") {
		throw new Error(`Deepgram TTS text is required (got: ${text})`);
	}
	if (!voiceId || typeof voiceId !== "string") {
		throw new Error(`Deepgram TTS voiceId is required (got: ${voiceId})`);
	}

	const apiKey = resolveDeepgramApiKey(config);
	if (!apiKey) {
		throw new Error("DEEPGRAM_API_KEY not set. Run /voice-settings to configure it, or export DEEPGRAM_API_KEY in your shell.");
	}

	const url = buildDeepgramSpeakUrl(voiceId, sampleRate);

	let response: Response;
	try {
		response = await fetch(url, {
			method: "POST",
			headers: {
				"Authorization": `Token ${apiKey}`,
				"Content-Type": "application/json",
				// Deepgram's TTS REST returns audio bytes regardless of
				// Accept, but setting it explicitly documents intent and
				// matches the official docs example.
				"Accept": "audio/wav",
			},
			body: JSON.stringify({ text }),
			signal,
		});
	} catch (err: any) {
		// fetch() aborts surface as DOMException with name "AbortError".
		// Re-throw as-is so callers can pattern-match on err.name.
		if (err?.name === "AbortError") throw err;
		throw new Error(`Deepgram TTS network error: ${err?.message ?? String(err)}`);
	}

	if (!response.ok) {
		// Try to capture the body for a useful error. Deepgram returns
		// JSON like {"err_code":"INVALID_AUTH",...} on auth failures and
		// plain text on others.
		let body = "";
		try { body = (await response.text()).slice(0, 300); } catch {}
		throw new Error(`Deepgram TTS HTTP ${response.status}${body ? `: ${body}` : ""}`);
	}

	// Size-bound the response. Trust Content-Length when present; fall back
	// to streaming with a running byte count when it's absent or chunked.
	// Either way we cap at DEEPGRAM_TTS_MAX_BYTES to defend against a
	// runaway error page or misconfigured account.
	const declared = parseInt(response.headers.get("content-length") ?? "", 10);
	if (Number.isFinite(declared) && declared > DEEPGRAM_TTS_MAX_BYTES) {
		throw new Error(
			`Deepgram TTS response too large (${declared} bytes, max ${DEEPGRAM_TTS_MAX_BYTES}). ` +
			`Reduce text length or check your Deepgram account.`,
		);
	}

	const wav = await readBoundedBody(response, DEEPGRAM_TTS_MAX_BYTES);
	return { wav, sampleRate };
}

/**
 * Read a fetch response body into a Uint8Array, aborting if cumulative
 * bytes exceed `maxBytes`. Falls back to `arrayBuffer()` when the body
 * isn't a stream (older runtimes / mocks).
 */
async function readBoundedBody(response: Response, maxBytes: number): Promise<Uint8Array> {
	if (!response.body) {
		// No stream available — buffer in one shot but verify size after.
		const buf = new Uint8Array(await response.arrayBuffer());
		if (buf.byteLength > maxBytes) {
			throw new Error(
				`Deepgram TTS response too large (${buf.byteLength} bytes, max ${maxBytes}).`,
			);
		}
		return buf;
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
				// Cancel the underlying fetch so we don't keep streaming.
				try { await reader.cancel(`Deepgram TTS response exceeded ${maxBytes} bytes`); } catch {}
				throw new Error(
					`Deepgram TTS response too large (>${maxBytes} bytes). ` +
					`Reduce text length or check your Deepgram account.`,
				);
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
	return out;
}

/**
 * Build the `/v1/speak` request URL. Exported so unit tests can verify the
 * exact query-string shape without making a network call.
 */
export function buildDeepgramSpeakUrl(voiceId: string, sampleRate: number): string {
	const params = new URLSearchParams({
		model: voiceId,
		encoding: "linear16",
		sample_rate: String(sampleRate),
		container: "wav",
	});
	return `https://api.deepgram.com/v1/speak?${params.toString()}`;
}

// ─── Voice catalog helpers ────────────────────────────────────────────────────

/** Look up a voice entry by id; returns undefined if not in the surfaced list. */
export function getDeepgramVoice(id: string): typeof DEEPGRAM_TTS_VOICES[number] | undefined {
	return DEEPGRAM_TTS_VOICES.find(v => v.id === id);
}

/**
 * Filter the surfaced voice list by language tag. Used by the settings
 * panel voice picker — when the user has `ttsLanguage = "en"` we only
 * show English Aura voices.
 *
 * Region matching is intentionally loose (base tag only) for Deepgram
 * because Aura voice ids carry language without region (e.g.
 * `aura-2-thalia-en` is American and `aura-2-amalthea-en` is Filipino
 * English — both list `language: "en"`).
 */
export function filterDeepgramVoicesByLanguage(lang: string): readonly (typeof DEEPGRAM_TTS_VOICES)[number][] {
	const base = (lang.split("-")[0] ?? "").toLowerCase();
	if (!base) return DEEPGRAM_TTS_VOICES;
	return DEEPGRAM_TTS_VOICES.filter(v => v.language === base);
}

/**
 * Validate that a Deepgram voice id matches a requested language. Used by
 * the speak orchestrator before any network call so language ↔ voice
 * mismatches surface immediately rather than after a wasted round trip.
 *
 * Voices not in the surfaced catalog (custom Aura-2 ids the user pasted
 * directly into config) are accepted on faith — Deepgram's server will
 * reject if invalid.
 */
export function assertLanguageForDeepgram(voiceId: string, language: string): void {
	if (!language || typeof language !== "string") {
		throw new Error(`TTS language is required (got: ${language})`);
	}
	if (!voiceId || typeof voiceId !== "string") {
		throw new Error(`TTS voice id is required (got: ${voiceId})`);
	}
	const voice = getDeepgramVoice(voiceId);
	if (!voice) {
		// Unknown id — let Deepgram validate it server-side.
		return;
	}
	const requestedBase = (language.split("-")[0] ?? "").toLowerCase();
	if (voice.language !== requestedBase) {
		throw new Error(
			`Deepgram voice ${voice.id} speaks ${voice.language} but ttsLanguage is ${language}. ` +
			`Pick a voice for ${language} via /voice-settings, or change ttsLanguage to ${voice.language}.`,
		);
	}
}

// ─── v7.1.3: Deepgram WebSocket streaming TTS ─────────────────────────────────

/**
 * Open a WebSocket connection to Deepgram's streaming TTS endpoint and
 * pipe binary audio frames into a `PlaybackStream` sink as they arrive.
 *
 * Endpoint: `wss://api.deepgram.com/v1/speak?model=<voice>&encoding=linear16&sample_rate=<rate>`
 * Auth:     `Authorization: Token <api-key>` request header
 * Send:     JSON `{"type":"Speak","text":"..."}` then `{"type":"Flush"}`
 * Receive:  Binary frames = raw 16-bit signed LE PCM (mono).
 *           Text frames = control / metadata / errors.
 *
 * Resolves when the server signals end-of-stream (after Flush). Rejects
 * on auth errors, network errors, or signal abort. The sink itself is
 * NOT ended/cancelled by this function — the caller controls the sink
 * lifecycle so multiple calls can stream into the same player.
 *
 * For sub-200ms time-to-first-audio (TTFB) the network path matters
 * most. We send Flush immediately after the text so Deepgram starts
 * emitting audio frames without waiting for more input.
 */
export interface DeepgramStreamingOpts {
	readonly text: string;
	readonly voiceId: string;
	readonly config: VoiceConfig;
	readonly sampleRate?: number;
	readonly signal?: AbortSignal;
	/** Sink to receive PCM frames as they arrive. Caller manages lifecycle. */
	readonly sink: PlaybackStream;
}

export async function deepgramSpeakStreaming(opts: DeepgramStreamingOpts): Promise<void> {
	const { text, voiceId, config, sink, signal } = opts;
	const sampleRate = opts.sampleRate ?? DEEPGRAM_TTS_SAMPLE_RATE;

	if (!text || typeof text !== "string") throw new Error(`Deepgram TTS text is required (got: ${text})`);
	if (!voiceId || typeof voiceId !== "string") throw new Error(`Deepgram TTS voiceId is required (got: ${voiceId})`);
	const apiKey = resolveDeepgramApiKey(config);
	if (!apiKey) throw new Error("DEEPGRAM_API_KEY not set.");
	if (signal?.aborted) throw makeAbortError();

	const wsUrl = `wss://api.deepgram.com/v1/speak?model=${encodeURIComponent(voiceId)}` +
		`&encoding=linear16&sample_rate=${sampleRate}`;

	// Node 22 ships a built-in WebSocket. The 3rd-arg `headers` form is
	// the ws-package extension; the global Node WebSocket accepts headers
	// via `WebSocket.HEADERS_INIT` constructor 2nd arg. To stay
	// compatible across Node versions we use the 3rd-arg shape that the
	// `ws` package supports — that's already pulled in transitively.
	let ws: any;
	try {
		// Try built-in (Node 22+ undici).
		ws = new (globalThis as any).WebSocket(wsUrl, {
			headers: { Authorization: `Token ${apiKey}` },
		});
	} catch {
		// Fall back to ws package if the built-in rejects the headers form.
		const { WebSocket } = await import("ws");
		ws = new WebSocket(wsUrl, { headers: { Authorization: `Token ${apiKey}` } });
	}
	ws.binaryType = "arraybuffer";

	return new Promise<void>((resolve, reject) => {
		let settled = false;
		const settle = (action: () => void) => {
			if (settled) return;
			settled = true;
			action();
		};
		const onAbort = () => settle(() => {
			try { ws.close(1000, "abort"); } catch {}
			reject(makeAbortError());
		});
		signal?.addEventListener("abort", onAbort);

		ws.addEventListener("open", () => {
			try {
				ws.send(JSON.stringify({ type: "Speak", text }));
				// Flush tells Deepgram "no more text — emit audio + close".
				ws.send(JSON.stringify({ type: "Flush" }));
			} catch (err: any) {
				settle(() => reject(new Error(`Deepgram WS send failed: ${err?.message ?? err}`)));
			}
		});
		ws.addEventListener("message", (ev: any) => {
			const data = ev?.data;
			if (data instanceof ArrayBuffer) {
				// Binary frame = raw int16 LE PCM. Pipe straight to sink.
				// `sink.writePcm` returns a Promise (writeTail chain
				// serializes internally; even fire-and-forget here
				// preserves order). Catch on the returned promise so a
				// late rejection (sink torn down mid-frame) doesn't
				// crash with UnhandledPromiseRejection.
				const i16 = new Int16Array(data);
				try {
					const p = sink.writePcm(i16);
					if (p && typeof (p as Promise<void>).catch === "function") {
						(p as Promise<void>).catch(() => { /* sink already errored, caller observes */ });
					}
				} catch { /* sync errors swallowed — sink handles state */ }
			} else if (typeof data === "string") {
				// Control frames: { "type": "Metadata" } / { "type": "Flushed" } / errors
				try {
					const msg = JSON.parse(data);
					if (msg?.type === "Flushed" || msg?.type === "Final") {
						// Server signals end of synthesis. Close and resolve.
						try { ws.close(1000, "done"); } catch {}
						settle(() => resolve());
					} else if (msg?.type === "Error" || msg?.error) {
						// godspeed glm finding: must close socket on error
						// path or sustained errors leak FDs + abort listeners.
						try { ws.close(1011, "error"); } catch {}
						settle(() => reject(new Error(`Deepgram WS error: ${msg.error ?? data}`)));
					}
				} catch { /* ignore unparsable text frames */ }
			}
		});
		ws.addEventListener("error", (ev: any) => {
			// godspeed glm finding: close on error path to release the
			// FD + abort listener. Without this, sustained errors leak.
			try { ws.close(1011, "error"); } catch {}
			settle(() => reject(new Error(`Deepgram WS error: ${ev?.message ?? "unknown"}`)));
		});
		ws.addEventListener("close", (ev: any) => {
			signal?.removeEventListener("abort", onAbort);
			// Server-initiated close after a successful stream resolves
			// the promise; abort/error paths already settled above.
			if (ev?.code === 1000) settle(() => resolve());
			else settle(() => reject(new Error(`Deepgram WS closed: code=${ev?.code} reason=${ev?.reason ?? ""}`)));
		});
	});
}

function makeAbortError(): Error {
	if (typeof DOMException === "function") return new DOMException("Deepgram TTS aborted", "AbortError");
	const e = new Error("Deepgram TTS aborted");
	(e as any).name = "AbortError";
	return e;
}

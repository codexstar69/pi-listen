/**
 * 60db cloud TTS — REST `/tts-synthesize` + WebSocket `/ws/tts` clients.
 *
 * Mirrors `./tts-deepgram.ts`: REST is the default path (a complete WAV
 * blob the playback layer writes to a temp file), and WebSocket streaming
 * is gated behind the `ttsSixtyDbStreaming` config flag for low
 * time-to-first-audio.
 *
 * Auth differs from Deepgram:
 *   - REST: `Authorization: Bearer <key>`
 *   - WS:   `?apiKey=<key>` query parameter
 *
 * REST endpoint contract (https://docs.60db.ai/api-reference/tts/text-to-speech):
 *   POST https://api.60db.ai/tts-synthesize
 *   Headers: Authorization: Bearer <key>, Content-Type: application/json
 *   Body:    { text, voice_id, output_format: "wav", speed, ... }
 *   Returns: JSON { success, audio_base64, sample_rate, encoding, output_format }
 *            — audio is BASE64 inside JSON, not raw bytes (unlike Deepgram).
 *
 * WebSocket contract (https://docs.60db.ai/websocket-api/tts):
 *   Connect: wss://api.60db.ai/ws/tts?apiKey=<key>
 *   Client → Server (type is the envelope key):
 *     { create_context: { context_id, voice_id, audio_config: { audio_encoding, sample_rate_hertz } } }
 *     { send_text:      { context_id, text } }
 *     { flush_context:  { context_id } }
 *     { close_context:  { context_id } }
 *   Server → Client:
 *     { connection_established: {...} } { context_created: {...} }
 *     { audio_chunk: { audioContent: <base64 PCM> } }
 *     { flush_completed: {...} } { context_closed: {...} } { error: {...} }
 *   Audio frames are base64-encoded 16-bit signed LE PCM (mono) for the
 *   LINEAR16 encoding.
 */

import { randomUUID } from "node:crypto";
import type { VoiceConfig } from "./config";
import { resolveSixtyDbApiKey, SIXTYDB_TTS_URL, SIXTYDB_WS_TTS_URL } from "./sixtydb";
import type { PlaybackStream } from "./tts-playback";

// ─── Constants ──────────────────────────────────────────────────────────────────

/**
 * Sample rate negotiated for both REST and WS. 24 kHz matches the
 * Deepgram/Kitten/Kokoro pipeline so the playback layer needs no
 * backend-specific buffer handling. 60db LINEAR16 supports
 * 8000/16000/24000/48000.
 */
export const SIXTYDB_TTS_SAMPLE_RATE = 24000;

/** Max text length per the 60db docs. Both REST and WS reject longer input. */
export const SIXTYDB_TTS_MAX_TEXT = 5000;

/**
 * Maximum REST response size (bytes). The body is base64 JSON, so it runs
 * ~33% larger than the decoded audio; 100 MB caps decoded audio at roughly
 * 26 min of 24 kHz mono PCM — far beyond any agent-response synthesis.
 * Defends against a misconfigured account returning a huge error page.
 */
const SIXTYDB_TTS_MAX_BYTES = 100_000_000; // ~100 MB

// ─── REST client ────────────────────────────────────────────────────────────────

export interface SixtyDbSpeakOpts {
	text: string;
	voiceId: string;
	config: VoiceConfig;
	signal?: AbortSignal;
	/** Speech rate multiplier (0.5–2.0). Default 1.0. */
	speed?: number;
}

export interface SixtyDbSpeakResult {
	/** WAV bytes (decoded from audio_base64) — write to temp file then play. */
	wav: Uint8Array;
	/** Effective sample rate for playback configuration. */
	sampleRate: number;
}

/**
 * Synthesize via 60db REST. Returns a complete WAV blob suitable for
 * `tts-playback.ts`. Errors normalized to a small user-facing set, matching
 * `deepgramSpeak`:
 *   - "SIXTYDB_API_KEY not set"
 *   - "60db TTS HTTP <status>: <body>"
 *   - "60db TTS returned no audio" (success=false or empty audio_base64)
 *   - AbortError (signal fired)
 *   - "60db TTS network error: <msg>"
 */
export async function sixtyDbSpeak(opts: SixtyDbSpeakOpts): Promise<SixtyDbSpeakResult> {
	const { text, voiceId, config, signal } = opts;
	const speed = clampSpeed(opts.speed);

	if (!text || typeof text !== "string") {
		throw new Error(`60db TTS text is required (got: ${text})`);
	}
	if (!voiceId || typeof voiceId !== "string") {
		throw new Error("60db TTS voice_id is required. Pick a voice via /voice-settings → Speak tab.");
	}

	const apiKey = resolveSixtyDbApiKey(config);
	if (!apiKey) {
		throw new Error("SIXTYDB_API_KEY not set. Run /voice-settings to configure it, or export SIXTYDB_API_KEY in your shell.");
	}

	let response: Response;
	try {
		response = await fetch(SIXTYDB_TTS_URL, {
			method: "POST",
			headers: {
				"Authorization": `Bearer ${apiKey}`,
				"Content-Type": "application/json",
				"Accept": "application/json",
			},
			body: JSON.stringify({
				text,
				voice_id: voiceId,
				output_format: "wav",
				speed,
			}),
			signal,
		});
	} catch (err: any) {
		if (err?.name === "AbortError") throw err;
		throw new Error(`60db TTS network error: ${err?.message ?? String(err)}`);
	}

	if (!response.ok) {
		let body = "";
		try { body = (await response.text()).slice(0, 300); } catch {}
		throw new Error(`60db TTS HTTP ${response.status}${body ? `: ${body}` : ""}`);
	}

	const declared = parseInt(response.headers.get("content-length") ?? "", 10);
	if (Number.isFinite(declared) && declared > SIXTYDB_TTS_MAX_BYTES) {
		throw new Error(
			`60db TTS response too large (${declared} bytes, max ${SIXTYDB_TTS_MAX_BYTES}). ` +
			`Reduce text length or check your 60db account.`,
		);
	}

	const rawText = await readBoundedText(response, SIXTYDB_TTS_MAX_BYTES);
	let json: { success?: boolean; message?: string; audio_base64?: string; sample_rate?: number };
	try {
		json = JSON.parse(rawText);
	} catch (err: any) {
		throw new Error(`60db TTS returned non-JSON: ${err?.message ?? String(err)}`);
	}

	if (json.success === false || !json.audio_base64) {
		throw new Error(`60db TTS returned no audio${json.message ? `: ${json.message}` : ""}`);
	}

	const wav = base64ToBytes(json.audio_base64);
	return {
		wav,
		sampleRate: Number.isFinite(json.sample_rate) ? Number(json.sample_rate) : SIXTYDB_TTS_SAMPLE_RATE,
	};
}

// ─── WebSocket streaming client ─────────────────────────────────────────────────

export interface SixtyDbStreamingOpts {
	readonly text: string;
	readonly voiceId: string;
	readonly config: VoiceConfig;
	readonly sampleRate?: number;
	readonly signal?: AbortSignal;
	/** Sink to receive PCM frames as they arrive. Caller manages lifecycle. */
	readonly sink: PlaybackStream;
}

/**
 * Open a WebSocket to 60db's streaming TTS endpoint and pipe decoded PCM
 * frames into a `PlaybackStream` sink as they arrive. Resolves when the
 * server closes the context after a flush. Rejects on auth/network/error
 * frames or signal abort. The sink is NOT ended/cancelled here — the caller
 * controls its lifecycle so multiple calls can stream into one player.
 *
 * Mirrors `deepgramSpeakStreaming`, adapted to 60db's `create_context` /
 * `send_text` / `flush_context` / `close_context` protocol and base64
 * audio frames.
 */
export async function sixtyDbSpeakStreaming(opts: SixtyDbStreamingOpts): Promise<void> {
	const { text, voiceId, config, sink, signal } = opts;
	const sampleRate = opts.sampleRate ?? SIXTYDB_TTS_SAMPLE_RATE;

	if (!text || typeof text !== "string") throw new Error(`60db TTS text is required (got: ${text})`);
	if (!voiceId || typeof voiceId !== "string") throw new Error("60db TTS voice_id is required.");
	const apiKey = resolveSixtyDbApiKey(config);
	if (!apiKey) throw new Error("SIXTYDB_API_KEY not set.");
	if (signal?.aborted) throw makeAbortError();

	// 60db authenticates the WebSocket via a query parameter rather than a
	// header, so the key never needs the header-form constructor.
	const wsUrl = `${SIXTYDB_WS_TTS_URL}?apiKey=${encodeURIComponent(apiKey)}`;
	const contextId = randomUUID();

	let ws: any;
	try {
		ws = new (globalThis as any).WebSocket(wsUrl);
	} catch {
		const { WebSocket } = await import("ws");
		ws = new WebSocket(wsUrl);
	}
	ws.binaryType = "arraybuffer";

	return new Promise<void>((resolve, reject) => {
		let settled = false;
		let textSent = false;
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

		// Send the synthesis request once the context exists. Guarded so a
		// duplicate context_created frame can't double-send.
		const sendText = () => {
			if (textSent) return;
			textSent = true;
			try {
				ws.send(JSON.stringify({ send_text: { context_id: contextId, text } }));
				// Flush tells 60db to synthesize buffered text; close tells it
				// "no more text — emit remaining audio and close the context".
				ws.send(JSON.stringify({ flush_context: { context_id: contextId } }));
				ws.send(JSON.stringify({ close_context: { context_id: contextId } }));
			} catch (err: any) {
				settle(() => reject(new Error(`60db WS send failed: ${err?.message ?? err}`)));
			}
		};

		ws.addEventListener("message", (ev: any) => {
			const data = ev?.data;
			if (typeof data !== "string") {
				// 60db delivers audio as base64 inside JSON, so any binary
				// frame is unexpected — ignore it defensively.
				return;
			}
			let msg: any;
			try {
				msg = JSON.parse(data);
			} catch {
				return; // ignore unparsable frames
			}

			if (msg.connection_established) {
				// Authenticated — open a context with our voice + audio config.
				try {
					ws.send(JSON.stringify({
						create_context: {
							context_id: contextId,
							voice_id: voiceId,
							audio_config: { audio_encoding: "LINEAR16", sample_rate_hertz: sampleRate },
						},
					}));
				} catch (err: any) {
					settle(() => reject(new Error(`60db WS create_context failed: ${err?.message ?? err}`)));
				}
			} else if (msg.context_created) {
				sendText();
			} else if (msg.audio_chunk) {
				const b64 = msg.audio_chunk?.audioContent;
				if (typeof b64 === "string" && b64.length > 0) {
					const i16 = base64ToInt16(b64);
					try {
						const p = sink.writePcm(i16);
						if (p && typeof (p as Promise<void>).catch === "function") {
							(p as Promise<void>).catch(() => { /* sink torn down — caller observes */ });
						}
					} catch { /* sync sink errors swallowed — sink owns its state */ }
				}
			} else if (msg.context_closed) {
				// Server finished this context — all audio sent.
				try { ws.close(1000, "done"); } catch {}
				settle(() => resolve());
			} else if (msg.error) {
				try { ws.close(1011, "error"); } catch {}
				const detail = typeof msg.error === "string" ? msg.error : (msg.error?.message ?? data);
				settle(() => reject(new Error(`60db WS error: ${detail}`)));
			}
			// flush_completed is informational — we wait for context_closed.
		});

		ws.addEventListener("error", (ev: any) => {
			try { ws.close(1011, "error"); } catch {}
			settle(() => reject(new Error(`60db WS error: ${ev?.message ?? "unknown"}`)));
		});

		ws.addEventListener("close", (ev: any) => {
			signal?.removeEventListener("abort", onAbort);
			if (ev?.code === 1000) settle(() => resolve());
			else settle(() => reject(new Error(`60db WS closed: code=${ev?.code} reason=${ev?.reason ?? ""}`)));
		});
	});
}

// ─── Helpers ────────────────────────────────────────────────────────────────────

function clampSpeed(speed: number | undefined): number {
	if (typeof speed !== "number" || !Number.isFinite(speed)) return 1.0;
	return Math.max(0.5, Math.min(2.0, speed));
}

/** Decode a base64 string to bytes (Buffer is available in Node/Bun). */
function base64ToBytes(b64: string): Uint8Array {
	return new Uint8Array(Buffer.from(b64, "base64"));
}

/**
 * Decode base64 PCM to an Int16Array. Copies into a fresh, 2-byte-aligned
 * buffer so the Int16Array view is valid even when Buffer hands back an
 * odd-length or unaligned slice of its shared pool.
 */
function base64ToInt16(b64: string): Int16Array {
	const buf = Buffer.from(b64, "base64");
	// Drop a trailing odd byte (can't form a complete int16 sample).
	const usable = buf.byteLength - (buf.byteLength % 2);
	const copy = new Uint8Array(usable);
	copy.set(buf.subarray(0, usable));
	return new Int16Array(copy.buffer, copy.byteOffset, usable / 2);
}

async function readBoundedText(response: Response, maxBytes: number): Promise<string> {
	if (!response.body) {
		const buf = new Uint8Array(await response.arrayBuffer());
		if (buf.byteLength > maxBytes) {
			throw new Error(`60db TTS response too large (${buf.byteLength} bytes, max ${maxBytes}).`);
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
				try { await reader.cancel(`60db TTS response exceeded ${maxBytes} bytes`); } catch {}
				throw new Error(`60db TTS response too large (>${maxBytes} bytes). Reduce text length or check your 60db account.`);
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

function makeAbortError(): Error {
	if (typeof DOMException === "function") return new DOMException("60db TTS aborted", "AbortError");
	const e = new Error("60db TTS aborted");
	(e as any).name = "AbortError";
	return e;
}

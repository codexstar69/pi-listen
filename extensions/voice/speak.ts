/**
 * High-level TTS orchestrator.
 *
 * Public API:
 *   - speak(text, opts)     — synthesize and play with backend dispatch
 *   - chunkText(text, lang) — split into playback-friendly sentences
 *
 * Pipeline:
 *   1. Validate inputs (non-empty text, valid backend choice)
 *   2. Resolve language from opts → config → fallback
 *   3. Backend dispatch: local (sherpa) or cloud (Deepgram)
 *   4. Sentence-aware chunking via Intl.Segmenter (with word-window
 *      fallback for locales/runtimes without segmenter support)
 *   5. Sequentially synthesize + play each chunk
 *   6. Cooperative abort via AbortSignal at every async boundary
 *
 * Sentence chunking uses Intl.Segmenter rather than naive `.`/`!`/`?`
 * splitting. The naive split misfires on common patterns (`Dr. Smith`,
 * `e.g.`, `v2.0`, `U.S.A.`, URLs, file paths, decimal numbers). Locale-
 * aware Segmenter handles these correctly without an abbreviation table
 * to maintain.
 */

import type { VoiceConfig } from "./config";
import { synthesize, type TtsAudio } from "./tts-engine";
import { openPlaybackStream, float32ToInt16, type PlaybackStream } from "./tts-playback";
import { getTtsModel, type TtsLocalModelInfo } from "./tts-local-models";
import {
	deepgramSpeak,
	deepgramSpeakStreaming,
	DEFAULT_DEEPGRAM_TTS_VOICE,
	DEEPGRAM_TTS_SAMPLE_RATE,
	assertLanguageForDeepgram,
} from "./tts-deepgram";
import { play } from "./tts-playback";

// ─── Public API ───────────────────────────────────────────────────────────────

export interface SpeakOpts {
	/** The full text to speak. Will be sentence-chunked before synthesis. */
	text: string;
	/** Active VoiceConfig — read for backend choice, voice, language, speed. */
	config: VoiceConfig;
	/**
	 * Override config.ttsLanguage. BCP-47 tag (e.g. "en", "es-ES").
	 * Defaults to `config.ttsLanguage ?? config.language ?? "en"`.
	 */
	language?: string;
	/** Cooperative cancellation. Aborts mid-chunk if user hits Escape. */
	signal?: AbortSignal;
	/**
	 * Resolver for installed local model directories. Required for the
	 * local backend; the orchestrator doesn't know where models live —
	 * that's a property of the download/install layer in voice.ts /
	 * model-download.ts. Throws if the model id isn't installed.
	 */
	resolveModelDir?: (modelId: string) => string;
	/** Override the play() function — used by tests to capture audio. */
	playAudio?: typeof play;
}

/**
 * Synthesize `opts.text` and play it through the user's audio output.
 * Resolves when playback completes; rejects on abort or error.
 *
 * Errors propagate as-thrown:
 *   - DOMException("AbortError") if signal fires
 *   - Error("...") with a user-facing message for everything else
 */
export async function speak(opts: SpeakOpts): Promise<void> {
	const { text, config, signal } = opts;

	if (!text || typeof text !== "string" || !text.trim()) {
		throw new Error("speak(): text is required and must be non-empty");
	}
	if (signal?.aborted) {
		throw makeAbortError();
	}

	const language = resolveLanguage(opts);
	const backend: "local" | "deepgram" = config.ttsBackend === "deepgram" ? "deepgram" : "local";

	// Chunk the text once up front. Each chunk is small enough to fit a
	// single synthesize call without the engine refusing it; sequential
	// playback keeps the audio in order.
	const chunks = chunkText(text, language);
	if (chunks.length === 0) {
		throw new Error("speak(): no synthesizable content after chunking");
	}

	const playAudio = opts.playAudio ?? play;

	// v7.1.3 — pipelined synth + true streaming playback when sox/paplay
	// is available. Two layers:
	//
	// (1) Pipeline: while chunk N plays, synthesize chunk N+1 in parallel
	//     so the gap between sentences disappears.
	// (2) Stream: when a streaming-capable player is on PATH (sox / paplay
	//     / Linux), open a long-lived player process and pipe int16 PCM
	//     bytes directly. Drops file-write/open/start latency. Falls back
	//     to file-based per-chunk playback when no streaming player is
	//     available (Windows, or systems without sox).
	//
	// Cancellation: if the signal aborts mid-pipeline, the in-flight synth
	// promise is left to settle naturally and the streaming sink is
	// cancelled (kills sox + drops buffered audio). A throwing synth
	// (broken model + sid) cancels the stream and propagates.
	// godspeed runtime/gemini-3.1-pro VETO fix: when we kick off
	// `nextSynth = synthOne(...)` and the loop later exits via abort
	// or an unrelated throw, the prefetched promise can reject after
	// no one is awaiting it → unhandledRejection crashes the Pi
	// process. Attach a safety `.catch()` that lives until the next
	// loop iteration's `await` consumes the (rejected) promise. The
	// awaiter still observes the rejection because the catch is on a
	// SHADOW promise — `synthOne()` itself returns the original.
	const synthOne = (chunk: string) => {
		const p = synthesizeChunk({
			chunk,
			backend,
			config,
			language,
			signal,
			resolveModelDir: opts.resolveModelDir,
		});
		// Shadow .catch — swallows so an orphaned promise never
		// surfaces as UnhandledPromiseRejection. The original `p` is
		// returned so the actual awaiter sees the rejection.
		p.catch(() => { /* see comment above */ });
		return p;
	};

	// Discover the audio sample rate for the streaming player. Local
	// engine returns it on the first synth result; for Deepgram we
	// know it's the buildDeepgramSpeakUrl rate. We open the stream
	// AFTER the first synth so we have a definitive rate to pass.
	let stream: PlaybackStream | null = null;

	const cleanupStream = () => {
		if (stream) {
			try { stream.cancel(); } catch { /* already cancelled */ }
			stream = null;
		}
	};

	// v7.1.3 — Deepgram WebSocket streaming TTS. When the user enables
	// `ttsDeepgramStreaming` AND we have a local stream sink, bypass the
	// REST/file-based chunk loop entirely: open the sink once, send each
	// chunk's text directly to Deepgram's WS, and let the binary frames
	// flow into the sink as they arrive. Sub-200ms TTFA in good network
	// conditions vs ~1-2s for REST/file path.
	if (backend === "deepgram" && config.ttsDeepgramStreaming === true) {
		const dgSampleRate = 24000;
		const sink = openPlaybackStream({ sampleRate: dgSampleRate, signal });
		if (sink) {
			try {
				const voiceId = config.ttsDeepgramVoiceId || "aura-asteria-en";
				for (const chunk of chunks) {
					if (signal?.aborted) throw makeAbortError();
					await deepgramSpeakStreaming({
						text: chunk,
						voiceId,
						config,
						sampleRate: dgSampleRate,
						signal,
						sink,
					});
				}
				await sink.end();
				await sink.done();
				return;
			} catch (err) {
				try { sink.cancel(); } catch {}
				throw err;
			}
		}
		// No streaming player available — fall through to REST/file path.
	}

	try {
		let nextSynth: ReturnType<typeof synthOne> | null = null;
		for (let i = 0; i < chunks.length; i++) {
			if (signal?.aborted) throw makeAbortError();
			// First iteration synthesizes inline; subsequent iterations use
			// the prefetched audio from the previous iteration.
			const audio = await (nextSynth ?? synthOne(chunks[i]!));
			nextSynth = null;
			if (signal?.aborted) throw makeAbortError();

			// Streaming path: open the sink lazily on first chunk so we
			// have the actual sample rate. PCM-yielding chunks (local
			// `{ samples, sampleRate }`) are write-and-go; pre-encoded
			// WAV chunks (Deepgram REST) cannot stream and fall through.
			if ("samples" in audio && audio.sampleRate) {
				if (stream === null && i === 0) {
					stream = openPlaybackStream({ sampleRate: audio.sampleRate, signal });
				}
				if (stream) {
					// Kick off NEXT chunk's synth BEFORE awaiting the
					// write — synthesis runs in parallel with stdin
					// write/drain. writePcm returns a promise that
					// resolves once the byte queue is accepted (or
					// drained on backpressure).
					if (i + 1 < chunks.length) nextSynth = synthOne(chunks[i + 1]!);
					await stream.writePcm(float32ToInt16(audio.samples));
					continue;
				}
			}

			// Non-streaming fallback (Windows, missing sox, or WAV chunks):
			// file-per-chunk playback. Pipeline next synth while playing.
			if (i + 1 < chunks.length) {
				nextSynth = synthOne(chunks[i + 1]!);
			}
			await playAudio({ source: audio, signal });
			if (signal?.aborted) throw makeAbortError();
		}

		// Streaming path: tell the player no more PCM is coming and wait
		// for it to drain. end() awaits all queued writes + appends a
		// silence tail before signaling EOF (compensates for sox closing
		// the audio device on EOF and dropping ~1s of buffered audio).
		if (stream) {
			await stream.end();
			await stream.done();
			stream = null;
		}
	} catch (err) {
		cleanupStream();
		throw err;
	}
}

// ─── Backend dispatch ─────────────────────────────────────────────────────────

interface SynthesizeChunkOpts {
	chunk: string;
	backend: "local" | "deepgram";
	config: VoiceConfig;
	language: string;
	signal?: AbortSignal;
	resolveModelDir?: (modelId: string) => string;
}

/**
 * Single chunk → audio. Returns either a Float32 PCM frame (local) or a
 * pre-encoded WAV blob (Deepgram). The playback layer accepts both.
 */
async function synthesizeChunk(opts: SynthesizeChunkOpts): Promise<{ samples: Float32Array; sampleRate: number } | { wav: Uint8Array }> {
	const { chunk, backend, config, language, signal } = opts;
	if (backend === "deepgram") {
		const voiceId = typeof config.ttsDeepgramVoiceId === "string" && config.ttsDeepgramVoiceId
			? config.ttsDeepgramVoiceId
			: DEFAULT_DEEPGRAM_TTS_VOICE;
		assertLanguageForDeepgram(voiceId, language);
		const result = await deepgramSpeak({ text: chunk, voiceId, config, signal });
		// The Deepgram REST endpoint we use returns a complete WAV blob;
		// the playback layer reads the sample rate from the WAV header so
		// `result.sampleRate` is informational only at this layer (v6.1
		// streaming will wire it into the playback layer directly).
		return { wav: result.wav };
	}

	// Local backend
	const modelId = config.ttsLocalModel || "kitten-nano-en-v0_2";
	const model: TtsLocalModelInfo = getTtsModel(modelId);
	if (!opts.resolveModelDir) {
		throw new Error(
			"speak(): local TTS requires a resolveModelDir resolver. " +
			"This is supplied by voice.ts when invoking speak() from the slash-command layer.",
		);
	}
	const modelDir = opts.resolveModelDir(modelId);

	const sid = typeof config.ttsLocalVoiceId === "number" && Number.isFinite(config.ttsLocalVoiceId)
		? config.ttsLocalVoiceId
		: model.defaultSid;
	const speed = typeof config.ttsSpeed === "number" && Number.isFinite(config.ttsSpeed)
		? config.ttsSpeed
		: 1.0;

	const audio: TtsAudio = await synthesize({
		text: chunk,
		model,
		modelDir,
		language,
		sid,
		speed,
		signal,
	});
	return { samples: audio.samples, sampleRate: audio.sampleRate };
}

// ─── Language resolution ──────────────────────────────────────────────────────

function resolveLanguage(opts: SpeakOpts): string {
	const candidate = opts.language ?? opts.config.ttsLanguage ?? opts.config.language ?? "en";
	if (typeof candidate !== "string" || !candidate.trim()) return "en";
	return candidate.trim();
}

// ─── Sentence chunking ────────────────────────────────────────────────────────

/**
 * Hard cap on a single chunk in characters. sherpa-onnx's max_num_sentences
 * default is 2; at conversational speech rate ~25 words / sentence, 2
 * sentences ≈ 50 words ≈ 300-400 characters. Setting cap at 600 gives a
 * safety margin while still keeping playback latency low.
 */
const MAX_CHUNK_CHARS = 600;

/**
 * Split `text` into playback-friendly chunks. Uses `Intl.Segmenter` for
 * locale-aware sentence boundaries when available; falls back to a
 * word-window splitter otherwise.
 *
 * Returned chunks are non-empty and each fits within MAX_CHUNK_CHARS.
 * The caller can iterate them and play sequentially.
 *
 * Verified against problematic inputs (Dr./e.g./v2.0/U.S.A./URLs/decimals):
 * Intl.Segmenter does NOT split on those abbreviations. See `tests/`
 * for the regression cases.
 */
export function chunkText(text: string, language: string): string[] {
	const trimmed = text.trim();
	if (!trimmed) return [];

	const sentences = segmentSentences(trimmed, language);
	const chunks: string[] = [];
	let buf = "";

	for (const sentence of sentences) {
		const s = sentence.trim();
		if (!s) continue;

		if (s.length > MAX_CHUNK_CHARS) {
			// Single sentence longer than cap — wrap-split on word boundaries.
			if (buf) { chunks.push(buf); buf = ""; }
			chunks.push(...wordWindowSplit(s));
			continue;
		}

		const candidate = buf ? `${buf} ${s}` : s;
		if (candidate.length > MAX_CHUNK_CHARS) {
			chunks.push(buf);
			buf = s;
		} else {
			buf = candidate;
		}
	}
	if (buf) chunks.push(buf);
	return chunks;
}

/**
 * Locale-aware sentence segmentation. Falls back to a simple word-window
 * splitter on environments where Intl.Segmenter is missing (e.g. older
 * Node without ICU full-data, or Bun if locale-specific segmentation is
 * unavailable for that lang).
 */
function segmentSentences(text: string, language: string): string[] {
	const SegmenterCtor: typeof Intl.Segmenter | undefined = (Intl as any).Segmenter;
	if (typeof SegmenterCtor === "function") {
		try {
			const seg = new SegmenterCtor(language, { granularity: "sentence" });
			const out: string[] = [];
			for (const piece of seg.segment(text)) {
				out.push(piece.segment);
			}
			return out;
		} catch {
			// Fall through to word-window fallback.
		}
	}
	return wordWindowSplit(text);
}

/**
 * Split on whitespace and re-pack into chunks of ~25 words each, never
 * splitting mid-token. Used as the segmentation fallback.
 */
function wordWindowSplit(text: string): string[] {
	const words = text.split(/(\s+)/);
	const chunks: string[] = [];
	let buf = "";
	let wordCount = 0;
	const TARGET_WORDS = 25;
	for (const w of words) {
		const isWhitespace = /^\s+$/.test(w);
		if (!isWhitespace && wordCount >= TARGET_WORDS && buf.length > 0) {
			chunks.push(buf.trim());
			buf = "";
			wordCount = 0;
		}
		buf += w;
		if (!isWhitespace) wordCount++;
		if (buf.length >= MAX_CHUNK_CHARS) {
			// Hard cap mid-text — flush.
			chunks.push(buf.trim());
			buf = "";
			wordCount = 0;
		}
	}
	const tail = buf.trim();
	if (tail) chunks.push(tail);
	return chunks;
}

// ─── Errors ───────────────────────────────────────────────────────────────────

function makeAbortError(): Error {
	if (typeof DOMException === "function") {
		return new DOMException("speak() aborted", "AbortError");
	}
	const e = new Error("speak() aborted");
	(e as any).name = "AbortError";
	return e;
}

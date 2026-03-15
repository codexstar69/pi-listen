/**
 * Local transcription backend — batch-mode STT using a local server.
 *
 * Supports any OpenAI-compatible transcription API:
 *   - whisper.cpp server (built-in /v1/audio/transcriptions)
 *   - faster-whisper-server
 *   - Any server implementing POST /v1/audio/transcriptions
 *
 * Models mirror what handy.computer supports via transcribe-rs.
 *
 * Architecture difference from Deepgram:
 *   Deepgram  → real-time streaming (WebSocket, interim results while speaking)
 *   Local     → batch mode (record complete audio, transcribe after stop)
 */

import type { ChildProcess } from "node:child_process";
import type { VoiceConfig } from "./config";
import { SAMPLE_RATE, CHANNELS } from "./deepgram";

// ─── Model catalog (mirrors handy.computer's transcribe-rs models) ───────────

export interface LocalModelInfo {
	id: string;
	name: string;
	size: string;
	notes: string;
	/** Language family — determines which language list to show */
	langSupport: "whisper" | "english-only" | "parakeet-multi";
}

export const LOCAL_MODELS: LocalModelInfo[] = [
	{ id: "whisper-small", name: "Whisper Small", size: "487 MB", notes: "Good balance of speed and accuracy", langSupport: "whisper" },
	{ id: "whisper-medium", name: "Whisper Medium", size: "492 MB", notes: "Better accuracy, moderate speed", langSupport: "whisper" },
	{ id: "whisper-turbo", name: "Whisper Turbo", size: "1.6 GB", notes: "Fast and accurate, needs GPU", langSupport: "whisper" },
	{ id: "whisper-large", name: "Whisper Large", size: "1.1 GB", notes: "Best accuracy, slowest", langSupport: "whisper" },
	{ id: "moonshine-tiny", name: "Moonshine Tiny", size: "~60 MB", notes: "Ultra-fast, 5x less compute than Whisper, English only", langSupport: "english-only" },
	{ id: "moonshine-base", name: "Moonshine Base", size: "~130 MB", notes: "Fast and accurate, edge-optimized, English only", langSupport: "english-only" },
	{ id: "parakeet-v2", name: "Parakeet V2", size: "473 MB", notes: "CPU-optimized, English only", langSupport: "english-only" },
	{ id: "parakeet-v3", name: "Parakeet V3", size: "478 MB", notes: "CPU-optimized, auto language detection", langSupport: "parakeet-multi" },
];

export const DEFAULT_LOCAL_ENDPOINT = "http://localhost:8080";
export const DEFAULT_LOCAL_MODEL = "whisper-small";

// ─── Language support per model family ───────────────────────────────────────
// Whisper uses simple ISO 639-1 codes (no regional variants like "en-AU").
// Parakeet V2 is English-only. Parakeet V3 shares Whisper's language set.

export interface LocalLangEntry { name: string; code: string; popular?: boolean; }

const WHISPER_LANGUAGES: LocalLangEntry[] = [
	// Popular — shown first
	{ name: "English", code: "en", popular: true },
	{ name: "Hindi", code: "hi", popular: true },
	{ name: "Spanish", code: "es", popular: true },
	{ name: "French", code: "fr", popular: true },
	{ name: "German", code: "de", popular: true },
	{ name: "Portuguese", code: "pt", popular: true },
	{ name: "Japanese", code: "ja", popular: true },
	{ name: "Korean", code: "ko", popular: true },
	{ name: "Chinese", code: "zh", popular: true },
	{ name: "Arabic", code: "ar", popular: true },
	{ name: "Russian", code: "ru", popular: true },
	{ name: "Italian", code: "it", popular: true },
	// All others alphabetically
	{ name: "Afrikaans", code: "af" },
	{ name: "Armenian", code: "hy" },
	{ name: "Azerbaijani", code: "az" },
	{ name: "Belarusian", code: "be" },
	{ name: "Bengali", code: "bn" },
	{ name: "Bosnian", code: "bs" },
	{ name: "Bulgarian", code: "bg" },
	{ name: "Catalan", code: "ca" },
	{ name: "Croatian", code: "hr" },
	{ name: "Czech", code: "cs" },
	{ name: "Danish", code: "da" },
	{ name: "Dutch", code: "nl" },
	{ name: "Estonian", code: "et" },
	{ name: "Finnish", code: "fi" },
	{ name: "Galician", code: "gl" },
	{ name: "Greek", code: "el" },
	{ name: "Hebrew", code: "he" },
	{ name: "Hungarian", code: "hu" },
	{ name: "Icelandic", code: "is" },
	{ name: "Indonesian", code: "id" },
	{ name: "Kannada", code: "kn" },
	{ name: "Kazakh", code: "kk" },
	{ name: "Latvian", code: "lv" },
	{ name: "Lithuanian", code: "lt" },
	{ name: "Macedonian", code: "mk" },
	{ name: "Malay", code: "ms" },
	{ name: "Maori", code: "mi" },
	{ name: "Marathi", code: "mr" },
	{ name: "Nepali", code: "ne" },
	{ name: "Norwegian", code: "no" },
	{ name: "Persian", code: "fa" },
	{ name: "Polish", code: "pl" },
	{ name: "Romanian", code: "ro" },
	{ name: "Serbian", code: "sr" },
	{ name: "Slovak", code: "sk" },
	{ name: "Slovenian", code: "sl" },
	{ name: "Swahili", code: "sw" },
	{ name: "Swedish", code: "sv" },
	{ name: "Tagalog", code: "tl" },
	{ name: "Tamil", code: "ta" },
	{ name: "Telugu", code: "te" },
	{ name: "Thai", code: "th" },
	{ name: "Turkish", code: "tr" },
	{ name: "Ukrainian", code: "uk" },
	{ name: "Urdu", code: "ur" },
	{ name: "Vietnamese", code: "vi" },
	{ name: "Welsh", code: "cy" },
];

const ENGLISH_ONLY_LANGUAGES: LocalLangEntry[] = [
	{ name: "English", code: "en", popular: true },
];

/**
 * Get the supported language list for a local model.
 * Returns englishOnly=true for Moonshine and Parakeet V2 (no picker needed).
 */
export function getLanguagesForLocalModel(modelId: string): { languages: LocalLangEntry[]; englishOnly: boolean } {
	const model = LOCAL_MODELS.find(m => m.id === modelId);
	if (!model) return { languages: WHISPER_LANGUAGES, englishOnly: false };

	switch (model.langSupport) {
		case "english-only":
			return { languages: ENGLISH_ONLY_LANGUAGES, englishOnly: true };
		case "parakeet-multi":
		case "whisper":
		default:
			return { languages: WHISPER_LANGUAGES, englishOnly: false };
	}
}

/**
 * Check if a language code is supported by a local model.
 * Used to validate /voice-language changes against current model.
 */
export function isLanguageSupportedByModel(modelId: string, langCode: string): boolean {
	const { languages } = getLanguagesForLocalModel(modelId);
	// Match base code (e.g. "en" matches "en", regional variants stripped for local)
	const baseCode = langCode.split("-")[0];
	return languages.some(l => l.code === baseCode || l.code === langCode);
}

/**
 * Find display name for a language code in local model context.
 */
export function localLanguageDisplayName(code: string): string {
	const entry = WHISPER_LANGUAGES.find(l => l.code === code);
	return entry ? `${entry.name} (${entry.code})` : code;
}

// ─── Local session type ──────────────────────────────────────────────────────

export interface LocalSession {
	backend: "local";
	recProcess: ChildProcess;
	audioChunks: Buffer[];
	closed: boolean;
	hadAudioData: boolean;
	onTranscript: (interim: string, finals: string[]) => void;
	onDone: (fullText: string, meta: { hadAudio: boolean; hadSpeech: boolean }) => void;
	onError: (err: string) => void;
}

// ─── WAV encoding ────────────────────────────────────────────────────────────

/** Create a WAV file buffer from raw PCM data (16-bit signed LE, 16kHz, mono). */
function createWavBuffer(pcmData: Buffer): Buffer {
	const header = Buffer.alloc(44);
	const dataSize = pcmData.length;
	const fileSize = 36 + dataSize;

	// RIFF header
	header.write("RIFF", 0);
	header.writeUInt32LE(fileSize, 4);
	header.write("WAVE", 8);

	// fmt chunk
	header.write("fmt ", 12);
	header.writeUInt32LE(16, 16); // chunk size
	header.writeUInt16LE(1, 20); // PCM format
	header.writeUInt16LE(CHANNELS, 22);
	header.writeUInt32LE(SAMPLE_RATE, 24);
	header.writeUInt32LE(SAMPLE_RATE * CHANNELS * 2, 28); // byte rate
	header.writeUInt16LE(CHANNELS * 2, 32); // block align
	header.writeUInt16LE(16, 34); // bits per sample

	// data chunk
	header.write("data", 36);
	header.writeUInt32LE(dataSize, 40);

	return Buffer.concat([header, pcmData]);
}

// ─── Transcription via local server ──────────────────────────────────────────

/**
 * POST audio to a local OpenAI-compatible transcription endpoint.
 * Tries /v1/audio/transcriptions first, falls back to /inference (whisper.cpp native).
 */
export async function transcribeWithServer(
	wavBuffer: Buffer,
	config: VoiceConfig,
): Promise<string> {
	const endpoint = config.localEndpoint || DEFAULT_LOCAL_ENDPOINT;
	const model = config.localModel || DEFAULT_LOCAL_MODEL;
	const language = config.language || "en";

	// Build multipart/form-data manually (no external deps)
	const boundary = `----PiVoice${Date.now()}`;
	const parts: Buffer[] = [];

	// file field
	parts.push(Buffer.from(
		`--${boundary}\r\n` +
		`Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n` +
		`Content-Type: audio/wav\r\n\r\n`,
	));
	parts.push(wavBuffer);
	parts.push(Buffer.from("\r\n"));

	// model field
	parts.push(Buffer.from(
		`--${boundary}\r\n` +
		`Content-Disposition: form-data; name="model"\r\n\r\n` +
		`${model}\r\n`,
	));

	// language field
	parts.push(Buffer.from(
		`--${boundary}\r\n` +
		`Content-Disposition: form-data; name="language"\r\n\r\n` +
		`${language}\r\n`,
	));

	// response_format field
	parts.push(Buffer.from(
		`--${boundary}\r\n` +
		`Content-Disposition: form-data; name="response_format"\r\n\r\n` +
		`json\r\n`,
	));

	parts.push(Buffer.from(`--${boundary}--\r\n`));

	const body = Buffer.concat(parts);

	// Try OpenAI-compatible endpoint first
	const urls = [
		`${endpoint}/v1/audio/transcriptions`,
		`${endpoint}/inference`,
	];

	let lastError = "";
	for (const url of urls) {
		try {
			const resp = await fetch(url, {
				method: "POST",
				headers: {
					"Content-Type": `multipart/form-data; boundary=${boundary}`,
					"Content-Length": String(body.length),
				},
				body,
				signal: AbortSignal.timeout(120_000), // 2 min timeout for large models
			});

			if (!resp.ok) {
				lastError = `HTTP ${resp.status}: ${await resp.text().catch(() => "unknown")}`;
				continue;
			}

			const contentType = resp.headers.get("content-type") || "";
			if (contentType.includes("application/json")) {
				const json = await resp.json() as { text?: string };
				return (json.text || "").trim();
			}
			// Plain text response
			return (await resp.text()).trim();
		} catch (err: any) {
			if (err?.name === "AbortError" || err?.name === "TimeoutError") {
				lastError = "Transcription timed out (120s)";
				break; // Don't retry on timeout
			}
			lastError = err?.message || String(err);
			// Connection refused = server not running, try next URL
			continue;
		}
	}

	throw new Error(lastError || "Could not connect to local transcription server");
}

// ─── Session lifecycle ───────────────────────────────────────────────────────

/**
 * Start a local recording session. Audio is buffered in memory.
 * Transcription happens when stopLocalSession() is called.
 */
export function startLocalSession(
	recProcess: ChildProcess,
	callbacks: {
		onTranscript: (interim: string, finals: string[]) => void;
		onDone: (fullText: string, meta: { hadAudio: boolean; hadSpeech: boolean }) => void;
		onError: (err: string) => void;
	},
): LocalSession {
	const session: LocalSession = {
		backend: "local",
		recProcess,
		audioChunks: [],
		closed: false,
		hadAudioData: false,
		onTranscript: callbacks.onTranscript,
		onDone: callbacks.onDone,
		onError: callbacks.onError,
	};

	recProcess.stdout?.on("data", (chunk: Buffer) => {
		if (!session.closed) {
			session.hadAudioData = true;
			session.audioChunks.push(Buffer.from(chunk));
		}
	});

	recProcess.stderr?.on("data", (d: Buffer) => {
		const msg = d.toString().trim();
		if (msg.includes("buffer overrun") || msg.includes("Discarding") || msg.includes("Last message repeated")) return;
	});

	recProcess.on("error", (err) => {
		if (!session.closed) {
			session.onError(`Audio capture error: ${err.message}`);
		}
	});

	return session;
}

/**
 * Stop recording and transcribe the buffered audio via local server.
 * This is async — transcription takes time with local models.
 */
export async function stopLocalSession(session: LocalSession, config: VoiceConfig): Promise<void> {
	if (session.closed) return;

	// Stop recording
	try { session.recProcess.kill("SIGTERM"); } catch {}

	// Wait briefly for any remaining audio data
	await new Promise((r) => setTimeout(r, 200));

	const pcmData = Buffer.concat(session.audioChunks);

	if (pcmData.length === 0) {
		session.closed = true;
		session.onDone("", { hadAudio: false, hadSpeech: false });
		return;
	}

	const wavBuffer = createWavBuffer(pcmData);

	try {
		const text = await transcribeWithServer(wavBuffer, config);
		session.closed = true;
		session.onDone(text, { hadAudio: true, hadSpeech: text.trim().length > 0 });
	} catch (err: any) {
		session.closed = true;
		session.onError(`Local transcription failed: ${err.message || err}`);
	}
}

/** Abort a local session — kill recording, discard audio. */
export function abortLocalSession(session: LocalSession | null): void {
	if (!session || session.closed) return;
	session.closed = true;
	try { session.recProcess.kill("SIGKILL"); } catch {}
}

/** Check if a local transcription server is reachable. */
export async function checkLocalServer(endpoint?: string): Promise<{ ok: boolean; error?: string }> {
	const url = endpoint || DEFAULT_LOCAL_ENDPOINT;
	try {
		const resp = await fetch(`${url}/v1/models`, {
			signal: AbortSignal.timeout(5000),
		}).catch(() =>
			// whisper.cpp server doesn't have /v1/models, try root
			fetch(url, { signal: AbortSignal.timeout(5000) }),
		);
		return { ok: resp.ok || resp.status === 404 }; // 404 = server is up, just no models endpoint
	} catch (err: any) {
		if (err?.cause?.code === "ECONNREFUSED") {
			return { ok: false, error: `Server not running at ${url}` };
		}
		return { ok: false, error: err?.message || String(err) };
	}
}

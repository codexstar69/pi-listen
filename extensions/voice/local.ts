/**
 * Local transcription backend — in-process STT via sherpa-onnx + external server fallback.
 *
 * Default: sherpa-onnx in-process inference (zero-config, auto-download models).
 * Fallback: External server via POST /v1/audio/transcriptions (advanced users).
 *
 * All 20+ models supported in-process via sherpa-onnx-node.
 * Int8 quantized models used by default for smaller downloads and lower RAM.
 *
 * Architecture:
 *   Deepgram  → real-time streaming (WebSocket, interim results while speaking)
 *   Local     → batch mode (record complete audio, transcribe after stop)
 */

import type { ChildProcess } from "node:child_process";
import type { VoiceConfig } from "./config";
import { SAMPLE_RATE, CHANNELS } from "./deepgram";

// ─── Model catalog ───────────────────────────────────────────────────────────

export interface SherpaModelConfig {
	/** Recognizer type for sherpa-onnx */
	type: "whisper" | "moonshine" | "sense_voice" | "nemo_ctc" | "transducer";
	/** Map of role → filename within model directory */
	files: Record<string, string>;
	/** Map of role → download URL (HuggingFace or GitHub releases) */
	downloadUrls: Record<string, string>;
}

export interface LocalModelInfo {
	id: string;
	name: string;
	/** Human-readable download size (int8 where available) */
	size: string;
	/** Download size in bytes (for progress tracking + fitness scoring) */
	sizeBytes: number;
	/** Peak runtime RAM in MB (~2.5x model file size) */
	runtimeRamMB: number;
	notes: string;
	/** Language family — determines which language list to show */
	langSupport: "whisper" | "english-only" | "parakeet-multi" | "sensevoice" | "russian-only"
		| "single-ar" | "single-zh" | "single-ja" | "single-ko" | "single-uk" | "single-vi" | "single-es";
	/** Device tier: edge (<256 MB), standard (256 MB–1 GB), heavy (>1 GB) */
	tier: "edge" | "standard" | "heavy";
	/** sherpa-onnx model configuration — file paths and download URLs */
	sherpaModel: SherpaModelConfig;
}

// HuggingFace base URL for sherpa-onnx models
const HF = "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-tiny/resolve/main";
const SHERPA_GH = "https://github.com/k2-fsa/sherpa-onnx/releases/download";

export const LOCAL_MODELS: LocalModelInfo[] = [
	// ── Moonshine v2 Streaming — English only, low-latency, best for edge ──
	{
		id: "moonshine-v2-tiny", name: "Moonshine v2 Tiny", size: "~31 MB", sizeBytes: 32_505_856, runtimeRamMB: 80,
		notes: "Streaming, 34M params, 50ms latency, English only", langSupport: "english-only", tier: "edge",
		sherpaModel: { type: "moonshine", files: { preprocessor: "preprocess.onnx", encoder: "encode.int8.onnx", uncachedDecoder: "uncached_decode.int8.onnx", cachedDecoder: "cached_decode.int8.onnx", tokens: "tokens.txt" }, downloadUrls: {} },
	},
	{
		id: "moonshine-v2-small", name: "Moonshine v2 Small", size: "~100 MB", sizeBytes: 104_857_600, runtimeRamMB: 250,
		notes: "Streaming, 123M params, 148ms latency, English only", langSupport: "english-only", tier: "edge",
		sherpaModel: { type: "moonshine", files: { preprocessor: "preprocess.onnx", encoder: "encode.int8.onnx", uncachedDecoder: "uncached_decode.int8.onnx", cachedDecoder: "cached_decode.int8.onnx", tokens: "tokens.txt" }, downloadUrls: {} },
	},
	{
		id: "moonshine-v2-medium", name: "Moonshine v2 Medium", size: "~192 MB", sizeBytes: 201_326_592, runtimeRamMB: 480,
		notes: "Streaming, 245M params, beats Whisper Large v3 WER, English only", langSupport: "english-only", tier: "standard",
		sherpaModel: { type: "moonshine", files: { preprocessor: "preprocess.onnx", encoder: "encode.int8.onnx", uncachedDecoder: "uncached_decode.int8.onnx", cachedDecoder: "cached_decode.int8.onnx", tokens: "tokens.txt" }, downloadUrls: {} },
	},
	// ── Moonshine v1 (Useful Sensors) — English only ──────────────────────
	{
		id: "moonshine-tiny", name: "Moonshine Tiny", size: "~60 MB", sizeBytes: 62_914_560, runtimeRamMB: 150,
		notes: "Ultra-fast, 5x less compute than Whisper, English only", langSupport: "english-only", tier: "edge",
		sherpaModel: { type: "moonshine", files: { preprocessor: "preprocess.onnx", encoder: "encode.int8.onnx", uncachedDecoder: "uncached_decode.int8.onnx", cachedDecoder: "cached_decode.int8.onnx", tokens: "tokens.txt" }, downloadUrls: {} },
	},
	{
		id: "moonshine-base", name: "Moonshine Base", size: "~130 MB", sizeBytes: 136_314_880, runtimeRamMB: 325,
		notes: "Fast and accurate, edge-optimized, English only", langSupport: "english-only", tier: "standard",
		sherpaModel: { type: "moonshine", files: { preprocessor: "preprocess.onnx", encoder: "encode.int8.onnx", uncachedDecoder: "uncached_decode.int8.onnx", cachedDecoder: "cached_decode.int8.onnx", tokens: "tokens.txt" }, downloadUrls: {} },
	},
	// ── Whisper (OpenAI) — multilingual, 57 languages ─────────────────────
	{
		id: "whisper-small", name: "Whisper Small", size: "~180 MB", sizeBytes: 188_743_680, runtimeRamMB: 450,
		notes: "Good balance of speed and accuracy, 57 languages", langSupport: "whisper", tier: "standard",
		sherpaModel: { type: "whisper", files: { encoder: "encoder.int8.onnx", decoder: "decoder.int8.onnx", tokens: "tokens.txt" }, downloadUrls: {} },
	},
	{
		id: "whisper-medium", name: "Whisper Medium", size: "~500 MB", sizeBytes: 524_288_000, runtimeRamMB: 1250,
		notes: "Better accuracy, moderate speed, 57 languages", langSupport: "whisper", tier: "standard",
		sherpaModel: { type: "whisper", files: { encoder: "encoder.int8.onnx", decoder: "decoder.int8.onnx", tokens: "tokens.txt" }, downloadUrls: {} },
	},
	{
		id: "whisper-turbo", name: "Whisper Turbo", size: "~574 MB", sizeBytes: 601_882_624, runtimeRamMB: 1400,
		notes: "Fast and accurate, 57 languages", langSupport: "whisper", tier: "heavy",
		sherpaModel: { type: "whisper", files: { encoder: "encoder.int8.onnx", decoder: "decoder.int8.onnx", tokens: "tokens.txt" }, downloadUrls: {} },
	},
	{
		id: "whisper-large", name: "Whisper Large v3", size: "~1.0 GB", sizeBytes: 1_073_741_824, runtimeRamMB: 2500,
		notes: "Best accuracy, slowest, 57 languages", langSupport: "whisper", tier: "heavy",
		sherpaModel: { type: "whisper", files: { encoder: "encoder.int8.onnx", decoder: "decoder.int8.onnx", tokens: "tokens.txt" }, downloadUrls: {} },
	},
	// ── Moonshine Flavors — single-language specialized tiny models ────────
	{
		id: "moonshine-tiny-ar", name: "Moonshine Tiny Arabic", size: "~60 MB", sizeBytes: 62_914_560, runtimeRamMB: 150,
		notes: "Arabic-specialized, 27M params", langSupport: "single-ar", tier: "edge",
		sherpaModel: { type: "moonshine", files: { preprocessor: "preprocess.onnx", encoder: "encode.int8.onnx", uncachedDecoder: "uncached_decode.int8.onnx", cachedDecoder: "cached_decode.int8.onnx", tokens: "tokens.txt" }, downloadUrls: {} },
	},
	{
		id: "moonshine-tiny-zh", name: "Moonshine Tiny Chinese", size: "~60 MB", sizeBytes: 62_914_560, runtimeRamMB: 150,
		notes: "Chinese-specialized, 27M params", langSupport: "single-zh", tier: "edge",
		sherpaModel: { type: "moonshine", files: { preprocessor: "preprocess.onnx", encoder: "encode.int8.onnx", uncachedDecoder: "uncached_decode.int8.onnx", cachedDecoder: "cached_decode.int8.onnx", tokens: "tokens.txt" }, downloadUrls: {} },
	},
	{
		id: "moonshine-tiny-ja", name: "Moonshine Tiny Japanese", size: "~60 MB", sizeBytes: 62_914_560, runtimeRamMB: 150,
		notes: "Japanese-specialized, 27M params", langSupport: "single-ja", tier: "edge",
		sherpaModel: { type: "moonshine", files: { preprocessor: "preprocess.onnx", encoder: "encode.int8.onnx", uncachedDecoder: "uncached_decode.int8.onnx", cachedDecoder: "cached_decode.int8.onnx", tokens: "tokens.txt" }, downloadUrls: {} },
	},
	{
		id: "moonshine-tiny-ko", name: "Moonshine Tiny Korean", size: "~60 MB", sizeBytes: 62_914_560, runtimeRamMB: 150,
		notes: "Korean-specialized, 27M params", langSupport: "single-ko", tier: "edge",
		sherpaModel: { type: "moonshine", files: { preprocessor: "preprocess.onnx", encoder: "encode.int8.onnx", uncachedDecoder: "uncached_decode.int8.onnx", cachedDecoder: "cached_decode.int8.onnx", tokens: "tokens.txt" }, downloadUrls: {} },
	},
	{
		id: "moonshine-tiny-uk", name: "Moonshine Tiny Ukrainian", size: "~60 MB", sizeBytes: 62_914_560, runtimeRamMB: 150,
		notes: "Ukrainian-specialized, 27M params", langSupport: "single-uk", tier: "edge",
		sherpaModel: { type: "moonshine", files: { preprocessor: "preprocess.onnx", encoder: "encode.int8.onnx", uncachedDecoder: "uncached_decode.int8.onnx", cachedDecoder: "cached_decode.int8.onnx", tokens: "tokens.txt" }, downloadUrls: {} },
	},
	{
		id: "moonshine-tiny-vi", name: "Moonshine Tiny Vietnamese", size: "~60 MB", sizeBytes: 62_914_560, runtimeRamMB: 150,
		notes: "Vietnamese-specialized, 27M params", langSupport: "single-vi", tier: "edge",
		sherpaModel: { type: "moonshine", files: { preprocessor: "preprocess.onnx", encoder: "encode.int8.onnx", uncachedDecoder: "uncached_decode.int8.onnx", cachedDecoder: "cached_decode.int8.onnx", tokens: "tokens.txt" }, downloadUrls: {} },
	},
	{
		id: "moonshine-base-es", name: "Moonshine Base Spanish", size: "~130 MB", sizeBytes: 136_314_880, runtimeRamMB: 325,
		notes: "Spanish-specialized, 61M params", langSupport: "single-es", tier: "standard",
		sherpaModel: { type: "moonshine", files: { preprocessor: "preprocess.onnx", encoder: "encode.int8.onnx", uncachedDecoder: "uncached_decode.int8.onnx", cachedDecoder: "cached_decode.int8.onnx", tokens: "tokens.txt" }, downloadUrls: {} },
	},
	// ── SenseVoice (Alibaba/FunAudioLLM) — 5 languages ───────────────────
	{
		id: "sensevoice-small", name: "SenseVoice Small", size: "~228 MB", sizeBytes: 239_075_328, runtimeRamMB: 570,
		notes: "5 languages (zh/en/ja/ko/yue), 244M params, ultra-fast batch", langSupport: "sensevoice", tier: "standard",
		sherpaModel: { type: "sense_voice", files: { model: "model.int8.onnx", tokens: "tokens.txt" }, downloadUrls: {} },
	},
	// ── GigaAM v3 (Sber/Salute) — Russian only ───────────────────────────
	{
		id: "gigaam-v3", name: "GigaAM v3", size: "~225 MB", sizeBytes: 235_929_600, runtimeRamMB: 560,
		notes: "Russian only, 220M params, 50% lower WER than Whisper on Russian", langSupport: "russian-only", tier: "standard",
		sherpaModel: { type: "nemo_ctc", files: { model: "model.int8.onnx", tokens: "tokens.txt" }, downloadUrls: {} },
	},
	// ── Parakeet (NVIDIA) ─────────────────────────────────────────────────
	{
		id: "parakeet-v2", name: "Parakeet V2", size: "~473 MB", sizeBytes: 495_976_448, runtimeRamMB: 1180,
		notes: "CPU-optimized, English only, NVIDIA NeMo", langSupport: "english-only", tier: "standard",
		sherpaModel: { type: "nemo_ctc", files: { model: "model.int8.onnx", tokens: "tokens.txt" }, downloadUrls: {} },
	},
	{
		id: "parakeet-v3", name: "Parakeet V3", size: "~478 MB", sizeBytes: 501_219_328, runtimeRamMB: 1195,
		notes: "CPU-optimized, auto language detection, NVIDIA NeMo", langSupport: "parakeet-multi", tier: "standard",
		sherpaModel: { type: "nemo_ctc", files: { model: "model.int8.onnx", tokens: "tokens.txt" }, downloadUrls: {} },
	},
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

const SENSEVOICE_LANGUAGES: LocalLangEntry[] = [
	{ name: "Chinese (Mandarin)", code: "zh", popular: true },
	{ name: "English", code: "en", popular: true },
	{ name: "Japanese", code: "ja", popular: true },
	{ name: "Korean", code: "ko", popular: true },
	{ name: "Cantonese", code: "yue", popular: true },
];

const RUSSIAN_ONLY_LANGUAGES: LocalLangEntry[] = [
	{ name: "Russian", code: "ru", popular: true },
];

// Single-language lists for Moonshine Flavors
const SINGLE_LANG: Record<string, LocalLangEntry[]> = {
	ar: [{ name: "Arabic", code: "ar", popular: true }],
	zh: [{ name: "Chinese", code: "zh", popular: true }],
	ja: [{ name: "Japanese", code: "ja", popular: true }],
	ko: [{ name: "Korean", code: "ko", popular: true }],
	uk: [{ name: "Ukrainian", code: "uk", popular: true }],
	vi: [{ name: "Vietnamese", code: "vi", popular: true }],
	es: [{ name: "Spanish", code: "es", popular: true }],
};

/**
 * Get the supported language list for a local model.
 * Returns englishOnly=true when only one language is supported (no picker needed).
 */
export function getLanguagesForLocalModel(modelId: string): { languages: LocalLangEntry[]; englishOnly: boolean } {
	const model = LOCAL_MODELS.find(m => m.id === modelId);
	if (!model) return { languages: WHISPER_LANGUAGES, englishOnly: false };

	switch (model.langSupport) {
		case "english-only":
			return { languages: ENGLISH_ONLY_LANGUAGES, englishOnly: true };
		case "russian-only":
			return { languages: RUSSIAN_ONLY_LANGUAGES, englishOnly: true };
		case "single-ar": return { languages: SINGLE_LANG.ar!, englishOnly: true };
		case "single-zh": return { languages: SINGLE_LANG.zh!, englishOnly: true };
		case "single-ja": return { languages: SINGLE_LANG.ja!, englishOnly: true };
		case "single-ko": return { languages: SINGLE_LANG.ko!, englishOnly: true };
		case "single-uk": return { languages: SINGLE_LANG.uk!, englishOnly: true };
		case "single-vi": return { languages: SINGLE_LANG.vi!, englishOnly: true };
		case "single-es": return { languages: SINGLE_LANG.es!, englishOnly: true };
		case "sensevoice":
			return { languages: SENSEVOICE_LANGUAGES, englishOnly: false };
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
	// Check all language lists
	const allLists = [WHISPER_LANGUAGES, SENSEVOICE_LANGUAGES, RUSSIAN_ONLY_LANGUAGES, ...Object.values(SINGLE_LANG)];
	for (const list of allLists) {
		const entry = list.find(l => l.code === code);
		if (entry) return `${entry.name} (${entry.code})`;
	}
	return code;
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
 * Stop recording and transcribe the buffered audio.
 *
 * Routes transcription based on config:
 *   - If localEndpoint is set → external server (advanced users)
 *   - Otherwise → sherpa-onnx in-process (default, zero-config)
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

	try {
		let text: string;

		if (config.localEndpoint) {
			// External server mode (advanced override)
			const wavBuffer = createWavBuffer(pcmData);
			text = await transcribeWithServer(wavBuffer, config);
		} else {
			// In-process via sherpa-onnx (default)
			text = await transcribeInProcess(pcmData, config);
		}

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

// ─── In-process transcription via sherpa-onnx ────────────────────────────────

/**
 * Transcribe PCM audio using sherpa-onnx in-process.
 * Auto-downloads model on first use.
 */
async function transcribeInProcess(pcmData: Buffer, config: VoiceConfig): Promise<string> {
	const { initSherpa, isSherpaAvailable, getSherpaError, getOrCreateRecognizer, transcribeBuffer } = await import("./sherpa-engine");
	const { ensureModelDownloaded } = await import("./model-download");

	// Initialize sherpa if needed
	if (!isSherpaAvailable()) {
		const ok = await initSherpa();
		if (!ok) {
			throw new Error(`sherpa-onnx not available: ${getSherpaError() || "unknown error"}. Set localEndpoint in config to use an external server instead.`);
		}
	}

	const model = LOCAL_MODELS.find(m => m.id === (config.localModel || DEFAULT_LOCAL_MODEL));
	if (!model) throw new Error(`Unknown model: ${config.localModel}`);

	// Ensure model files are downloaded
	const modelDir = await ensureModelDownloaded(
		model.id,
		model.sherpaModel.downloadUrls,
		model.sizeBytes,
	);

	// Create/reuse recognizer and transcribe
	const recognizer = getOrCreateRecognizer(model, modelDir, config.language || "en");
	return transcribeBuffer(pcmData, recognizer);
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

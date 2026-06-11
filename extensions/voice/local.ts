/**
 * Local transcription backend — in-process STT via sherpa-onnx + external server fallback.
 *
 * Default: sherpa-onnx in-process inference (zero-config, auto-download models).
 * Fallback: External server via POST /v1/audio/transcriptions (advanced users).
 *
 * Model catalog verified against HuggingFace repos:
 *   - csukuangfj/ and csukuangfj2/ repos on huggingface.co
 *   - k2-fsa/sherpa-onnx GitHub releases (asr-models tag)
 *
 * Architecture:
 *   Deepgram  → real-time streaming (WebSocket, interim results while speaking)
 *   Local     → batch mode (record complete audio, transcribe after stop)
 */

import type { ChildProcess } from "node:child_process";
import type { VoiceConfig } from "./config";
import { isLoopbackEndpoint } from "./config";
import { SAMPLE_RATE, CHANNELS } from "./deepgram";
import { resolveSixtyDbApiKey, SIXTYDB_STT_URL } from "./sixtydb";

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
	/** Human-readable download size */
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
	/** Preferred model — best-in-class for its language/use case. Only these get [recommended]. */
	preferred?: boolean;
	/** Accuracy rating 1-5 (5 = best). Based on published WER benchmarks. */
	accuracy: 1 | 2 | 3 | 4 | 5;
	/** Speed rating 1-5 (5 = fastest). Based on real-time factor and latency benchmarks. */
	speed: 1 | 2 | 3 | 4 | 5;
	/** sherpa-onnx model configuration — file paths and download URLs */
	sherpaModel: SherpaModelConfig;
}

// HuggingFace base URLs for sherpa-onnx models (verified repos)
const HF1 = "https://huggingface.co/csukuangfj";
const HF2 = "https://huggingface.co/csukuangfj2";
const GH = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models";

// Helper to build HuggingFace resolve URLs
function hf1(repo: string, file: string): string {
	return `${HF1}/${repo}/resolve/main/${file}`;
}
function hf2(repo: string, file: string): string {
	return `${HF2}/${repo}/resolve/main/${file}`;
}

/**
 * Model catalog — verified against actual HuggingFace repos and file listings.
 *
 * Evidence:
 * - Moonshine v1 (csukuangfj): {preprocess.onnx, encode.int8.onnx, uncached_decode.int8.onnx, cached_decode.int8.onnx, tokens.txt}
 * - Moonshine v2 (csukuangfj2): {encoder_model.ort, decoder_model_merged.ort, tokens.txt}
 * - Whisper (csukuangfj): {SIZE-encoder.int8.onnx, SIZE-decoder.int8.onnx, SIZE-tokens.txt}
 * - SenseVoice (csukuangfj): {model.int8.onnx, tokens.txt}
 * - GigaAM CTC (csukuangfj): {model.int8.onnx, tokens.txt}
 * - Parakeet TDT (csukuangfj): {encoder.int8.onnx, decoder.int8.onnx, joiner.int8.onnx, tokens.txt} — transducer!
 *
 * Note on Moonshine v2 Small/Medium:
 *   These exist ONLY as streaming models (moonshine-ai/moonshine) with a different 5-file
 *   architecture (encoder.ort, frontend.ort, decoder_kv.ort, cross_kv.ort, adapter.ort)
 *   that is incompatible with sherpa-onnx's moonshine recognizer (which expects 2-file
 *   encoder+mergedDecoder or 4-file v1 structure). Only Tiny and Base have non-streaming
 *   variants compatible with sherpa-onnx. See: https://github.com/moonshine-ai/moonshine
 */
export const LOCAL_MODELS: LocalModelInfo[] = [
	// ═══════════════════════════════════════════════════════════════════════
	// TOP PICKS — best overall models, shown first
	// ═══════════════════════════════════════════════════════════════════════
	{
		id: "parakeet-v3", name: "Parakeet TDT v3", size: "~671 MB", sizeBytes: 703_594_496, runtimeRamMB: 1675,
		notes: "Best multilingual — 25 languages, auto language detection, WER 6.3%", langSupport: "parakeet-multi", tier: "standard", preferred: true, accuracy: 4, speed: 4,
		sherpaModel: {
			type: "transducer",
			files: { encoder: "encoder.int8.onnx", decoder: "decoder.int8.onnx", joiner: "joiner.int8.onnx", tokens: "tokens.txt" },
			downloadUrls: {
				encoder: hf1("sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8", "encoder.int8.onnx"),
				decoder: hf1("sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8", "decoder.int8.onnx"),
				joiner: hf1("sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8", "joiner.int8.onnx"),
				tokens: hf1("sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8", "tokens.txt"),
			},
		},
	},
	{
		id: "parakeet-v2", name: "Parakeet TDT v2", size: "~661 MB", sizeBytes: 693_109_760, runtimeRamMB: 1650,
		notes: "Best English — lowest WER (6.0%), fast, NVIDIA NeMo", langSupport: "english-only", tier: "standard", preferred: true, accuracy: 5, speed: 4,
		sherpaModel: {
			type: "transducer",
			files: { encoder: "encoder.int8.onnx", decoder: "decoder.int8.onnx", joiner: "joiner.int8.onnx", tokens: "tokens.txt" },
			downloadUrls: {
				encoder: hf1("sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8", "encoder.int8.onnx"),
				decoder: hf1("sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8", "decoder.int8.onnx"),
				joiner: hf1("sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8", "joiner.int8.onnx"),
				tokens: hf1("sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8", "tokens.txt"),
			},
		},
	},
	// ═══════════════════════════════════════════════════════════════════════
	// WHISPER — OpenAI, broadest language support (57 languages)
	// ═══════════════════════════════════════════════════════════════════════
	{
		id: "whisper-turbo", name: "Whisper Turbo", size: "~1.0 GB", sizeBytes: 1_087_373_312, runtimeRamMB: 2590,
		notes: "57 languages, good accuracy, faster than Medium and Large", langSupport: "whisper", tier: "heavy", accuracy: 4, speed: 2,
		sherpaModel: {
			type: "whisper",
			files: { encoder: "turbo-encoder.int8.onnx", decoder: "turbo-decoder.int8.onnx", tokens: "turbo-tokens.txt" },
			downloadUrls: {
				encoder: hf1("sherpa-onnx-whisper-turbo", "turbo-encoder.int8.onnx"),
				decoder: hf1("sherpa-onnx-whisper-turbo", "turbo-decoder.int8.onnx"),
				tokens: hf1("sherpa-onnx-whisper-turbo", "turbo-tokens.txt"),
			},
		},
	},
	{
		id: "whisper-medium", name: "Whisper Medium", size: "~946 MB", sizeBytes: 991_952_896, runtimeRamMB: 2365,
		notes: "57 languages, good accuracy, medium speed", langSupport: "whisper", tier: "standard", accuracy: 4, speed: 3,
		sherpaModel: {
			type: "whisper",
			files: { encoder: "medium-encoder.int8.onnx", decoder: "medium-decoder.int8.onnx", tokens: "medium-tokens.txt" },
			downloadUrls: {
				encoder: hf1("sherpa-onnx-whisper-medium", "medium-encoder.int8.onnx"),
				decoder: hf1("sherpa-onnx-whisper-medium", "medium-decoder.int8.onnx"),
				tokens: hf1("sherpa-onnx-whisper-medium", "medium-tokens.txt"),
			},
		},
	},
	{
		id: "whisper-small", name: "Whisper Small", size: "~375 MB", sizeBytes: 393_216_000, runtimeRamMB: 940,
		notes: "57 languages, fast, good for low-power devices", langSupport: "whisper", tier: "standard", accuracy: 3, speed: 4,
		sherpaModel: {
			type: "whisper",
			files: { encoder: "small-encoder.int8.onnx", decoder: "small-decoder.int8.onnx", tokens: "small-tokens.txt" },
			downloadUrls: {
				encoder: hf1("sherpa-onnx-whisper-small", "small-encoder.int8.onnx"),
				decoder: hf1("sherpa-onnx-whisper-small", "small-decoder.int8.onnx"),
				tokens: hf1("sherpa-onnx-whisper-small", "small-tokens.txt"),
			},
		},
	},
	{
		id: "whisper-large", name: "Whisper Large v3", size: "~1.8 GB", sizeBytes: 1_863_319_552, runtimeRamMB: 4440,
		notes: "57 languages, highest Whisper accuracy, slow on CPU", langSupport: "whisper", tier: "heavy", accuracy: 4, speed: 1,
		sherpaModel: {
			type: "whisper",
			files: { encoder: "large-v3-encoder.int8.onnx", decoder: "large-v3-decoder.int8.onnx", tokens: "large-v3-tokens.txt" },
			downloadUrls: {
				encoder: hf1("sherpa-onnx-whisper-large-v3", "large-v3-encoder.int8.onnx"),
				decoder: hf1("sherpa-onnx-whisper-large-v3", "large-v3-decoder.int8.onnx"),
				tokens: hf1("sherpa-onnx-whisper-large-v3", "large-v3-tokens.txt"),
			},
		},
	},
	// ═══════════════════════════════════════════════════════════════════════
	// MOONSHINE — ultra-fast edge models
	// ═══════════════════════════════════════════════════════════════════════
	{
		id: "moonshine-base", name: "Moonshine Base", size: "~287 MB", sizeBytes: 300_940_288, runtimeRamMB: 720,
		notes: "English only, very fast, handles accents well", langSupport: "english-only", tier: "standard", accuracy: 3, speed: 5,
		sherpaModel: {
			type: "moonshine",
			files: { preprocessor: "preprocess.onnx", encoder: "encode.int8.onnx", uncachedDecoder: "uncached_decode.int8.onnx", cachedDecoder: "cached_decode.int8.onnx", tokens: "tokens.txt" },
			downloadUrls: {
				preprocessor: hf1("sherpa-onnx-moonshine-base-en-int8", "preprocess.onnx"),
				encoder: hf1("sherpa-onnx-moonshine-base-en-int8", "encode.int8.onnx"),
				uncachedDecoder: hf1("sherpa-onnx-moonshine-base-en-int8", "uncached_decode.int8.onnx"),
				cachedDecoder: hf1("sherpa-onnx-moonshine-base-en-int8", "cached_decode.int8.onnx"),
				tokens: hf1("sherpa-onnx-moonshine-base-en-int8", "tokens.txt"),
			},
		},
	},
	{
		id: "moonshine-tiny", name: "Moonshine Tiny", size: "~124 MB", sizeBytes: 130_023_424, runtimeRamMB: 310,
		notes: "English only, 5x faster than Whisper Tiny, low accuracy", langSupport: "english-only", tier: "edge", accuracy: 2, speed: 5,
		sherpaModel: {
			type: "moonshine",
			files: { preprocessor: "preprocess.onnx", encoder: "encode.int8.onnx", uncachedDecoder: "uncached_decode.int8.onnx", cachedDecoder: "cached_decode.int8.onnx", tokens: "tokens.txt" },
			downloadUrls: {
				preprocessor: hf1("sherpa-onnx-moonshine-tiny-en-int8", "preprocess.onnx"),
				encoder: hf1("sherpa-onnx-moonshine-tiny-en-int8", "encode.int8.onnx"),
				uncachedDecoder: hf1("sherpa-onnx-moonshine-tiny-en-int8", "uncached_decode.int8.onnx"),
				cachedDecoder: hf1("sherpa-onnx-moonshine-tiny-en-int8", "cached_decode.int8.onnx"),
				tokens: hf1("sherpa-onnx-moonshine-tiny-en-int8", "tokens.txt"),
			},
		},
	},
	{
		id: "moonshine-v2-tiny", name: "Moonshine v2 Tiny", size: "~43 MB", sizeBytes: 45_088_768, runtimeRamMB: 110,
		notes: "English only, smallest model, 34ms latency, Raspberry Pi friendly", langSupport: "english-only", tier: "edge", preferred: true, accuracy: 2, speed: 5,
		sherpaModel: {
			type: "moonshine",
			files: { encoder: "encoder_model.ort", mergedDecoder: "decoder_model_merged.ort", tokens: "tokens.txt" },
			downloadUrls: {
				encoder: hf2("sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27", "encoder_model.ort"),
				mergedDecoder: hf2("sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27", "decoder_model_merged.ort"),
				tokens: hf2("sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27", "tokens.txt"),
			},
		},
	},
	// ═══════════════════════════════════════════════════════════════════════
	// SPECIALIST — best-in-class for specific languages
	// ═══════════════════════════════════════════════════════════════════════
	{
		id: "sensevoice-small", name: "SenseVoice Small", size: "~228 MB", sizeBytes: 239_075_328, runtimeRamMB: 570,
		notes: "Chinese, English, Japanese, Korean, Cantonese — very fast", langSupport: "sensevoice", tier: "edge", preferred: true, accuracy: 3, speed: 5,
		sherpaModel: {
			type: "sense_voice",
			files: { model: "model.int8.onnx", tokens: "tokens.txt" },
			downloadUrls: {
				model: hf1("sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17", "model.int8.onnx"),
				tokens: hf1("sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17", "tokens.txt"),
			},
		},
	},
	{
		id: "gigaam-v3", name: "GigaAM v3", size: "~225 MB", sizeBytes: 235_929_600, runtimeRamMB: 560,
		notes: "Russian — fast and accurate, 50% lower WER than Whisper", langSupport: "russian-only", tier: "edge", preferred: true, accuracy: 4, speed: 4,
		sherpaModel: {
			type: "nemo_ctc",
			files: { model: "model.int8.onnx", tokens: "tokens.txt" },
			downloadUrls: {
				model: hf1("sherpa-onnx-nemo-ctc-giga-am-v3-russian-2025-12-16", "model.int8.onnx"),
				tokens: hf1("sherpa-onnx-nemo-ctc-giga-am-v3-russian-2025-12-16", "tokens.txt"),
			},
		},
	},
	// ═══════════════════════════════════════════════════════════════════════
	// MOONSHINE v2 LANGUAGE VARIANTS — fast, single-language specialized
	// ═══════════════════════════════════════════════════════════════════════
	{
		id: "moonshine-v2-tiny-ja", name: "Moonshine v2 Tiny Japanese", size: "~69 MB", sizeBytes: 72_351_744, runtimeRamMB: 175,
		notes: "Japanese-specialized, ultra-fast", langSupport: "single-ja", tier: "edge", accuracy: 3, speed: 5,
		sherpaModel: {
			type: "moonshine",
			files: { encoder: "encoder_model.ort", mergedDecoder: "decoder_model_merged.ort", tokens: "tokens.txt" },
			downloadUrls: {
				encoder: hf2("sherpa-onnx-moonshine-tiny-ja-quantized-2026-02-27", "encoder_model.ort"),
				mergedDecoder: hf2("sherpa-onnx-moonshine-tiny-ja-quantized-2026-02-27", "decoder_model_merged.ort"),
				tokens: hf2("sherpa-onnx-moonshine-tiny-ja-quantized-2026-02-27", "tokens.txt"),
			},
		},
	},
	{
		id: "moonshine-v2-tiny-ko", name: "Moonshine v2 Tiny Korean", size: "~69 MB", sizeBytes: 72_351_744, runtimeRamMB: 175,
		notes: "Korean-specialized, ultra-fast", langSupport: "single-ko", tier: "edge", accuracy: 3, speed: 5,
		sherpaModel: {
			type: "moonshine",
			files: { encoder: "encoder_model.ort", mergedDecoder: "decoder_model_merged.ort", tokens: "tokens.txt" },
			downloadUrls: {
				encoder: hf2("sherpa-onnx-moonshine-tiny-ko-quantized-2026-02-27", "encoder_model.ort"),
				mergedDecoder: hf2("sherpa-onnx-moonshine-tiny-ko-quantized-2026-02-27", "decoder_model_merged.ort"),
				tokens: hf2("sherpa-onnx-moonshine-tiny-ko-quantized-2026-02-27", "tokens.txt"),
			},
		},
	},
	{
		id: "moonshine-v2-base-ar", name: "Moonshine v2 Base Arabic", size: "~135 MB", sizeBytes: 141_557_760, runtimeRamMB: 340,
		notes: "Arabic-specialized", langSupport: "single-ar", tier: "edge", accuracy: 3, speed: 5,
		sherpaModel: {
			type: "moonshine",
			files: { encoder: "encoder_model.ort", mergedDecoder: "decoder_model_merged.ort", tokens: "tokens.txt" },
			downloadUrls: {
				encoder: hf2("sherpa-onnx-moonshine-base-ar-quantized-2026-02-27", "encoder_model.ort"),
				mergedDecoder: hf2("sherpa-onnx-moonshine-base-ar-quantized-2026-02-27", "decoder_model_merged.ort"),
				tokens: hf2("sherpa-onnx-moonshine-base-ar-quantized-2026-02-27", "tokens.txt"),
			},
		},
	},
	{
		id: "moonshine-v2-base-zh", name: "Moonshine v2 Base Chinese", size: "~135 MB", sizeBytes: 141_557_760, runtimeRamMB: 340,
		notes: "Chinese-specialized", langSupport: "single-zh", tier: "edge", accuracy: 3, speed: 5,
		sherpaModel: {
			type: "moonshine",
			files: { encoder: "encoder_model.ort", mergedDecoder: "decoder_model_merged.ort", tokens: "tokens.txt" },
			downloadUrls: {
				encoder: hf2("sherpa-onnx-moonshine-base-zh-quantized-2026-02-27", "encoder_model.ort"),
				mergedDecoder: hf2("sherpa-onnx-moonshine-base-zh-quantized-2026-02-27", "decoder_model_merged.ort"),
				tokens: hf2("sherpa-onnx-moonshine-base-zh-quantized-2026-02-27", "tokens.txt"),
			},
		},
	},
	{
		id: "moonshine-v2-base-ja", name: "Moonshine v2 Base Japanese", size: "~135 MB", sizeBytes: 141_557_760, runtimeRamMB: 340,
		notes: "Japanese-specialized", langSupport: "single-ja", tier: "edge", accuracy: 3, speed: 4,
		sherpaModel: {
			type: "moonshine",
			files: { encoder: "encoder_model.ort", mergedDecoder: "decoder_model_merged.ort", tokens: "tokens.txt" },
			downloadUrls: {
				encoder: hf2("sherpa-onnx-moonshine-base-ja-quantized-2026-02-27", "encoder_model.ort"),
				mergedDecoder: hf2("sherpa-onnx-moonshine-base-ja-quantized-2026-02-27", "decoder_model_merged.ort"),
				tokens: hf2("sherpa-onnx-moonshine-base-ja-quantized-2026-02-27", "tokens.txt"),
			},
		},
	},
	{
		id: "moonshine-v2-base-uk", name: "Moonshine v2 Base Ukrainian", size: "~135 MB", sizeBytes: 141_557_760, runtimeRamMB: 340,
		notes: "Ukrainian-specialized", langSupport: "single-uk", tier: "edge", accuracy: 3, speed: 4,
		sherpaModel: {
			type: "moonshine",
			files: { encoder: "encoder_model.ort", mergedDecoder: "decoder_model_merged.ort", tokens: "tokens.txt" },
			downloadUrls: {
				encoder: hf2("sherpa-onnx-moonshine-base-uk-quantized-2026-02-27", "encoder_model.ort"),
				mergedDecoder: hf2("sherpa-onnx-moonshine-base-uk-quantized-2026-02-27", "decoder_model_merged.ort"),
				tokens: hf2("sherpa-onnx-moonshine-base-uk-quantized-2026-02-27", "tokens.txt"),
			},
		},
	},
	{
		id: "moonshine-v2-base-vi", name: "Moonshine v2 Base Vietnamese", size: "~135 MB", sizeBytes: 141_557_760, runtimeRamMB: 340,
		notes: "Vietnamese-specialized", langSupport: "single-vi", tier: "edge", accuracy: 3, speed: 4,
		sherpaModel: {
			type: "moonshine",
			files: { encoder: "encoder_model.ort", mergedDecoder: "decoder_model_merged.ort", tokens: "tokens.txt" },
			downloadUrls: {
				encoder: hf2("sherpa-onnx-moonshine-base-vi-quantized-2026-02-27", "encoder_model.ort"),
				mergedDecoder: hf2("sherpa-onnx-moonshine-base-vi-quantized-2026-02-27", "decoder_model_merged.ort"),
				tokens: hf2("sherpa-onnx-moonshine-base-vi-quantized-2026-02-27", "tokens.txt"),
			},
		},
	},
	{
		id: "moonshine-v2-base-es", name: "Moonshine v2 Base Spanish", size: "~63 MB", sizeBytes: 66_060_288, runtimeRamMB: 160,
		notes: "Spanish-specialized", langSupport: "single-es", tier: "edge", accuracy: 3, speed: 5,
		sherpaModel: {
			type: "moonshine",
			files: { encoder: "encoder_model.ort", mergedDecoder: "decoder_model_merged.ort", tokens: "tokens.txt" },
			downloadUrls: {
				encoder: hf2("sherpa-onnx-moonshine-base-es-quantized-2026-02-27", "encoder_model.ort"),
				mergedDecoder: hf2("sherpa-onnx-moonshine-base-es-quantized-2026-02-27", "decoder_model_merged.ort"),
				tokens: hf2("sherpa-onnx-moonshine-base-es-quantized-2026-02-27", "tokens.txt"),
			},
		},
	},
];

export const DEFAULT_LOCAL_ENDPOINT = "http://localhost:8080";
export const DEFAULT_LOCAL_MODEL = "parakeet-v3";

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

	// Security: refuse to send audio to non-loopback endpoints
	if (!isLoopbackEndpoint(endpoint)) {
		throw new Error(`Refusing to send audio to non-local endpoint: ${endpoint}. Only localhost/127.0.0.1/::1 allowed.`);
	}

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

// ─── Transcription via 60db cloud (batch /stt) ───────────────────────────────

/**
 * POST the recorded clip to 60db's `/stt` endpoint and return the
 * transcript. 60db STT is batch-only (no live interim transcripts), so it
 * reuses the same buffer-then-upload machinery as the local backend — the
 * only difference is the destination and `Authorization: Bearer` auth.
 *
 * Endpoint contract (https://docs.60db.ai/api-reference/stt/speech-to-text):
 *   POST https://api.60db.ai/stt   (multipart/form-data)
 *   Headers: Authorization: Bearer <key>
 *   Fields:  file=<audio.wav>, language=<iso|auto>
 *   Returns: JSON { text, language, ... }
 *
 * Max upload is 10 MB; at 16 kHz mono 16-bit that's ~5 min of audio, well
 * above the MAX_RECORDING_SECS cap, so we don't pre-check size here.
 */
export async function transcribeWith60db(
	wavBuffer: Buffer,
	config: VoiceConfig,
): Promise<string> {
	const apiKey = resolveSixtyDbApiKey(config);
	if (!apiKey) {
		throw new Error("SIXTYDB_API_KEY not set. Run /voice-settings to configure it, or export SIXTYDB_API_KEY in your shell.");
	}

	const language = config.language || "auto";

	// Build multipart/form-data manually (no external deps), mirroring
	// transcribeWithServer above.
	const boundary = `----PiVoice60db${Date.now()}`;
	const parts: Buffer[] = [];

	parts.push(Buffer.from(
		`--${boundary}\r\n` +
		`Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n` +
		`Content-Type: audio/wav\r\n\r\n`,
	));
	parts.push(wavBuffer);
	parts.push(Buffer.from("\r\n"));

	parts.push(Buffer.from(
		`--${boundary}\r\n` +
		`Content-Disposition: form-data; name="language"\r\n\r\n` +
		`${language}\r\n`,
	));

	parts.push(Buffer.from(`--${boundary}--\r\n`));
	const body = Buffer.concat(parts);

	let resp: Response;
	try {
		resp = await fetch(SIXTYDB_STT_URL, {
			method: "POST",
			headers: {
				"Authorization": `Bearer ${apiKey}`,
				"Content-Type": `multipart/form-data; boundary=${boundary}`,
				"Content-Length": String(body.length),
				"Accept": "application/json",
			},
			body,
			signal: AbortSignal.timeout(120_000),
		});
	} catch (err: any) {
		if (err?.name === "AbortError" || err?.name === "TimeoutError") {
			throw new Error("60db transcription timed out (120s)");
		}
		throw new Error(`60db network error: ${err?.message ?? String(err)}`);
	}

	if (!resp.ok) {
		const detail = await resp.text().catch(() => "unknown");
		throw new Error(`60db /stt HTTP ${resp.status}: ${detail.slice(0, 200)}`);
	}

	const json = await resp.json().catch(() => null) as { text?: string } | null;
	if (!json || typeof json.text !== "string") {
		throw new Error("60db /stt returned no transcript");
	}
	return json.text.trim();
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
			session.audioChunks.push(chunk);
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

	// Recheck after await — abort may have fired during the 200ms wait.
	// Still call onDone so the voice state machine transitions back to idle.
	if (session.closed) {
		session.onDone("", { hadAudio: false, hadSpeech: false });
		return;
	}

	const pcmData = Buffer.concat(session.audioChunks);
	// Free individual chunk references during transcription
	session.audioChunks.length = 0;

	if (pcmData.length === 0) {
		session.closed = true;
		session.onDone("", { hadAudio: false, hadSpeech: false });
		return;
	}

	try {
		let text: string;

		if (config.backend === "sixtydb") {
			// 60db cloud batch — upload the recorded clip, get a transcript.
			const wavBuffer = createWavBuffer(pcmData);
			text = await transcribeWith60db(wavBuffer, config);
		} else if (config.localEndpoint) {
			// External server mode (advanced override)
			const wavBuffer = createWavBuffer(pcmData);
			text = await transcribeWithServer(wavBuffer, config);
		} else {
			// In-process via sherpa-onnx (default, 120s timeout)
			let timeoutHandle: ReturnType<typeof setTimeout> | undefined;
			text = await Promise.race([
				transcribeInProcess(pcmData, config),
				new Promise<never>((_, reject) => {
					timeoutHandle = setTimeout(() => reject(new Error("Transcription timed out (120s)")), 120_000);
				}),
			]).finally(() => {
				if (timeoutHandle) clearTimeout(timeoutHandle);
			});
		}

		// Recheck after await — abort may have fired during transcription.
		// Still call onDone so state machine transitions back to idle.
		if (session.closed) {
			session.onDone("", { hadAudio: false, hadSpeech: false });
			return;
		}

		session.closed = true;
		session.onDone(text, { hadAudio: true, hadSpeech: text.trim().length > 0 });
	} catch (err: any) {
		if (session.closed) {
			// Session was aborted during transcription — still surface the error
			// so the user knows transcription failed, not just "no speech"
			session.onError(`Local transcription aborted: ${err.message || err}`);
			return;
		}
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

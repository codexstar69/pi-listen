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

	// ─── Pluggable engine fields (Phase 1, additive + back-compat) ───────────
	// All three are optional. Existing entries pre-date these fields and
	// behave identically to before; the defaults are documented below.

	/** Which STT engine runs this model. `undefined` → "sherpa-onnx" (the
	 *  legacy default; preserves zero-touch back-compat for every existing
	 *  catalog entry). Phase 2 adds "whisper-cpp" entries. */
	engine?: import("./engine").EngineId;

	/** Per-runtime support constraints. `undefined` (or empty) → universally
	 *  supported (matches the sherpa-onnx legacy assumption). When set, at
	 *  least one entry must match the runtime env for the model to be
	 *  selectable. Resolution lives in `model-compat.isModelSupported`. */
	supports?: import("./model-compat").ModelSupport[];

	/** On-disk file format. `undefined` → "onnx" (sherpa-onnx legacy). GGUF
	 *  models (whisper.cpp) ship as a single .bin file; ONNX models ship
	 *  multiple files via `sherpaModel.files`. */
	format?: "onnx" | "gguf";

	/** whisper.cpp model configuration — used when `format === "gguf"`.
	 *  Phase 1 declares the field for forward-compat; no catalog entries
	 *  populate it yet. Phase 2 adds GGUF entries. */
	whisperCppModel?: {
		/** On-disk filename, e.g. "ggml-base.en-q5_1.bin". */
		files: { model: string };
		/** GGUF source URL. */
		downloadUrls: { model: string };
	};
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
	// WHISPER.CPP METAL — Apple Silicon top picks (Phase 2)
	//   GGUF source: https://huggingface.co/ggerganov/whisper.cpp (Apache 2.0)
	//   Engine: smart-whisper (NAPI binding around whisper.cpp + Metal)
	//   Platforms: darwin-arm64 only — Metal kernels are macOS-specific.
	// ═══════════════════════════════════════════════════════════════════════
	{
		id: "whisper-base-en-metal", name: "Whisper Base EN (Metal)",
		size: "~57 MB", sizeBytes: 60_000_000, runtimeRamMB: 200,
		notes: "Default for fresh Apple Silicon installs. Fast, ~96% accuracy.",
		langSupport: "english-only", tier: "edge", preferred: true, accuracy: 4, speed: 5,
		engine: "whisper-cpp",
		supports: [{ platform: "darwin", arch: "arm64" }],
		format: "gguf",
		whisperCppModel: {
			files: { model: "ggml-base.en-q5_1.bin" },
			downloadUrls: {
				model: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en-q5_1.bin",
			},
		},
		// Note: sherpaModel is required by the existing LocalModelInfo
		// shape but unused for whisper-cpp entries. Stub with safe-but-
		// non-functional values so callers that read the field for
		// non-engine purposes (e.g. `langSupport` UI hints) stay happy.
		// The engine factory routes via `model.engine` so the stub is
		// never dispatched to sherpa-onnx.
		sherpaModel: { type: "whisper", files: {}, downloadUrls: {} },
	},
	{
		id: "whisper-large-v3-turbo-metal", name: "Whisper Turbo (Metal)",
		size: "~574 MB", sizeBytes: 602_000_000, runtimeRamMB: 1700,
		notes: "Best accuracy on Apple Silicon. Multilingual, ~99% accuracy, GPU-accelerated.",
		langSupport: "whisper", tier: "standard", accuracy: 5, speed: 4,
		engine: "whisper-cpp",
		supports: [{ platform: "darwin", arch: "arm64" }],
		format: "gguf",
		whisperCppModel: {
			files: { model: "ggml-large-v3-turbo-q5_0.bin" },
			downloadUrls: {
				model: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin",
			},
		},
		sherpaModel: { type: "whisper", files: {}, downloadUrls: {} },
	},
	{
		id: "whisper-tiny-en-metal", name: "Whisper Tiny EN (Metal)",
		size: "~31 MB", sizeBytes: 33_000_000, runtimeRamMB: 110,
		notes: "Smallest English model — fastest, lowest accuracy.",
		langSupport: "english-only", tier: "edge", accuracy: 3, speed: 5,
		engine: "whisper-cpp",
		supports: [{ platform: "darwin", arch: "arm64" }],
		format: "gguf",
		whisperCppModel: {
			files: { model: "ggml-tiny.en-q5_1.bin" },
			downloadUrls: {
				model: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en-q5_1.bin",
			},
		},
		sherpaModel: { type: "whisper", files: {}, downloadUrls: {} },
	},
	// ═══════════════════════════════════════════════════════════════════════
	// WHISPER.CPP CUDA — Linux GPU (Phase 3)
	//   Same GGUF source as Metal entries; different supports constraint.
	//   Requires `requiresCUDA: true` so device.hasCUDA gates selection.
	// ═══════════════════════════════════════════════════════════════════════
	{
		id: "whisper-large-v3-turbo-cuda", name: "Whisper Turbo (CUDA)",
		size: "~574 MB", sizeBytes: 602_000_000, runtimeRamMB: 1700,
		notes: "Best accuracy on Linux x64 + NVIDIA GPU. Multilingual, GPU-accelerated.",
		langSupport: "whisper", tier: "standard", accuracy: 5, speed: 4,
		engine: "whisper-cpp",
		supports: [{ platform: "linux", arch: "x64", requiresCUDA: true }],
		format: "gguf",
		whisperCppModel: {
			files: { model: "ggml-large-v3-turbo-q5_0.bin" },
			downloadUrls: {
				model: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin",
			},
		},
		sherpaModel: { type: "whisper", files: {}, downloadUrls: {} },
	},
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

/**
 * Optional notifier the engine layer calls when a transcribe-time
 * download begins / progresses / settles. Wired up by `voice.ts` at
 * `session_start` so the user sees clear UI when the first recording
 * triggers a model download (round-2 audit UX gap: previously the user
 * saw a "hung" recording while a 57-800 MB GGUF was fetched).
 *
 * `null` until set; `transcribeInProcess` no-ops the notifier path when
 * unset (CLI / test contexts without a UI surface).
 */
export type DownloadNotifierEvent =
	| { type: "start"; modelId: string; sizeBytes: number; humanSize: string }
	| { type: "progress"; modelId: string; raw: import("./model-download").DownloadProgress }
	| { type: "done"; modelId: string }
	| { type: "fail"; modelId: string; error: string };
export type DownloadNotifier = (event: DownloadNotifierEvent) => void;

let _downloadNotifier: DownloadNotifier | null = null;

/** Set the transcribe-time download notifier. Called by voice.ts at
 *  session_start (and `null` at session_shutdown). Single global slot
 *  — only one session runs at a time. */
export function setDownloadNotifier(fn: DownloadNotifier | null): void {
	_downloadNotifier = fn;
}

/**
 * Pick the role→URL map for downloading a model, based on its `format`.
 * GGUF models (whisper.cpp) use `whisperCppModel.downloadUrls`; ONNX
 * models use `sherpaModel.downloadUrls` (the legacy default for entries
 * that pre-date the `format` field).
 *
 * Both maps share the same `Record<string, string>` shape so
 * `model-download.ensureModelDownloaded` + `isModelDownloaded` can
 * consume either without further branching.
 */
export function getDownloadUrlsForModel(model: LocalModelInfo): Record<string, string> {
	if (model.format === "gguf") {
		const urls = model.whisperCppModel?.downloadUrls;
		if (!urls || Object.keys(urls).length === 0) {
			throw new Error(
				`Catalog entry "${model.id}" is format=gguf but has no ` +
				`whisperCppModel.downloadUrls — fix in extensions/voice/local.ts.`,
			);
		}
		return urls;
	}
	return model.sherpaModel.downloadUrls;
}

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

		if (config.localEndpoint) {
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
	const { ensureModelDownloaded, isModelDownloaded } = await import("./model-download");
	const { getEngineForModel } = await import("./engine-factory");

	const modelId = config.localModel || DEFAULT_LOCAL_MODEL;
	const model = LOCAL_MODELS.find(m => m.id === modelId);
	if (!model) throw new Error(`Unknown model: ${modelId}`);

	// Ensure model files are downloaded BEFORE engine init/warmup so the
	// engine never has to fetch. (Engines deliberately do not own model
	// downloading — separation of concerns.) `getDownloadUrlsForModel`
	// picks the right role→URL map based on `model.format`: ONNX entries
	// use `sherpaModel.downloadUrls`, GGUF entries use
	// `whisperCppModel.downloadUrls`.
	const downloadUrls = getDownloadUrlsForModel(model);

	// Notifier hook for the round-2 audit UX gap: surface a clear download
	// notification when the first recording triggers a fetch. The notifier
	// is set by voice.ts at session_start; `null` in CLI/test contexts.
	const willDownload = !isModelDownloaded(model.id, downloadUrls);
	if (willDownload) _downloadNotifier?.({
		type: "start", modelId: model.id, sizeBytes: model.sizeBytes, humanSize: model.size,
	});

	let modelDir: string;
	try {
		modelDir = await ensureModelDownloaded(
			model.id,
			downloadUrls,
			model.sizeBytes,
			willDownload && _downloadNotifier
				? (raw) => _downloadNotifier?.({ type: "progress", modelId: model.id, raw })
				: undefined,
		);
	} catch (err) {
		if (willDownload) _downloadNotifier?.({
			type: "fail", modelId: model.id, error: err instanceof Error ? err.message : String(err),
		});
		throw err;
	}
	if (willDownload) _downloadNotifier?.({ type: "done", modelId: model.id });

	const engine = getEngineForModel(model);
	if (!engine.available) {
		const ok = await engine.init();
		if (!ok) {
			// Round-6 audit (GPT-5.5 Amp polish): name the actual engine
			// in the error so users with whisper-cpp selected don't see
			// "sherpa-onnx not available" — they correctly see
			// "whisper-cpp not available" and can act on it.
			throw new Error(
				`${engine.id} not available: ${engine.error || "unknown error"}. ` +
				`Set localEndpoint in config to use an external server instead.`,
			);
		}
	}

	return engine.transcribe(pcmData, {
		model,
		modelDir,
		lang: config.language || "en",
	});
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

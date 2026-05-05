/**
 * sherpa-onnx in-process transcription engine.
 *
 * Provides zero-config local STT by loading ONNX models directly
 * into the extension process via sherpa-onnx-node (N-API bindings).
 *
 * Supports: Whisper, Moonshine v1/v2, SenseVoice, GigaAM, Parakeet
 *
 * Recognizer instances are cached and reused (model loading is expensive).
 * Destroyed on: model change, language change, extension deactivation.
 *
 * API verified against:
 *   https://github.com/k2-fsa/sherpa-onnx/tree/master/nodejs-addon-examples
 *   - acceptWaveform({sampleRate, samples}) — object parameter
 *   - Config requires featConfig: {sampleRate: 16000, featureDim: 80}
 *   - Moonshine v2: {encoder, mergedDecoder} (2 files)
 *   - Moonshine v1: {preprocessor, encoder, uncachedDecoder, cachedDecoder} (4 files)
 *   - SenseVoice: useInverseTextNormalization is 1/0, not true/false
 */

import * as path from "node:path";
import * as os from "node:os";
import type { LocalModelInfo } from "./local";
import { loadSherpa, getSherpaModule, getSherpaError, isSherpaAvailable } from "./sherpa-loader";

// ─── Types ───────────────────────────────────────────────────────────────────

/** sherpa-onnx recognizer — opaque handle from the native module */
type SherpaRecognizer = any;

/** Cached recognizer for the currently loaded model + language */
let cachedRecognizer: { modelId: string; language: string; recognizer: SherpaRecognizer } | null = null;

// ─── Initialization ──────────────────────────────────────────────────────────

/**
 * Initialize the sherpa-onnx module. Kept as a thin alias around `loadSherpa()`
 * so existing call sites (extension lifecycle, settings panel, /voice test)
 * continue to compile and behave identically. New code should call
 * `loadSherpa()` directly.
 */
export async function initSherpa(): Promise<boolean> {
	return loadSherpa();
}

// Re-export the loader's status helpers so STT call sites that imported them
// from `./sherpa-engine` keep working.
export { getSherpaError, isSherpaAvailable };

// ─── Recognizer management ──────────────────────────────────────────────────

/**
 * Get or create a recognizer for a model.
 * Returns a cached instance if the model hasn't changed.
 */
export function getOrCreateRecognizer(model: LocalModelInfo, modelDir: string, language: string): SherpaRecognizer {
	// getSherpaModule() throws if loadSherpa() hasn't run yet — same contract as
	// the previous "if (!sherpaModule) throw" check, but routed through the
	// shared loader so STT and TTS share the single-flight init.
	getSherpaModule();

	// Strip regional suffix and lowercase. Round-5 audit (Kimi 2.5T #4):
	// lowercase MUST match the adapter's `recognizerKey()` normalization,
	// otherwise `lang="EN-US"` produces adapter key `"...\0en"` but sherpa-
	// engine's internal key is `(model, "EN")` — cache incoherence between
	// the two layers.
	const baseLang = (language.split("-")[0] || language).toLowerCase();

	if (cachedRecognizer && cachedRecognizer.modelId === model.id && cachedRecognizer.language === baseLang) {
		return cachedRecognizer.recognizer;
	}

	// Destroy previous recognizer
	clearRecognizerCache();

	const recognizer = createRecognizer(model, modelDir, baseLang);
	cachedRecognizer = { modelId: model.id, language: baseLang, recognizer };
	return recognizer;
}

/** Destroy cached recognizer and free memory. */
export function clearRecognizerCache(): void {
	if (cachedRecognizer) {
		// sherpa-onnx-node ships no `.free()` / `.release()` / `.dispose()`
		// method on `OfflineRecognizer` (verified in
		// node_modules/sherpa-onnx-node/non-streaming-asr.js). Native ONNX
		// resources are released via N-API finalizers when the JS object is
		// garbage-collected. Dropping our last reference (cachedRecognizer
		// set to null) makes the recognizer eligible for GC, which the
		// platform-level concurrent / generational GC eventually reclaims.
		// If sherpa-onnx-node ever exposes an explicit dispose API, this is
		// the call site to wire it up.
		cachedRecognizer = null;
	}
}

// ─── Transcription ───────────────────────────────────────────────────────────

/**
 * Transcribe PCM audio buffer using a sherpa recognizer.
 *
 * @param pcmData - Raw 16-bit signed LE PCM at 16kHz mono
 * @param recognizer - sherpa OfflineRecognizer instance
 * @returns Transcribed text
 */
export async function transcribeBuffer(pcmData: Buffer, recognizer: SherpaRecognizer): Promise<string> {
	// Throws if loadSherpa() hasn't run; callers always call initSherpa first.
	getSherpaModule();

	// Convert 16-bit PCM to Float32Array (sherpa expects float samples in [-1, 1])
	const samples = pcmToFloat32(pcmData);

	// Create a stream, accept waveform, decode asynchronously
	// API: stream.acceptWaveform({sampleRate, samples}) — verified from official examples
	// decodeAsync runs inference on ONNX Runtime's background thread pool (N-API AsyncWorker),
	// keeping the event loop free for UI updates during the 5-15s decode
	const stream = recognizer.createStream();
	stream.acceptWaveform({ sampleRate: 16000, samples });
	await recognizer.decodeAsync(stream);

	const result = recognizer.getResult(stream);
	return (result?.text || "").trim();
}

// ─── Internal: Recognizer creation per model type ────────────────────────────

function createRecognizer(model: LocalModelInfo, modelDir: string, language: string): SherpaRecognizer {
	const modelType = model.sherpaModel?.type;

	switch (modelType) {
		case "whisper":
			return createWhisperRecognizer(model, modelDir, language);
		case "moonshine":
			return createMoonshineRecognizer(model, modelDir);
		case "sense_voice":
			return createSenseVoiceRecognizer(model, modelDir, language);
		case "nemo_ctc":
			return createNemoCtcRecognizer(model, modelDir);
		case "transducer":
			return createTransducerRecognizer(model, modelDir);
		default:
			throw new Error(`Unknown sherpa model type: ${modelType} for model ${model.id}`);
	}
}

// Verified against: nodejs-addon-examples/test_asr_non_streaming_whisper.js
function createWhisperRecognizer(model: LocalModelInfo, modelDir: string, language: string): SherpaRecognizer {
	const files = model.sherpaModel!.files;
	const sherpa = getSherpaModule();
	return new sherpa.OfflineRecognizer({
		featConfig: {
			sampleRate: 16000,
			featureDim: 80,
		},
		modelConfig: {
			whisper: {
				encoder: path.join(modelDir, files.encoder!),
				decoder: path.join(modelDir, files.decoder!),
				language: language || "en",
				task: "transcribe",
			},
			tokens: path.join(modelDir, files.tokens!),
			numThreads: getNumThreads(),
			provider: getProvider("whisper"),
		},
	});
}

// Verified against: nodejs-addon-examples/test_asr_non_streaming_moonshine_v2.js
// Moonshine v2: {encoder, mergedDecoder} — 2 files
// Moonshine v1: {preprocessor, encoder, uncachedDecoder, cachedDecoder} — 4 files
function createMoonshineRecognizer(model: LocalModelInfo, modelDir: string): SherpaRecognizer {
	const files = model.sherpaModel!.files;

	// Detect v1 vs v2 by presence of mergedDecoder field
	const moonshineConfig: Record<string, string> = {};

	if (files.mergedDecoder) {
		// Moonshine v2: encoder + mergedDecoder
		moonshineConfig.encoder = path.join(modelDir, files.encoder!);
		moonshineConfig.mergedDecoder = path.join(modelDir, files.mergedDecoder!);
	} else {
		// Moonshine v1: preprocessor + encoder + uncachedDecoder + cachedDecoder
		moonshineConfig.preprocessor = path.join(modelDir, files.preprocessor!);
		moonshineConfig.encoder = path.join(modelDir, files.encoder!);
		moonshineConfig.uncachedDecoder = path.join(modelDir, files.uncachedDecoder!);
		moonshineConfig.cachedDecoder = path.join(modelDir, files.cachedDecoder!);
	}

	const sherpa = getSherpaModule();
	return new sherpa.OfflineRecognizer({
		featConfig: {
			sampleRate: 16000,
			featureDim: 80,
		},
		modelConfig: {
			moonshine: moonshineConfig,
			tokens: path.join(modelDir, files.tokens!),
			numThreads: getNumThreads(),
			provider: getProvider("moonshine"),
		},
	});
}

// Verified against: nodejs-addon-examples/test_asr_non_streaming_sense_voice.js
function createSenseVoiceRecognizer(model: LocalModelInfo, modelDir: string, language: string): SherpaRecognizer {
	const files = model.sherpaModel!.files;
	const sherpa = getSherpaModule();
	return new sherpa.OfflineRecognizer({
		featConfig: {
			sampleRate: 16000,
			featureDim: 80,
		},
		modelConfig: {
			senseVoice: {
				model: path.join(modelDir, files.model!),
				language: language || "auto",
				useInverseTextNormalization: 1,
			},
			tokens: path.join(modelDir, files.tokens!),
			numThreads: getNumThreads(),
			provider: getProvider("sense_voice"),
		},
	});
}

// Verified against: nodejs-addon-examples/test_asr_non_streaming_nemo_ctc.js
function createNemoCtcRecognizer(model: LocalModelInfo, modelDir: string): SherpaRecognizer {
	const files = model.sherpaModel!.files;
	const sherpa = getSherpaModule();
	return new sherpa.OfflineRecognizer({
		featConfig: {
			sampleRate: 16000,
			featureDim: 80,
		},
		modelConfig: {
			nemoCtc: {
				model: path.join(modelDir, files.model!),
			},
			tokens: path.join(modelDir, files.tokens!),
			numThreads: getNumThreads(),
			provider: getProvider("nemo_ctc"),
		},
	});
}

// Verified against: nodejs-addon-examples/test_asr_non_streaming_transducer.js
//
// Tuning notes:
//   - Transducer stays on CPU regardless of platform — see getProvider().
//     CoreML regresses TDT graphs (sherpa-onnx issue #2910, RTF 0.470 vs
//     0.427 on M2 Max). Revisit when the partition-aware CoreML EP lands.
//   - `numThreads` uses the higher transducer cap (6 vs Whisper's 4). Parakeet
//     TDT v3's encoder-decoder-joiner scales to ~6 P-cores; 4 leaves modern
//     M-series chips idle. Per RTF curves at
//     https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/nemo-transducer-models.html.
function createTransducerRecognizer(model: LocalModelInfo, modelDir: string): SherpaRecognizer {
	const files = model.sherpaModel!.files;
	const sherpa = getSherpaModule();
	return new sherpa.OfflineRecognizer({
		featConfig: {
			sampleRate: 16000,
			featureDim: 80,
		},
		modelConfig: {
			transducer: {
				encoder: path.join(modelDir, files.encoder!),
				decoder: path.join(modelDir, files.decoder!),
				joiner: path.join(modelDir, files.joiner!),
			},
			tokens: path.join(modelDir, files.tokens!),
			numThreads: getNumThreads(TRANSDUCER_MAX_THREADS),
			provider: getProvider("transducer"),
			// debug: 1 — uncomment to log per-stage (encoder/decoder/joiner) timings.
		},
	});
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/** Convert 16-bit signed LE PCM buffer to Float32Array.
 * Uses Int16Array typed view for ~5-10x speedup over per-sample readInt16LE.
 * Safe on all sherpa-onnx platforms (x86/ARM LE). Respects Buffer.byteOffset for pooled buffers.
 */
function pcmToFloat32(pcm: Buffer): Float32Array {
	const numSamples = Math.floor(pcm.length / 2);
	const float32 = new Float32Array(numSamples);

	// Pooled Buffer instances can start at odd byte offsets, which makes a
	// direct Int16Array view throw. Fall back to readInt16LE in that case.
	if ((pcm.byteOffset & 1) !== 0) {
		for (let i = 0; i < numSamples; i++) {
			float32[i] = pcm.readInt16LE(i * 2) / 32768.0;
		}
		return float32;
	}

	const int16 = new Int16Array(pcm.buffer, pcm.byteOffset, numSamples);
	for (let i = 0; i < numSamples; i++) {
		float32[i] = int16[i]! / 32768.0;
	}
	return float32;
}

/**
 * Get optimal thread count for inference (leave 1-2 cores free for the rest
 * of the agent UI / Pi runtime).
 *
 * `maxThreads` is the per-model-class cap. ONNX Runtime's intra-op pool stops
 * scaling well past a model-specific elbow:
 *   - Whisper: autoregressive decoder, threadpool-bound differently → cap 4
 *   - SenseVoice / NeMo CTC: encoder-only, modest scaling → cap 4
 *   - Transducer (Parakeet TDT, Zipformer): encoder-decoder-joiner, scales
 *     to ~6 threads on Apple Silicon performance cores. Caller passes 6.
 *
 * Numbers come from sherpa-onnx published RTF curves
 * (https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/nemo-transducer-models.html)
 * and the threadpool-saturation discussion in
 * https://github.com/k2-fsa/sherpa-onnx/issues/2910.
 */
function getNumThreads(maxThreads = 4): number {
	const cpus = os.cpus().length || 2;
	if (cpus <= 2) return 1;
	if (cpus <= 4) return 2;
	return Math.min(maxThreads, cpus - 2);
}

/** Transducer (Parakeet, Zipformer) thread budget — see getNumThreads(). */
const TRANSDUCER_MAX_THREADS = 6;

/**
 * Pick an ONNX Runtime execution provider for `modelType`.
 *
 * Apple Silicon ships a bundled CoreML EP in `libonnxruntime.dylib`
 * (verified via `nm -gU` → `OrtSessionOptionsAppendExecutionProvider_CoreML`).
 * Routing encoder-heavy graphs through it offloads matmuls to the Neural
 * Engine / GPU and typically lands 1.5–3× speedups vs. the CPU EP.
 *
 *   - "transducer" (Parakeet TDT, Zipformer) → CPU. CoreML is a measured
 *     regression on M2 Max (sherpa-onnx issue #2910, RTF 0.470 vs 0.427)
 *     until upstream lands the partition-aware EP.
 *   - "whisper" / "moonshine" / "sense_voice" / "nemo_ctc" → CoreML on
 *     darwin-arm64. If the int8/quantized graph has ops CoreML can't
 *     compile, ORT silently partitions the unsupported subgraph back to
 *     CPU, so the worst case is "no change," not a crash.
 *   - Any other platform → CPU.
 *
 * Env escape hatch: `PI_VOICE_FORCE_CPU=1` forces CPU across the board.
 * Use it to A/B test or work around a CoreML driver regression without
 * rebuilding.
 */
function getProvider(modelType: string): string {
	if (process.env.PI_VOICE_FORCE_CPU === "1") return "cpu";
	if (modelType === "transducer") return "cpu";
	if (process.platform === "darwin" && process.arch === "arm64") return "coreml";
	return "cpu";
}

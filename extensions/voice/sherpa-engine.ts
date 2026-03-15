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
 */

import * as path from "node:path";
import * as fs from "node:fs";
import * as os from "node:os";
import type { LocalModelInfo } from "./local";

// ─── Types ───────────────────────────────────────────────────────────────────

/** sherpa-onnx recognizer — opaque handle from the native module */
type SherpaRecognizer = any;

/** sherpa-onnx module — dynamically imported */
let sherpaModule: any = null;
let sherpaInitialized = false;
let sherpaError: string | null = null;

/** Cached recognizer for the currently loaded model */
let cachedRecognizer: { modelId: string; recognizer: SherpaRecognizer } | null = null;

// ─── Initialization ──────────────────────────────────────────────────────────

/**
 * Initialize the sherpa-onnx module.
 * Must be called once before any transcription.
 *
 * Sets LD_LIBRARY_PATH/DYLD_LIBRARY_PATH to find native binaries,
 * then loads the sherpa-onnx-node module.
 *
 * Returns true if initialization succeeded.
 */
export async function initSherpa(): Promise<boolean> {
	if (sherpaInitialized) return !sherpaError;
	if (sherpaError) return false;

	try {
		// Find the platform-specific native binary directory
		const platformPkgDir = findNativePkgDir();
		if (platformPkgDir) {
			// Set library path so the native module can find its shared libs
			if (process.platform === "darwin") {
				process.env.DYLD_LIBRARY_PATH = `${platformPkgDir}:${process.env.DYLD_LIBRARY_PATH || ""}`;
			} else {
				process.env.LD_LIBRARY_PATH = `${platformPkgDir}:${process.env.LD_LIBRARY_PATH || ""}`;
			}
		}

		sherpaModule = await import("sherpa-onnx-node");
		sherpaInitialized = true;
		return true;
	} catch (err: any) {
		sherpaError = err?.message || String(err);
		sherpaInitialized = true;
		return false;
	}
}

/** Get the sherpa initialization error, if any. */
export function getSherpaError(): string | null {
	return sherpaError;
}

/** Check if sherpa-onnx is available. */
export function isSherpaAvailable(): boolean {
	return sherpaInitialized && !sherpaError && sherpaModule != null;
}

// ─── Recognizer management ──────────────────────────────────────────────────

/**
 * Get or create a recognizer for a model.
 * Returns a cached instance if the model hasn't changed.
 */
export function getOrCreateRecognizer(model: LocalModelInfo, modelDir: string, language: string): SherpaRecognizer {
	if (!sherpaModule) throw new Error("sherpa-onnx not initialized. Call initSherpa() first.");

	if (cachedRecognizer && cachedRecognizer.modelId === model.id) {
		return cachedRecognizer.recognizer;
	}

	// Destroy previous recognizer
	clearRecognizerCache();

	const recognizer = createRecognizer(model, modelDir, language);
	cachedRecognizer = { modelId: model.id, recognizer };
	return recognizer;
}

/** Destroy cached recognizer and free memory. */
export function clearRecognizerCache(): void {
	if (cachedRecognizer) {
		try {
			// sherpa-onnx recognizers may have a free/delete method
			if (typeof cachedRecognizer.recognizer?.free === "function") {
				cachedRecognizer.recognizer.free();
			}
		} catch {
			// Ignore cleanup errors
		}
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
export function transcribeBuffer(pcmData: Buffer, recognizer: SherpaRecognizer): string {
	if (!sherpaModule) throw new Error("sherpa-onnx not initialized");

	// Convert 16-bit PCM to Float32Array (sherpa expects float samples)
	const samples = pcmToFloat32(pcmData);

	// Create a stream, accept waveform, decode
	const stream = recognizer.createStream();
	stream.acceptWaveform(16000, samples);
	recognizer.decode(stream);

	const text = recognizer.getResult(stream)?.text || "";

	// Clean up stream
	if (typeof stream.free === "function") stream.free();

	return text.trim();
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

function createWhisperRecognizer(model: LocalModelInfo, modelDir: string, language: string): SherpaRecognizer {
	const files = model.sherpaModel!.files;
	return new sherpaModule.OfflineRecognizer({
		modelConfig: {
			whisper: {
				encoder: path.join(modelDir, files.encoder!),
				decoder: path.join(modelDir, files.decoder!),
				language: language || "en",
				task: "transcribe",
			},
			tokens: path.join(modelDir, files.tokens!),
			numThreads: getNumThreads(),
			provider: "cpu",
		},
	});
}

function createMoonshineRecognizer(model: LocalModelInfo, modelDir: string): SherpaRecognizer {
	const files = model.sherpaModel!.files;
	return new sherpaModule.OfflineRecognizer({
		modelConfig: {
			moonshine: {
				preprocessor: path.join(modelDir, files.preprocessor!),
				encoder: path.join(modelDir, files.encoder!),
				uncachedDecoder: path.join(modelDir, files.uncachedDecoder!),
				cachedDecoder: path.join(modelDir, files.cachedDecoder!),
			},
			tokens: path.join(modelDir, files.tokens!),
			numThreads: getNumThreads(),
			provider: "cpu",
		},
	});
}

function createSenseVoiceRecognizer(model: LocalModelInfo, modelDir: string, language: string): SherpaRecognizer {
	const files = model.sherpaModel!.files;
	return new sherpaModule.OfflineRecognizer({
		modelConfig: {
			senseVoice: {
				model: path.join(modelDir, files.model!),
				language: language || "auto",
				useInverseTextNormalization: true,
			},
			tokens: path.join(modelDir, files.tokens!),
			numThreads: getNumThreads(),
			provider: "cpu",
		},
	});
}

function createNemoCtcRecognizer(model: LocalModelInfo, modelDir: string): SherpaRecognizer {
	const files = model.sherpaModel!.files;
	return new sherpaModule.OfflineRecognizer({
		modelConfig: {
			nemoCtc: {
				model: path.join(modelDir, files.model!),
			},
			tokens: path.join(modelDir, files.tokens!),
			numThreads: getNumThreads(),
			provider: "cpu",
		},
	});
}

function createTransducerRecognizer(model: LocalModelInfo, modelDir: string): SherpaRecognizer {
	const files = model.sherpaModel!.files;
	return new sherpaModule.OfflineRecognizer({
		modelConfig: {
			transducer: {
				encoder: path.join(modelDir, files.encoder!),
				decoder: path.join(modelDir, files.decoder!),
				joiner: path.join(modelDir, files.joiner!),
			},
			tokens: path.join(modelDir, files.tokens!),
			numThreads: getNumThreads(),
			provider: "cpu",
		},
	});
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/** Convert 16-bit signed LE PCM buffer to Float32Array. */
function pcmToFloat32(pcm: Buffer): Float32Array {
	const numSamples = pcm.length / 2;
	const float32 = new Float32Array(numSamples);
	for (let i = 0; i < numSamples; i++) {
		float32[i] = pcm.readInt16LE(i * 2) / 32768.0;
	}
	return float32;
}

/** Get optimal thread count for inference (leave 1-2 cores free). */
function getNumThreads(): number {
	const cpus = os.cpus().length || 2;
	if (cpus <= 2) return 1;
	if (cpus <= 4) return 2;
	return Math.min(4, cpus - 2);
}

/** Find the sherpa-onnx platform-specific native binary directory. */
function findNativePkgDir(): string | null {
	const platformMap: Record<string, string> = {
		"linux-x64": "sherpa-onnx-linux-x64",
		"linux-arm64": "sherpa-onnx-linux-arm64",
		"darwin-x64": "sherpa-onnx-darwin-x64",
		"darwin-arm64": "sherpa-onnx-darwin-arm64",
		"win32-x64": "sherpa-onnx-win-x64",
	};

	const key = `${process.platform}-${process.arch}`;
	const pkgName = platformMap[key];
	if (!pkgName) return null;

	// Try to resolve the platform package
	try {
		const pkgPath = require.resolve(`${pkgName}/package.json`);
		return path.dirname(pkgPath);
	} catch {
		// Try relative to sherpa-onnx-node
		try {
			const mainPkg = require.resolve("sherpa-onnx-node/package.json");
			const nodeModules = path.resolve(path.dirname(mainPkg), "..");
			const platformDir = path.join(nodeModules, pkgName);
			if (fs.existsSync(platformDir)) return platformDir;
		} catch {
			// Not installed
		}
	}

	return null;
}

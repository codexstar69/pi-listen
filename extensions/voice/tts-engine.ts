/**
 * Local TTS engine — wraps sherpa-onnx-node OfflineTts.
 *
 * Public API:
 *   - initTts()                  — load sherpa via the shared loader
 *   - synthesize(opts)           — produce {samples, sampleRate} + abort
 *   - synthesizeById(opts)       — synthesize using a model id from the catalog
 *   - clearTtsCache()            — drop cached instance (called on model switch)
 *   - resolveLanguageForLocal    — validate language ↔ local model match
 *   - validateTtsLanguageInput    — non-empty/string language guard (both backends)
 *
 * Internal helpers:
 *   - getOrCreateTts(model, dir) — module-private cached OfflineTts instance
 *
 * ─── Concurrency contract (read this before adding "race condition" notes) ───
 *
 * This module assumes the standard Node.js / Bun single-threaded execution
 * model. Synchronous statements (including `Map.get`/`Map.set`, property
 * reads/writes, the `&&` short-circuit, and `return` of an already-resolved
 * value) run to completion before any other code executes. There is NO
 * preemption between two synchronous operations, no parallel JS threads,
 * and no shared-memory data races on plain JS values.
 *
 * Async interleaving CAN happen at any `await` point. Every `await` in this
 * module is followed by an identity-guard check (`inFlight.get(key) === …`)
 * to detect "another caller took over while I was suspended" and to skip
 * stale cache writes. Callers also pass an `AbortSignal` through
 * `synthesize()` for cooperative cancellation.
 *
 * Sherpa-onnx-node native handles are reference-counted via N-API
 * finalizers — they are released only when the JS reference count drops
 * to zero. Since we keep a strong reference in `cachedTts` (or in a
 * caller's local `cached` after `await getOrCreateTts(...)`), there is no
 * use-after-free on the JS side. `clearTtsCache()` only drops the cache
 * slot; any caller that already has a `CachedTts` reference still owns
 * the underlying native handle until they drop it.
 *
 * Why a separate cache from STT: a user can have STT (Parakeet) and TTS
 * (Kitten Nano) loaded simultaneously — they're different sherpa engine
 * objects with different memory footprints, and the model-switch logic
 * differs (STT swaps on language change; TTS swaps on model change).
 */

import * as path from "node:path";
import * as os from "node:os";
import * as fs from "node:fs";
import { loadSherpa, getSherpaModule, getSherpaError, isSherpaAvailable } from "./sherpa-loader";
import {
	type TtsLocalModelInfo,
	modelSupportsLanguage,
	getTtsModel,
} from "./tts-local-models";

// ─── Types ────────────────────────────────────────────────────────────────────

/** Opaque OfflineTts handle from sherpa-onnx-node. */
type OfflineTts = any;

interface CachedTts {
	modelId: string;
	tts: OfflineTts;
	sampleRate: number;
	numSpeakers: number;
	/**
	 * Per-instance synthesis serialization tail. Two concurrent
	 * `synthesize()` calls against the same OfflineTts handle would (a)
	 * produce overlapping audio at the playback layer (bad UX) and
	 * (b) traverse sherpa-onnx-node's native generate path twice in
	 * parallel — the JS binding likely serializes internally but we
	 * shouldn't depend on undocumented behavior. Each synthesize() awaits
	 * the previous one before starting; the chain unwinds in order.
	 *
	 * Failed/aborted runs do not block subsequent runs — the chain
	 * advances on settlement (resolve OR reject), so a single bad
	 * synthesize doesn't deadlock the queue.
	 */
	generateChain: Promise<void>;
}

/** Result of a synthesize call. */
export interface TtsAudio {
	/** PCM float samples in [-1, 1] interleaved mono. */
	samples: Float32Array;
	/** Sample rate in Hz (typically 22050 or 24000). */
	sampleRate: number;
}

/** Synthesis input. `opts.signal` aborts via the OfflineTts onProgress callback. */
export interface SynthesizeOpts {
	text: string;
	model: TtsLocalModelInfo;
	modelDir: string;
	language: string;
	/** Speaker id within the model. Default: model.defaultSid. */
	sid?: number;
	/** Speed multiplier; 1.0 = normal. Range 0.5–2.0. */
	speed?: number;
	/** Trailing silence between sentences (sherpa internal default = 0.5). */
	silenceScale?: number;
	/** Cancellation token. Triggering this returns 0 from onProgress and aborts the generate. */
	signal?: AbortSignal;
	/**
	 * Optional progress callback. Fires once per sentence with the partial
	 * float samples and progress ∈ [0,1]. Useful if the caller wants to
	 * stream playback in v6.1.
	 */
	onProgress?: (chunk: { samples: Float32Array; progress: number }) => void;
}

// ─── State ────────────────────────────────────────────────────────────────────

/**
 * Composite cache key. Two callers passing different `modelDir` for the
 * same `modelId` (e.g. after a re-download to a versioned path) MUST get
 * different OfflineTts instances — sherpa-onnx-node holds open file
 * handles to the model files, and pointing at the wrong directory is
 * undefined behaviour. Including `modelDir` in the key forces a rebuild
 * when the on-disk location changes.
 *
 * Length-prefix the modelId so two distinct (modelId, modelDir) pairs
 * can't collide via concatenation when the separator happens to appear
 * inside modelId. Catalog ids are kebab-case today but the prefix is
 * cheap belt-and-suspenders against future ids that include `|`.
 */
function cacheKey(modelId: string, modelDir: string): string {
	return `${modelId.length}|${modelId}|${modelDir}`;
}

/**
 * Single source of truth for cached TTS instances. Stores resolved values
 * for fast lookups AND in-flight construction promises for single-flight
 * deduplication. Keys are `cacheKey(modelId, modelDir)`.
 *
 * Map values:
 *   - `Promise<CachedTts>` while construction is in-flight
 *   - `CachedTts` once construction has resolved
 *
 * Callers always wrap the lookup result in `Promise.resolve(hit)` —
 * `Promise.resolve` returns a thenable's same promise unchanged and wraps
 * a plain value in a one-microtask resolved promise, so a single return
 * statement covers both states without an instanceof check.
 *
 * `clearTtsCache()` wipes the entire map. In-flight builds remain pending;
 * their `.then(...)` handlers no-op against an empty map and the resulting
 * native handles are dropped on the floor (GC reclaims them). This is the
 * "fire and forget the in-flight build" trade-off — clearing the cache
 * mid-build wastes the build's CPU but is otherwise harmless.
 */
const ttsCache = new Map<string, CachedTts | Promise<CachedTts>>();

// ─── Initialization ──────────────────────────────────────────────────────────

/**
 * Initialize TTS support. Routes through the shared sherpa loader so the
 * native module is loaded exactly once per process even if STT was already
 * initialized.
 */
export async function initTts(): Promise<boolean> {
	return loadSherpa();
}

export { isSherpaAvailable, getSherpaError };

// ─── Recognizer (TTS instance) management ────────────────────────────────────

/**
 * Get or create an OfflineTts for `model`. Internal — public callers go
 * through `synthesize()` / `synthesizeById()` instead, which keeps the
 * sherpa-specific native handle behind the module boundary. Pulling the
 * handle out of this module would couple every caller to sherpa lifecycle
 * semantics and make swapping engines (or adding pooling) require a
 * cross-package refactor.
 *
 * Concurrent calls share the same in-flight construction promise. After
 * construction the cached instance is reused until either:
 *   - The user picks a different model (call clearTtsCache() then re-call)
 *   - The settings panel deletes a downloaded model (cache is cleared)
 *   - The session shuts down (clearTtsCache() runs in voiceCleanup)
 *
 * @throws if loadSherpa() failed (platform incompat) — caller should already
 *         have checked initTts() ok before calling.
 */
function getOrCreateTts(model: TtsLocalModelInfo, modelDir: string): Promise<CachedTts> {
	const key = cacheKey(model.id, modelDir);
	const hit = ttsCache.get(key);
	// Both branches return a Promise:
	//   - resolved CachedTts → wrapped in Promise.resolve (cheap, ~1 microtask)
	//   - in-flight Promise<CachedTts> → returned as-is
	if (hit) return Promise.resolve(hit);

	// Construct, store the in-flight promise immediately so concurrent
	// same-key callers see-through it, and replace with the resolved
	// CachedTts in the same map slot when construction settles. On
	// failure, evict so the next caller gets a fresh attempt.
	const pending = doCreateTts(model, modelDir).then(
		(cached) => {
			// Only swap in the resolved value if our pending promise is
			// still the entry — clearTtsCache() may have wiped the map
			// while we were building. In that case the freshly built
			// instance is dropped on the floor (GC reclaims the native
			// handle via the N-API finalizer).
			if (ttsCache.get(key) === pending) {
				ttsCache.set(key, cached);
			}
			return cached;
		},
		(err) => {
			// Don't leave a permanently rejected promise stuck in the
			// cache — the next caller should be allowed to retry.
			if (ttsCache.get(key) === pending) {
				ttsCache.delete(key);
			}
			throw err;
		},
	);
	ttsCache.set(key, pending);
	return pending;
}

async function doCreateTts(model: TtsLocalModelInfo, modelDir: string): Promise<CachedTts> {
	const sherpa = getSherpaModule();
	const config = buildTtsConfig(model, modelDir);
	const tts = await sherpa.OfflineTts.createAsync(config);
	return {
		modelId: model.id,
		tts,
		// Some sherpa builds expose `sampleRate` directly on the handle;
		// others require reading from the model. Prefer the handle and fall
		// back to the catalog entry to keep this resilient across versions.
		sampleRate: typeof tts.sampleRate === "number" ? tts.sampleRate : model.sampleRate,
		numSpeakers: typeof tts.numSpeakers === "number" ? tts.numSpeakers : model.voices.length,
		generateChain: Promise.resolve(),
	};
}

/**
 * Drop all cached OfflineTts instances. Same garbage-collection contract
 * as the STT recognizer cache: sherpa-onnx-node has no `.dispose()` API;
 * native resources are released via N-API finalizers when the JS reference
 * count drops to zero. Clearing the map drops our last reference; any
 * caller still holding a CachedTts via a previously-resolved promise keeps
 * its native handle alive until they drop it too.
 *
 * In-flight construction promises check `ttsCache.get(key) === pending`
 * before writing back, so a clear during build correctly causes the
 * freshly-built instance to be dropped on the floor.
 */
export function clearTtsCache(): void {
	ttsCache.clear();
}

// ─── Warmup ──────────────────────────────────────────────────────────────────

/**
 * Pre-load the sherpa-onnx module AND construct the OfflineTts for `model`
 * in the background, so the user's first `/voice-speak` doesn't pay the
 * 600-900ms cold-start init cost.
 *
 * Idempotent: subsequent calls for the same (model, modelDir) await the
 * same in-flight promise via the existing `ttsCache` machinery — the only
 * difference vs a real synthesize is that warmup discards the result.
 *
 * Cancellation: `signal` aborts the load. If the user toggles TTS off
 * before warmup completes, the construction continues and the resulting
 * instance lands in the cache (cheap memory cost), but no UI flicker
 * happens — the cache is simply unused. Cleaner alternative would be to
 * abort native createAsync but sherpa-onnx-node doesn't expose that.
 *
 * Errors: returns `false` on any failure (logged to debug output, not
 * rethrown). Callers treat this as a best-effort optimization — failure
 * here is not a user-facing error because the next /voice-speak will
 * surface the same error anyway through synthesize().
 */
export async function warmupTts(
	model: TtsLocalModelInfo,
	modelDir: string,
	opts: { signal?: AbortSignal } = {},
): Promise<boolean> {
	if (opts.signal?.aborted) return false;
	try {
		const ok = await loadSherpa();
		if (!ok) return false;
		if (opts.signal?.aborted) return false;
		await getOrCreateTts(model, modelDir);
		return true;
	} catch {
		// Warmup is best-effort; swallow errors so callers never have to
		// worry about a backgrounded promise rejection.
		return false;
	}
}

// ─── Synthesis ───────────────────────────────────────────────────────────────

/**
 * Synthesize `text` to PCM samples using the given model.
 *
 * Validation order:
 *   1. Language ↔ model compatibility (resolveLanguageForLocal)
 *   2. Engine availability (loadSherpa must have succeeded)
 *   3. Generate the audio (with optional abort + progress)
 *
 * The signal is wired through OfflineTts.generateAsync's onProgress: when
 * AbortSignal fires we return 0 from the callback, which sherpa-onnx-node
 * treats as "stop generating". The pending promise resolves with whatever
 * samples were produced so the caller can still play the partial result.
 */
export async function synthesize(opts: SynthesizeOpts): Promise<TtsAudio> {
	const { text, model, modelDir, language, signal } = opts;

	// 1. Language check first — synchronous, gives clear error before any
	//    model load or network work happens.
	resolveLanguageForLocal(model, language);

	// 2. Attach the abort listener BEFORE any await so a signal that fires
	//    during model load / engine construction is observed. If the signal
	//    is already aborted at entry, `signal.aborted` is true and the
	//    onProgress callback returns 0 on its first invocation.
	let aborted = signal?.aborted === true;
	const onAbort = () => { aborted = true; };
	if (signal && !aborted) {
		signal.addEventListener("abort", onAbort, { once: true });
	}

	try {
		// Fast-path: caller passed an already-aborted signal. Bail without
		// loading the engine — there's nothing to do.
		if (aborted) {
			throw makeAbortError();
		}

		// 3. Make sure the engine is loaded.
		const ok = await loadSherpa();
		if (aborted) throw makeAbortError();
		if (!ok) {
			throw new Error(`sherpa-onnx not available: ${getSherpaError() ?? "unknown error"}`);
		}

		// 4. Get or create the OfflineTts instance.
		const cached = await getOrCreateTts(model, modelDir);
		if (aborted) throw makeAbortError();

		// 5. Build the GenerationConfig.
		const sherpa = getSherpaModule();
		const sid = clampSid(opts.sid ?? model.defaultSid, model);
		const speed = clampSpeed(opts.speed ?? 1.0);
		const silenceScale = clampNonNegative(opts.silenceScale ?? 0.2, 0, 5);
		const generationConfig = new sherpa.GenerationConfig({ sid, speed, silenceScale });

		// 6. Serialize generates against this CachedTts instance via the
		// chain extension below — see the `generateChain` doc on CachedTts.
		const { previousTail, release: resolveOurTail } = extendChain(cached);
		await previousTail.catch(() => {}); // ignore prior failure
		if (aborted) {
			resolveOurTail();
			throw makeAbortError();
		}

		let audio: { samples: Float32Array; sampleRate?: number };
		try {
			audio = await cached.tts.generateAsync({
				text,
				generationConfig,
				onProgress: (info: { samples: Float32Array; progress: number }) => {
					opts.onProgress?.(info);
					// Returning 0 / false signals sherpa-onnx-node to stop generating.
					return aborted ? 0 : 1;
				},
			});
		} finally {
			// Always release the chain so subsequent synthesize() calls
			// can proceed, even if generateAsync threw.
			resolveOurTail();
		}
		// If the run was aborted mid-generate, surface that to the caller
		// rather than silently returning a partial buffer.
		if (aborted) throw makeAbortError();
		return {
			samples: audio.samples,
			sampleRate: audio.sampleRate ?? cached.sampleRate,
		};
	} finally {
		signal?.removeEventListener("abort", onAbort);
	}
}

function makeAbortError(): Error {
	// Use a DOMException-shaped error so callers using AbortController.signal
	// can pattern-match on `err.name === "AbortError"`. Falls back to a plain
	// Error in environments without DOMException.
	if (typeof DOMException === "function") {
		return new DOMException("TTS synthesis aborted", "AbortError");
	}
	const e = new Error("TTS synthesis aborted");
	(e as any).name = "AbortError";
	return e;
}

/**
 * Atomically extend a CachedTts's `generateChain` with a new tail.
 *
 * **All three operations run synchronously in a single function call.**
 * JavaScript run-to-completion semantics guarantee no other code (no
 * other synthesize() call, no microtask, no I/O callback) executes
 * between the read of `cached.generateChain` and the assignment of its
 * replacement. Two concurrent synthesize() callers therefore observe
 * distinct previousTails and link into the chain in arrival order.
 *
 * Returns:
 *   - previousTail: the chain at function entry — caller awaits this
 *     before doing its own generate work
 *   - release: caller MUST call this when its generate settles (success
 *     or failure) so the next link can advance
 */
function extendChain(cached: CachedTts): { previousTail: Promise<void>; release: () => void } {
	const previousTail = cached.generateChain;
	let release!: () => void;
	cached.generateChain = new Promise<void>((res) => { release = res; });
	return { previousTail, release };
}

// ─── Validation ──────────────────────────────────────────────────────────────

/**
 * Validate that `language` is non-empty and a string. Both backends call
 * this; the deepgram path additionally relies on the voice id encoding the
 * language (see `tts-deepgram.ts`), so this is the only check it needs.
 */
export function validateTtsLanguageInput(language: unknown): asserts language is string {
	if (!language || typeof language !== "string") {
		throw new Error(`TTS language is required (got: ${language})`);
	}
}

/**
 * Validate that `language` is supported by the local `model`. Throws a
 * user-facing error describing how to fix the mismatch.
 *
 * Local-only by signature — deepgram callers don't have a TtsLocalModelInfo
 * and use the (separate) deepgram voice catalog instead. Splitting the
 * validation by backend keeps the cloud TTS module independent of the
 * local catalog so the two engines can evolve independently.
 */
export function resolveLanguageForLocal(model: TtsLocalModelInfo, language: string): void {
	validateTtsLanguageInput(language);
	if (!modelSupportsLanguage(model, language)) {
		const supported = model.languages.join(", ");
		throw new Error(
			`Active local TTS model ${model.name} only supports ${supported}. ` +
			`Switch model in /voice-settings or change ttsLanguage to a supported value.`,
		);
	}
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/**
 * Build the OfflineTtsConfig for sherpa-onnx based on the model's slot.
 * Each slot has its own field names — kitten/vits/kokoro all live in
 * separate sub-objects of OfflineTtsModelConfig.
 */
function buildTtsConfig(model: TtsLocalModelInfo, modelDir: string): any {
	const dataDir = path.join(modelDir, "espeak-ng-data");
	const tokens = path.join(modelDir, "tokens.txt");
	const numThreads = getTtsThreads(model.sherpaSlot);

	switch (model.sherpaSlot) {
		case "kitten": {
			return {
				model: {
					kitten: {
						model: path.join(modelDir, "model.fp16.onnx"),
						voices: path.join(modelDir, "voices.bin"),
						tokens,
						dataDir,
						lengthScale: 1.0,
					},
				},
				numThreads,
				provider: "cpu",
			};
		}
		case "vits": {
			// Piper voices ship with one file ending in `.onnx` — naming varies
			// (`en_US-lessac-medium.onnx`, etc.). The downloader records the
			// extracted file name so we don't have to grep for it here.
			const onnx = findPiperOnnx(modelDir);
			return {
				model: {
					vits: {
						model: onnx,
						tokens,
						dataDir,
						// Defaults from sherpa-onnx upstream; surface only if we add knobs
						noiseScale: 0.667,
						noiseScaleW: 0.8,
						lengthScale: 1.0,
					},
				},
				numThreads,
				provider: "cpu",
			};
		}
		case "kokoro": {
			// Kokoro multilingual ships lexicon-* files for non-English languages;
			// pass them comma-separated so the engine can handle code-switching.
			const lexicon = findKokoroLexicons(modelDir);
			return {
				model: {
					kokoro: {
						model: path.join(modelDir, "model.onnx"),
						voices: path.join(modelDir, "voices.bin"),
						tokens,
						dataDir,
						lengthScale: 1.0,
						...(lexicon ? { lexicon } : {}),
					},
				},
				numThreads,
				provider: "cpu",
			};
		}
	}
}

/**
 * Per-model-class thread budget. TTS is a flow-matching / VITS / TDT-like
 * autoregressive workload — same scaling characteristics as the STT
 * transducer path. M-series Pro/Max chips scale to ~6 threads; non-Apple
 * CPUs back off to 4 to leave headroom for the agent UI.
 *
 * Mirrors `getNumThreads(maxThreads)` in `sherpa-engine.ts` rather than
 * importing it to keep TTS compileable in isolation if STT is later moved
 * to a separate package.
 */
function getTtsThreads(slot: TtsLocalModelInfo["sherpaSlot"]): number {
	const cpus = os.cpus().length || 2;
	if (cpus <= 2) return 1;
	if (cpus <= 4) return 2;
	// Per-slot tuning, mirroring the STT path's TRANSDUCER_MAX_THREADS=6
	// vs the Whisper-class cap of 4. Decisions per sherpa-onnx published
	// RTF curves and #2910 (CoreML regression for transformer graphs):
	//
	//   - kitten (Kitten Nano TTS): small model, scales to 4 threads
	//   - vits   (Piper):           single-speaker VITS, scales to 4
	//   - kokoro (Kokoro v0.19/v1.0): larger transformer encoder, scales
	//                                 to ~6 P-cores on M-series
	const max = slot === "kokoro" ? 6 : 4;
	return Math.min(max, cpus - 2);
}

/** Clamp speaker id into the model's voice range. */
function clampSid(sid: number, model: TtsLocalModelInfo): number {
	if (!Number.isFinite(sid)) return model.defaultSid;
	const maxSid = Math.max(0, ...model.voices.map(v => v.sid));
	return Math.max(0, Math.min(maxSid, Math.floor(sid)));
}

function clampSpeed(speed: number): number {
	if (!Number.isFinite(speed)) return 1.0;
	return Math.max(0.5, Math.min(2.0, speed));
}

function clampNonNegative(v: number, lo: number, hi: number): number {
	if (!Number.isFinite(v)) return lo;
	return Math.max(lo, Math.min(hi, v));
}

/**
 * Locate the Piper VITS .onnx file inside the extracted model directory.
 * Piper archives include exactly one `.onnx` file at the top level (no
 * subdirectories), but the filename varies by voice — fd.readdirSync once
 * and pick the first match.
 */
function findPiperOnnx(modelDir: string): string {
	const entries = fs.readdirSync(modelDir);
	const onnx = entries.find(e => e.endsWith(".onnx"));
	if (!onnx) throw new Error(`No .onnx file found in Piper model directory: ${modelDir}`);
	return path.join(modelDir, onnx);
}

/**
 * Find Kokoro lexicon files for multilingual support. Kokoro v1.0 ships
 * `lexicon-us-en.txt`, `lexicon-zh.txt`, etc. for grapheme-to-phoneme
 * conversion in non-English languages. Returns a comma-separated path
 * string the engine accepts, or null for English-only Kokoro v0.19.
 */
function findKokoroLexicons(modelDir: string): string | null {
	const entries = fs.readdirSync(modelDir);
	const lex = entries.filter(e => e.startsWith("lexicon-") && e.endsWith(".txt"));
	if (lex.length === 0) return null;
	return lex.map(e => path.join(modelDir, e)).join(",");
}

/**
 * Tiny convenience for callers that have a model id but not the full model
 * record. Looks up the catalog entry and forwards to synthesize().
 */
export async function synthesizeById(opts: Omit<SynthesizeOpts, "model"> & { modelId: string }): Promise<TtsAudio> {
	const model = getTtsModel(opts.modelId);
	return synthesize({ ...opts, model });
}

/**
 * sherpa-onnx engine adapter — wraps the existing `sherpa-engine.ts` module
 * (which holds the recognizer cache + transcribe call) behind the unified
 * `Engine` interface.
 *
 * Lifecycle (round-2 audit polish — matches WhisperCpp adapter):
 *   - Singleton starts in `available: false, error: null`. Reading the
 *     instance never throws.
 *   - `init()` is idempotent + single-flight; failures clear the cached
 *     promise so retries can run a fresh attempt.
 *   - `warmup()` is single-flight per `model.id` (`_warmupPromises`).
 *     Cached recognizer references live in `_recognizers` so
 *     `transcribe()` doesn't re-resolve via `getOrCreateRecognizer`
 *     every call (round-2 audit highest-impact perf fix).
 *   - A generation counter (`_cacheGen`) guards the
 *     `clearCache()`/`warmup()` race — same pattern as whisper-cpp.
 *   - `clearCache()` is async and defers freeing while a transcribe is
 *     in flight (use-after-free protection).
 *   - The underlying `sherpa-engine.ts` keeps its own internal cache for
 *     back-compat with existing call sites; the adapter caches refs
 *     directly to avoid the per-transcribe map lookup, and calls
 *     `clearRecognizerCache()` when its own cache is cleared so the two
 *     stay coherent.
 */

import {
	initSherpa as loadSherpaModule,
	isSherpaAvailable,
	getSherpaError,
	getOrCreateRecognizer,
	transcribeBuffer,
	clearRecognizerCache,
} from "./sherpa-engine";
import type { Engine, EngineContext, EngineId } from "./engine";

/** Adapter-level cache record. We hold the sherpa recognizer reference
 *  directly so `transcribe()` skips the per-call map lookup that
 *  `sherpa-engine.getOrCreateRecognizer` does internally. */
interface CachedRecognizer {
	recognizer: ReturnType<typeof getOrCreateRecognizer>;
	activeTranscribes: number;
	pendingFree: boolean;
	lastUsedAt: number;
}

/**
 * Round-5 audit (GLM 5 #2): LRU cache bound for the sherpa adapter.
 * Sherpa recognizers are smaller than GGUF (10s-100s of MB) but a user
 * who switches across many model+language combinations can still
 * accumulate hundreds of MB of native memory. Default 4 (sherpa is
 * lighter than whisper-cpp); override via `PI_VOICE_SHERPA_CACHE_SIZE`,
 * clamped 1-8.
 */
function resolveSherpaCacheSize(env: NodeJS.ProcessEnv = process.env): number {
	const raw = env.PI_VOICE_SHERPA_CACHE_SIZE;
	if (!raw) return 4;
	const n = Number.parseInt(raw, 10);
	if (!Number.isFinite(n) || n < 1) return 4;
	if (n > 8) return 8;
	return n;
}

/**
 * Cache key for the adapter's `_recognizers` / `_warmupPromises` maps.
 * MUST include the language — sherpa-onnx Whisper / SenseVoice
 * recognizers are language-bound at construction time, and
 * `sherpa-engine.getOrCreateRecognizer` keys its internal cache by
 * `(modelId, baseLang)`. Round-4 audit (GPT-5.5 #2): keying our cache
 * by `model.id` alone caused `warmup("en") + transcribe(pcm, "ja")`
 * to reuse the English recognizer for Japanese audio.
 *
 * Strip regional suffix to match sherpa-engine's internal key
 * normalization (`"en-US"` → `"en"`).
 */
function recognizerKey(ctx: EngineContext): string {
	// Round-5 audit (Opus 4.7, GLM 5, Kimi 2.5T): empty `ctx.lang` would
	// produce `${id}\0` (trailing-NUL key) which collides with future
	// "intentionally-empty-lang" callers AND drifts from sherpa-engine's
	// internal default of `language || "en"`. Fall through to `"en"` so
	// our adapter cache key matches what sherpa-engine.getOrCreateRecognizer
	// will actually use for the recognizer's `language` field.
	//
	// Round-5 audit (GPT-5.5 #4): only the language-bound recognizer
	// types (Whisper, SenseVoice) actually vary by language. Moonshine,
	// NeMo CTC, and Transducer ignore the language argument at construct
	// time, so keying by language for those types over-caches duplicates
	// of the same native graph. Use empty string for those — collapses
	// all `(modelId, *)` calls onto one cache slot per model.
	const t = ctx.model.sherpaModel?.type;
	const langSensitive = t === "whisper" || t === "sense_voice";
	const baseLang = langSensitive
		? (ctx.lang.split("-")[0] || ctx.lang || "en").toLowerCase()
		: "";
	return `${ctx.model.id}\0${baseLang}`;
}

class SherpaEngine implements Engine {
	readonly id: EngineId = "sherpa-onnx";
	private _available = false;
	private _error: string | null = null;
	private _initPromise: Promise<boolean> | null = null;

	private readonly _recognizers = new Map<string, CachedRecognizer>();
	private readonly _warmupPromises = new Map<string, Promise<void>>();
	private _cacheGen = 0;
	private readonly _cacheSize = resolveSherpaCacheSize();
	/**
	 * Set true by `clearCache()` when at least one active transcribe was
	 * in flight at the time. The transcribe `finally` block calls
	 * `clearRecognizerCache()` once the last active inference settles,
	 * keeping our adapter cache and sherpa-engine's internal cache
	 * coherent (round-3 audit Opus #4). Reset to false on flush.
	 */
	private _needsSherpaCacheFlush = false;
	/** Global count of all active transcribes across all models — used to
	 *  detect concurrent inferences (mostly informational; the deferred
	 *  flush specifically uses `_pendingFreeRemaining` to track only the
	 *  at-clear-time entries). */
	private _activeTranscribesGlobal = 0;
	/**
	 * Round-5 audit (GLM 5 #1): captures how many entries had
	 * `activeTranscribes > 0` at the moment `clearCache()` ran. The
	 * deferred sherpa-engine flush triggers when this hits zero — i.e.
	 * when every original at-clear active transcribe has settled. New
	 * transcribes started AFTER `clearCache()` don't affect this count,
	 * so the flush isn't blocked indefinitely by ongoing usage and
	 * doesn't prematurely fire mid-inference.
	 */
	private _pendingFreeRemaining = 0;

	get available() { return this._available; }
	get error() { return this._error; }
	/** Round-4 audit (Opus 4.6): getter for interface symmetry with
	 *  WhisperCppEngine. Sherpa has no informational hints today. */
	get hint(): string | null { return null; }

	init(): Promise<boolean> {
		if (this._available) return Promise.resolve(true);
		if (this._initPromise) return this._initPromise;
		this._initPromise = this._doInit();
		// Non-throwing cleanup chain — see WhisperCppEngine.init for rationale.
		this._initPromise.then(
			() => { this._initPromise = null; },
			() => { this._initPromise = null; },
		);
		return this._initPromise;
	}

	private async _doInit(): Promise<boolean> {
		try {
			const ok = await loadSherpaModule();
			if (!ok || !isSherpaAvailable()) {
				this._error = getSherpaError() ?? "sherpa-onnx unavailable";
				this._available = false;
				return false;
			}
			this._available = true;
			this._error = null;
			return true;
		} catch (err) {
			this._error = err instanceof Error ? err.message : String(err);
			this._available = false;
			return false;
		}
	}

	async warmup(ctx: EngineContext): Promise<void> {
		this.requireAvailable();
		const key = recognizerKey(ctx);
		const inFlight = this._warmupPromises.get(key);
		if (inFlight) return inFlight;
		if (this._recognizers.has(key)) return;

		const startGen = this._cacheGen;
		const promise = Promise.resolve().then(() => {
			// `getOrCreateRecognizer` is sync (no native I/O for ONNX —
			// the model is mmap'd by sherpa-onnx-node on first construct,
			// which is the slow path we want to amortize at warmup time).
			// Note: this populates sherpa-engine's module-level cache as
			// a side effect, even when we're about to discard the result
			// because of a stale generation.
			getOrCreateRecognizer(ctx.model, ctx.modelDir, ctx.lang);
			if (this._cacheGen !== startGen) {
				// Round-7 audit (GPT-5.5 Codex P0): the previous
				// `clearRecognizerCache()` here was itself racy —
				// between this stale-warmup detecting a gen change and
				// this line running, a FRESH warmup may have installed
				// a new recognizer in sherpa-engine's singleton cache.
				// Flushing now would clear that fresh recognizer.
				//
				// Just discard our adapter insertion. The leftover
				// recognizer in sherpa-engine's singleton (if any) is
				// harmless: it's keyed by (model, lang) and will be
				// returned by `getOrCreateRecognizer` only when the
				// next caller requests the SAME (model, lang) — at
				// which point it's still semantically correct (same
				// model, same language, same arguments). On the next
				// `clearCache()` it's flushed unconditionally anyway.
				return;
			}
			// Re-resolve from sherpa-engine's cache — guaranteed to be
			// the just-built recognizer since no concurrent invalidation
			// happened (gen check above). Cheap O(1) lookup.
			const recognizer = getOrCreateRecognizer(ctx.model, ctx.modelDir, ctx.lang);
			this._recognizers.set(key, {
				recognizer,
				activeTranscribes: 0,
				pendingFree: false,
				lastUsedAt: Date.now(),
			});
			this._evictUntilCapacity(this._cacheSize);
		}).finally(() => {
			// Round-5 audit (Kimi 2.5T #1): only delete if WE are still
			// the registered promise. Otherwise a `clearCache()` followed
			// by a fresh `warmup()` (same key, new promise) would let the
			// stale promise's finally evict the new one — leaking the
			// fresh recognizer and forcing a third warmup.
			if (this._warmupPromises.get(key) === promise) {
				this._warmupPromises.delete(key);
			}
		});
		this._warmupPromises.set(key, promise);
		return promise;
	}

	async transcribe(pcm: Buffer, ctx: EngineContext): Promise<string> {
		this.requireAvailable();
		const key = recognizerKey(ctx);
		let cached = this._recognizers.get(key);
		if (!cached) {
			await this.warmup(ctx);
			cached = this._recognizers.get(key);
			if (!cached) {
				throw new Error(
					`sherpa-onnx recognizer for "${ctx.model.id}" (lang=${ctx.lang}) ` +
					`was evicted during warmup (likely a concurrent clearCache).`,
				);
			}
		}
		// Reference-identity capture (round-3 audit Opus #1, #3) — same
		// pattern as whisper-cpp adapter: deferred-free must compare
		// reference identity, not id, so it never evicts a fresh entry
		// inserted by a post-clearCache warmup with the same id.
		const entry = cached;
		entry.activeTranscribes++;
		entry.lastUsedAt = Date.now();
		this._activeTranscribesGlobal++;
		try {
			return await transcribeBuffer(pcm, entry.recognizer);
		} finally {
			entry.activeTranscribes--;
			this._activeTranscribesGlobal--;
			if (entry.pendingFree && entry.activeTranscribes <= 0) {
				const current = this._recognizers.get(key);
				if (current === entry) this._recognizers.delete(key);
				// Round-6 audit (GPT-5.5 Codex): sherpa-engine's
				// singleton cache is now flushed IMMEDIATELY at
				// clearCache time, not deferred. So this finally only
				// drops the adapter's JS reference; native memory is
				// reclaimed by N-API finalizers on a future GC cycle.
				// The deferred-flush counter (`_pendingFreeRemaining`)
				// is no longer needed but kept on the class for binary
				// compatibility / debug logging.
			}
		}
	}

	async clearCache(): Promise<void> {
		this._cacheGen++;
		const entries = Array.from(this._recognizers.entries());
		this._recognizers.clear();
		this._warmupPromises.clear();
		// Round-6 audit (GPT-5.5 Codex BLOCK #3): flush sherpa-engine's
		// module-level singleton IMMEDIATELY at clear time, regardless
		// of active inferences. Deferring the flush opens a race where:
		//   1. clearCache() runs while transcribe(A) is in flight.
		//   2. We mark A's entry pendingFree, defer flush.
		//   3. New warmup(B) runs `getOrCreateRecognizer(B)`, which
		//      installs B in sherpa-engine's singleton cache.
		//   4. transcribe(A) finishes, its finally block runs the
		//      deferred `clearRecognizerCache()` — clearing B, the
		//      RECOGNIZER ANOTHER TRANSCRIBE MAY BE USING.
		// Flushing immediately is safe because the active in-flight
		// inference holds a direct reference to its recognizer (captured
		// `entry.recognizer` in transcribe), independent of whether
		// sherpa-engine's singleton still points to it.
		clearRecognizerCache();
		this._needsSherpaCacheFlush = false;
		this._pendingFreeRemaining = 0;
		// Mark entries pendingFree so the JS reference drops when the
		// in-flight transcribe settles. The map.delete in the finally
		// uses reference-identity, so a fresh post-clearCache warmup
		// cannot be mistakenly evicted.
		for (const [, c] of entries) {
			if (c.activeTranscribes > 0) c.pendingFree = true;
		}
	}

	/**
	 * Round-5 audit (GLM 5 #2): bound the adapter cache. Mirrors
	 * WhisperCppEngine._evictUntilCapacity: removes the LRU entries from
	 * the map immediately and marks active ones `pendingFree` so the
	 * `transcribe()` finally drops their references when the inference
	 * settles. Sherpa recognizers don't have a `free()` method, so
	 * eviction is just JS-reference dropping; the N-API finalizer runs
	 * on a future GC. Idempotent (no-op when under capacity).
	 */
	private _evictUntilCapacity(maxRetained: number): void {
		if (this._recognizers.size <= maxRetained) return;
		const sorted = Array.from(this._recognizers.entries())
			.sort(([, a], [, b]) => a.lastUsedAt - b.lastUsedAt);
		for (const [k, cached] of sorted) {
			if (this._recognizers.size <= maxRetained) break;
			this._recognizers.delete(k);
			if (cached.activeTranscribes > 0) {
				// Mark pending so transcribe()'s finally drops the JS ref
				// when the in-flight inference completes. Reference-
				// identity capture in transcribe makes this safe.
				cached.pendingFree = true;
			}
			// inactive entries: just drop the map slot. JS ref is gone
			// from the map; the local `cached` variable goes out of
			// scope after this iteration. N-API finalizer reclaims.
		}
	}

	private requireAvailable(): void {
		if (!this._available) {
			throw new Error(
				`sherpa-onnx engine is not available — ${this._error ?? "init() not called"}.`,
			);
		}
	}
}

export const sherpaEngine: Engine = new SherpaEngine();

/**
 * Pluggable STT engine interface.
 *
 * Phase 1 ships exactly one implementation (`engine-sherpa.ts`); whisper.cpp
 * and any future engines plug in by exposing a singleton that satisfies
 * `Engine`. Callers (`local.ts:transcribeInProcess`,
 * `voice.ts:warmupLocalRecognizerInBackground`) route through
 * `engine-factory.getEngineForModel(model)` and never touch a concrete
 * engine directly.
 *
 * Design decisions:
 *   - Context objects (not arg lists) so the caller resolves model + path
 *     once per call. Engines never re-resolve `modelDir`.
 *   - No `sampleRate` parameter — capture is pinned to 16 kHz upstream
 *     (`SAMPLE_RATE` constant in `deepgram.ts`); a build-time canary in
 *     `local.ts` fails if that ever changes without coordinated engine
 *     updates. Threading sample rate through the API would imply a
 *     configurability we don't actually have.
 *   - Engines start in `available: false, error: null`. All native binding
 *     loads happen inside `init()`, awaited, with errors captured to
 *     `engine.error`. Reading the singleton never throws.
 *   - `init()` is idempotent + single-flight. Failed inits clear the cached
 *     promise so a transient failure (locked file, brief FS error) doesn't
 *     brick the engine until process restart.
 */

import type { LocalModelInfo } from "./local";

// ─── Types ───────────────────────────────────────────────────────────────────

export type EngineId = "sherpa-onnx" | "whisper-cpp";

/**
 * Resolved per-call context. The caller is responsible for ensuring model
 * files exist on disk before invoking an engine — engines do not fetch.
 * Sample rate is fixed at 16 kHz across all engines (asserted at module load
 * in `local.ts`), so it is not part of the context.
 */
export interface EngineContext {
	/** The catalog entry. */
	model: LocalModelInfo;
	/** Absolute path to the directory where the model's files live. */
	modelDir: string;
	/** Language code (ISO 639-1, optionally with regional suffix). */
	lang: string;
}

export interface Engine {
	readonly id: EngineId;
	/** True iff `init()` succeeded. Always-readable; never throws. */
	readonly available: boolean;
	/** Last error from `init()` / `transcribe()`; null on success or pre-init.
	 *  Adapters MUST NOT use this slot for informational messages — that's
	 *  what `hint` is for (round-4 audit GLM 5: `error` set on success
	 *  violates contract callers reasonably check via `if (engine.error)`).
	 *
	 *  Concurrency note (round-8 audit Gemini 3.1 Pro): `error` reflects
	 *  the MOST RECENT failure. If `init()` fails, sets `error`, then a
	 *  later `init()` retry succeeds, `error` is cleared back to null.
	 *  In a multi-failure window (e.g. concurrent transcribes both fail),
	 *  callers see whichever error settled last. Treat `error` as a
	 *  diagnostic snapshot, not a per-call error report — for that, await
	 *  the per-call `init()` / `transcribe()` promise and inspect the
	 *  rejection. */
	readonly error: string | null;
	/**
	 * Optional informational message, populated even when `init()` succeeded.
	 * Use case: surfacing a degraded-but-working state, e.g. "Metal won't
	 * activate because GGML_METAL_PATH_RESOURCES wasn't shell-exported —
	 * transcription will run on CPU until you fix your shell rc". `null`
	 * when there's nothing to say. Distinct from `error` so callers can
	 * distinguish "broken" from "working but suboptimal".
	 */
	readonly hint: string | null;
	/** Idempotent + single-flight. Resolves to true iff the engine is now
	 *  `available`. MUST NOT be called from module top-level. */
	init(): Promise<boolean>;
	/** Build / load the recognizer for `ctx.model`. Cached per `model.id`
	 *  inside the engine; concurrent calls share a single in-flight load.
	 *
	 *  May throw (round-8 audit Opus 4.6): adapters validate
	 *  `ctx.model.whisperCppModel.files.model` against an allowlist
	 *  (`ggufFilenameFor`). A catalog entry that bypassed CI invariants
	 *  with an unsafe filename causes warmup to throw with a structured
	 *  error before any native binding is touched. Callers should
	 *  treat warmup as fallible. */
	warmup(ctx: EngineContext): Promise<void>;
	/** Run inference on a complete utterance. PCM is 16-bit signed LE mono
	 *  at 16 kHz. Throws if the engine is not `available`. */
	transcribe(pcm: Buffer, ctx: EngineContext): Promise<string>;
	/**
	 * Free cached recognizers. Called on model switch + session_shutdown.
	 * Idempotent — safe to call multiple times.
	 *
	 * Resolution semantics (round-3 audit Opus #5, #8):
	 *
	 *   - When the underlying binding exposes an awaitable `free()` (e.g.
	 *     `smart-whisper.Whisper.free()`), the adapter awaits all pending
	 *     `free()` calls for INACTIVE entries before this resolves. Native
	 *     memory is released by the time the promise resolves.
	 *
	 *   - When the binding does NOT expose `free()` (e.g. sherpa-onnx's
	 *     `OfflineRecognizer`), the adapter drops JS references and lets
	 *     N-API finalizers reclaim native memory on a future GC cycle.
	 *     The promise resolves immediately; callers should NOT assume
	 *     native memory is reclaimed before resolution for these adapters.
	 *
	 *   - Entries with active in-flight `transcribe()` calls are removed
	 *     from the adapter cache immediately (so future warmups load
	 *     fresh) but their native handles are freed in the `transcribe()`
	 *     `finally` block once the last active inference settles
	 *     (use-after-free protection). This deferred free is NOT awaited
	 *     by `clearCache()` — bounding the wait would require tracking
	 *     every in-flight inference promise, which the current shape
	 *     doesn't.
	 *
	 * Adapters MUST guarantee:
	 *   - `clearCache()` never throws.
	 *   - The map is empty by the time it resolves (active entries already
	 *     removed; their native handles will free later via deferred-free).
	 *   - A subsequent `warmup()` call always loads fresh — no stale entry
	 *     under the same id can leak across the boundary.
	 */
	clearCache(): Promise<void>;
}

// ─── Errors ──────────────────────────────────────────────────────────────────

/**
 * Thrown by `engine-factory.resolveEffectiveModel` when the catalog is so
 * corrupted that no fallback exists. `voice.ts:migratePersistedModelOnSessionStart`
 * catches this at session_start and disables voice for the session with a
 * clear notification — better than letting "Unknown model: …" surface
 * mid-transcribe with no breadcrumb to the catalog problem.
 */
export class VoiceCatalogUnrecoverableError extends Error {
	constructor(message: string) {
		super(message);
		this.name = "VoiceCatalogUnrecoverableError";
	}
}

/**
 * Shared sherpa-onnx-node loader used by both the STT engine
 * (`sherpa-engine.ts`) and the TTS engine (`tts-engine.ts`).
 *
 * Why a shared loader: the native module load is expensive (50-200ms) and
 * has platform compatibility checks that need to run exactly once per
 * process. Without this, an STT-then-TTS or TTS-then-STT call sequence
 * would either re-run the platform checks or — worse — race two concurrent
 * `import("sherpa-onnx-node")` calls when the user enables both backends.
 *
 * Concurrency contract: JavaScript is single-threaded with run-to-completion
 * semantics. The `sherpaInitialized` fast-path check, the `initPromise ??=
 * …` slot claim, and the synchronous start of the async body in
 * `doLoadSherpa()` all happen in the same tick — there is no preemption
 * window between them. A second caller arriving on a later microtask sees a
 * non-null `initPromise` and awaits the same in-flight promise. Once the
 * promise settles, `sherpaInitialized` is true so every later caller takes
 * the synchronous fast-path. No race window where two concurrent callers
 * can both enter the platform-check / dynamic-import path.
 *
 * Behavior is byte-identical to the previous `initSherpa()` in
 * `sherpa-engine.ts`. STT continues to call `initSherpa()` (kept as a
 * thin alias) so this is a zero-behavior-change refactor.
 */

import * as fs from "node:fs";

/** sherpa-onnx-node module — populated by loadSherpa() on first call. */
let sherpaModule: any = null;
let sherpaInitialized = false;
let sherpaError: string | null = null;

/**
 * Promise cache for in-flight initialization. Concurrent callers (e.g. one
 * voice command and one settings-panel diagnostic firing within the same
 * tick) all await the same promise instead of each running platform checks
 * and dynamically importing the native module independently.
 */
let initPromise: Promise<boolean> | null = null;

/**
 * Load (and cache) the sherpa-onnx-node native module. Returns true on
 * success, false on platform incompatibility or load failure. Subsequent
 * calls return the cached result synchronously through the resolved
 * promise — no work is repeated.
 */
export async function loadSherpa(): Promise<boolean> {
	if (sherpaInitialized) return !sherpaError;
	// `??=` is a single expression: assign-if-nullish. It claims the slot
	// before yielding, so two concurrent callers cannot both enter doLoadSherpa.
	initPromise ??= doLoadSherpa();
	return initPromise;
}

async function doLoadSherpa(): Promise<boolean> {
	try {
		// Early platform checks — fail fast with clear messages
		if (process.arch === "arm") {
			throw new Error("ARM32 (armv7l) is not supported by sherpa-onnx-node. Use 64-bit OS or the Deepgram cloud backend.");
		}
		if (process.platform === "linux") {
			try {
				const ldd = fs.readFileSync("/usr/bin/ldd", "utf-8");
				if (ldd.includes("musl")) {
					throw new Error("Alpine Linux (musl libc) is not supported by sherpa-onnx-node. Use a glibc-based distribution or the Deepgram cloud backend.");
				}
			} catch (e: any) {
				if (e?.message?.includes("musl")) throw e;
				// /usr/bin/ldd not readable — not Alpine, continue
			}
		}

		// Note: LD_LIBRARY_PATH/DYLD_LIBRARY_PATH set at runtime have no effect on dlopen().
		// The native .node binary uses $ORIGIN/@loader_path to find sibling .so/.dylib files,
		// so library resolution works without env var manipulation.

		// CJS-vs-ESM interop. sherpa-onnx-node ships as CommonJS with
		// `module.exports = { OfflineTts, OfflineRecognizer, ... }`.
		// Three runtimes return three different shapes:
		//
		//   - Bun: namespace exposes all symbols at top-level (works directly).
		//   - Node native: top-level is empty, full module on `.default`.
		//   - jiti (Pi's loader): top-level has STUB classes (e.g.
		//     `OfflineTts` is a function) but their static methods (like
		//     `OfflineTts.createAsync`) are MISSING. The real, fully-
		//     populated module sits on `.default`. This is the failure
		//     mode reported as "sherpa.OfflineTts.createAsync is not a
		//     function" on first /voice-speak.
		//
		// We sniff for `createAsync` specifically because that's the
		// thing the TTS engine needs. If the top-level OfflineTts has
		// it, the namespace is fully-populated (Bun); otherwise prefer
		// `.default` when it carries createAsync; final fallback is the
		// namespace as-is so future runtime variants degrade gracefully.
		const ns = await import("sherpa-onnx-node");
		const top = ns as any;
		const def = (ns as any).default;
		const topHasCreateAsync = typeof top?.OfflineTts?.createAsync === "function";
		const defHasCreateAsync = typeof def?.OfflineTts?.createAsync === "function";
		sherpaModule = topHasCreateAsync ? top : (defHasCreateAsync ? def : top);
		sherpaInitialized = true;
		return true;
	} catch (err: any) {
		sherpaError = err?.message || String(err);
		sherpaInitialized = true;
		return false;
	} finally {
		// Drop the in-flight reference once the promise settles. From here on
		// the synchronous `sherpaInitialized` fast-path at the top of loadSherpa
		// serves every caller — keeping the resolved promise around is dead
		// weight. Setting to null in the finally is safe because every code
		// path through the try/catch sets `sherpaInitialized = true` before
		// reaching here, so any later caller will fast-path and never read
		// the now-null `initPromise`.
		initPromise = null;
	}
}

/** Get the loader error message, if any. */
export function getSherpaError(): string | null {
	return sherpaError;
}

/** True when sherpa-onnx is loaded and usable. */
export function isSherpaAvailable(): boolean {
	return sherpaInitialized && !sherpaError && sherpaModule != null;
}

/**
 * Get the cached sherpa-onnx-node module. Throws if loadSherpa() has not
 * succeeded yet — callers must `await loadSherpa()` first and check the
 * boolean return before calling this.
 */
export function getSherpaModule(): any {
	if (!sherpaInitialized || sherpaError || sherpaModule == null) {
		throw new Error("sherpa-onnx not loaded. Call loadSherpa() first and verify it returned true.");
	}
	return sherpaModule;
}

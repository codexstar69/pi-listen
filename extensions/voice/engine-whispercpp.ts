/**
 * whisper.cpp engine adapter — wraps `smart-whisper` (a NAPI binding around
 * the official whisper.cpp library).
 *
 * Why smart-whisper:
 *   - PCM Float32Array input (matches our 16 kHz capture; we just convert
 *     Int16 → Float32 inline)
 *   - GPU flag selects Metal on macOS and CUDA on Linux when whisper.cpp
 *     was built with the right backend
 *   - Single-file GGUF model load (matches our `whisperCppModel.files.model`
 *     catalog convention)
 *   - MIT-licensed
 *
 * Install caveat (documented in HANDOFF.md): `smart-whisper` builds via
 * `node-gyp rebuild` on install — requires Xcode CLT (macOS) or
 * build-essential + cmake (Linux), and the project install path must not
 * contain spaces (smart-whisper's binding.gyp doesn't quote paths).
 *
 * Lifecycle contract (per `engine.ts` + audit round 2):
 *   - Singleton starts in `available: false, error: null`. Reading the
 *     instance never throws.
 *   - Native binding load is deferred to `init()`. Failures map to
 *     friendly messages with platform-specific install commands.
 *   - `init()` is idempotent + single-flight; failures clear the cached
 *     promise so transient errors don't permanently brick the engine.
 *   - `warmup()` is single-flight per `model.id`. The `_whispers` cache
 *     is bounded — `PI_VOICE_ENGINE_CACHE_SIZE` (default 1, max 4)
 *     LRU-evicts older models when a new one is loaded.
 *   - A generation counter (`_cacheGen`) prevents the
 *     `clearCache()`/`warmup()` race: a warmup that completes after
 *     `clearCache()` ran finds its captured generation stale and frees
 *     the just-loaded model instead of caching it.
 *   - `clearCache()` is async and defers freeing models that are
 *     mid-`transcribe()` until the active inference settles
 *     (use-after-free protection).
 *
 * Path-traversal hardening: `ggufFilenameFor` enforces an allowlist
 * regex on `whisperCppModel.files.model`, refusing path separators,
 * URL-encoded escapes (`%2F`, `%2E`), Windows reserved names, and any
 * non-`[A-Za-z0-9._-]` character. CI mirrors the same predicate over the
 * catalog so unsafe filenames cannot ship.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { createRequire } from "node:module";
import type { Engine, EngineContext, EngineId } from "./engine";
import type { LocalModelInfo } from "./local";

// ─── Filename safety ────────────────────────────────────────────────────────

/**
 * Allowlist for GGUF filenames — alphanumerics, dot, hyphen, underscore.
 * Any character outside this set (path separators, NUL, control chars,
 * URL-encoded escapes that contain `%`, Unicode homoglyphs, …) is
 * rejected. Keeping the allowlist narrow IS the security contract.
 */
const SAFE_GGUF_FILENAME_RE = /^[A-Za-z0-9._-]+$/;

/**
 * Windows reserved device names. Defense-in-depth — we don't currently
 * ship whisper.cpp on win32, but the catalog tests still apply this guard
 * so a hypothetical future Windows-target entry can't trip the parser.
 *
 *   MATCHES (REJECTED):  CON, PRN, AUX, NUL, COM1-COM9, COM10-COM999...,
 *                        LPT1-LPT99..., COM1.bin, lpt1.gguf
 *   DOES NOT MATCH:      COM0, COM00, LPT0 (NOT Windows-reserved per docs)
 *
 * `[1-9]\d*` requires non-zero leading digit; `[^/\\]` on the optional
 * extension is belt-and-braces (allowlist already rejects path separators).
 */
const WINDOWS_RESERVED_NAME_RE =
	/^(CON|PRN|AUX|NUL|COM[1-9]\d*|LPT[1-9]\d*)(\.[^/\\]*)?$/i;

/**
 * Resolve the on-disk filename for a GGUF model entry.
 *
 * Reads `whisperCppModel.files.model` directly — no URL parsing, no
 * `new URL()` call that could throw on a malformed catalog entry. URL
 * validity is enforced once at CI time by `tests/catalog-invariants`.
 *
 * @throws Error when `model` isn't GGUF, or when `files.model` fails the
 *  path-traversal allowlist (separators, URL escapes, NUL, parent-dir
 *  traversal, Windows reserved names). CI runs the same predicate over
 *  every catalog entry so unsafe filenames cannot ship.
 */
export function ggufFilenameFor(model: LocalModelInfo): string {
	if (model.format !== "gguf") {
		throw new Error(
			`ggufFilenameFor called on non-GGUF model: ${model.id} ` +
			`(format=${model.format ?? "undefined"})`,
		);
	}
	const filename = model.whisperCppModel?.files?.model;
	if (!filename) {
		throw new Error(
			`Catalog entry "${model.id}" is format=gguf but has no ` +
			`whisperCppModel.files.model — fix in extensions/voice/local.ts.`,
		);
	}
	if (
		filename.length === 0 ||
		!SAFE_GGUF_FILENAME_RE.test(filename) ||
		filename === "." || filename === ".." ||
		WINDOWS_RESERVED_NAME_RE.test(filename)
	) {
		throw new Error(
			`Catalog entry "${model.id}" has unsafe filename "${filename}" — ` +
			`must match ${SAFE_GGUF_FILENAME_RE.source} and not be a reserved ` +
			`OS name. URL-encoded sequences (e.g. %2F, %2E) are rejected by ` +
			`the allowlist.`,
		);
	}
	return filename;
}

// ─── Audio conversion ───────────────────────────────────────────────────────

/**
 * Convert 16-bit signed LE PCM (the on-the-wire format from sox/ffmpeg/
 * arecord) to Float32Array in [-1, 1] (the format smart-whisper consumes).
 * Pure — no native call.
 *
 * Odd-offset path (round-2 audit polish): instead of N readInt16LE calls
 * we copy once into an aligned Buffer then use the typed-array view. The
 * single copy is faster than per-sample reads for any non-trivial input.
 */
function pcmInt16ToFloat32(pcm: Buffer): Float32Array {
	const numSamples = Math.floor(pcm.length / 2);
	const float32 = new Float32Array(numSamples);
	const aligned = (pcm.byteOffset & 1) === 0
		? pcm
		: Buffer.from(pcm); // single copy — beats per-sample readInt16LE
	const int16 = new Int16Array(aligned.buffer, aligned.byteOffset, numSamples);
	for (let i = 0; i < numSamples; i++) {
		float32[i] = int16[i]! / 32768.0;
	}
	return float32;
}

// ─── Metal kernel resolution ────────────────────────────────────────────────

/**
 * On macOS arm64, smart-whisper's bundled Metal backend looks for
 * `ggml-metal.metal` (the Metal kernel source) at runtime. The file ships
 * inside `node_modules/smart-whisper/whisper.cpp/ggml/src/`; smart-whisper
 * itself doesn't set the resource path. Without it, Metal init falls
 * back to looking in `cwd` and fails on most invocation paths.
 *
 * **Upstream limitation (round-2 audit discovery):** macOS
 * `NSProcessInfo.environment` caches the process environment at process
 * start. Setting `process.env.GGML_METAL_PATH_RESOURCES` from inside Node
 * does NOT propagate to `NSProcessInfo`, so smart-whisper's
 * `[NSProcessInfo objectForKey:@"GGML_METAL_PATH_RESOURCES"]` lookup
 * returns nil. Metal then falls through to the cwd probe and silently
 * uses CPU.
 *
 * To actually enable Metal, the user must export the env var BEFORE
 * Node starts (typically in their shell rc). We set it from JS as
 * defense-in-depth — doesn't help on macOS but works on Linux Metal-
 * absent contexts where we want predictable behavior — AND we surface
 * the limitation via a friendly hint when init succeeds without the env
 * var being set on darwin-arm64.
 */
/** True iff `GGML_METAL_PATH_RESOURCES` was already set when this module
 *  loaded — i.e. the user shell-exported it before Node started. This is
 *  the only reliable signal that Metal will actually find its kernel
 *  source on macOS (NSProcessInfo cache pre-dates Node startup). */
const _metalResourcesShellExported = !!process.env.GGML_METAL_PATH_RESOURCES;

function configureMetalResources(): void {
	if (process.env.GGML_METAL_PATH_RESOURCES) return; // user-set wins
	if (process.platform !== "darwin") return;
	try {
		const req = createRequire(import.meta.url);
		const packageJson = req.resolve("smart-whisper/package.json");
		const ggmlSrcDir = path.join(path.dirname(packageJson), "whisper.cpp", "ggml", "src");
		if (!fs.existsSync(path.join(ggmlSrcDir, "ggml-metal.metal"))) return;
		process.env.GGML_METAL_PATH_RESOURCES = ggmlSrcDir;
	} catch {
		// smart-whisper not installed — env stays unset, init() catches.
	}
}

/**
 * Hint string when init succeeds on darwin-arm64 but Metal will silently
 * run on CPU because the env var wasn't shell-exported. Used by `init()`
 * to populate `engine.error` with an informational message — the engine
 * is still `available`, just slower than ideal until the user fixes
 * their shell.
 */
function metalShellExportHint(): string | null {
	if (process.platform !== "darwin" || process.arch !== "arm64") return null;
	if (_metalResourcesShellExported) return null;
	const req = createRequire(import.meta.url);
	let ggmlSrcDir: string;
	try {
		const packageJson = req.resolve("smart-whisper/package.json");
		ggmlSrcDir = path.join(path.dirname(packageJson), "whisper.cpp", "ggml", "src");
	} catch {
		return null;
	}
	return (
		`whisper-cpp engine ready, but Metal acceleration won't activate ` +
		`until GGML_METAL_PATH_RESOURCES is shell-exported (macOS NSProcessInfo ` +
		`caches the env at process start; Node-set env doesn't propagate). ` +
		`Add to your shell rc:\n` +
		`  export GGML_METAL_PATH_RESOURCES="${ggmlSrcDir}"\n` +
		`Then restart pi-voice. Until then, transcription runs on CPU.`
	);
}

// ─── Cache size config ──────────────────────────────────────────────────────

/**
 * How many warm whisper models to keep simultaneously. Each cached model
 * costs whatever GGUF size it loaded as native memory (~57 MB for Base EN,
 * ~600 MB for Turbo). Default 1 — most users only run one model at a time
 * and switching models is rare. Power users can bump via env (clamped 1-4).
 */
function resolveCacheSize(env: NodeJS.ProcessEnv = process.env): number {
	const raw = env.PI_VOICE_ENGINE_CACHE_SIZE;
	if (!raw) return 1;
	const n = Number.parseInt(raw, 10);
	if (!Number.isFinite(n) || n < 1) return 1;
	if (n > 4) return 4;
	return n;
}

// ─── Friendly error mapping ─────────────────────────────────────────────────

/**
 * Map a low-level binding load failure to an actionable user-facing
 * message. The user almost never sees the raw "Cannot find module
 * 'smart-whisper'" or libstdc++ symbol-resolution noise; they see what
 * they should run to fix it.
 */
function friendlyInitError(rawMessage: string): string {
	const msg = rawMessage.toLowerCase();
	const platform = process.platform;
	// Round-5 audit (GPT-5.5 #3, Kimi 2.5T #7): clang's binding.gyp
	// path-with-spaces failure has a specific signature distinct from
	// Node's "Cannot find module 'foo bar'":
	//   - clang: emits "no such file or directory: 'Files/Pi'" (the
	//     fragment is RELATIVE — no leading slash — because the path got
	//     split on whitespace into broken arg pieces).
	//   - Node:  emits "Cannot find module 'foo bar'" (no "no such file
	//     or directory:" prefix).
	// We match the clang signature: "no such file or directory:" AND a
	// node_modules reference somewhere in the same error. This catches
	// the binding.gyp failure without false-firing on normal missing-
	// module errors that happen to contain a space.
	const isClangPathFragment = rawMessage.includes("no such file or directory:");
	const referencesNodeModules = rawMessage.includes("node_modules");
	if (isClangPathFragment && referencesNodeModules) {
		return (
			`smart-whisper failed to build because the install path contains ` +
			`spaces — smart-whisper@0.8.1's binding.gyp doesn't quote paths. ` +
			`Move the project to a path without spaces and reinstall.\n` +
			`(Original: ${rawMessage})`
		);
	}
	if (msg.includes("cannot find module") || msg.includes("cannot find package")) {
		if (platform === "darwin") {
			return (
				`smart-whisper native binding failed to load. Install Xcode ` +
				`Command Line Tools and rebuild:\n` +
				`  xcode-select --install\n` +
				`  bun install --force\n` +
				`(Original: ${rawMessage})`
			);
		}
		if (platform === "linux") {
			return (
				`smart-whisper native binding failed to load. Install build ` +
				`prerequisites and rebuild:\n` +
				`  sudo apt install build-essential cmake python3\n` +
				`  bun install --force\n` +
				`For NVIDIA CUDA: WHISPER_CUDA=1 bun install --force\n` +
				`(Original: ${rawMessage})`
			);
		}
		return (
			`smart-whisper native binding failed to load. Reinstall with ` +
			`a working node-gyp toolchain:\n` +
			`  bun install --force\n(Original: ${rawMessage})`
		);
	}
	if (msg.includes("ggml") || msg.includes("metal")) {
		return (
			`smart-whisper Metal backend init failed. Falling back to CPU. ` +
			`Set GGML_METAL_PATH_RESOURCES manually if your install layout ` +
			`differs from smart-whisper@0.8.1.\n(Original: ${rawMessage})`
		);
	}
	if (msg.includes("cuda")) {
		return (
			`smart-whisper CUDA backend init failed. Confirm the binding was ` +
			`built with CUDA support:\n  WHISPER_CUDA=1 bun install --force\n` +
			`(Original: ${rawMessage})`
		);
	}
	return rawMessage;
}

// ─── Engine ─────────────────────────────────────────────────────────────────

/** Cached model record. `lastUsedAt` powers the LRU eviction order;
 *  `activeTranscribes` defers `free()` while the recognizer is in use. */
interface CachedWhisper {
	readonly instance: import("smart-whisper").Whisper;
	lastUsedAt: number;
	activeTranscribes: number;
	pendingFree: boolean;
}

class WhisperCppEngine implements Engine {
	readonly id: EngineId = "whisper-cpp";
	private _available = false;
	private _error: string | null = null;
	/** Informational message — populated even on successful init. See
	 *  `Engine.hint` JSDoc + round-4 audit GLM 5 finding. */
	private _hint: string | null = null;
	private _initPromise: Promise<boolean> | null = null;
	private _smartWhisper: typeof import("smart-whisper") | null = null;

	private readonly _whispers = new Map<string, CachedWhisper>();
	private readonly _warmupPromises = new Map<string, Promise<void>>();
	/**
	 * Generation counter incremented every `clearCache()`. Each warmup
	 * captures the gen at start; if it changed by the time `whisper.load()`
	 * resolves, the just-loaded model is freed instead of cached. Prevents
	 * the round-2 audit race: `clearCache → set → microtask drain → warmup
	 * resolves → re-inserts a freshly-loaded native handle that the user
	 * meant to be cleared`.
	 */
	private _cacheGen = 0;
	private readonly _cacheSize = resolveCacheSize();

	get available() { return this._available; }
	get error() { return this._error; }
	get hint() { return this._hint; }

	init(): Promise<boolean> {
		if (this._available) return Promise.resolve(true);
		if (this._initPromise) return this._initPromise;
		this._initPromise = this._doInit();
		// Non-throwing cleanup — _doInit() catches all errors and resolves
		// `false`, but a future edit might let an exception escape. Using
		// .then(_, _) instead of .finally() keeps the cleanup path
		// rejection-safe (round-2 audit: no unhandled-rejection noise).
		this._initPromise.then(
			() => { this._initPromise = null; },
			() => { this._initPromise = null; },
		);
		return this._initPromise;
	}

	private async _doInit(): Promise<boolean> {
		try {
			configureMetalResources();
			const mod = await import("smart-whisper");
			// Interface-shape check (defense-in-depth against a tampered or
			// partially-corrupted binding). `Whisper` is the constructor we
			// use; nothing else is required for the adapter contract.
			if (typeof (mod as any)?.Whisper !== "function") {
				this._error = `smart-whisper missing required export "Whisper"`;
				this._available = false;
				return false;
			}
			this._smartWhisper = mod;
			this._available = true;
			this._error = null;
			// Round-4 audit (GLM 5): surface the Metal degradation message
			// via `hint`, NOT `error`. The interface contract says
			// `error: null on success` — callers checking
			// `if (engine.error)` rely on it. The Metal-on-CPU fallback
			// is "working but suboptimal", which is exactly what `hint`
			// is for.
			this._hint = metalShellExportHint();
			return true;
		} catch (err) {
			const raw = err instanceof Error ? err.message : String(err);
			this._error = friendlyInitError(raw);
			this._available = false;
			return false;
		}
	}

	async warmup(ctx: EngineContext): Promise<void> {
		this.requireAvailable();
		const inFlight = this._warmupPromises.get(ctx.model.id);
		if (inFlight) return inFlight;
		const cached = this._whispers.get(ctx.model.id);
		if (cached) {
			cached.lastUsedAt = Date.now();
			return;
		}

		const startGen = this._cacheGen;
		const promise = Promise.resolve().then(async () => {
			const filename = ggufFilenameFor(ctx.model);
			const fullPath = path.join(ctx.modelDir, filename);
			const whisper = new this._smartWhisper!.Whisper(fullPath, {
				gpu: true,
				offload: 60,
			});
			try {
				await whisper.load();
			} catch (err) {
				// Free the Whisper instance if load fails
				try { await whisper.free(); } catch { /* native finalizer */ }
				throw err;
			}
			// Generation check: if `clearCache()` ran while we were loading,
			// the user wanted no cached models. Free the just-loaded one
			// instead of caching it.
			if (this._cacheGen !== startGen) {
				try { await whisper.free(); } catch { /* native finalizer */ }
				return;
			}
			// Insert first, then evict - ensures correct LRU ordering
			this._whispers.set(ctx.model.id, {
				instance: whisper,
				lastUsedAt: Date.now(),
				activeTranscribes: 0,
				pendingFree: false,
			});
			this._evictUntilCapacity(this._cacheSize);
		}).finally(() => {
			// Round-5 audit (Kimi 2.5T #1): identity-check before delete.
			// If `clearCache()` ran and a fresh warmup inserted a new
			// promise at this id, our stale promise's finally must NOT
			// evict the fresh one.
			if (this._warmupPromises.get(ctx.model.id) === promise) {
				this._warmupPromises.delete(ctx.model.id);
			}
		});
		this._warmupPromises.set(ctx.model.id, promise);
		return promise;
	}

	async transcribe(pcm: Buffer, ctx: EngineContext): Promise<string> {
		this.requireAvailable();
		let cached = this._whispers.get(ctx.model.id);
		if (!cached) {
			await this.warmup(ctx);
			cached = this._whispers.get(ctx.model.id);
			if (!cached) {
				throw new Error(
					`whisper-cpp recognizer for "${ctx.model.id}" was evicted ` +
					`during warmup (likely a concurrent clearCache).`,
				);
			}
		}
		// Capture the entry reference here. round-3 audit (Opus #1, #3):
		// the deferred-free path MUST act against THIS reference, not by
		// re-looking-up `ctx.model.id`. Otherwise a `clearCache()` followed
		// by a fresh `warmup()` (same id, new entry) would let our finally
		// block delete the FRESH recognizer.
		const entry = cached;
		entry.activeTranscribes++;
		entry.lastUsedAt = Date.now();
		try {
			const float32 = pcmInt16ToFloat32(pcm);
			const baseLang = (ctx.lang.split("-")[0] || "en").toLowerCase();
			const task = await entry.instance.transcribe(float32, { language: baseLang });
			const segments = await task.result;
			return segments.map(s => s.text).join("").trim();
		} finally {
			// Guard against negative activeTranscribes if error thrown before increment
			if (entry.activeTranscribes > 0) entry.activeTranscribes--;
			if (entry.pendingFree && entry.activeTranscribes <= 0) {
				// Reference-identity check: only remove the map entry if
				// it's STILL the same record we just used. A new
				// warmup() may have inserted a fresh entry under this id;
				// we must not evict it.
				const current = this._whispers.get(ctx.model.id);
				if (current === entry) this._whispers.delete(ctx.model.id);
				try { await entry.instance.free(); } catch { /* native finalizer */ }
			}
		}
	}

	async clearCache(): Promise<void> {
		this._cacheGen++;
		const entries = Array.from(this._whispers.entries());
		this._whispers.clear();
		this._warmupPromises.clear();
		const frees: Promise<void>[] = [];
		for (const [, cached] of entries) {
			if (cached.activeTranscribes > 0) {
				// Defer — transcribe()'s finally block frees this entry
				// once the last active inference settles. The active
				// transcribe holds the entry by REFERENCE (captured local
				// `entry` var), not by map lookup, so removing from the
				// map here doesn't break the in-flight inference. The
				// finally block calls `entry.instance.free()` against the
				// captured ref; map identity is only consulted to skip
				// evicting a fresh entry inserted by a post-clearCache
				// warmup. Do NOT re-insert the active entry — a future
				// caller looking up this id should see an empty map and
				// load a fresh recognizer (the old one is being freed).
				cached.pendingFree = true;
				continue;
			}
			// Wrap in Promise.resolve().then() so a synchronous throw from
			// `free()` (e.g., binding-side validation error before the
			// promise resolves) becomes a rejection — round-4 audit
			// (GPT-5.5 #3): without this, `clearCache()` could throw
			// despite contract claiming it never throws.
			frees.push(Promise.resolve().then(() => cached.instance.free()).catch(() => {/* native finalizer */}));
		}
		await Promise.all(frees);
	}

	private _evictUntilCapacity(maxRetained: number): void {
		if (this._whispers.size <= maxRetained) return;
		// Round-8 audit (GPT-5.5 Codex + GPT-5.5 Amp consensus, P1): an
		// earlier rewrite changed this loop to `continue` past active
		// entries WITHOUT removing them from the map. With cache size 1,
		// model A active, warmup B inserts B (size=2). Iteration: A
		// (active) → skip-without-delete; B (inactive) → DELETE B. The
		// just-loaded B was the WRONG entry to evict; a subsequent
		// `transcribe(B)` then throws "evicted during warmup" because
		// the warmup-finally lookup returns undefined.
		//
		// Correct behavior: remove the LRU entry from the map even when
		// it has active transcribes — `transcribe()`'s finally block
		// holds the entry by reference (captured `entry` local var),
		// not by map lookup, so removing the map slot doesn't break the
		// in-flight inference. Mark `pendingFree`; the finally calls
		// `entry.instance.free()` UNCONDITIONALLY when both flags hit
		// (reference-identity check is only used to gate the *map*
		// delete, not the `free()` call).
		const sorted = Array.from(this._whispers.entries())
			.sort(([, a], [, b]) => a.lastUsedAt - b.lastUsedAt);
		for (const [id, cached] of sorted) {
			if (this._whispers.size <= maxRetained) break;
			this._whispers.delete(id);
			if (cached.activeTranscribes > 0) {
				// Active — defer free to transcribe()'s finally.
				cached.pendingFree = true;
				continue;
			}
			// Inactive — free immediately. Sync-throw guard keeps
			// `clearCache`'s "never throws" contract symmetric here.
			void Promise.resolve().then(() => cached.instance.free()).catch(() => {/* native finalizer */});
		}
	}

	private requireAvailable(): void {
		if (!this._available) {
			throw new Error(
				`whisper-cpp engine is not available — ${this._error ?? "init() not called"}.`,
			);
		}
	}
}

export const whispercppEngine: Engine = new WhisperCppEngine();

// Test seam — used by tests/engine-whispercpp.test.ts to set the cache
// size directly without env-var manipulation between runs.
export const __test = {
	resolveCacheSize,
	friendlyInitError,
	pcmInt16ToFloat32,
};

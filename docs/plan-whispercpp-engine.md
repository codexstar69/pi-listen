# Plan: Pluggable STT Engine — sherpa-onnx + whisper.cpp Metal

## Changes since v14 (godspeed round-14)

This v15 fully addresses the catalog-corruption surface area:

- **sonnet — three-tier fallback duplicated**: extracted to a single
  `findResilientFallbackId()` helper used by both
  `degradedResolve()` (outer ensureCatalogValid catch) and
  `safeDefault()` (inner pickDefault catch). Logic now lives in one
  place; future fallback-policy changes are a single-site edit.
- **grok + minimax — tier-3 still returned a missing id**: added a
  `LOCAL_MODELS[0]` last resort search. If LITERALLY no model
  exists in the catalog, `findResilientFallbackId()` returns `null`
  and the resolver throws a typed `VoiceCatalogUnrecoverableError`.
- **`migratePersistedModelOnSessionStart` catches
  `VoiceCatalogUnrecoverableError`**: disables voice for the session
  with a clear `notify("error")` instead of letting "Unknown model:
  …" surface mid-transcribe with no breadcrumb. Voice off + clear
  message > broken-but-running. Persisted config untouched, so
  reinstalling pi-voice (or copying the config to a working machine)
  restores normal operation.
- **`degraded: true` cases also warn the user**: previously the v14
  `migratePersistedModelOnSessionStart` only notified on `fellBack`,
  so catalog-corruption "degraded" sessions were silent. v15 notifies
  on either condition, with `notify("warning")` for the degraded
  branch (a louder color than `info`).
- **qwen — false positive on warmup/transcribe sequencing**: traced
  the flow; transcribe checks `_recognizers.has()` then `await
  warmup()` if missing, then reads from `_recognizers` after the
  await. Cache is set inside the warmup promise's success branch,
  which resolves before the await returns. No race. Added a unit
  test asserting that two concurrent transcribe() calls observe the
  recognizer in the cache after their shared warmup() resolves.

## Changes since v13 (godspeed round-13)

This v14 closes the remaining 8/10 → ship gap:

- **grok — degraded path can return non-existent fallback id**: the
  v13 `safeDefault()` and outer `ensureCatalogValid` catch both
  returned `ULTIMATE_FALLBACK_MODEL_ID` directly without verifying it
  was still in `LOCAL_MODELS`. If catalog corruption removed even
  the fallback, downstream `transcribeInProcess` would throw
  "Unknown model: moonshine-v2-tiny" with no breadcrumb to the
  catalog problem. v14 adds a three-tier fallback search:
    1. Use `ULTIMATE_FALLBACK_MODEL_ID` if it's still in the catalog
       (the common case — most invariant violations don't delete
       the fallback model).
    2. Otherwise search `LOCAL_MODELS` for any platform-agnostic
       sherpa model and use it.
    3. Last resort: return the hardcoded id even if missing — at
       least surfaces "Unknown model: …" with a previously-logged
       catalog-corruption warning, making the failure mode
       diagnosable from logs.
- **qwen — `warmup()` concurrent-trace claim** (advisory, not
  blocking by corroboration but worth addressing): the
  `Promise.resolve().then(…)` ordering is correct, but reasoning
  about microtask scheduling is finicky. v14 adds an explicit
  execution-trace unit test that pins all four invariants
  (single-flight, set-before-drain, cleanup-on-settle,
  retry-after-drain) for both the success and sync-throw branches.
  Machine-verifiable, not just "I promise the trace works".

## Changes since v12 (godspeed round-12)

This v13 closes the remaining 8/11 → 10/11 gap:

- **gemini misread + minimax misread on warmup ordering**: both
  reviewers ended up arguing about whether the v12 `async` IIFE
  semantics actually preserved the `set → microtask-cleanup` order.
  The v12 code WAS correct (async functions yield to microtask after
  any sync throw, so `set()` ran first), but the reasoning lived in
  comments and reviewers couldn't verify by inspection. v13 rewrites
  `warmup()` to use `Promise.resolve().then(async () => …)` —
  `Promise.resolve()` is immediately-resolved and `.then(cb)`
  schedules `cb` as a microtask, so `cb`'s synchronous throw is
  IMPOSSIBLE to observe before `_warmupPromises.set(…)` returns. The
  ordering is now obvious from the structure, not from runtime
  semantics. Step-by-step execution trace included in the code
  comment.
- **gemini round-12 — Windows regex comment misread**: the v12
  comment "REJECTS COM0 / LPT0 / COM00" read as a contradiction with
  the regex `COM[1-9]\d*` (which DOESN'T match `COM0`). v13 splits
  the comment into two explicit lists ("MATCHES → REJECTS" vs.
  "DOES NOT MATCH → ACCEPTS"), removing the ambiguity. No regex
  change.
- **qwen round-12 nit — comment attribution**: cleaned up the
  warmup-comment to reference the actual fix (microtask scheduling
  via `Promise.resolve().then`) rather than the round-11 history
  about IIFE finally placement.

## Changes since v11 (godspeed round-11)

This v12 closes the last two NO_SHIPs:

- **gemini (advisory but real concurrency bug) — sync-throw before
  `_warmupPromises.set`**: the v11 `warmup()` had `try/finally` INSIDE
  the async IIFE. If `ggufFilenameFor` threw synchronously (corrupted
  catalog), the `finally` ran as part of the IIFE's synchronous
  prelude — BEFORE the outer `set()` registered the entry. The
  `delete` no-op'd on an empty map, then `set` cached the rejected
  promise, permanently bricking the engine for that model id. v12
  moves cleanup to **outer** `.finally()` (microtask-scheduled) so
  the order is: IIFE returns rejected promise → `set()` registers it
  → microtask drain → `.finally` deletes the entry → next caller
  retries cleanly. Test added that injects a sync-throwing
  `ggufFilenameFor` and asserts a second `warmup()` call after the
  microtask drain runs a fresh attempt.
- **moonshot + grok — `COM\d+` / `LPT\d+` matches COM0 / LPT0 / COM00
  which aren't actually Windows reserved**: tightened the regex to
  `COM[1-9]\d*` / `LPT[1-9]\d*` — non-zero leading digit, optional
  trailing digits. Covers Microsoft's documented COM1–COM9 +
  multi-port extensions (COM10–COM256) without over-rejecting
  legitimate filenames whose names happen to start with COM0 / LPT0.
  Test cases asserting both — reject `COM10`, accept `COM0` /
  `LPT0` / `COM00`.

## Changes since v10 (godspeed round-10)

This v11 closes the last two NO_SHIPs to push the SHIP count from 9/11
to threshold:

- **minimax — Windows reserved-name regex misses COM10+**: `COM[1-9]` /
  `LPT[1-9]` left COM10–COM256 (and LPT equivalents) as a defense-in-
  depth gap on Windows multi-port serial-card setups. v11 widens to
  `COM\d+` / `LPT\d+` so the entire class of NT device-name patterns
  is rejected. Tests cover `COM10`, `COM256`, `LPT99`, `COM10.bin`.
- **grok — allowlist rejects spaces but HuggingFace might use them**:
  research-grade false positive — every file under
  `huggingface.co/ggerganov/whisper.cpp` uses hyphen-separated
  lowercase, no spaces. v11 documents the convention in code, calls
  out the catalog-invariants CI test that enforces URL-basename ↔
  `files.model` consistency, and adds an explicit test asserting
  spaces ARE rejected (intentional invariant). If a future binding
  needs spaces, it's a one-line allowlist update + test refresh; the
  security surface stays intact.
- **security (advisory) — `\..*` extension suffix**: tightened
  `WINDOWS_RESERVED_NAME_RE` extension group to `\.[^/\\]*` —
  defense-in-depth even though the allowlist already rejects `/` and
  `\` upstream of this regex.
- **qwen — comment nit**: `_initPromise` finally-cleanup comment
  reworded to acknowledge it handles both success and failure paths,
  not just failure.

## Changes since v9 (godspeed round-9 — ONE confirmed VETO)

This v10 addresses every NO_SHIP from round 9, including the critical
URL-encoded traversal VETO confirmed by the advisor:

- **grok (VETO, critical) — URL-encoded traversal bypass**: the v9
  guard relied on `path.basename(filename) === filename` as
  belt-and-braces, but Node's `path.basename` does NOT decode URL
  escapes — `path.basename("..%2F..%2Fetc%2Fpasswd") ===
  "..%2F..%2Fetc%2Fpasswd"` (true), so the check passed and the
  decoded path would later traverse. v10 replaces the blocklist with
  an **allowlist regex**, `/^[A-Za-z0-9._-]+$/`, which rejects the
  entire class of escape sequences (`%`, Unicode look-alikes, NULs,
  control chars) by construction. The allowlist is tighter than any
  enumerable blocklist and matches every legitimate GGUF filename we
  ship today.
- **runtime — `ensureCatalogValid()` outside the `safeDefault()`
  try/catch**: the v8 plan claimed catalog corruption couldn't crash
  session_start, but the validator call itself was at the top of
  `resolveEffectiveModel`, outside any catch. v10 wraps it in its own
  try/catch that returns the hardcoded `ULTIMATE_FALLBACK_MODEL_ID`
  with `degraded: true` if invariants fail, and adds a unit test that
  injects a corrupted catalog mock to verify session_start gets a
  useable result instead of a thrown exception.
- **security2 (advisory) — Windows reserved names**: added
  `WINDOWS_RESERVED_NAME_RE` rejecting `CON`, `PRN`, `AUX`, `NUL`,
  `COM[1-9]`, `LPT[1-9]` (case-insensitive, with optional extension).
  We don't ship whisper.cpp for win32 today, but defense-in-depth is
  cheap and the test fails CI if a Windows-target catalog entry ever
  uses one of these names.
- **deepseek / qwen — observation comments**: noted but no behavior
  change. `_warmupPromises.delete()` in finally is intentional
  defense for the failure-retry path (qwen called it "redundant" but
  it's the failure-cleanup branch). `_initPromise` after success is
  cleared by `.finally()` (already addressed in v9).

## Changes since v8 (godspeed round-8 findings)

This v9 addresses every NO_SHIP from round 8. (Both round-8 VETOs were
*cleared* by the advisor as INVALID, but reviewer votes still count
toward the SHIP threshold, so we tighten anyway.)

- **runtime — warmup check ordering ambiguity**: the v8 code was
  technically race-free (single-threaded JS, no `await` between the
  has() check and the .set()) but read ambiguously to gemini. v9
  reorders to check `_warmupPromises` FIRST, then `_recognizers`. The
  in-flight gate is now the obvious first line of defense; future
  reviewers don't have to reason about the JS task model. Comment
  explicitly explains the order rationale.
- **minimax — `startsWith("..")` over-rejects `"..."`**: narrowed the
  parent-dir check to `startsWith("../")` and `startsWith("..\\")`.
  Plain `"..."` (three-dot ellipsis filename, valid on POSIX) and
  `"....bin"` are no longer false-positives. The
  `path.basename(filename) === filename` belt-and-braces still rejects
  any encoded-traversal exotica.
- **qwen — `_initPromise` cleanup misses rejected promises**: the
  `.then(ok => …)` callback never fires if the inner promise rejects.
  v9 extracts `_doInit()` (which catches all errors and returns
  `false` via try/catch) and uses `.finally()` for cleanup, which
  fires on resolve OR reject. Belt-and-braces against future edits to
  `_doInit()` that might let an exception escape.
- **deepseek — minor `_initPromise` leak after success**: the v8
  cleanup only ran on failure; success kept the resolved promise
  pinned. v9's `.finally()` clears unconditionally — `_available` is
  the source of truth for the success case, so dropping the cached
  promise is safe and avoids the leak.

## Changes since v7 (godspeed round-7 — TWO VETOs)

This v8 addresses both VETOed findings + the security2 advisory:

- **security (VETO, high) — path traversal in `ggufFilenameFor`**: a
  malicious catalog `files.model = "../../etc/passwd"` would let
  `path.join(ctx.modelDir, filename)` escape the models dir. v8 adds
  an explicit `assertSafeBasename`-style guard inside
  `ggufFilenameFor`: rejects empty, NUL, path separators, leading
  `..`, and any value where `path.basename(filename) !== filename`.
  CI mirrors the same predicate over every GGUF catalog entry, so
  unsafe filenames can never ship. The guard is also reused at the
  download-write site in `model-download.ts` — defense in depth on
  both read and write paths.
- **runtime (VETO, high) — concurrent warmup double-loads 800 MB
  model**: `WhisperCppEngine` now keeps a per-model `_warmupPromises`
  Map. Concurrent `warmup()` calls for the same model id share one
  in-flight promise (single-flight), preventing double native-memory
  load and the OOM risk gemini flagged. The map entry is cleared on
  both success and failure so retries work.
- **security2 (advisory) — failed `init()` permanently bricks the
  engine**: `_initPromise` is now cleared on a failed result via
  `p.then(ok => { if (!ok) this._initPromise = null; })`. Transient
  failures (locked file, brief FS error) no longer require a process
  restart — the next caller starts a fresh init. Successful init is
  still cached via `_available`.

## Changes since v6 (godspeed round-6 findings)

This v7 addresses every NO_SHIP from round 6:

- **security2 — `pickDefaultModelForPlatform` can throw, crashing
  session_start**: `resolveEffectiveModel` now wraps the call in a
  `safeDefault()` helper that catches and degrades to
  `ULTIMATE_FALLBACK_MODEL_ID` directly, with a `degraded: true` flag
  surfaced in the return value so `voice.ts` can show a different
  notification. Catalog corruption no longer crashes startup.
- **runtime + sonnet — `new URL()` runtime crash on bad catalog**:
  removed `new URL()` from `ggufFilenameFor`. Filename is now read
  directly from `model.whisperCppModel.files.model` (a string field on
  the catalog entry) — no parsing at runtime. URL validity is enforced
  once at CI time via `tests/catalog-invariants.test.ts`.
- **minimax — `ensureCatalogValid` doesn't check fallback's `engine`**:
  added third invariant — fallback must have `engine` undefined or
  `"sherpa-onnx"`, otherwise `PI_VOICE_FORCE_SHERPA=1` would have no
  valid path. Catalog test mirrors the runtime check.
- **grok — `clearCache()` missing from WhisperCppEngine sketch**: full
  class sketch now implements every `Engine` interface method —
  `init`, `warmup`, `transcribe`, `clearCache`, plus a
  `requireAvailable()` private guard and a `_recognizers` Map keyed by
  `model.id` (mirrors sherpa-engine's caching pattern).
- **qwen — ambiguous catalog shape for GGUF (`sherpaModel`-typed)**:
  introduced parallel `whisperCppModel` field on `LocalModelInfo` for
  GGUF models. `sherpaModel` continues to own ONNX. Catalog invariant
  test asserts each entry uses exactly one of the two, never both /
  neither. All four whisper.cpp catalog entries updated to the new
  shape with concrete filenames + URLs.

## Changes since v5 (godspeed round-5 findings)

This v6 addresses every NO_SHIP from round 5:

- **architect — misleading sample-rate parameter**: removed `sampleRate`
  from `EngineContext` / engine API entirely. The 16 kHz invariant is now
  enforced once at module load in `local.ts` (`if (SAMPLE_RATE !== 16000)
  throw …`) — a build-time canary instead of a parameter that pretends
  to be configurable. Future 48 kHz support is a single coordinated
  change to the constant + per-engine `acceptWaveform` reads, no
  cross-cutting interface churn.
- **security (advisory) — supply-chain integrity on dynamic import**:
  added an interface-shape check after `await import("nodejs-whisper")`
  — `engine.error` is set if the binding is missing required exports,
  callers fall back via `migratePersistedModelOnSessionStart`. Documented
  defense-in-depth: lockfile-pinned exact version, no dynamic binding
  paths, HANDOFF.md note on the trust boundary. Full SHA-512 binary
  pinning called out as follow-up.
- **grok — undefined `ggufFilenameFor`**: defined explicitly inside
  `engine-whispercpp.ts`. Filename is derived from the catalog entry's
  `downloadUrls.model` (basename of URL pathname), matching the
  on-disk filename `model-download.ts` writes via `fileNameFromUrl()`.
- **qwen — `PI_VOICE_FORCE_SHERPA` not implemented**: factory exposes
  `forceSherpa()` reading `process.env.PI_VOICE_FORCE_SHERPA === "1"`.
  Both `pickDefaultModelForPlatform` and `resolveEffectiveModel` skip
  whisper-cpp models when set. `isSherpaModel()` helper handles the
  `model.engine === undefined` default-to-sherpa case (catches existing
  catalog entries without the field).

## Changes since v4 (godspeed round-4 findings)

This v5 addresses every NO_SHIP from round 4:

- **sonnet — wrong default for fresh installs**: `DEFAULT_MODEL_PRIORITY`
  now lists `whisper-base-en-metal` (57 MB) before
  `whisper-large-v3-turbo-metal` (800 MB). Matches the plan body and
  Phase 2 rollout goal: fresh M-series installs get the small model,
  Turbo is opt-in via settings.
- **minimax — whitespace-only CUDA env values**: `CUDA_DISABLE_VALUES`
  documentation explicitly covers `"   "`, `"\t"`, `" -1 "`. Behavior
  is unchanged (whitespace-only trims to `""` → disabled) but the
  intent is now stated in code-comment form so a future maintainer
  doesn't "fix" it the wrong way. Tests cover all variants.
- **grok — engine adapter load-time safety**: documented an explicit
  contract — adapters MUST defer native binding loads to `init()`,
  start in `available: false, error: null`, expose idempotent +
  single-flight `init()`, and `transcribe()`/`warmup()` throw a
  structured error rather than a native stack trace when the engine
  isn't available. Sketch of the WhisperCppEngine class shape included.
- **qwen — unreachable return**: replaced the unreachable trailing
  `return ULTIMATE_FALLBACK_MODEL_ID` in `pickDefaultModelForPlatform`
  with an explicit `throw` carrying diagnostic env context. If the
  invariant is ever broken (catalog mutated mid-iteration, validator
  bypassed), failure is loud, not silent.

## Changes since v3 (godspeed round-3 findings)

This v4 addresses every NO_SHIP from round 3:

- **runtime + architect — circular import / TDZ**: `engine-factory.ts` no
  longer reads `LOCAL_MODELS` at module load. Catalog invariants are
  enforced (a) by `tests/catalog-invariants.test.ts` at CI time and
  (b) by a lazy `ensureCatalogValid()` on first call to any factory
  export at runtime. The mutual import (`local.ts ↔ engine-factory.ts`)
  is safe because no top-level read crosses the cycle.
- **security2 + sonnet — checklist contradiction**: removed the v2
  leftover line in the regression checklist that claimed
  `assertModelSupported throws on activation`. v3+ behavior
  (`migratePersistedModelOnSessionStart` graceful fallback, no persisted-
  config mutation) is now the only documented path.
- **minimax — case-insensitive CUDA env**: explicit test cases for
  `"NONE"`, `"None"`, `"VOID"`, whitespace-padded `" -1 "`, etc.,
  exercising the `.trim().toLowerCase()` path.
- **qwen — ASI / minified-build risk in IIFE block**: replaced the
  module-scoped `{ … }` block (which qwen flagged as ASI-risky) with
  a named function `ensureCatalogValid()`. Function declarations are
  ASI-safe by construction.

## Changes since v2 (godspeed round-2 findings)

This v3 addresses every NO_SHIP from round 2:

- **runtime + minimax — `hasCUDA` env var override skip**: env-var presence
  is now authoritative; `CUDA_VISIBLE_DEVICES=""` / `="-1"` return
  `false` instead of falling through to `/dev/nvidia0` probing.
  `CUDA_DISABLE_VALUES` set centralises the disabling tokens.
- **sonnet — engine path contract**: replaced `(pcm, model, lang, sampleRate)`
  arg lists with `EngineContext` / `TranscribeContext` objects that carry
  the resolved `modelDir`. Engines never re-resolve.
- **architect (advisory) — config portability**: removed the hard
  `assertModelSupported` guards from `transcribeInProcess` and
  `ensureModelDownloaded`. Replaced with `resolveEffectiveModel()` at
  `session_start` that falls back gracefully and surfaces a one-time
  notification, leaving persisted config untouched.
- **qwen — fallback safety**: module-load assertion in `engine-factory.ts`
  enforces that `moonshine-v2-tiny` exists in the catalog AND has no
  `supports` constraint. Bad catalog fails at import, not at first
  transcribe.
- **grok — short-circuit bypass concern (false positive but reviewer
  clarity matters)**: extracted `matchesSupport(s, env)` with three
  explicit independent checks — platform, arch, CUDA — so the guarantee
  is obvious by construction.
- **tribunal — fragile fallback chains**: replaced nested OR/find chains
  in `pickDefaultModelForPlatform` with an explicit
  `DEFAULT_MODEL_PRIORITY: readonly string[]` table iterated linearly;
  missing IDs are skipped (not treated as wildcards), supporting
  incremental rollout.

## Problem

pi-voice's local STT is 5–15× slower than native Mac dictation apps (Handy,
Muesli, MacWhisper) on Apple Silicon. Root cause is architectural, not
configuration:

- `sherpa-onnx-node` runs ONNX Runtime CPU-only by default for transducers, and
  even with the CoreML EP enabled (just landed for non-transducer types) ORT's
  generic kernel translation underperforms `whisper.cpp`'s hand-tuned Metal
  shaders by 5–15×.
- Decode is non-streaming: audio buffers until release, then runs in one shot.
- Every call goes through Node → N-API → C++; whisper.cpp apps go native →
  FFI → C++ in one fewer hop with zero V8/event-loop overhead.

## Goal

Add **whisper.cpp + Metal/CUDA** as a second engine alongside sherpa-onnx,
auto-selected at install-time based on platform. Apple Silicon users get
Handy-class speed; Linux/Intel/RPi users keep working unchanged.

## Non-goals

- Replacing sherpa-onnx. It stays as the cross-platform fallback and the only
  path for Parakeet/transducer models, which whisper.cpp doesn't support.
- Streaming/chunked decode. Separate follow-up — plan keeps batch semantics.
- TTS engine — sherpa-onnx-node continues to own TTS via `non-streaming-tts.js`.
  Only STT routing changes.
- macOS Speech framework / `SFSpeechRecognizer` — not callable from Node
  without a Swift bridge; out of scope.
- **Windows CUDA support** — explicitly deferred. We have no verified
  win32-x64-CUDA prebuilt for whisper.cpp; adding a Windows model entry would
  ship broken downloads. Re-evaluate when we adopt a binding with verified
  Windows prebuilts.

## Architecture

### Audio sample rate is fixed by upstream capture

The audio capture layer (`extensions/voice/deepgram.ts:10` —
`SAMPLE_RATE = 16000`) already pins capture to 16 kHz mono linear16 across
SoX, ffmpeg, and arecord. The same constant is consumed by sherpa
(`sherpa-engine.ts:114`: `acceptWaveform({ sampleRate: 16000, samples })`).

The Engine interface does **not** accept a sample-rate parameter — that
would imply runtime configurability the pipeline doesn't actually
support, and bake a contradictory contract into a public API (architect
v5 finding). Instead, sample-rate correctness is enforced **above** the
engine boundary, at the single place that owns the assumption:

```ts
// extensions/voice/local.ts (top of file, near other invariants)
import { SAMPLE_RATE } from "./deepgram";
// Engine adapters assume 16 kHz. If the audio capture constant ever
// changes (e.g. to support 48 kHz), this is the canary that fails the
// build / unit test before silent garbage transcription reaches users.
if (SAMPLE_RATE !== 16000) {
  throw new Error(
    `Voice engines require 16 kHz capture; got SAMPLE_RATE=${SAMPLE_RATE}. ` +
    `Update engine adapters before changing the capture rate.`,
  );
}
```

This single assertion replaces what was previously a per-engine
`sampleRate !== 16000` runtime check. Effects:
- Engine API gets simpler — `transcribe(pcm, ctx)` and `warmup(ctx)`,
  no `sampleRate` field that pretends to be configurable.
- The qwen-flagged failure mode (48 kHz pipeline → silently garbage
  transcripts) is still caught — at module load, with a descriptive
  error, before any user-visible behavior.
- Future 48 kHz support is a single coordinated change: bump the
  constant, update each engine adapter's internal `acceptWaveform({
  sampleRate, samples })` call to read `SAMPLE_RATE`, drop the
  assertion. No cross-cutting interface churn.

### Engine abstraction (`extensions/voice/engine.ts` — new)

The interface uses **context objects** instead of long argument lists so the
caller (always `local.ts:transcribeInProcess`) resolves model + path once and
hands a complete record to the engine. Engines never re-resolve the model
directory — that contract is explicit, not folklore.

**Engine adapter implementation contract** (load-time safety):

1. **No top-level throws**. The adapter module's top-level code MUST NOT
   `import` (synchronously / require) the native binding or do anything
   that can fail. All native-module loading happens inside `init()`,
   awaited, with errors captured into `engine.error`.
2. **Singleton starts in failed-but-readable state**: `available = false`,
   `error = null` (no error yet — just not ready). Reading the singleton
   never throws.
3. **`init()` is idempotent**. Multiple awaiters get the same outcome;
   subsequent calls after success are O(1) no-ops. Concurrent calls share
   the same in-flight promise (single-flight) so a fast second caller
   doesn't double-load the native module.
4. **`available === true` is the only signal callers should trust**.
   Callers that try `transcribe`/`warmup` while `available === false`
   get a thrown `Error` with `engine.error` as the message — never a
   confusing native-binding stack trace.

This contract is what makes `getEngineForModel(model)` safe to call
unconditionally: the singleton is always there, even if the native
binding is broken; the caller-facing failure is structured.

```ts
export type EngineId = "sherpa-onnx" | "whisper-cpp";

/** Resolved per-call context. Caller is responsible for ensuring model
 *  files exist on disk before invoking an engine — engines do not fetch.
 *  Sample rate is fixed at 16 kHz across all engines (asserted at module
 *  load in local.ts) — see "Audio sample rate is fixed by upstream
 *  capture" above. */
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
  /** True if init() has succeeded and the engine is ready for warmup
   *  / transcribe. Always-readable; never throws. */
  readonly available: boolean;
  /** Last error from init() / transcribe(); null if no error yet OR
   *  init() succeeded. */
  readonly error: string | null;
  /** Idempotent + single-flight. Returns true if the engine is now
   *  `available`. MUST NOT be called from module top-level. */
  init(): Promise<boolean>;
  warmup(ctx: EngineContext): Promise<void>;
  /** Throws if the engine is not `available` — callers should check
   *  `available` first or call `init()`. PCM is 16-bit signed LE mono
   *  at 16 kHz. */
  transcribe(pcm: Buffer, ctx: EngineContext): Promise<string>;
  clearCache(): void;
}
```

Sketch of a compliant adapter (engine-whispercpp.ts shape) — every
method on the `Engine` interface is implemented:

```ts
class WhisperCppEngine implements Engine {
  readonly id: EngineId = "whisper-cpp";
  private _available = false;
  private _error: string | null = null;
  /** In-flight init() promise. Cached on success; CLEARED on failure so
   *  a retry path exists (transient FS / locked file shouldn't brick
   *  the engine until process restart — security2 finding). */
  private _initPromise: Promise<boolean> | null = null;
  private _binding: typeof import("nodejs-whisper") | null = null;
  /** Loaded recognizers, keyed by model.id. `unknown` because the
   *  binding's recognizer type isn't surfaced by nodejs-whisper. */
  private readonly _recognizers = new Map<string, unknown>();
  /** In-flight warmup promises keyed by model.id — single-flight per
   *  model so concurrent callers (background warmup + transcribe)
   *  don't double-load an 800 MB model into native memory (runtime
   *  finding). */
  private readonly _warmupPromises = new Map<string, Promise<void>>();

  get available() { return this._available; }
  get error() { return this._error; }

  init(): Promise<boolean> {
    if (this._available) return Promise.resolve(true);
    if (this._initPromise) return this._initPromise;
    this._initPromise = this._doInit();
    // Clear the cached promise on EITHER success or failure once it
    // settles:
    //   - success: `_available` becomes the source of truth, so the
    //     resolved promise reference is just memory pinned for no
    //     reason (deepseek round-8 nit).
    //   - failure: the cleared slot lets the next caller retry instead
    //     of replaying the cached failure forever (security2 round-7
    //     "permanently bricked" finding).
    // `.finally()` is the right primitive — it fires on both resolve
    // AND reject, unlike `.then(ok => …)` which silently no-ops on
    // rejection (qwen round-8 finding).
    void this._initPromise.finally(() => { this._initPromise = null; });
    return this._initPromise;
  }

  private async _doInit(): Promise<boolean> {
    try {
      // Dynamic import — never at module top-level.
      const binding = await import("nodejs-whisper");
      // Interface-shape check (security defense-in-depth).
      for (const name of ["transcribe"] as const) {
        if (typeof (binding as any)?.[name] !== "function") {
          this._error = `nodejs-whisper binding missing required export "${name}"`;
          this._available = false;
          return false;
        }
      }
      this._binding = binding;
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
    // Check in-flight FIRST. The check + set below run in a single
    // synchronous block (no `await` between them in this method's
    // body), so a concurrent caller cannot interleave between get()
    // and set(). In-flight wins over recognizer-cache so concurrent
    // callers piggy-back on the running load.
    const inFlight = this._warmupPromises.get(ctx.model.id);
    if (inFlight) return inFlight;
    if (this._recognizers.has(ctx.model.id)) return;

    // Why `Promise.resolve().then(…)` instead of an `async` IIFE:
    //
    // If `ggufFilenameFor(ctx.model)` throws synchronously (corrupted
    // catalog entry that escaped CI), we still need `_warmupPromises`
    // to register the resulting rejected promise BEFORE its `.finally`
    // cleanup deletes the entry — otherwise a second concurrent
    // warmup() call could race past register-then-delete and either
    // see no entry (double-load) or see a stale rejection.
    //
    // `Promise.resolve()` produces an immediately-resolved promise.
    // The `.then(cb)` schedules `cb` as a MICROTASK; cb does not start
    // running synchronously. Therefore even a sync `throw` inside cb
    // is captured as a rejected promise that propagates only after
    // the surrounding synchronous code (the `_warmupPromises.set(…)`
    // below) has finished. The `.finally(cleanup)` chained on top
    // also runs in a later microtask, after the rejection settles.
    //
    // Order of execution, sync throw branch:
    //   1. `Promise.resolve()` → resolved promise (sync).
    //   2. `.then(cb)` returns a new pending promise (sync). cb queued.
    //   3. `.finally(cleanup)` returns final promise (sync). cleanup queued.
    //   4. `_warmupPromises.set(ctx.model.id, promise)` runs (sync).
    //   5. Microtasks drain: cb runs → throws → step-2 promise rejects;
    //      cleanup runs → `delete(ctx.model.id)`.
    //   6. Future callers see an empty map and retry.
    //
    // This makes the single-flight + cleanup contract obvious without
    // relying on `async`-function-prelude ordering subtleties.
    const promise = Promise.resolve().then(async () => {
      const filename = ggufFilenameFor(ctx.model); // path-traversal guarded
      const fullPath = path.join(ctx.modelDir, filename);
      const rec = await (this._binding as any).loadModel(fullPath, { lang: ctx.lang });
      this._recognizers.set(ctx.model.id, rec);
    }).finally(() => {
      this._warmupPromises.delete(ctx.model.id);
    });
    this._warmupPromises.set(ctx.model.id, promise);
    return promise;
  }

  async transcribe(pcm: Buffer, ctx: EngineContext): Promise<string> {
    this.requireAvailable();
    if (!this._recognizers.has(ctx.model.id)) {
      // warmup() is single-flight — concurrent transcribe + background
      // warmup share the same in-flight promise.
      await this.warmup(ctx);
    }
    const rec = this._recognizers.get(ctx.model.id);
    return (this._binding as any).transcribe(rec, pcm);
  }

  clearCache(): void {
    // Drop references — native finalizers reclaim on GC. Same pattern
    // as sherpa-engine.ts:clearRecognizerCache. Idempotent. Does NOT
    // cancel in-flight warmups; those resolve into a fresh _recognizers
    // map (which is fine — just a redundant load).
    this._recognizers.clear();
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
```

The `engine-sherpa.ts` adapter follows the same shape — singleton,
`available`/`error` flags, idempotent `init()`, `clearCache()` calls
through to `clearRecognizerCache()` from existing `sherpa-engine.ts`.

### Engine implementations

- **`engine-sherpa.ts`** — wrapper around the existing `sherpa-engine.ts` code.
  Hoists `initSherpa`, `getOrCreateRecognizer`, `transcribeBuffer` behind the
  `Engine` interface. Uses `ctx.modelDir` directly; never re-resolves.
- **`engine-whispercpp.ts`** — new. Calls into the validated whisper.cpp
  Node binding (`nodejs-whisper` is the leading candidate — MIT,
  prebuilt darwin-arm64 + linux-x64 binaries with Metal/CUDA compiled
  in; finalised in the spike). Loads GGUF file from
  `path.join(ctx.modelDir, ggufFilenameFor(ctx.model))` where
  `ggufFilenameFor` is a pure function on the catalog entry (defined
  below). After dynamic import, validates the binding's exported shape
  (defense-in-depth for supply-chain integrity — see Security note).

**Catalog shape for GGUF (whisper.cpp) models**

GGUF model entries do **not** reuse `sherpaModel` — that field is
sherpa-onnx-specific (encoder/decoder/joiner roles, ONNX file map). A
parallel optional field, `whisperCppModel`, mirrors its shape but for
the GGUF single-file case:

```ts
interface LocalModelInfo {
  // ... existing fields unchanged
  engine?: EngineId;          // undefined → "sherpa-onnx" (back-compat)
  supports?: ModelSupport[];  // undefined → all platforms
  format?: "onnx" | "gguf";   // undefined → "onnx"

  /** Existing field — used when format is "onnx". */
  sherpaModel?: {
    type: "whisper" | "moonshine" | "sense_voice" | "nemo_ctc" | "transducer";
    files: Record<string, string>;
    downloadUrls: Record<string, string>;
  };

  /** New field — used when format is "gguf". Single-file GGUF model. */
  whisperCppModel?: {
    files: { model: string };          // on-disk filename, e.g. "ggml-base.en-q5_1.bin"
    downloadUrls: { model: string };   // GGUF source URL
  };
}
```

A model entry uses **exactly one** of `sherpaModel` / `whisperCppModel`
(asserted in `tests/catalog-invariants.test.ts`).

**`ggufFilenameFor()` (defined in `engine-whispercpp.ts`)**

The on-disk filename is read directly from the catalog entry — no URL
parsing at runtime, no `new URL()` call that could throw. URL validity
is enforced once at CI time by `tests/catalog-invariants.test.ts`.

**Path-traversal guard.** A malicious / corrupted catalog entry could
declare `files.model = "../../etc/passwd"`, and a downstream
`path.join(ctx.modelDir, filename)` would escape the models directory
on read. Even though the catalog is in-tree (vendored), CI doesn't
catch every path-injection class of bug, and a future feature that
imports community catalog entries (Handy-style) would re-expose this.
We harden at the read site:

```ts
// extensions/voice/engine-whispercpp.ts
function ggufFilenameFor(model: LocalModelInfo): string {
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
  // Defense-in-depth: filename MUST be a plain basename, validated by
  // an allowlist of safe characters. Allowlists are stronger than
  // blocklists — they reject the entire class of escape sequences
  // (URL-encoded `%2F`/`%2E`/`%5C`, Unicode look-alikes, NULs, control
  // chars) that a blocklist must enumerate. CI runs the same predicate
  // over every GGUF catalog entry, so a bad filename fails before it
  // ships.
  //
  // GGUF model files in our catalog match the pattern
  // `ggml-<name>-q5_1.bin`, `ggml-large-v3-turbo-q5_0.bin`, etc. The
  // allowed character set [A-Za-z0-9._-] covers every legitimate name
  // we ship today and is intentionally narrow.
  //
  // Filename convention rationale (grok round-10 concern about spaces):
  // every GGUF model on the upstream `huggingface.co/ggerganov/
  // whisper.cpp` repo uses hyphenated lowercase names — there are NO
  // spaces in any whisper.cpp ggml-*.bin filename. The catalog-
  // invariants CI test asserts `path.basename(new URL(url).pathname)
  // === files.model` for every entry, so a future URL-with-spaces
  // (URL-encoded as %20) would fail CI long before this allowlist
  // rejects it at runtime. If we ever need to support a binding that
  // ships filenames with spaces, expanding the allowlist is a
  // single-line change accompanied by a test update — the security
  // surface is unaffected because separators stay banned.
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

/** Allowlist for GGUF filenames — alphanumerics, dot, hyphen, underscore.
 *  Any character outside this set (path separators, NUL, control chars,
 *  URL-encoded escapes that contain `%`, Unicode homoglyphs, …) is
 *  rejected. Keeping this narrow is the security contract. */
const SAFE_GGUF_FILENAME_RE = /^[A-Za-z0-9._-]+$/;

/** Windows reserved device names — defense-in-depth on platforms we
 *  don't currently target for whisper.cpp but might in the future.
 *  Match `COM[1-9]\d*` / `LPT[1-9]\d*`: the first digit is non-zero,
 *  with optional trailing digits.
 *
 *  Names this REGEX MATCHES (i.e. ggufFilenameFor REJECTS):
 *    CON, PRN, AUX, NUL, COM1, COM9, COM10, COM256, LPT1, LPT99,
 *    COM1.bin, lpt1.gguf  (case-insensitive)
 *  Names this REGEX DOES NOT MATCH (i.e. ggufFilenameFor ACCEPTS):
 *    COM0, COM00, LPT0, LPT00, COM0a, ...  (NOT Windows reserved
 *    per Microsoft's spec; an over-broad `\d+` here would cause
 *    false-positive rejections — moonshot + grok round 11.)
 *
 *  Coverage rationale:
 *    - Microsoft's documented COM1–COM9 / LPT1–LPT9 reserved range
 *      (security2 round 8).
 *    - Multi-port-serial-card extensions COM10–COM256 etc. that the
 *      NT device-name parser also reserves (minimax round 10).
 *
 *  Optional `.<ext>` suffix bounded to non-separator chars; `[^/\\]`
 *  is belt-and-braces even though the upstream allowlist already
 *  rejects `/` and `\` (security round-10 nit). */
const WINDOWS_RESERVED_NAME_RE =
  /^(CON|PRN|AUX|NUL|COM[1-9]\d*|LPT[1-9]\d*)(\.[^/\\]*)?$/i;
```

The same guard is reused (as `assertSafeBasename(filename, modelId)`)
inside `model-download.ts` when computing the on-disk write path for
GGUF downloads — defense-in-depth on both the read and write paths.

**Supply-chain integrity (security finding)**

Native binding load happens via `await import("nodejs-whisper")` in
`init()`. Defense-in-depth:

1. **Lockfile-pinned exact version**: `optionalDependencies` pins to a
   specific version (no caret/range). bun.lock / package-lock.json
   commits the resolved tarball SHA-512. Any post-merge tarball change
   fails `bun install` integrity check.
2. **Interface shape check after import**: immediately after the dynamic
   import, the engine asserts the binding exports the expected callable
   interface. A tampered or partially-corrupted module fails fast, with
   `engine.error` set to a structured message — callers fall back via
   `migratePersistedModelOnSessionStart` instead of unknowingly running
   a hostile wrapper.
   ```ts
   // engine-whispercpp.ts init() — after `await import(...)`
   const required = ["transcribe"] as const;
   for (const name of required) {
     if (typeof (this._binding as any)?.[name] !== "function") {
       this._error = `nodejs-whisper binding is missing required export "${name}"`;
       this._available = false;
       return false;
     }
   }
   ```
3. **No dynamic / runtime-resolved binding paths**: `import("nodejs-whisper")`
   is a literal string, not a variable — the bundler / loader resolves
   to the locked package, not user-influenced input.
4. **HANDOFF.md note**: documents that the supply-chain trust boundary
   is the lockfile, not the running process; reviewers must vet
   binding upgrades like any other dependency.

This isn't full hash verification (which would require shipping our own
SHA manifest and validating the on-disk binary against it — a real
project but out of scope for this PR). The interface-shape check is
the cheap defense against accidental corruption + the most common
tampering pattern (export-replacement); proper binary pinning is
follow-up work tracked alongside the binding-vetting checklist.

### Engine factory (`engine-factory.ts` — new)

**No top-level catalog reads.** `engine-factory.ts` and `local.ts` form a
mutual import (`local.ts:transcribeInProcess` calls
`pickDefaultModelForPlatform`; `engine-factory.ts` reads `LOCAL_MODELS`).
A top-level assertion that touches `LOCAL_MODELS` at module-eval time hits
the TDZ window and crashes with `ReferenceError: Cannot access 'LOCAL_MODELS'
before initialization` whichever side the loader picks up first.

The fix is a **lazy, idempotent validator** that runs on first call to any
exported function — by that point all modules have finished evaluating and
`LOCAL_MODELS` is fully populated. The CI test
(`tests/catalog-invariants.test.ts`, below) is the up-front gate that
catches a bad catalog before ship; the lazy validator is the runtime
backstop for hand-edited installs.

```ts
import { LOCAL_MODELS, type LocalModelInfo } from "./local";
import { currentVoiceRuntimeEnv, isModelSupported, type VoiceRuntimeEnv } from "./model-compat";

/** The single ID guaranteed to work on every supported runtime. Picked
 *  for: small (~43 MB), no `supports` constraint, fast on CPU, English-only
 *  but acceptable as a compatibility fallback. Exported so tests + the
 *  catalog-invariants test can reference the same constant. */
export const ULTIMATE_FALLBACK_MODEL_ID = "moonshine-v2-tiny";

/** Default models in priority order. The order matches the plan's stated
 *  Phase 2 goal: fresh Apple Silicon installs default to Whisper Base EN
 *  Metal (57 MB, fast) — Turbo (800 MB) is opt-in via settings. Linux
 *  CUDA + sherpa-onnx fallback follow.
 *
 *  Entries may reference IDs not yet present in the catalog (during
 *  incremental rollout); missing IDs are skipped, NOT treated as
 *  wildcards. ULTIMATE_FALLBACK_MODEL_ID terminates the chain and is
 *  guaranteed to exist + be platform-agnostic by ensureCatalogValid(). */
const DEFAULT_MODEL_PRIORITY: readonly string[] = [
  "whisper-base-en-metal",         // 57 MB — fast first-install on M-series
  "whisper-large-v3-turbo-metal",  // 800 MB — opt-in upgrade for accuracy
  "whisper-large-v3-turbo-cuda",   // Linux GPU
  ULTIMATE_FALLBACK_MODEL_ID,      // moonshine-v2-tiny — universal CPU fallback
];

let _catalogValidated = false;

/** Lazy invariant check — runs once on first export call. Reading
 *  LOCAL_MODELS only inside this function avoids the circular-import
 *  TDZ trap that a top-level read would create. The three invariants:
 *  1. Fallback model exists in LOCAL_MODELS.
 *  2. Fallback has no `supports` constraint (works on every runtime).
 *  3. Fallback is a sherpa-onnx model (engine undefined OR "sherpa-onnx")
 *     so that PI_VOICE_FORCE_SHERPA always has a path that satisfies it.
 */
function ensureCatalogValid(): void {
  if (_catalogValidated) return;
  const m = LOCAL_MODELS.find(x => x.id === ULTIMATE_FALLBACK_MODEL_ID);
  if (!m) {
    throw new Error(
      `Catalog is missing the ultimate fallback model ` +
      `"${ULTIMATE_FALLBACK_MODEL_ID}" — check extensions/voice/local.ts.`,
    );
  }
  if (m.supports && m.supports.length > 0) {
    throw new Error(
      `Ultimate fallback "${ULTIMATE_FALLBACK_MODEL_ID}" must be ` +
      `platform-agnostic (no \`supports\` constraint).`,
    );
  }
  if (m.engine !== undefined && m.engine !== "sherpa-onnx") {
    throw new Error(
      `Ultimate fallback "${ULTIMATE_FALLBACK_MODEL_ID}" must be a ` +
      `sherpa-onnx model so PI_VOICE_FORCE_SHERPA=1 always resolves; ` +
      `got engine="${m.engine}".`,
    );
  }
  _catalogValidated = true;
}

/** Thrown by resolveEffectiveModel when the catalog is so corrupted
 *  that no fallback exists. voice.ts catches this at session_start
 *  and disables voice for the session with a clear notification —
 *  better than silently routing to a non-existent model and getting
 *  "Unknown model: …" mid-transcribe. */
export class VoiceCatalogUnrecoverableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "VoiceCatalogUnrecoverableError";
  }
}

/** Find an id that is likely (but not guaranteed) to exist in
 *  LOCAL_MODELS, walking three tiers of preference:
 *    1. ULTIMATE_FALLBACK_MODEL_ID if it's in the catalog.
 *    2. Any platform-agnostic sherpa-onnx model.
 *    3. Any model at all in the catalog (last-resort, may be
 *       platform-locked but at least exists, so transcribe gets a
 *       descriptive engine error instead of "Unknown model").
 *  Returns `null` only if LOCAL_MODELS is empty — at which point the
 *  install is genuinely unrecoverable and the caller should disable
 *  voice for the session. (grok + minimax round 14 — no missing-id
 *  return values.)
 */
function findResilientFallbackId(): string | null {
  const ultimate = LOCAL_MODELS.find(m => m.id === ULTIMATE_FALLBACK_MODEL_ID);
  if (ultimate) return ULTIMATE_FALLBACK_MODEL_ID;
  const anyAgnosticSherpa = LOCAL_MODELS.find(m =>
    isSherpaModel(m) && (!m.supports || m.supports.length === 0),
  );
  if (anyAgnosticSherpa) return anyAgnosticSherpa.id;
  if (LOCAL_MODELS.length > 0 && LOCAL_MODELS[0]) return LOCAL_MODELS[0].id;
  return null;
}

/** Build the degraded ResolveEffectiveModel result given catalog
 *  corruption. Throws VoiceCatalogUnrecoverableError if the catalog
 *  is empty (unrecoverable). One implementation, two call sites
 *  (sonnet round 14 — no logic duplication). */
function degradedResolve(
  persistedId: string | undefined,
  rootCause: unknown,
): { id: string; fellBack: boolean; from?: string; degraded: true } {
  const id = findResilientFallbackId();
  if (id === null) {
    throw new VoiceCatalogUnrecoverableError(
      `Catalog invariants failed and LOCAL_MODELS is empty — ` +
      `${String(rootCause)}. Reinstall pi-voice or check for a ` +
      `corrupted extension build.`,
    );
  }
  // eslint-disable-next-line no-console
  console.error(
    `[pi-voice] resolveEffectiveModel: catalog invariants failed — ` +
    `${String(rootCause)}. Falling back to "${id}".`,
  );
  return {
    id,
    fellBack: persistedId !== undefined && persistedId !== id,
    from: persistedId,
    degraded: true,
  };
}

export function getEngineForModel(model: LocalModelInfo): Engine {
  // model.engine is optional — undefined defaults to sherpa-onnx.
  return model.engine === "whisper-cpp" ? whispercppEngine : sherpaEngine;
}

/** True when the user has set PI_VOICE_FORCE_SHERPA=1, asking the
 *  picker / resolver to skip whisper.cpp models even when supported.
 *  Consulted as `forceSherpa()` so tests can stub `process.env`. */
function forceSherpa(env: NodeJS.ProcessEnv = process.env): boolean {
  return env.PI_VOICE_FORCE_SHERPA === "1";
}

/** Treat a model as a sherpa-onnx model when its `engine` is undefined
 *  OR explicitly "sherpa-onnx". This matches getEngineForModel's
 *  default. */
function isSherpaModel(model: LocalModelInfo): boolean {
  return model.engine === undefined || model.engine === "sherpa-onnx";
}

export function pickDefaultModelForPlatform(
  env: VoiceRuntimeEnv = currentVoiceRuntimeEnv(),
  processEnv: NodeJS.ProcessEnv = process.env,
): string {
  ensureCatalogValid();
  const skipWhisperCpp = forceSherpa(processEnv);
  for (const id of DEFAULT_MODEL_PRIORITY) {
    const model = LOCAL_MODELS.find(m => m.id === id);
    if (!model) continue;                       // skip un-shipped IDs
    if (skipWhisperCpp && !isSherpaModel(model)) continue;
    if (isModelSupported(model, env)) return id;
  }
  // ensureCatalogValid() guarantees the ultimate fallback is in
  // LOCAL_MODELS with no `supports` constraint AND it's a sherpa model
  // (it has no `engine` field), so the loop above MUST have returned
  // for any combination of env + processEnv. Reaching this line means
  // the catalog was mutated mid-iteration or the validator was bypassed
  // — fail loudly with diagnostic context, never silently substitute.
  throw new Error(
    `pickDefaultModelForPlatform: no supported model found in catalog ` +
    `for ${env.platform}-${env.arch} (hasCUDA=${env.hasCUDA}, ` +
    `forceSherpa=${skipWhisperCpp}). The ultimate fallback ` +
    `"${ULTIMATE_FALLBACK_MODEL_ID}" should always match — this is a ` +
    `catalog invariant violation.`,
  );
}

/** Resolve the effective model for a session, given (possibly stale)
 *  persisted config. Falls back to the platform default when:
 *    - persisted id isn't in the catalog, OR
 *    - persisted id isn't supported on the current runtime, OR
 *    - PI_VOICE_FORCE_SHERPA=1 and the persisted model is whisper-cpp.
 *  Returns { fellBack: true, from } so the caller can surface a notice
 *  via `migratePersistedModelOnSessionStart` in voice.ts.
 *
 *  Robustness: `pickDefaultModelForPlatform` can throw (catalog
 *  invariant violation). We catch + degrade so a corrupted catalog
 *  doesn't crash session_start — the user gets the hardcoded ultimate
 *  fallback id with a structured warning. The real fix is the CI test
 *  catching catalog corruption before ship; this is the runtime
 *  belt-and-braces. */
export function resolveEffectiveModel(
  persistedId: string | undefined,
  env: VoiceRuntimeEnv = currentVoiceRuntimeEnv(),
  processEnv: NodeJS.ProcessEnv = process.env,
): { id: string; fellBack: boolean; from?: string; degraded?: boolean } {
  // ensureCatalogValid() can throw on a corrupted catalog. Wrap it so
  // session_start still gets a working session — even if the result is
  // a wrong-but-platform-agnostic fallback id. This is the runtime-
  // round-9 finding: an exception from the validator must NOT bubble up
  // and crash startup. CI's catalog-invariants test is the up-front gate
  // that should make this branch unreachable in practice.
  try {
    ensureCatalogValid();
  } catch (err) {
    return degradedResolve(persistedId, err);
  }

  const skipWhisperCpp = forceSherpa(processEnv);

  function safeDefault(): { id: string; degraded: boolean } {
    try {
      return { id: pickDefaultModelForPlatform(env, processEnv), degraded: false };
    } catch (err) {
      // pickDefault should never throw if ensureCatalogValid above
      // succeeded — defense-in-depth against any future invariant we
      // forget to enforce. Re-use the same degraded resolver so the
      // fallback search lives in ONE place (sonnet round-14 nit).
      const id = findResilientFallbackId();
      if (id === null) {
        throw new VoiceCatalogUnrecoverableError(
          `pickDefaultModelForPlatform threw and no resilient fallback ` +
          `found in LOCAL_MODELS — ${String(err)}`,
        );
      }
      // eslint-disable-next-line no-console
      console.error(
        `[pi-voice] resolveEffectiveModel: pickDefault threw — ${String(err)}. ` +
        `Falling back to "${id}".`,
      );
      return { id, degraded: true };
    }
  }

  if (persistedId) {
    const m = LOCAL_MODELS.find(x => x.id === persistedId);
    const blockedByForceSherpa = m && skipWhisperCpp && !isSherpaModel(m);
    if (m && isModelSupported(m, env) && !blockedByForceSherpa) {
      return { id: persistedId, fellBack: false };
    }
    const def = safeDefault();
    return { id: def.id, fellBack: true, from: persistedId, degraded: def.degraded };
  }
  const def = safeDefault();
  return { id: def.id, fellBack: false, degraded: def.degraded };
}
```

Why lazy and not top-level: ESM module evaluation order is loader-dependent.
With `local.ts → engine-factory.ts → local.ts` mutual imports, evaluating
either module first puts the cycle through TDZ. A top-level read of
`LOCAL_MODELS` from inside `engine-factory.ts` may see an uninitialized
binding. Deferring the read to first call breaks the trap — by then both
modules have finished evaluation.

CI catches structural problems early; the runtime guard catches tamper /
hand-edits in the field.

`resolveEffectiveModel` is the **only place** that handles platform-config
mismatch. All downstream callers (`transcribeInProcess`,
`warmupLocalRecognizerInBackground`, settings panel display) use the
already-resolved id and never need to re-check support.

`hasCUDA()` (in `device.ts`) is replaced with a platform-aware version that
no longer assumes Linux. The env-var path is **authoritative when set** —
its value (including disabling values) determines the result; only an
unset env var falls through to native probing. This eliminates the silent
override-skip bug where `CUDA_VISIBLE_DEVICES=""` or `="-1"` was ignored
and execution still hit the Linux probe.

```ts
// extensions/voice/device.ts

/**
 * Tokens that disable CUDA when set as the env-var value.
 * Match is case-insensitive after `.trim()`, so:
 *   - ""              → disabled (explicit empty)
 *   - "   "           → disabled (trims to "" — same intent as empty)
 *   - "\t" / "\n"     → disabled (whitespace-only)
 *   - "-1"  / " -1 "  → disabled (CUDA convention)
 *   - "NONE" / "None" → disabled (case-insensitive)
 *   - "VOID"          → disabled (NVIDIA convention)
 *
 * Anything else (e.g. "0", "all", "0,1,2") enables CUDA. To re-enable
 * CUDA after disabling via env, unset the variable entirely or set it
 * to a device id like "0".
 */
const CUDA_DISABLE_VALUES = new Set(["", "-1", "none", "void"]);

export function hasCUDA(
  env: NodeJS.ProcessEnv = process.env,
  platform: NodeJS.Platform = process.platform,
): boolean {
  // Step 1: honor explicit env override on every platform.
  // If the env var is SET (even to whitespace), its value is
  // authoritative — we never fall through to native probing. This
  // prevents the override-skip bug where an empty / "-1" override was
  // silently ignored. Whitespace-only values trim to "" and disable —
  // matches user intent that "I set this to nothing meaningful" === off.
  const visible = env.CUDA_VISIBLE_DEVICES ?? env.NVIDIA_VISIBLE_DEVICES;
  if (visible !== undefined) {
    return !CUDA_DISABLE_VALUES.has(visible.trim().toLowerCase());
  }
  // Step 2: env var unset — probe native paths on platforms we ship for.
  if (platform === "linux") {
    return fs.existsSync("/dev/nvidia0") ||
           fs.existsSync("/proc/driver/nvidia/version");
  }
  // Windows is intentionally NOT probed — see Non-goals (no verified
  // win32-x64-CUDA prebuilt). Add detection only when prebuilts ship.
  return false;
}
```

### Model compatibility contract (`extensions/voice/model-compat.ts` — new)

The original plan stored `platforms: NodeJS.Platform[]` per model — but
`NodeJS.Platform` only encodes "darwin"/"linux"/"win32" and can't represent
arch (`darwin-arm64` vs `darwin-x64`) or GPU requirement. Replace with a
structured contract that all consumers (settings filter, default picker,
download guard, runtime activation) call through:

```ts
// extensions/voice/model-compat.ts
import { hasCUDA } from "./device";
import type { LocalModelInfo } from "./local";

export interface ModelSupport {
  platform: NodeJS.Platform;
  arch?: NodeJS.Architecture;
  requiresCUDA?: boolean;
}

export interface VoiceRuntimeEnv {
  platform: NodeJS.Platform;
  arch: NodeJS.Architecture;
  hasCUDA: boolean;
}

export function currentVoiceRuntimeEnv(): VoiceRuntimeEnv {
  return { platform: process.platform, arch: process.arch, hasCUDA: hasCUDA() };
}

/** A single ModelSupport row matches the runtime if every constraint
 *  it specifies is satisfied. The three checks are independent — arch
 *  and CUDA validation are separate gates and neither is bypassed by
 *  the other. */
function matchesSupport(s: ModelSupport, env: VoiceRuntimeEnv): boolean {
  if (s.platform !== env.platform) return false;
  if (s.arch !== undefined && s.arch !== env.arch) return false;
  if (s.requiresCUDA === true && !env.hasCUDA) return false;
  return true;
}

export function isModelSupported(
  model: LocalModelInfo,
  env: VoiceRuntimeEnv = currentVoiceRuntimeEnv(),
): boolean {
  // No `supports` constraint = available everywhere (sherpa-onnx default).
  if (!model.supports || model.supports.length === 0) return true;
  return model.supports.some(s => matchesSupport(s, env));
}

export function describeUnsupported(
  model: LocalModelInfo,
  env: VoiceRuntimeEnv = currentVoiceRuntimeEnv(),
): string {
  const arch = `${env.platform}-${env.arch}`;
  const needsCUDA = model.supports?.some(s => s.requiresCUDA) ?? false;
  return `Model ${model.id} is not supported on ${arch}` +
         (needsCUDA && !env.hasCUDA ? " (CUDA unavailable)" : "");
}
```

Note: `assertModelSupported` is **deliberately not exported**. The plan's
v2 used it as a hard guard inside `transcribeInProcess` and
`ensureModelDownloaded`, which made persisted config non-portable across
machines (architect's advisory finding). v3 instead resolves the effective
model once at `session_start`, falling back to the platform default with
a one-time notification if the persisted `localModel` is unsupported on
the current runtime — see `resolveEffectiveModel` below.

### Model catalog changes (`local.ts`)

`LocalModelInfo` gains `engine`, `supports`, `format` — all optional, all
back-compat:

```ts
interface LocalModelInfo {
  id: string;
  engine?: EngineId;          // undefined → "sherpa-onnx" (back-compat)
  supports?: ModelSupport[];  // undefined → all platforms (sherpa default)
  format?: "onnx" | "gguf";   // undefined → "onnx"
  // ... existing fields unchanged
}
```

New whisper.cpp catalog entries (illustrative — final filenames /
URLs locked by the spike before merge):

```ts
{
  id: "whisper-base-en-metal",
  name: "Whisper Base EN (Metal)",
  engine: "whisper-cpp",
  supports: [{ platform: "darwin", arch: "arm64" }],
  format: "gguf",
  size: "~57 MB",
  whisperCppModel: {
    files: { model: "ggml-base.en-q5_1.bin" },
    downloadUrls: {
      model: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en-q5_1.bin",
    },
  },
}
{
  id: "whisper-large-v3-turbo-metal",
  name: "Whisper Turbo (Metal)",
  engine: "whisper-cpp",
  supports: [{ platform: "darwin", arch: "arm64" }],
  format: "gguf",
  size: "~800 MB",
  whisperCppModel: {
    files: { model: "ggml-large-v3-turbo-q5_0.bin" },
    downloadUrls: {
      model: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin",
    },
  },
}
{
  id: "whisper-tiny-en-metal",
  name: "Whisper Tiny EN (Metal)",
  engine: "whisper-cpp",
  supports: [{ platform: "darwin", arch: "arm64" }],
  format: "gguf",
  size: "~31 MB",
  whisperCppModel: {
    files: { model: "ggml-tiny.en-q5_1.bin" },
    downloadUrls: {
      model: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en-q5_1.bin",
    },
  },
}
{
  id: "whisper-large-v3-turbo-cuda",
  name: "Whisper Turbo (CUDA)",
  engine: "whisper-cpp",
  supports: [{ platform: "linux", arch: "x64", requiresCUDA: true }],
  format: "gguf",
  size: "~800 MB",
  whisperCppModel: {
    files: { model: "ggml-large-v3-turbo-q5_0.bin" },
    downloadUrls: {
      model: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin",
    },
  },
}
```

GGUF source: `https://huggingface.co/ggerganov/whisper.cpp` (Apache 2.0).

### Settings panel changes (`settings-panel.ts`)

The settings UI only shows models that are supported on the current
runtime — a Linux user never sees Metal-only models, so they can't
select one accidentally:

```ts
// Before (v1)
return LOCAL_MODELS.filter(model =>
  model.platforms?.includes(currentPlatform) ?? true);

// After (v3)
const env = currentVoiceRuntimeEnv();
return LOCAL_MODELS.filter(model => isModelSupported(model, env));
```

`scanHandyModels()` gets the same filter. The activation path stores the
new `localModel` id without any compatibility guard — by construction
the user can only pick visible (= supported) models. Hand-edited config
files are caught at session-start by `resolveEffectiveModel`, not here.

### Session-start config migration (new helper in `voice.ts`)

```ts
function migratePersistedModelOnSessionStart() {
  let resolved: ReturnType<typeof resolveEffectiveModel>;
  try {
    resolved = resolveEffectiveModel(config.localModel);
  } catch (err) {
    if (err instanceof VoiceCatalogUnrecoverableError) {
      // Catalog is empty / completely broken. Voice can't function at
      // all this session — disable it with a clear notification rather
      // than letting "Unknown model: …" surface mid-transcribe.
      // (grok + minimax round 14.)
      voiceDebug("session_start: voice catalog unrecoverable", { err: err.message });
      if (ctx?.hasUI) {
        ctx.ui.notify(
          `Voice catalog is corrupted (${err.message}). Voice features ` +
          `disabled for this session. Reinstall pi-voice or check the ` +
          `extension files.`,
          "error",
        );
      }
      // Disable voice for THIS session only. Persisted config untouched
      // so a subsequent session with a fixed install just works.
      config = { ...config, enabled: false };
      return;
    }
    throw err; // unexpected — let the upper handler decide
  }
  if (!resolved.fellBack && !resolved.degraded) return;
  // Persisted id was unsupported here, OR catalog was degraded — use
  // the platform default for THIS session, surface a one-time
  // notification, and leave persisted config untouched so a subsequent
  // session on the original machine still works.
  voiceDebug("model fallback", {
    from: resolved.from, to: resolved.id, degraded: !!resolved.degraded,
  });
  if (ctx?.hasUI) {
    const reason = resolved.degraded
      ? `Voice catalog degraded — using "${resolved.id}" for this session.`
      : `Voice model "${resolved.from}" isn't supported on this machine — ` +
        `using "${resolved.id}" for this session.`;
    ctx.ui.notify(
      `${reason} Run /voice-settings to pick a different default.`,
      resolved.degraded ? "warning" : "info",
    );
  }
  // session-scoped only — does NOT mutate persisted config
  config = { ...config, localModel: resolved.id };
}
```

Called from both `session_start` paths immediately after `config` is
loaded. Persisted config is **never overwritten** on fallback — that's
the architect's portability concern: a config that travels between
machines should keep working in both directions.

### Local session routing (`local.ts:789` `transcribeInProcess`)

By the time we hit `transcribeInProcess`, `config.localModel` is already
the resolved-effective id (`migratePersistedModelOnSessionStart` ran at
session start). No re-resolution needed.

```ts
async function transcribeInProcess(pcmData: Buffer, config: VoiceConfig) {
  const modelId = config.localModel || pickDefaultModelForPlatform();
  const model = LOCAL_MODELS.find(m => m.id === modelId);
  if (!model) throw new Error(`Unknown model: ${modelId}`);  // resolved id, not raw config

  const modelDir = await ensureModelDownloaded(model);
  const engine = getEngineForModel(model);

  if (!engine.available) {
    const ok = await engine.init();
    if (!ok) throw new Error(`Engine ${engine.id} unavailable: ${engine.error}`);
  }
  const ctx: EngineContext = { model, modelDir, lang: config.language || "en" };
  return engine.transcribe(pcmData, ctx);
}
```

The 16 kHz invariant is enforced once at module load in `local.ts`
(see "Audio sample rate is fixed by upstream capture"), not threaded
through every transcribe call.

### Download routing (`model-download.ts`)

`ensureModelDownloaded` accepts the full `LocalModelInfo` so it can
pick filenames by `model.format` (GGUF: single `.bin`; ONNX: multi-file
map). No `assertModelSupported` call here — the activation path filters
unsupported models out of the UI, and `migratePersistedModelOnSessionStart`
handles stale config. Adding a guard would just gate downloads behind
platform checks on every transcribe, which is wasteful and re-introduces
the portability problem.

### Warmup (`voice.ts`)

After session-start resolution, `config.localModel` is already the
effective id. Warmup just needs to confirm it's downloaded and hand a
context to the engine:

```ts
function warmupLocalRecognizerInBackground() {
  if (config.backend !== "local") return;
  if (config.localEndpoint) return; // External server — no in-process load
  (async () => {
    const modelId = config.localModel || pickDefaultModelForPlatform();
    const model = LOCAL_MODELS.find(m => m.id === modelId);
    if (!model) return;
    if (!isModelDownloaded(modelId, model)) return;  // warmup never downloads

    const engine = getEngineForModel(model);
    if (!engine.available) {
      const ok = await engine.init();
      if (!ok) return;
    }
    const modelDir = getModelDir(modelId);
    await engine.warmup({ model, modelDir, lang: config.language || "en" });
    voiceDebug("warmup: engine recognizer preloaded", { modelId, engine: engine.id });
  })().catch(err => voiceDebug("warmup: failed (non-fatal)", { error: String(err) }));
}
```

## Files changed

**New files (5)**
- `extensions/voice/engine.ts` — interface + `EngineContext` /
  `TranscribeContext` types (~90 lines)
- `extensions/voice/engine-sherpa.ts` — sherpa adapter (~150 lines)
- `extensions/voice/engine-whispercpp.ts` — whisper.cpp adapter (~200 lines)
- `extensions/voice/engine-factory.ts` — picker + `resolveEffectiveModel`
  + module-load catalog assertions (~80 lines)
- `extensions/voice/model-compat.ts` — `ModelSupport`/`matchesSupport`/
  `isModelSupported`/`describeUnsupported` (~70 lines).
  Note: `assertModelSupported` is **deliberately absent** — graceful
  fallback at session-start replaces hard-fail.

**Modified files (6)**
- `extensions/voice/local.ts` — `LocalModelInfo` gets optional
  `engine`/`supports`/`format` fields (all back-compat defaulted);
  `transcribeInProcess` uses resolved id and `EngineContext`; new
  whisper.cpp model entries appended to `LOCAL_MODELS`.
- `extensions/voice/model-download.ts` — `ensureModelDownloaded` accepts
  full `LocalModelInfo`; switches on `model.format` for filename
  resolution (GGUF single-file vs. ONNX multi-file).
- `extensions/voice/settings-panel.ts` — UI list filtered by
  `isModelSupported`; "Recommended" sort on supported platforms; engine
  routing on activate (no compat guard — UI-filtered visibility is the
  only gate).
- `extensions/voice/device.ts` — `hasCUDA(env, platform)` is now
  authoritative on env-var presence (returns based on value when SET);
  native probe only when env unset; Linux-only native paths.
- `extensions/voice.ts` — adds `migratePersistedModelOnSessionStart()`
  helper (called from both `session_start` paths after config load);
  `warmupLocalRecognizerInBackground` uses `EngineContext`;
  `pickDefaultModelForPlatform()` used for new installs.
- `extensions/voice/sherpa-engine.ts` — no API change. Consumed by
  `engine-sherpa.ts`.

**Package changes**
- Add the validated whisper.cpp Node binding to `optionalDependencies` so
  bun/npm only fetches it on supported platforms. Spike picks the binding
  (`nodejs-whisper` is the leading candidate, MIT, prebuilt
  darwin-arm64 + linux-x64). License audit before merge.

## Migration / back-compat

1. **Existing users with downloaded ONNX models**: zero impact. `engine`
   defaults to `"sherpa-onnx"`, `supports` defaults to `undefined` (=
   "supported everywhere"). Their workflow doesn't change.
2. **Existing users on parakeet-v3**: still on parakeet-v3 (transducer →
   sherpa-onnx → CPU). One-time, dismissible notification on first
   `session_start` after upgrade: *"Whisper Turbo (Metal) is available — run
   /voice-settings for 5–15× faster transcription on Apple Silicon."*
3. **Fresh installs**: `pickDefaultModelForPlatform()` picks the platform-
   appropriate default; onboarding shows filtered model list.
4. **Onboarding**: model picker shows the platform-filtered list with
   recommended model pre-selected.

## Risks & mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| nodejs-whisper prebuilts missing for linux-arm64 (Raspberry Pi) | Med | `supports: [{ platform: "linux", arch: "x64" }]` excludes Pi from whisper.cpp models; sherpa-onnx remains the only path on Pi. |
| Selected binding requires `node-gyp` build at install | Low | Spike validates prebuilt fetch on darwin-arm64 + linux-x64 before merge. If it fails, switch to alternative or block the PR. |
| whisper.cpp Metal regresses on a future macOS update | Low | `PI_VOICE_FORCE_SHERPA=1` env falls a whisper-cpp model selection back to its sherpa equivalent. |
| 800 MB Turbo download deters first-time use | Med | Default to **Whisper Base EN Metal** (57 MB); Turbo is opt-in via settings. |
| GGUF and ONNX clash in `models/` | Low | Same dir, different filenames; `isModelDownloaded` switches on `model.format`. |
| ~25 MB extra binary on Apple Silicon | Low | Acceptable trade for 5–15× speedup. `optionalDependencies` keeps Linux/RPi unaffected. |
| Two engines diverge on punctuation / numbers / language quirks | Med | Document per-engine quirks in HANDOFF.md. Test matrix covers common cases. |
| nodejs-whisper unmaintained / single-maintainer | Med | Spike validates current viability; sherpa path keeps working regardless; escape hatches: fork, alternative binding, custom napi-rs wrapper. |
| whisper.cpp prebuilt vendors libonnxruntime/libgomp conflicting with sherpa-onnx | Low | Both engines link statically; verify with `otool -L` during spike. If conflict, scope a load-order fix. |
| `supports` array misconfigured → user with valid hardware sees an empty model list | Low | Default model picker always falls back to `moonshine-v2-tiny` (no `supports` constraint). Catalog tests assert at least one model is supported on each {darwin-arm64, darwin-x64, linux-x64, linux-arm64} pair. |
| Audio pipeline accidentally produces non-16kHz PCM | Low | `Engine.transcribe()` rejects `sampleRate !== 16000` with explicit error — fails loud instead of silently transcribing garbage. |
| `hasCUDA()` false-positive on a Linux box without whisper-cpp CUDA binary | Low | `engine.init()` for whisper-cpp will fail at first transcribe; error message identifies the missing binary. User can `PI_VOICE_FORCE_SHERPA=1` to skip. |

## Testing strategy

**Unit tests (new)**
- `tests/catalog-invariants.test.ts` — runs at every CI build, asserts:
  - `LOCAL_MODELS` contains an entry with id `ULTIMATE_FALLBACK_MODEL_ID`.
  - That entry has `supports === undefined` (or empty array) — i.e. is
    platform-agnostic.
  - That entry's `engine` is undefined or `"sherpa-onnx"` (so
    `PI_VOICE_FORCE_SHERPA=1` always finds a path).
  - Each catalog entry has exactly one of `sherpaModel` /
    `whisperCppModel` set — never both, never neither.
  - GGUF entries (`format === "gguf"`) have `whisperCppModel.files.model`
    and `whisperCppModel.downloadUrls.model` both present.
  - Every GGUF entry's `files.model` passes the `assertSafeBasename`
    predicate — no `..`, no path separators, no NUL, basename matches
    itself. **This is the CI-side gate for the security/VETO finding;
    a malicious / corrupted catalog entry can never ship.**
  - Every `downloadUrls.model` URL passes `new URL(url)` (catches the
    runtime-finding malformed-URL path at CI time).
  - `path.basename(new URL(url).pathname) === files.model` — the on-disk
    filename matches the URL basename (consistency with `model-download`).
  - For each declared `supports[]` entry, `platform` is a valid
    `NodeJS.Platform` value. A bad catalog change fails CI before ship.
- `tests/model-compat.test.ts` — `matchesSupport` / `isModelSupported`
  matrix across {darwin-arm64, darwin-x64, linux-x64, linux-x64+CUDA,
  linux-arm64, win32-x64} × all catalog entries. Specifically asserts:
  - arch is checked independently from CUDA (no short-circuit bypass)
  - CUDA-required models on linux-arm64 are NOT supported even with CUDA
  - at least one model is supported on each tested runtime
- `tests/engine-factory.test.ts` — covers:
  - lazy `ensureCatalogValid()` fires the first time any export is
    called (not at module load) — proves the circular-import fix
  - validator throws when ultimate fallback is missing from catalog
  - validator throws when ultimate fallback has a `supports` constraint
  - validator throws when ultimate fallback has a non-sherpa engine
  - **`resolveEffectiveModel()` does NOT throw when the validator fails**
    — instead returns `{ degraded: true, id: ULTIMATE_FALLBACK_MODEL_ID }`.
    Test injects a corrupted catalog mock so `ensureCatalogValid` throws,
    then asserts session_start sees a useable result. (runtime round-9.)
  - `pickDefaultModelForPlatform` returns the right id on each runtime
  - missing IDs in `DEFAULT_MODEL_PRIORITY` are skipped (not treated as
    wildcards) — important for incremental rollout
  - the post-loop `throw` fires when a tampered catalog leaves zero
    matches (covers the qwen "unreachable" path explicitly so the
    diagnostic message regression is caught)
  - `resolveEffectiveModel({ persistedId: "whisper-…-metal" }, linuxEnv)`
    falls back with `fellBack: true, from: "whisper-…-metal"`
  - `resolveEffectiveModel({ persistedId: "moonshine-v2-tiny" }, anyEnv)`
    returns `fellBack: false`
  - `PI_VOICE_FORCE_SHERPA=1`:
    - skips whisper-cpp entries during default selection
    - falls a persisted whisper-cpp model id back to the sherpa default
    - keeps a sherpa model id (engine === undefined or "sherpa-onnx")
      unchanged
    - is read from `processEnv` arg (test stubs `process.env`)
- `tests/engine-sherpa.test.ts` — interface contract; rejects
  `ctx.sampleRate !== 16000` with explicit message; uses `ctx.modelDir`
  without re-resolving; module imports without throwing even when
  sherpa-onnx native binding is missing (deferred-load contract).
- `tests/engine-whispercpp.test.ts` — mocked binding; verifies:
  - module imports without throwing when `nodejs-whisper` is missing
  - singleton starts in `available: false, error: null`
  - failed `init()` sets `error` to the import-failure message and
    `available: false`
  - **failed `init()` clears `_initPromise` so the next caller starts a
    fresh attempt** (security2 finding — engine isn't bricked until
    process restart on a transient FS error)
  - `init()` rejects a binding missing `transcribe` (interface-shape
    validation — supply-chain defense-in-depth)
  - `transcribe()` while `available: false` throws a structured error
    (not a native stack trace) carrying `engine.error` as context
  - concurrent `init()` calls share one in-flight promise (single-flight)
  - subsequent `init()` after success is a cheap no-op
  - **`warmup()` is single-flight per `model.id`** — two concurrent
    `warmup()` calls for the same model share one `_binding.loadModel`
    invocation; no double-load (runtime VETO finding)
  - **`warmup()` failure clears `_warmupPromises[model.id]`** so a
    retry can run a fresh load
  - **`warmup()` sync-throw cleans up correctly** — if
    `ggufFilenameFor` throws synchronously (corrupted catalog), the
    `.finally()` on the OUTER promise still fires as a microtask
    AFTER `_warmupPromises.set(…)` has registered the entry, so the
    cleanup successfully deletes it. A second `warmup()` call after
    the microtask drain sees an empty map and retries instead of
    inheriting the cached rejection. (gemini round-11 finding.)
  - **`warmup()` concurrent-call execution-trace test** — explicit
    test of the `Promise.resolve().then()` ordering. Stub the binding
    with a `loadModel` that hangs on a manual deferred. Issue
    `warmup(ctx)` twice synchronously. Assert:
      1. Both calls return the SAME promise reference (single-flight).
      2. `_warmupPromises.size === 1` immediately after both calls
         (entry registered before any microtask drain).
      3. Resolve the deferred. After awaiting,
         `_warmupPromises.size === 0` (cleanup ran via `.finally`).
      4. A third `warmup(ctx)` call after step 3 issues a fresh
         `loadModel` (no stale entry).
    Same pattern for the sync-throw branch: stub `ggufFilenameFor`
    to throw, verify steps 1–4 still hold (order: register → reject
    → cleanup → next-call retry). (qwen round-13 advisory — this
    test makes the trace machine-verified.)
  - **`transcribe()` racing with warmup()** awaits the in-flight
    warmup, doesn't double-load
  - loads file from `path.join(ctx.modelDir, ggufFilenameFor(model))`
  - `ggufFilenameFor()` throws on non-GGUF models
  - **`ggufFilenameFor()` rejects path-traversal filenames** — `".."`,
    `"../foo"`, `"..\\foo"`, `"foo/bar"`, `"foo\\bar"`, `"\0foo"`,
    empty string. (security VETO finding from round 7.)
  - **`ggufFilenameFor()` rejects URL-encoded traversal** —
    `"%2E%2E%2F"`, `"..%2F..%2Fetc%2Fpasswd"`, `"foo%2Fbar.bin"`,
    `"%2E%2Ebin"`. The allowlist regex `/^[A-Za-z0-9._-]+$/` rejects
    `%` characters by construction, closing the
    `path.basename`-doesn't-decode bypass. (grok round-9 critical VETO.)
  - **`ggufFilenameFor()` rejects Windows reserved names** —
    `"CON"`, `"COM1"`, `"LPT1.bin"`, `"NUL"`, `"con.bin"`,
    `"COM10"`, `"COM256"`, `"LPT99"`, `"COM10.bin"`
    (case-insensitive; `COM[1-9]\d*` / `LPT[1-9]\d*` covers the
    multi-port serial-card extensions).
  - **`ggufFilenameFor()` ACCEPTS leading-zero / non-reserved
    look-alikes** — `"COM0"`, `"COM00"`, `"COM0.bin"`, `"LPT0"`,
    `"COM0a"`, `"LPT01"` are NOT Windows reserved names and the
    regex must not over-reject them. (moonshot + grok round-11 false
    positives on `\d+` over-broadness; v11 narrows to `[1-9]\d*`.)
  - **`ggufFilenameFor()` rejects spaces** — `"ggml base.bin"`,
    `"foo bar.bin"`. The allowlist permits no whitespace; valid
    whisper.cpp catalog filenames never contain spaces. CI catalog
    test asserts `path.basename(URL.pathname) === files.model` for
    every entry, so a space-containing URL would fail CI before
    runtime. (grok round-10 concern, addressed by documentation +
    test.)
  - **`ggufFilenameFor()` accepts `"..."` and `"....bin"`** — three-dot
    (Unicode ellipsis ASCII repr) and longer-leading-dot filenames are
    valid POSIX names that fit the allowlist; the guard must not
    over-reject. (minimax round-8 false-positive.)
  - **`ggufFilenameFor()` accepts canonical names** — every shipped
    GGUF filename (`ggml-base.en-q5_1.bin`,
    `ggml-large-v3-turbo-q5_0.bin`, `ggml-tiny.en-q5_1.bin`) passes.
    Test parameterized over the entire catalog so a future entry that
    accidentally adds e.g. a space fails CI loudly.
  - **`init()` `_initPromise` cleanup uses `.finally()`** — the cleanup
    runs whether the inner promise resolves or rejects. Test simulates
    both: a normal resolve-false (binding load fails cleanly), and a
    synthetic reject path (e.g., the inner async fn throws an unhandled
    error). After either, a fresh `init()` call starts a new attempt.
    (qwen round-8 finding.)
- `tests/device.test.ts` (extended) — `hasCUDA`:
  - returns false on darwin / win32 with no env override
  - returns true on linux when `/dev/nvidia0` exists
  - **`CUDA_VISIBLE_DEVICES=""` returns false** (the override-skip bug)
  - **`CUDA_VISIBLE_DEVICES="-1"` returns false** (negative override)
  - `CUDA_VISIBLE_DEVICES="NONE"` / `"None"` / `"none"` all return false
    (case-insensitive match against `CUDA_DISABLE_VALUES`)
  - `CUDA_VISIBLE_DEVICES="VOID"` / `"Void"` / `"void"` all return false
  - `CUDA_VISIBLE_DEVICES=" -1 "` returns false (whitespace trimmed)
  - `CUDA_VISIBLE_DEVICES="0"` returns true on every platform
  - `NVIDIA_VISIBLE_DEVICES="all"` returns true
  - env-var SET takes precedence over native probe (no fallthrough)
  - env-var unset on Linux + `/dev/nvidia0` present → returns true

**Integration tests**
- Existing `tests/sherpa-engine.test.ts` continues to pass.
- New `tests/integration-whispercpp.test.ts` (gated on `INTEGRATION=1`) —
  downloads tiny GGUF, runs end-to-end transcription on a known WAV fixture.

**Manual benchmarks** (recorded in HANDOFF.md before merge)
- 30-second WAV transcription on M-series:
  - parakeet-v3 sherpa-CPU (baseline)
  - moonshine-v2-tiny sherpa-CoreML
  - whisper-base-en-metal whisper.cpp-Metal
  - whisper-large-v3-turbo-metal whisper.cpp-Metal
- Targets: Turbo-Metal ≤ 0.10 RTF, Base-EN-Metal ≤ 0.05 RTF.

**Regression checklist**
- Linux/Intel/RPi: no whisper.cpp deps installed, no whisper.cpp models in UI,
  sherpa-onnx flow unchanged.
- Apple Silicon with parakeet-v3 selected: still routes to sherpa-onnx, still
  CPU-bound (we don't break working setups).
- Apple Silicon with no model selected: onboarding picks
  `whisper-large-v3-turbo-metal` (or `whisper-base-en-metal` if the smaller
  default is chosen).
- `PI_VOICE_FORCE_CPU=1` still forces CPU EP for sherpa.
- New `PI_VOICE_FORCE_SHERPA=1` falls a whisper.cpp selection back to its
  closest sherpa equivalent (or platform default).
- Hand-edited config with `localModel: "whisper-base-en-metal"` on a Linux
  box: `migratePersistedModelOnSessionStart` falls back to the platform
  default (`moonshine-v2-tiny`) for the current session, surfaces a
  one-time `notify("info")` explaining the fallback, and **does not mutate
  persisted config** so the same file still works when the user returns to
  Apple Silicon. (v3 replaces the v2 `assertModelSupported`-throws-on-
  activation behavior — see "Changes since v2".)

## Spike (do this BEFORE the rest)

Before any architecture work lands, validate `nodejs-whisper` (or chosen
binding) on a single Apple Silicon machine:

1. Add to a scratch repo: `bun add nodejs-whisper`.
2. Verify prebuilt Metal binary downloads on darwin-arm64 (no node-gyp build).
3. Download `ggml-base.en-q5_1.bin` (~57 MB).
4. Transcribe a 30-second test WAV.
5. Measure latency. Target: < 3 s wall time on M1+.
6. `otool -L` the bundled `.dylib` to check for libonnxruntime conflicts.
7. Repeat (1–6) on linux-x64 (no GPU) to confirm CPU prebuilt works there too.

**If spike fails** (build-from-source required, no Metal prebuilt, or
benchmark misses target by >2×): stop and switch to alternative binding
(`whisper-cpp-node`) or scope a custom napi-rs wrapper. Don't proceed with
architecture changes until the spike passes.

## Rollout

**Phase 1 (this PR)** — engine abstraction + sherpa adapter + `model-compat`
scaffolding (no whisper.cpp models yet, but `supports`/`engine`/`format`
fields land in `LocalModelInfo`). Pure refactor; nothing user-visible
changes. Verifies the interface holds before adding the second engine.

**Phase 2** — whisper.cpp adapter + Metal model entries + platform-aware
default. New install on Apple Silicon picks Whisper Base EN Metal. Existing
installs unchanged.

**Phase 3** — Linux CUDA support + Turbo model + onboarding hint for existing
parakeet-v3 users on Apple Silicon. Windows CUDA still deferred.

Each phase is independently shippable + revertable. Worst-case rollback is
`git revert <phase-commit>` — no data migration, no model deletion needed
(GGUF and ONNX coexist in `models/`).

## Out of scope (follow-ups)

- Streaming partial transcripts (whisper.cpp + VAD chunking).
- macOS Speech framework integration via Swift bridge.
- Windows CUDA support (no verified prebuilt).
- WebGPU / wgpu fallback for non-Metal/non-CUDA platforms.
- Dynamic model swap mid-session.
- Auto-quantization based on detected RAM.

## Decision points needing user sign-off

1. **Default English model on Apple Silicon**: Whisper Base EN (57 MB,
   ~96% accuracy) or Whisper Turbo (800 MB, ~99% accuracy)? Recommendation:
   Base EN as default, Turbo as opt-in upgrade in settings.
2. **CUDA support in Phase 2 or Phase 3?** Recommendation: Phase 3 to keep
   Phase 2 small and Apple-Silicon-focused.
3. **Surface a one-time "switch to Metal" hint** for existing parakeet-v3
   users on Apple Silicon, or stay silent? Recommendation: hint, dismissible,
   shown once.

## Estimated effort

- Spike: ~2 hours
- Phase 1 (engine abstraction + model-compat): ~1.5 days
- Phase 2 (whisper.cpp adapter + model catalog + settings UI): ~2 days
- Phase 3 (CUDA + onboarding hint + final polish): ~1 day
- Tests: ~1 day across phases

**Total: ~5–6 working days** for a complete, tested, reverted-safe rollout.

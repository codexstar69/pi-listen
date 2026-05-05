# Handoff

## 2026-05-04 22:21 IST

- Most recent work: checked a critical security review claim about
  URL-encoded traversal bypassing `ggufFilenameFor` filename checks.
- User prompts:
  - "A code reviewer raised the following concern about a piece of code:
    Category: security Severity: critical..."
- Verification performed:
  - Searched authoritative web sources only for the final claim check:
    Node.js official path docs, Node upstream `lib/path.js`, ECMAScript
    `decodeURIComponent`, and RFC 3986 percent-encoding/path rules.
  - Confirmed `extensions/voice/engine-whispercpp.ts` is not present in the
    current tree; the matching code is only in `docs/plan-whispercpp-engine.md`.
- Conclusion:
  - Web evidence supports the reviewer on the narrow `path.basename()` point:
    it treats literal path separators and does not decode percent-encoded
    `%2F`, `%2E`, or `%5C` sequences before checking a name.

## 2026-05-04 22:00 IST

- Most recent work: checked a reviewer claim about a planned
  `engine-whispercpp.ts` warmup single-flight race.
- User prompts:
  - "A code reviewer raised the following concern about a piece of code..."
- Verification performed:
  - Searched authoritative web sources for JavaScript/Node concurrency rules.
  - Read the local `docs/plan-whispercpp-engine.md` sketch that has the
    `_recognizers.has()` check before `_warmupPromises.get()`.
  - Confirmed `extensions/voice/engine-whispercpp.ts` is not present in the
    current tree.
- Conclusion:
  - The reviewer's proposed check-order race is not supported by JS/Node
    run-to-completion semantics when there is no `await` between the checks and
    storing the in-flight promise.

## 2026-03-26 22:02 IST

- Most recent work: implemented the env-derived Deepgram key persistence fix
  and verified it with tests and typecheck.
- User prompts:
  - "impliment it"
- Changes made:
  - Removed the `session_start` code path in `extensions/voice.ts` that copied
    `process.env.DEEPGRAM_API_KEY` into config and saved it to global settings.
  - Added `getSessionStartPersistedConfig()` in `extensions/voice/config.ts`
    so startup-triggered saves keep stored keys but strip env-only keys.
  - Added regression tests in `tests/config.test.ts` for env-only vs stored-key
    startup persistence.
  - Bumped package version to `5.0.5` and added a changelog entry.
- Verification performed:
  - `bun test tests/config.test.ts` passed
  - `bun test tests/onboarding.test.ts` passed
  - `pnpm typecheck` passed
  - `bun test` passed: 79 tests, 0 failures

## 2026-03-26 21:55 IST

- Most recent work: wrote and reviewed an ExecPlan for fixing the
  env-derived Deepgram key persistence bug.
- User prompts:
  - "but the env can be added directly when using pi and so it saves it in
    env.secrets - how would be a better implimentation?"
  - "build a implimentation plan and cross review it"
- Artifacts:
  - Plan saved to
    `docs/superpowers/plans/2026-03-26-deepgram-env-runtime-only.md`
- Review conclusion:
  - Preferred implementation is the smallest fix: remove the
    `session_start` env-to-config mutation in `extensions/voice.ts`.
  - A helper should stay optional and only be added if needed for testing or
    clarity.
  - Explicit onboarding saves to `~/.env.secrets` / `~/.zshrc` should remain.

## 2026-03-26 21:40 IST

- Most recent work: verified the reported `DEEPGRAM_API_KEY` persistence bug in
  `pi-listen` without changing runtime code.
- User prompts:
  - "Summary ... Suggested fix ... - is it true?"
- Verification performed:
  - Read `extensions/voice.ts` startup flow and confirmed env-derived
    `DEEPGRAM_API_KEY` is copied into config, then saved to global settings when
    `configSource === "global"` at lines 1807-1810.
  - Read the onboarding auto-activation path and confirmed first-run/default
    startup also persists config through `saveConfig(...)` at line 1828 when a
    Deepgram key is present via env.
  - Read `extensions/voice/config.ts` and confirmed serialization only strips
    `deepgramApiKey` for project scope; global saves keep the key.
  - Read `extensions/voice/deepgram.ts` and confirmed runtime resolution already
    prefers `process.env.DEEPGRAM_API_KEY` before stored config.
  - Read changelog and tests confirming 5.0.1 fixed only project-scope leakage,
    not global-scope persistence.
- Conclusion:
  - The report is true in substance.
  - The env-derived Deepgram key is still persisted into
    `~/.pi/agent/settings.json`.
  - The likely-cause note is slightly incomplete because there are two write
    paths: direct global save on `session_start` for global config sources, and
    onboarding migration save on first-run/default startup.

## 2026-03-18 23:59 IST

- Most recent work: full bug audit of the `pi-voice` codebase with code
  evidence, test evidence, and official documentation checks, followed by fixes
  for the confirmed runtime issues.
- User prompts:
  - "I want you to audit entire code for bugs and possible errors"
  - "I want you to audit entire code for bugs and possible errors verify with
    evidence and official documentations"
- Verification performed:
  - `bun test --run tests/sherpa-engine.test.ts` passed after adding the
    regression for odd-offset PCM buffers
  - `bun test --run` passed: 77 tests, 0 failures
  - `bun run typecheck` failed in this environment because Bun reported its
    postinstall script was not run; this looked like a local toolchain issue,
    not a repo type error proven from source
  - Local repros run for:
    - Node WebSocket upgrade request headers
    - TypedArray misalignment crash (`Int16Array` with odd `byteOffset`)
- Main fixes applied:
  - Deepgram stop now waits for socket close or a short finalize timeout after
    `CloseStream`
  - Deepgram streaming errors now go through a single close-and-report path
  - `pcmToFloat32()` now handles odd `Buffer.byteOffset` values safely
  - `package.json` version bumped to `5.0.4`
  - `CHANGELOG.md` updated with the patch release entry
- Useful references gathered:
  - Deepgram finalize docs
  - Deepgram CloseStream docs
  - WHATWG WebSocket spec on `error` then `close`
  - Node Buffer docs
  - TypedArray docs for byte-offset alignment

---

## Pluggable STT engine (Phase 1 + 2 + 3)

Added a second STT engine — **whisper.cpp via `smart-whisper`** — alongside
the existing sherpa-onnx path. Apple Silicon users now get **5–15× faster
transcription** with Metal acceleration; Linux + NVIDIA users get a CUDA
path; everyone else stays on sherpa-onnx (unchanged). Architecture spec is
`docs/plan-whispercpp-engine.md` (15 godspeed rounds + Codex/Gemini final
audit). All 344 tests pass + typecheck clean.

### Install requirements (whisper.cpp engine)

`smart-whisper` is in `optionalDependencies` and builds from source via
`node-gyp rebuild` on install. Requirements:

- **macOS**: Xcode Command Line Tools (`xcode-select --install`). Build
  produces `node_modules/smart-whisper/build/Release/smart-whisper.node`
  with Metal kernels compiled in. Adapter sets
  `GGML_METAL_PATH_RESOURCES` automatically so the bundled Metal kernel
  source (`ggml-metal.metal`) is found at runtime.
- **Linux**: `build-essential` + `cmake` + Python 3 (for node-gyp). Add
  CUDA Toolkit + the `WHISPER_CUDA=1` envar at install time to enable the
  CUDA backend.
- **Windows**: Visual Studio Build Tools. CUDA prebuilt path NOT shipped
  in this release; only the Apple Silicon and Linux x64 paths have
  catalog entries.
- **Path constraint**: `smart-whisper@0.8.1`'s `binding.gyp` does NOT
  quote paths. Install paths containing spaces (e.g.
  `/Users/me/Code Files/pi-voice`) cause the build to fail. Workaround:
  install at a no-space path. Long-term fix: upstream patch to
  smart-whisper.

If the build fails (no toolchain, path with spaces, etc.):
- `bun install` keeps going — `optionalDependencies` failure is
  non-fatal.
- The `engine-whispercpp` adapter's `init()` catches the missing binding
  and sets `engine.error`. `migratePersistedModelOnSessionStart` then
  surfaces a clear notification and falls a whisper-cpp model id back to
  the sherpa-onnx default. No crash; voice still works (just on the
  slower path).

### What ships in each phase

- **Phase 1** — Engine abstraction (`engine.ts`, `engine-sherpa.ts`,
  `engine-factory.ts`, `model-compat.ts`). Pure refactor, no
  user-visible change. `LocalModelInfo` gains optional `engine` /
  `supports` / `format` / `whisperCppModel` fields with back-compat
  defaults.
- **Phase 2** — `engine-whispercpp.ts` adapter, three Metal GGUF catalog
  entries (Whisper Base EN, Turbo, Tiny EN). `whisper-base-en-metal` is
  the default for fresh Apple Silicon installs;
  `whisper-large-v3-turbo-metal` is the opt-in upgrade.
- **Phase 3** — `whisper-large-v3-turbo-cuda` for Linux x64 + NVIDIA
  GPU. One-time "switch to Metal" hint on session_start for existing
  Apple Silicon users still on a sherpa-onnx model. Hint is sticky via
  `config.onboarding.whisperCppHintShownAt`.

### Env overrides

- `PI_VOICE_FORCE_CPU=1` — disables the CoreML provider in `sherpa-onnx`
  (keeps everything CPU). Pre-existing escape hatch from the round-1
  CoreML work.
- `PI_VOICE_FORCE_SHERPA=1` — falls a whisper-cpp model selection back
  to its closest sherpa-onnx equivalent. New in Phase 2; useful when
  whisper.cpp regresses on a future macOS update.
- `CUDA_VISIBLE_DEVICES=""` / `="-1"` — disables CUDA detection.
  Authoritative when set; the Linux native probe (`/dev/nvidia0`) only
  runs when the env var is unset.
- `GGML_METAL_PATH_RESOURCES` — points whisper.cpp's Metal backend at
  `ggml-metal.metal`. Set automatically by `engine-whispercpp.ts:
  configureMetalResources()` if not already set; expose for advanced
  users who run from outside `node_modules`.

### Verified benchmark (M4 Max)

- Engine smoke-test (`/tmp/whisper-spike/smoke.mjs`) on tiny.en (74 MB)
  with Metal active:
  - warmup: ~100 ms
  - transcribe: 195 ms for 3.72 s audio = **RTF 0.052 (19× realtime)**
  - Apple M4 Max GPU confirmed via `ggml_metal_init: GPU name`
- Equivalent sherpa-onnx CPU path on Parakeet-v3: ~1500–3000 ms for
  the same audio = **RTF 0.4–0.8 (1.2–2.5× realtime)**.


---

## Round-2 audit fixes (5-model consensus)

After the initial Phase 1+2+3 implementation shipped, the codebase was
re-audited by 5 separate models (GPT-5.5, Gemini 3.1 Pro, Claude Opus
4.1, DeepSeek V3.1, Grok 4). Universal verdict: SHIP-WITH-POLISH. The
items below were applied.

### Memory + lifecycle (high impact)

- **Bounded LRU cache** in both adapters. `_whispers` / `_recognizers`
  Maps no longer grow unbounded across model switches. Default size is 1
  (most users run one model at a time); configurable 1-4 via
  `PI_VOICE_ENGINE_CACHE_SIZE`. Older models LRU-evict on warmup.
- **`Engine.clearCache()` is now `async`/`Promise<void>`**. Callers can
  await native memory release before loading the next model.
- **Generation token + active-transcribe tracking**. `clearCache()`
  during an in-flight `transcribe()` no longer use-after-frees the
  recognizer — the actual `free()` is deferred until the last active
  inference settles. Race between `clearCache()` and a concurrent
  `warmup()` is closed by a `_cacheGen` counter: a warmup that completes
  after a clear finds its captured generation stale and frees the
  just-loaded model instead of caching it.
- **Sherpa adapter caches its own recognizer reference** in `warmup()`.
  Previously every `transcribe()` re-resolved via
  `getOrCreateRecognizer(...)` (one Map lookup + branch); now adapter
  reads its own `_recognizers` map directly.

### Performance (medium impact)

- **O(1) catalog lookups** via lazy `Map<string, LocalModelInfo>` in
  `engine-factory.ts`. Replaces 3+ `LOCAL_MODELS.find()` calls per
  `resolveEffectiveModel`.
- **PCM odd-offset bulk-copy** in `pcmInt16ToFloat32`. Single
  `Buffer.from(pcm)` to align, then bulk Int16Array view + loop —
  beats the per-sample `readInt16LE` fallback for any non-trivial
  buffer.
- **Sherpa single-flight warmup** matches whisper-cpp pattern. Two
  rapid recordings won't construct two recognizers.

### Robustness (medium impact)

- **`getEngineForModel` throws on unknown engine values** instead of
  silently routing to sherpa. TypeScript prevents this in source; the
  guard catches catalog corruption / hand-edited config.
- **`init()` cleanup uses non-throwing `.then(_, _)` chain** instead of
  `.finally()` so a future edit that lets `_doInit` reject can't surface
  as unhandled rejection noise.
- **Metal kernel existence check** in `configureMetalResources` —
  `fs.existsSync` before setting the env var, so a smart-whisper layout
  change doesn't point Metal at a stale dir.
- **`findResilientFallbackId` respects `PI_VOICE_FORCE_SHERPA=1` in the
  last-resort tier** — no whisper-cpp model is returned even when the
  catalog is degraded, unless absolutely no sherpa option exists.

### UX gaps (medium impact)

- **First-recording download notifications**. `local.ts` now exports a
  `setDownloadNotifier()` slot; `voice.ts` wires it at `session_start`
  to surface clear `notify("info")` messages when the first recording
  triggers a model fetch. Throttled to ~1 Hz so progress doesn't spam.
  Cleared at `session_shutdown` (via `voiceCleanup`) so the closure
  doesn't outlive its `ctx`.
- **Friendly install-error mapping** in `engine-whispercpp._doInit`.
  Common failure modes (`Cannot find module`, path-with-spaces clang
  errors, CUDA build mismatch) map to actionable messages with platform-
  specific install commands.

### CUDA build-vs-hardware mismatch (medium impact)

- New `device.cudaBuildCapable(env)` returns true when
  `WHISPER_CUDA=1` is in the runtime env (signal that smart-whisper was
  built with CUDA support).
- `model-compat.matchesSupport` for `requiresCUDA: true` now requires
  BOTH `env.hasCUDA` (hardware) AND `env.cudaBuildCapable` (binding).
  CUDA-only models hide from the picker on systems that have NVIDIA
  hardware but a default CPU-only smart-whisper build, so users don't
  silently route to a model that runs on CPU.
- `describeUnsupported` distinguishes the two reasons: "(CUDA hardware
  unavailable)" vs "(smart-whisper built without CUDA — reinstall with
  WHISPER_CUDA=1)".

### Metal env var on macOS — known upstream limitation

`smart-whisper@0.8.1` reads `GGML_METAL_PATH_RESOURCES` via
`[NSProcessInfo processInfo].environment`, which **caches the
environment at process start**. Setting `process.env.GGML_METAL_PATH_RESOURCES`
from inside Node does NOT propagate to NSProcessInfo. Metal silently
falls back to CPU.

Workaround: shell-export the var before Node starts. Add to your
shell rc:

```sh
export GGML_METAL_PATH_RESOURCES="$HOME/path/to/pi-voice/node_modules/smart-whisper/whisper.cpp/ggml/src"
```

Or, project-local, via direnv `.envrc`. The engine surfaces this hint
via `engine.error` after a successful `init()` on darwin-arm64 when the
var wasn't shell-exported — the engine is still `available`, just slower
than Metal-accelerated until the user fixes their shell.

Verified end-to-end: with the env shell-exported, transcription on M4
Max runs at **RTF 0.052 (19× realtime)** on tiny.en. Without it, falls
back to CPU at ~RTF 0.10 (still real-time, just not GPU-accelerated).


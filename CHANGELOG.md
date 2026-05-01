# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [7.2.0] - 2026-05-01

World-class voice UX: end-to-end TTS streaming via stdin pipe, sub-cell
audio waveforms in truecolor, floating-island chrome, plus a stack of
lifecycle/correctness fixes shaken out by multi-model code review.

### Visual language (new)

- **Liquid Braille audio waveform** — sub-cell vertical bars (4 levels
  × 2 columns per cell via braille U+2800-28FF) replace the chunky
  `▁▂▃▄▅▆▇█` block bars. 8 effective vertical levels per cell, 2
  audio samples per cell width — twice the density of v7.1.
- **Aurora truecolor gradient** — Catppuccin-inspired ramp (lavender →
  mauve → pink → peach) interpolated at runtime via 24-bit ANSI
  escapes (`\x1b[38;2;R;G;Bm`). Loud peaks "burn" warmer; soft tails
  stay cool. No theme dependency, falls back gracefully on legacy TTYs.
- **Floating Island chrome** — 3-line bordered card with rounded
  corners (`╭─╮│╰╯`), inline-titled top edge, optional inline footer.
  Width-tight (36–48 cols). Borders ANSI-aware so right edge always
  lands cleanly even with embedded color escapes.
- **Breathing title** — slow aurora-stop cycle on the floating island
  title (`Voice Input` / `Voice Mode`); 4-second sine, in phase with
  the global `RenderTicker` so all widgets breathe together.
- **Activity chip** — one-glance audio-level indicator next to the
  timer: `▁ quiet` / `▃ voice` / `▅ active` / `▇ loud`, each with its
  own aurora hue.
- **Fade-in transition** — recording widget's wave amplitude scales
  0→1 over 300ms so it grows out of the warmup state instead of
  snapping in.
- **Branded picker cursor** — selected rows show a thin accent left
  bar (`│`) with full-saturation text; non-selected rows dim. HIG
  "deference" — chrome subtle, content hierarchy via dim contrast.
- **Status-dot badges** — `● ready` / `○ download` / `✗ broken`
  colored-dot indicators replace the v7.1 plain-word labels.

### TTS streaming (new)

- **Long-lived stdin-piped player** — `ffplay -f s16le -ar <rate>
  -ch_layout mono -i pipe:0` (preferred) / `paplay --raw --format=s16le`
  (Linux fallback) / `sox` (last-resort, has CoreAudio underrun bug).
  Drops file-write/open/start latency entirely.
- **Async writePcm with serialized writeTail chain** — each write
  awaits the previous one's drain (or stream tear-down) before the
  next, no unbounded memory growth, no premature pipe close. Replaces
  the broken multi-`once('drain')` listener pattern that resolved
  prematurely.
- **1 s silence padding before EOF** — compensates for sox/CoreAudio
  closing the audio device on EOF and dropping ~25% of buffered audio.
- **Pipelined synthesis** — chunk N+1 synthesizes while chunk N
  plays, producing seamless multi-sentence output.
- **Deepgram WebSocket TTS** — `wss://api.deepgram.com/v1/speak`
  binary frames pipe straight into the same playback sink.
  Sub-200 ms TTFA on good network. Toggle via `/voice-stream on`.

### STT auto-submit (new)

- **`/voice-autosubmit on/off`** — when enabled, transcribed text is
  sent directly to the agent via `pi.sendUserMessage(text,
  { deliverAs: "followUp" })` and the editor clears.
- **`agentBusy` guard** — when the agent is mid-turn (especially
  mid-retry), STT text stays in the editor and a notify surfaces
  *"Agent is busy — voice text held in editor. Press [↵] to send when
  ready."* — no `followUp` queue pile-up onto a failing turn.
- **Editor preserves user edits** — `clearAfterSuccess` only clears
  if the editor still contains exactly the dispatched text; if the
  user typed something else while the send promise was pending, their
  edits are preserved.

### Lifecycle / correctness

- **Hold-threshold 1200 ms → 700 ms** (configurable via
  `voice.holdThresholdMs`, range 200–3000, slash command
  `/voice-hold-delay <ms>`).
- **Caller-signal cascade** — `runInstallWithWidget` accepts caller's
  `AbortSignal`; `/voice-speak-stop` or TTS-disable now cancels
  in-flight install downloads cleanly.
- **Owner-checked `activeInstallWidgets.delete`** — stale finally
  blocks can't evict a newer same-id widget.
- **`sizeBytes ?? 0` guard** — prevents NaN in install progress when
  catalog entry lacks the field.
- **kokoro-multi-lang flagged incompatible** — int8 voices.bin
  produces NaN samples per upstream issue #1923 (sherpa-onnx-node
  1.12.29 + 1.13.0 both affected). Synthesize() refuses; smart-default
  recommender skips it; picker shows `✗ broken` badge.
- **Frozen-error tagging guard** — `__alreadyNotified` assignment is
  try/catch wrapped (handles non-extensible / cross-realm errors).
- **Unhandled-rejection safety** — donePromise + nextSynth prefetch
  promises have shadow `.catch()` so cancel-without-await flows don't
  surface as `UnhandledPromiseRejection`.

### Text normalization

- **Emoji + decoration stripping** — `\p{Extended_Pictographic}` +
  dingbats (✓✗★) + arrows (→⇒) + box drawing + repeated punctuation
  collapsed.
- **Abbreviation expansion** — `Dr.` → "Doctor", `e.g.` → "for
  example", `API/CLI/URL/HTTP/JSON/YAML/CSS/HTML` spelled or expanded
  for engine-friendly pronunciation.
- **Cardinal number expansion** — `100` → "one hundred", smart skip
  for years, decimals, version strings, unit-suffixed values.

### Catalog

- **+3 Kokoro variants** — `kokoro-int8-multi-lang-v1_1` (140 MB,
  upstream re-quantized), `kokoro-multi-lang-v1_0` (333 MB fp32, no
  NaN risk), `kokoro-en-v0_19` (304 MB fp32, highest-quality English).

### Tests

- **+97 new tests** across 8 new test files. Total **271 tests
  passing** (1 123 expect calls, 20 test files). Typecheck clean.

### Changed

- `sherpa-onnx-node` peer dep `^1.12.29` → `^1.13.0`.
- `extensions/voice.ts` — wires Floating Island widgets, agentBusy,
  ffplay streaming, Aurora gradient.
- `extensions/voice/tts-playback.ts` — new `openPlaybackStream` /
  `PlaybackStream` / `float32ToInt16` exports.
- `extensions/voice/tts-deepgram.ts` — new `deepgramSpeakStreaming`.
- 9 new files in `extensions/voice/` (ui-aura, ui-icons, ui-width,
  ui-locale-labels, ui-picker, ui-widget-base, ui-render-ticker,
  ui-help-overlay, tts-onboarding-overlay).

### Documentation

- `docs/v7.1.0-ui-plan.md` — the v5 architecture plan archived
  in-tree (8 design-review iterations from Codex + Gemini).

## [7.1.2] - 2026-04-29

### Dependency

- **Bumped `sherpa-onnx-node` from `^1.12.29` → `^1.13.0`** based on
  Gemini deep-research findings: PRs #3362-#3365 in k2-fsa/sherpa-onnx
  fix the `GenerationConfig`/`Generate` API for VITS / Kokoro / Matcha
  / Kitten (the "Not implemented yet" path). 1.13.0 still ships the
  napi_create_arraybuffer crash on `onProgress` (binding-thread bug —
  needs `napi_threadsafe_function` upstream). Verified 1.13.0 fixes
  GenerationConfig but does NOT resolve the kokoro multilingual NaN
  issue (root cause: int8 quantization of `voices.bin` speaker
  embeddings, per upstream issue #1923).



Hotfix continuing the v7.1.1 install-time TTS surgery. Live testing
on Pi (jiti-loaded extension under Node) surfaced three more failure
modes that produce the same user symptom — silent or erroring TTS —
each with a different root cause.

### Fixed

- **`sherpa.OfflineTts.createAsync is not a function` (jiti edition).**
  v7.1.1 sniffed the namespace for `OfflineRecognizer` to pick the
  ESM-vs-CJS shape. Pi's actual loader is **jiti**, which exposes
  *stub* class constructors at the top level (so `OfflineTts` looks
  like a function) but the real fully-populated module — including
  static `createAsync` — lives on `.default`. v7.1.2 sniffs for
  `OfflineTts.createAsync` directly: if it's on the namespace, use
  the namespace (Bun); otherwise prefer `.default` when it carries
  `createAsync`; final fallback is the namespace as-is so future
  runtime variants degrade gracefully.
- **`Failed to create OfflineTts. Check your config!` for kokoro and
  kitten.** `tts-engine.ts` hardcoded `model.onnx` for kokoro and
  `model.fp16.onnx` for kitten. The actual sherpa-onnx model archives
  ship variant filenames depending on quantization (kokoro v1.0
  multilingual ships `model.int8.onnx`, kitten ships
  `model.fp16.onnx`). v7.1.2 introduces `findFirstOnnx(modelDir,
  candidates)` which probes a priority list of likely filenames and
  falls back to "any .onnx in the dir" — defends against future
  quantization variants without code changes.
- **Kokoro multilingual: silent playback on every voice.** The
  picker, sherpa-onnx-node 1.12.29, and the kokoro v1.0
  multilingual `voices.bin` combine to produce all-NaN samples for
  every speaker id (sid 0..50, all 17 listed voices). The encoded
  WAV plays as silence, so users see "Playing 19s" with no sound.
  Fix is multi-pronged:
  - **Catalog** — new `incompatible` field marks the model with a
    one-line user-facing reason. Picker shows it with `✗ broken`
    badge and surfaces the reason in the detail row.
  - **`synthesize()`** — refuses incompatible models upfront with an
    explicit error pointing at `/voice-speak-models`.
  - **NaN detection** — synthesize also detects all-NaN samples
    *after* generation (defends against future broken voices in
    other models) and throws the same clear error.
  - **Smart-default recommender** — skips kokoro for ja/ko locales
    while flagged, falls back to English with `fallback: true` so
    onboarding surfaces the situation honestly. Auto re-enables once
    the `incompatible` flag is removed.
  - **Picker activation guard** — pressing enter on an incompatible
    model is a no-op so users can still see future-fix candidates
    in the list without accidentally activating one.

### Verified

- Sweep across all installed models × all sids: `kitten-nano-en`
  voices 0–7 all produce real audio (max amplitudes 0.40–0.80);
  `piper-ru_RU-denis` produces audio; `kokoro-int8-multi-lang` 0/1/2/
  …/50 all return NaN under sherpa-onnx-node 1.12.29 (confirmed in
  fresh process, no engine-cache contamination).
- 271 tests passing (smart-default ja/ko tests rewritten to expect
  the English fallback while kokoro is flagged).
- Typecheck clean. `bun publish --dry-run` packs cleanly.

## [7.1.1] - 2026-04-29

Hotfix immediately following v7.1.0 install testing.

### Fixed

- **`sherpa.OfflineTts.createAsync is not a function` on first
  /voice-speak under Pi (Node)** — `sherpa-loader.ts` now normalizes
  the synthetic ESM namespace returned by Node's `await import(...)`
  for CommonJS modules. Under Bun, the namespace exposes
  `OfflineTts` directly; under Node, the entire CJS `module.exports`
  lives on `.default`. The loader now picks whichever shape carries
  `OfflineRecognizer`, fixing TTS for everyone running pi-listen on
  Pi's bundled Node runtime.
- **`Not implemented yet. Only some models support this` from
  sherpa-onnx during synthesis** — sherpa-onnx-node 1.12.29 ships two
  bugs in the new `generateAsync({ generationConfig, onProgress })`
  API: the GenerationConfig path throws "Not implemented yet" and the
  onProgress callback variant crashes the process in
  `napi_create_arraybuffer`. v7.1.1 falls back to the legacy
  `generateAsync({ text, sid, speed })` path which is verified
  end-to-end on macOS arm64 with kitten / vits / kokoro. Cost: we lose
  intra-synthesis progress callbacks (synthesis is fast enough that
  the missing UI signal is invisible) and the optional `silenceScale`
  knob (sherpa default is reasonable).

### Added

- **`autoSubmitOnSpeak` config + `/voice-autosubmit` command** —
  when ON, transcribed STT text is sent to the agent immediately
  instead of just being placed in the editor. Defaults OFF so
  existing users aren't surprised.
- **TTS auto-speak default flipped to ON** — `ttsAutoSpeak` now
  defaults to `true` so users hear agent responses out of the box
  once TTS is enabled. Disable via `/voice-settings` or
  `voice.ttsAutoSpeak = false` in `settings.json`.

## [7.1.0] - 2026-04-29

World-class Settings UI redesign. Plan reviewed and SHIP'd by Codex (v5)
+ Gemini (v5); implementation reviewed across 7 iterations until both
reviewers + self-review converged on SHIP.

### Added — UI foundation (`extensions/voice/ui-*`)

- **`ui-widget-base.ts`** — `WidgetRegistry` + `BaseDisposableWidget`
  with the lifecycle contract that survived 5 review iterations:
  - `register(w)` synchronously disposes any same-key incumbent.
  - `unregister(key, owner)` is owner-checked — only deletes the Map
    entry if it currently points to `owner`.
  - `disposeAll()` iterates a CLONED snapshot (so widget `dispose()`
    can re-enter `unregister` safely) and wraps each call in
    `try/catch`.
  - `BaseDisposableWidget` enforces dispose ordering: idempotency
    guard reads PRIOR state → set `disposed = true` → unsubscribe
    ticker → `onDispose()` → clear slot → owner-checked unregister.
  - `installWidgetKey(modelId)` per-model-id slot keys so concurrent
    installs of different models coexist.
- **`ui-render-ticker.ts`** — single shared 10 Hz frame coalescer.
  Subscribers pass an explicit `TickerSubscriber { tick, dispose?, label? }`
  object so ownership is unambiguous. Per-tick `try/catch` isolates
  throwing subscribers; 3-throws-in-a-row auto-evicts and the
  eviction `dispose()` is itself wrapped in `try/catch`.
- **`ui-picker.ts`** — `PickerChassis<T>` with heading-aware nav,
  search filtering that retains a heading only when at least one
  child row matches, cursor restoration on search clear, compact
  mode for narrow terminals.
- **`ui-icons.ts`** — geometric Unicode glyph table (no emoji per the
  v7.1 hard constraint).
- **`ui-width.ts`** — `visualWidth()` handles surrogate pairs + EAW
  Wide/Fullwidth code points; `truncateToVisualWidth()` never slices
  a wide glyph; width tier helpers (wide/mid/narrow).
- **`ui-locale-labels.ts`** — hand-curated native-script names for
  12 languages; ar/hi intentionally omitted (RTL hazards + Devanagari
  combining marks would need a grapheme segmenter that violates the
  zero-dependency constraint).

### Added — Widgets

- **`tts-install-progress.ts`** — sticky download progress widget,
  per-model-id key, progress bar with size/speed/ETA on wide
  terminals (graceful trim on narrow), `[esc]`-cancellable.
- **`tts-playback-indicator.ts`** — honest playback state (spinner +
  state word). No fake amplitude meter (the v1 plan proposed a
  `sin + random walk` meter; both reviewers flagged it as misleading
  and v7.1 ships the honest version).
- **`tts-onboarding-overlay.ts`** — rich first-run overlay with three
  explicit actions (`[↵]` try / `[m]` pick another / `[esc]` skip).
  All three actions persist `ttsOnboardingShown = true` BEFORE any
  async work (§9 event-ordering contract).
- **`ui-help-overlay.ts`** — keyboard / command reference. `F1` or
  `/voice-help` opens it. `?` is intentionally NOT bound globally
  (would block typing `?` in the editor).

### Added — Settings panel integration

- TTS Models picker uses `PickerChassis` with v7.1 grouping
  (Recommended / Per-language / Multilingual heavyweight) and
  width-tier compact mode.
- Voice picker rows now show native-script language labels (`中文`,
  `日本語`, `한국어`, etc.) when the language has a curated entry.
- Two-row status header — at-a-glance system state (STT `●`/`○` +
  backend, TTS `●`/`○` + backend, current language). Width-aware.
- Hard block at <60 cols — single-line "terminal too narrow" message
  pointing users at slash commands.

### Added — voice.ts wiring

- Per-session `WidgetRegistry` + `RenderTicker` (lazy init).
- `voiceCleanup()` cancels in-flight install controllers FIRST so
  downloads abort on session shutdown, then drains the registry,
  then disposes the ticker.
- `runInstallWithWidget()` replaces the v7.0.x notify-spam loop with
  the sticky widget. Rethrows install failures (with
  `__alreadyNotified` marker) so callers short-circuit instead of
  proceeding with a missing model. Same contract on the headless
  branch.
- `[esc]` priority routing: install cancel (most-recent first) →
  playback stop → fallthrough.
- `F1` opens help overlay; `/voice-help` slash command.

### Tests

- **+97 new tests** across 8 new test files. Total: **271 tests
  passing** (up from 174 in v7.0.1). Typecheck clean.

### Changed

- `extensions/voice/settings-panel.ts` — TTS Models picker via
  chassis; voice picker emits native-script labels; two-row status
  header; hard block at <60 cols.
- `extensions/voice.ts` — wires the v7.1 widgets, [esc] routing,
  F1 help, rich onboarding overlay; `voiceCleanup` aborts installs
  before tearing down UI.
- `package.json` — 7.0.1 → 7.1.0.

### Documentation

- `docs/v7.1.0-ui-plan.md` — the approved v5 plan archived in-tree.

## [7.0.1] - 2026-04-29

Hotfix targeting concurrency + corruption hazards in the v7.0.0 install
pipeline. Surfaced by godspeed multi-model review after publish.

### Fixed
- **Concurrent install race** — `ensureTtsModelInstalled()` now uses a
  per-modelId in-flight `Map` so two concurrent install calls for the
  same model share one promise. Without this, two calls would both open
  `<modelId>.partial.tar.bz2` for writing (`fs.createWriteStream` with
  `flags: "w"` truncates), corrupt each other's bytes, and either fail
  tar extraction or race on the rename to the final dir with `ENOTEMPTY`.
- **HTTP 416 on completed partial archives** — when a prior run died
  AFTER the download finished but BEFORE unlinking the partial, the
  next attempt sent `Range: bytes=<size>-` and the server returned 416
  ("Range Not Satisfiable"). v7.0.0 threw and required manual cleanup;
  v7.0.1 treats 416 as "you already have the full bytes" and proceeds
  to SHA verification + extract.
- **Stream error during download** — added an upfront `error` listener
  on the file write stream and an error-aware drain wait in the
  body-streaming loop. v7.0.0 could throw an unhandled exception (and
  crash the agent) on disk-full / EIO / EPERM during a write.
- **onProgress callback in finally path** — `onProgress({ phase: "done" })`
  now fires AFTER the install try/catch completes successfully. v7.0.0
  fired it inside the try, which meant a user-supplied callback that
  threw could trigger the catch block's cleanup and delete a freshly-
  installed model.
- **Partial archive cleanup on extract failure** — `doInstall` now
  tracks which phase reached and deletes the partial archive when the
  failure is at the `verify` (SHA mismatch) or `extract` (tar failed)
  phases, where the bytes are known-corrupt. Network failures during
  the `download` phase still keep the partial for resume.
- **Pt-PT vs pt-BR fallback flag** — `recommendDefaultModel("pt-PT")`
  now correctly returns `fallback: true` because the catalog only ships
  Brazilian Portuguese. Reason text updated to surface "accent will
  differ from your locale". v7.0.0 set `fallback: false` on this path,
  which would have caused the v7.1 onboarding picker to suppress the
  language-mismatch warning.
- **Defensive `phaseReached === "done"` guard** in the catch block
  ensures cleanup never runs after a successful install completes,
  even under refactor-induced rearrangement.
- **Upfront modelId validation** in `ensureTtsModelInstalled` — unknown
  ids now throw synchronously before the in-flight Map is touched,
  rather than after the first await yield in `doInstall`.

### Verification
- `bun test` 174/174 passing (3 new concurrency-shape tests added)
- godspeed multi-model review: 5/7 SHIP, VETO=0 (the in-flight Map
  pattern was confirmed by deepseek, security, sonnet, moonshot,
  architect; remaining 2 NO_SHIPs are noise-floor — substring-match
  false positive and a low-severity AbortSignal-deduplication concern
  the advisor downgraded in the previous round)
- Real `pi 0.70.5` RPC smoke: extension loads cleanly

## [7.0.0] - 2026-04-29

**v7 makes the TTS picker an actual UI, downloads happen on selection,
auto-speak finally works, and onboarding tells you what to do.**

The user-facing gap in v6.0.0 was: "you have to edit settings.json to
pick a model, you can't auto-download, and ttsAutoSpeak doesn't do
anything." v7 fixes all three.

### New — Speak tab Model + Voice pickers (replaces JSON-editing)

- **Models picker** in `/voice-settings` Speak tab: full 14-model
  catalog with per-row size, language coverage, and install state.
  ✓ for installed, ⬇ for download-on-select. Search-as-you-type filter.
  Activating a not-installed entry triggers automatic download with
  progress notifications (every 10% step).
- **Voices picker**: numeric speaker ids with display names from the
  catalog (Kitten Nano shows 8 voices, Kokoro v0.19 shows 11, Kokoro
  multilingual v1.0 shows 17 hand-picked voices spanning 9 languages,
  Piper LibriTTS-R shows 904). Backend-aware — picks Aura voice ids
  for Deepgram, filtered by current language.
- **Always-visible status row** at the top of the Speak tab —
  `Local · Kitten Nano · Expr-Voice-2-M · 1.0× · EN`. One glance tells
  you exactly what's configured.

### New — First-run TTS onboarding

- When you enable TTS for the first time (`/voice-speak-toggle` or
  Speak tab), pi-listen shows a one-shot hint with a smart-default
  recommendation based on your `systemLocale`:
  - `en-*` → Kitten Nano (25 MB, default)
  - `es/fr/de/hi/it/ru/ar/tr/nl/zh` → matching Piper voice (~20 MB)
  - `ja/ko` → Kokoro multilingual (126 MB, covers 9 langs)
  - unknown locale → English fallback with explicit warning
- Subsequent toggles are quiet — `ttsOnboardingShown` flag persists in
  config so we don't spam the same hint.

### New — `/voice-speak-info` and `/voice-speak-models` commands

- `/voice-speak-info` — diagnostic that prints backend, model, voice,
  language, install state, sample rate, voice catalog size, and a
  quick command reference. Mirrors `/voice test` for STT.
- `/voice-speak-models` — opens settings panel directly on the Speak
  tab. Faster path than `/voice-settings → ←→`.

### New — Auto-speak after assistant turns

- `ttsAutoSpeak: true` now actually fires. Subscribes to
  pi-coding-agent's `turn_end` event; pipes the assistant's text
  content through the same `prepareForSpeech()` filter the manual
  command uses (code blocks dropped, ANSI escapes stripped, markdown
  links collapsed, length capped at 2000 chars).
- **Rate limit**: max one auto-speak per 3 seconds. Rapid-fire short
  responses won't queue up unread audio.
- **Mic-feedback guard**: skipped when STT is recording or finalizing.
- **Abort on next user input**: integrates with the existing
  `activeSpeak` AbortController so starting a new message immediately
  cancels in-flight playback.

### New — Download integrity + resume

- `ensureTtsModelInstalled()` now downloads the archive to disk first
  (`~/.pi/models/tts/<id>.partial.tar.bz2`), then extracts. Interrupted
  downloads resume via `Range: bytes=N-` header on retry; falls back
  to full re-download if the server returns 200.
- SHA-256 streamed during write and compared against the catalog's
  `archiveSha256` field (when set). v7.0.0 ships catalog without
  pinned hashes; the verification path is built and the computed hash
  is returned alongside install completion so v7.1+ can pin it.
- Disk-space pre-check via existing `getFreeDiskSpace`.

### New — `tts-text-filter.ts` module

- `prepareForSpeech(text, opts)` — handles the auto-speak preprocessing.
  Drops fenced code blocks (`\`\`\`...\`\`\`` and `~~~...~~~`),
  strips ANSI escapes (CSI + OSC sequences), collapses markdown link
  syntax `[text](url)` → `text`, drops image syntax `![alt](url)`
  entirely, normalizes whitespace, enforces length cap. Returns
  `{ skipped, text, reason, stats }` so callers can surface skip
  reasons in `/voice-speak-info`.
- `lightNormalize(text)` — minimal trim-and-collapse for the manual
  `/voice-speak <text>` path. The user typed exactly what they want
  spoken; we don't second-guess.
- `normalizeBCP47(tag)` — single canonical form for language tags
  used everywhere. Properly handles language / script / region /
  variant per RFC 5646 (lowercase lang, Title-case script,
  uppercase region, lowercase variant). `zh-Hant-TW` round-trips.
- `baseLanguage(tag)` — extract base lang from any BCP-47 form.

### New — Engine warmup hook + per-model threadpool tuning

- `warmupTts(model, dir)` — best-effort background load of the
  sherpa-onnx engine + OfflineTts construction. Cuts the user's first
  `/voice-speak` from 600-900ms cold-start to ~50ms (the cache hit
  path). Best-effort: errors don't surface — they'd surface again on
  the next real synthesize() anyway.
- `getTtsThreads(slot)` now per-class tuned: kokoro scales to 6
  threads on M-series Pro/Max, kitten/vits cap at 4. Mirrors the
  TRANSDUCER_MAX_THREADS=6 logic from the v5.0.9 STT release.

### Configuration schema (additive — v6 configs load unchanged)

- `ttsOnboardingShown` (bool, default false) — gates the first-run hint.

### Architecture

- **New module**: `voice/tts-text-filter.ts` (auto-speak preprocessing)
- **New module**: `voice/tts-onboarding.ts` (first-run hint flow)
- **Catalog gains** `recommendDefaultModel(locale)`, `getTtsModelDir()`,
  `getInstalledTtsModelDir()`, `isTtsModelInstalled()`, refactored
  install pipeline with download-then-extract + Range resume.
- **Engine gains** `warmupTts(model, dir)` and per-slot threadpool
  tuning.
- **Settings panel gains** two new sub-pickers (`tts-model-picker`,
  `tts-voice-picker`) reusing the existing `lang-picker` chassis.
- **voice.ts gains** `turn_end` subscription for auto-speak.

### Verification

- `bunx tsc -p tsconfig.json --noEmit` clean against pi-coding-agent 0.70.5
- `bun test` — 171/171 passing (41 new tests covering text filter
  regression cases, BCP-47 normalization including zh-Hant-TW,
  smart-default selector across all locales, ANSI/code-block stripping)
- godspeed multi-model review on plan (8/8 SHIP) + per-step gates
  (mostly 6-8 SHIP; the recurring "JS thread race" reviewer noise
  consistently advisor-cleared as INVALID per the v6 pattern)
- Real `pi 0.70.5` RPC smoke: extension loads, all 11 commands
  register, lifecycle handlers fire cleanly
- End-to-end audio test in this session: ✅ confirmed `/voice-speak-test`
  produces audible output via Kitten Nano on M3 Pro

### Known v7.1 follow-ups (deferred from the plan to keep scope tight)

- Streaming local playback (ffplay stdin pipe for sub-350ms TTFB —
  current temp-WAV path is ~700ms)
- Real-pi integration test in CI with stubbed afplay
- Inline voice-picker preview ("press P on a voice to hear a sample")
- Download progress UI improvements (sticky progress bar in panel
  rather than transient notify lines)

## [6.0.0] - 2026-04-28

**Major release — pi-listen is now bidirectional voice for Pi.** Voice in
(hold-to-talk STT, unchanged from v5.x) plus voice out (manual `/voice-speak`
TTS, opt-in). The package's `description` and tagline expand to reflect
that — major bump signals the new capability.

### New — Local TTS engine (offline, sherpa-onnx)

- **12 local TTS models in the catalog**, downloaded on demand:
  - **Kitten Nano v0.2 fp16** (default, 25.4 MB, English, 8 voices,
    Apache-2.0) — smallest viable English TTS, sub-real-time on M-series
  - **Piper voices**, ~20 MB each: en_US-lessac, en_US-amy,
    en_US-libritts_r (904 voices), es_ES, fr_FR, de_DE, hi_IN, pt_BR,
    zh_CN, it_IT, ru_RU, ar_JO, tr_TR, nl_NL — MIT licensed
  - **Kokoro v1.0 multilingual int8** (~126 MB) — 9 languages
    (en/zh/ja/ko/es/fr/hi/it/pt) with 53 voices, opt-in for users who
    prefer one-download-fits-all over per-language Piper voices
  - **Kokoro en v0.19 int8** (~99 MB) — English HQ alternative with 11
    voices and the best prosody in the catalog
- **Region-strict language matching** — `pt-PT` cannot silently route to
  a Brazilian voice. Mismatches surface as actionable errors.
- **Sentence-aware text chunking** via `Intl.Segmenter`, with a word-window
  fallback. Locked regression cases for `Dr. Smith`, `e.g.`, `v2.0`,
  `U.S.A.`, URLs, and decimal numbers — none break sentence boundaries.

### New — Cloud TTS (Deepgram REST `/v1/speak`)

- 12 Aura voices surfaced in the picker (asteria, luna, stella, athena,
  hera, orion, arcas, perseus, angus, orpheus, helios, zeus). Custom
  Aura-2 voice ids that aren't in the catalog are accepted on faith and
  validated server-side.
- Reuses the existing `DEEPGRAM_API_KEY` configured for STT — one key
  drives both directions of the voice loop.
- Response size guarded at 75 MB (~26 minutes of 24 kHz PCM) to defend
  against runaway error pages or misconfigured accounts.
- Streaming WebSocket TTS (`wss://api.deepgram.com/v1/speak`) is gated
  behind `ttsDeepgramStreaming` for v6.1.

### New — Slash commands

- `/voice-speak <text>` — synthesize and play
- `/voice-speak-stop` — abort in-flight playback
- `/voice-speak-toggle` — flip `ttsEnabled`
- `/voice-speak-test` — synthesize "The quick brown fox..."

### New — Settings panel "Speak" tab

- Toggle TTS enabled, swap backend (Local ↔ Deepgram), cycle speed
  (0.5x → 2.0x), and run a test synthesis from inside `/voice-settings`.
- Voice selection in v6.0 is by editing config (`ttsLocalVoiceId` numeric
  for local, `ttsDeepgramVoiceId` string for Deepgram). Inline picker
  comes in v6.1.

### Architecture

- New `voice/sherpa-loader.ts` extracted from `sherpa-engine.ts` so STT
  and TTS share a single-flight native module load via `??=`.
- New modules: `voice/tts-local-models.ts` (catalog + install pipeline),
  `voice/tts-engine.ts` (sherpa OfflineTts wrapper), `voice/tts-deepgram.ts`
  (REST client), `voice/tts-playback.ts` (cross-platform audio spawn),
  `voice/speak.ts` (orchestrator with backend dispatch).
- Audio playback uses argument-array spawn (no shell) on all platforms;
  Windows passes the WAV path via `$env:PI_SPEAK_PATH` so single quotes
  in TMPDIR can't inject. Files are written `0600` and unlinked in a
  single-ownership `finally` block. `AbortSignal` plumbed through
  Node's native spawn signal option for atomic mid-playback cancellation.
- TTS instance cache uses a single `Map<key, CachedTts | Promise<CachedTts>>`
  keyed by `cacheKey(modelId, modelDir)` — no cross-key races, no
  multi-field lockstep. Per-instance generate serialization via a
  `generateChain` promise prevents overlapping audio from concurrent
  `synthesize()` calls.

### Configuration schema (additive)

- `ttsEnabled` (bool, default false)
- `ttsBackend` ("local" | "deepgram", default "local")
- `ttsLocalModel` (string, default "kitten-nano-en-v0_2")
- `ttsLocalVoiceId` (number, type-validated at load)
- `ttsDeepgramVoiceId` (string, type-validated at load)
- `ttsSpeed` (number, clamped 0.5–2.0)
- `ttsAutoSpeak` (bool, reserved for v6.1)
- `ttsLanguage` (BCP-47 string, optional)
- `ttsDeepgramStreaming` (bool, v6.1 feature flag)

Existing v5 configs load unchanged with TTS disabled.

### Verification

- `bunx tsc -p tsconfig.json --noEmit` clean against pi-coding-agent 0.70.5
- `bun test` 130/130 passing (51 new TTS tests covering catalog shape,
  Deepgram URL/voice/language validation, sentence chunking regression
  cases, WAV encoder edge cases including NaN/Infinity, and abort
  signaling)
- Module-load smoke (mock pi): all 9 commands register, lifecycle handlers
  fire cleanly, gates behave as documented
- Real `pi 0.70.5` RPC smoke: extension loads, voice status updates emit
- godspeed multi-model review run on every step (Steps 0-9, plus full-bundle
  release gate). Many noise-floor false positives on the engine's
  concurrency code consistently advisor-cleared as INVALID by Opus 4.7
  (advisor confirms: Node single-threaded, run-to-completion). Substantive
  reviewers (security, runtime, sonnet, moonshot, glm) consistently
  SHIP'd.

### Known gaps for v6.1

- Streaming local playback (currently temp-WAV + spawn — adds ~50-100ms
  latency over a streaming pipe)
- `ttsAutoSpeak` hook firing after each assistant turn (the config field
  exists but is unwired in v6.0 — manual `/voice-speak <text>` is the
  primary entry point)
- Inline voice picker in the Speak tab (v6.0 ships read-only display)

## [5.1.0] - 2026-04-28

### Changed (Settings panel UX overhaul)

The `/voice-settings` panel got a structural cleanup focused on the Models
tab. The previous flat 19-model list with mixed status glyphs and duplicated
ratings made it hard to scan; the new layout groups by family and shows a
single source of truth per row.

- **Models tab now groups by family.** Top picks for the current device
  appear first (fitness-recommended, capped at 4), followed by Parakeet,
  Whisper, Moonshine, and Specialist sections. Each section has a heading
  and short subtitle (e.g. "OpenAI · 57 languages") so the user knows what
  they're scanning before reading model names. Search filters across all
  groups; empty groups are dropped so the user never sees an orphan heading.
- **Cleaner row layout.** Right-aligned size column, language-coverage hint
  inline ("57 langs", "English", "zh/en/ja/ko", "Russian"), single status
  cell on the right ("active" / "ready" / fitness label). The redundant
  inline `●●●●○/●●●●○` ratings dropped — accuracy/speed bars now appear only
  once, on the expanded selected row.
- **Theme-aware colors.** The panel now uses the host `Theme` from
  `ctx.ui.custom()` for accent, success, warning, error, and dim colors.
  Catppuccin Mocha, Solarized, and other non-default themes render
  correctly. Falls back to raw ANSI when no theme is provided so unit
  tests keep working.
- **Two-step delete on the Downloaded tab.** Pressing `x` once arms the
  delete with a 1.5s confirmation window — the row shows
  `press x again to delete`. A second `x` within the window commits;
  any other navigation aborts. Whisper Large is 1.8 GB; a single stray
  keypress should not nuke a multi-minute download.
- **Enter hint is contextual.** The footer hint on the Models tab now
  reads `↵ activate` for already-downloaded models or
  `↵ download (1.8 GB) + activate` for fresh ones, so users know what
  they're committing to before pressing Enter.
- **Tab key works as alias for `→`** to advance tabs. `←→` still works.
- **Tab bar visual polish.** Active tab is bold + accent without bracket
  noise; tabs separated by `·` instead of being mashed together.
- **Device tab "Disk space" line shows fits-largest-model check.**
  Example: `45.2 GB free (largest model needs 1.8 GB ✓)` so users can
  pre-flight a download.
- **Render cache removed.** Was keyed only on width; mutating any of
  tab / row / search / sub-picker / delete-pending state would have
  served stale frames. The panel renders ~12-30 lines per frame —
  uncached is well within the budget.

### Internal
- New `groupModels()` helper synthesizes the family grouping from the
  flat `LOCAL_MODELS` catalog. Models can appear under both "Top picks"
  and their family group; intentional — top picks is the fast path,
  family is the comparison path.
- `formatLangHint(model)` and `formatFitness(fitness)` extracted as
  pure helpers for the Models tab row layout.
- `PanelDeps` gains an optional `theme?: Theme`; `voice.ts:openSettingsPanel`
  now constructs the panel inside the `ctx.ui.custom()` callback so the
  host theme is in scope when the panel is built.
- Imports `Theme` and `ThemeColor` from `@mariozechner/pi-coding-agent`'s
  public surface.

### Verification
- `bunx tsc -p tsconfig.json --noEmit` — clean against pi-coding-agent 0.70.5
- `bun test` — 79/79 passing
- Real `pi 0.70.5` RPC smoke: panel constructs, status bar populates, clean teardown
- godspeed multi-model review: 6/8 SHIP (threshold 6), 0 NO_SHIP-VETO
  (the two NO_SHIPs flagged removed code that doesn't exist in the diff —
  reviewer false positives)

## [5.0.9] - 2026-04-28

### Performance
- **Parakeet TDT v3 transcription latency cut by 30-50% on Apple Silicon
  Pro/Max** — `getNumThreads()` was capped at 4 for every model class. On
  modern M-series chips (10+ performance cores) that left more than half the
  P-cores idle for transducer-style ASR. The cap is now per-model-class:
  Parakeet (and other transducer models) get up to 6 threads, while Whisper /
  SenseVoice / NeMo CTC stay at 4 where their decoder shape doesn't benefit
  from more.
- **Why not CoreML?** Tested and rejected. Sherpa-onnx ships the CoreML
  execution provider in its bundled `libonnxruntime.dylib`, but for transducer
  / transformer ASR graphs CoreML currently regresses by ~10% on M2 Max
  (sherpa-onnx [#2910](https://github.com/k2-fsa/sherpa-onnx/issues/2910) —
  RTF 0.470 CoreML vs 0.427 CPU). Revisit when the partition-aware CoreML EP
  lands.
- Tuning citations: [sherpa-onnx NeMo transducer RTF
  table](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/nemo-transducer-models.html),
  [#2910 CoreML regression](https://github.com/k2-fsa/sherpa-onnx/issues/2910).

### Internal
- Added `TRANSDUCER_MAX_THREADS = 6` and made `getNumThreads(maxThreads = 4)`
  parameterized so future model classes can declare their own threadpool
  budget without forking the helper.
- Inline tuning notes added to `createTransducerRecognizer` documenting the
  CPU-vs-CoreML decision and the thread-cap rationale.

## [5.0.8] - 2026-04-28

### Changed
- **Peer dependency floor raised to `>=0.70.0`** for both
  `@mariozechner/pi-coding-agent` and `@mariozechner/pi-tui`. Pi-mono 0.65.0
  removed `session_switch` and added `event.reason` metadata to
  `session_start` / `session_shutdown`; pi-tui 0.61.0 renamed
  `getEditorKeybindings` → `getKeybindings` and namespaced action ids
  (`selectUp` → `tui.select.up`). Older Pi hosts can no longer install this
  extension. If you're on Pi < 0.70, run `pi update` first, then reinstall
  pi-listen.

### Fixed
- **Session lifecycle now actually honors `event.reason`** — `session_start`
  extracts the reason (compat-narrowed against pre-0.65 typings), runs setup
  wiring on every transition, and gates the first-run install hint on
  `reason === "startup"`. The v5.0.7 changelog claimed this was implemented
  but the code did not match; this entry closes that gap.
- **`session_shutdown` ordering hardened** — `ctx = null` now runs synchronously
  before any `await`, and the sherpa recognizer cache is cleared only on
  `reason === "quit" | undefined` to avoid racing the recognizer init in the
  replacement session on `/new`, `/fork`, `/resume`. `voiceCleanup()` is wrapped
  in try/catch in all lifecycle handlers so a single child-process kill EPERM
  cannot leak ctx or skip the cache clear.
- **Stale local-transcription callbacks neutralized** — `abortSession()`
  replaces `onTranscript` / `onDone` / `onError` with no-ops before the
  backend-specific abort, preventing post-abort sherpa transcription or late
  WebSocket events from writing into a replacement session's editor or firing
  notifications on the new ctx.
- **`initSherpa()` re-entrancy** — concurrent callers now share a single
  in-flight promise via `initPromise ??= doInitSherpa()` (atomic claim under
  JS run-to-completion). The cached promise is released in `finally` once
  `sherpaInitialized` flips, so the synchronous fast-path serves every later
  caller. Previously two callers arriving on the same tick could both run
  platform checks and re-import the native module.
- **Status bar refresh on session_start when voice is disabled** —
  `updateVoiceStatus()` now runs on every transition (even when
  `config.enabled === false`), clearing the status entry instead of leaving
  stale `MIC STREAM` text from the prior session.

### Removed
- **Legacy `session_switch` handler** — pi-mono 0.65.0 dropped the event and
  the new peer floor (`>=0.70.0`) makes the shim unreachable. Cleanup is now
  handled entirely via `session_shutdown` → `session_start` (with `reason`).

### Internal
- **Drop `as any` on `pi.registerShortcut`** — replaced with `as KeyId`
  assertion; `isValidShortcut()` validates the runtime config string at load
  time, so the assertion documents intent instead of hiding type info.
- **`pi-tui` keybinding API migration** — `getEditorKeybindings` →
  `getKeybindings`, action ids updated to namespaced form
  (`selectUp` → `tui.select.up`, `selectConfirm` → `tui.select.confirm`, etc.).
- **Dev dependencies pinned** — `@mariozechner/pi-coding-agent` and
  `@mariozechner/pi-tui` now appear in `devDependencies` at `^0.70.5` so local
  typecheck runs against the same API surface the runtime expects.

### Verification
- `bunx tsc -p tsconfig.json --noEmit` — clean against pi-coding-agent 0.70.5
- `bun test` — 79/79 passing (564 expect calls)
- Module-load smoke (mock pi): every `event.reason` value handled cleanly,
  concurrent `initSherpa()` calls return identical results
- Real `pi 0.70.5` RPC smoke: extension loads, `session_start` runs,
  status bar populates, clean teardown
- godspeed multi-model review: 7/7 SHIP, 0 NO_SHIP, 0 VETO

## [5.0.7] - 2026-04-03

### Added
- **Configurable voice toggle shortcut** — the toggle shortcut (default
  `Ctrl+Shift+V`) can now be customized via `toggleShortcut` in global
  `~/.pi/agent/settings.json` under the `"voice"` key. Project-scoped overrides
  are intentionally ignored since Pi registers shortcuts statically at load time.
- **Shortcut validation** — invalid shortcut values are rejected with a warning
  and fall back to the default `ctrl+shift+v`. Validation requires modifier+key
  format (e.g. `ctrl+shift+v`, `alt+r`, `meta+shift+m`).
- **Debug output** — `/voice debug` now shows the configured toggle shortcut.

### Fixed
- **Pi API compatibility: session events** — `session_start` now uses per-event
  `event.reason` detection for new Pi API. Non-startup transitions trigger
  `voiceCleanup()` and sherpa recognizer cache clear before re-initialization.
  Legacy `session_switch` handler preserved for backward compatibility with older
  Pi versions.
- No auth API changes needed — pi-voice does not make direct LLM calls via
  `ModelRegistry`.
## [5.0.5] - 2026-03-26

### Fixed
- **Env-derived Deepgram keys stay runtime-only** — `DEEPGRAM_API_KEY` from the
  shell is no longer copied into `~/.pi/agent/settings.json` during startup or
  first-run auto-activation.
- **Explicit secret saves remain intentional** — keys entered during onboarding
  still go to `~/.env.secrets` or `~/.zshrc`, while runtime env resolution
  continues to take priority over stored config.
- **Report credit** — thanks to [@dvic](https://github.com/dvic) for reporting
  the remaining global config leak.

## [5.0.4] - 2026-03-18

### Fixed
- **Deepgram shutdown waits for server finals** — stream stop now sends
  `CloseStream`, waits for Deepgram to return the last finalized transcript, and
  only falls back to local finalization after a short timeout.
- **Single-fire Deepgram error handling** — streaming failures now close the
  session once, preventing duplicate error notifications and conflicting state
  transitions from `error` plus `close`.
- **Misaligned PCM buffers in sherpa path** — odd-offset `Buffer` inputs now
  fall back to `readInt16LE()` instead of throwing `RangeError` when converting
  audio samples.

### Added
- **Regression test for odd-offset PCM input** — `transcribeBuffer()` is now
  covered for pooled `Buffer` slices with non-2-byte-aligned offsets.

## [5.0.1] - 2026-03-16

### Security
- **API key no longer leaks into project config** — `deepgramApiKey` is stripped at serialization time when saving to project scope. Previously, env-derived API keys could be auto-persisted into `.pi/settings.json` inside repos, risking accidental credential commits.
- **Mic audio exfiltration blocked** — `localEndpoint` in project config is now restricted to loopback addresses only (localhost/127.0.0.1/::1). A malicious repo can no longer redirect microphone audio to a remote server.
- **Shell injection prevented in API key onboarding** — API keys are now escaped using single-quote shell escaping before writing to `~/.env.secrets` or `~/.zshrc`. Keys with embedded newlines are rejected. New secrets files are created with `0600` permissions.

### Fixed
- **Atomic config writes** — settings are now written to a temp file and renamed, preventing corruption from partial writes or concurrent saves.
- **Deleting active model no longer leaves broken config** — when the active local model is deleted from the settings panel, config switches to another downloaded model (or clears the selection) instead of leaving a dangling reference.
- **Timeout timer cleanup** — the 120s transcription timeout in local mode is now properly cleared when transcription finishes early, preventing resource leaks.
- **Config parse errors logged** — `readJsonFile()` now logs warnings to stderr instead of silently swallowing parse/read errors.
- **Inconsistent default model** — settings panel now uses `parakeet-v3` as fallback instead of `whisper-small`, matching `DEFAULT_LOCAL_MODEL`.

### Added
- 19 new regression tests covering secret stripping, endpoint validation, atomic writes, shell escaping, and loopback detection.

## [4.0.0] - 2026-03-14

### Removed
- **All voice commands and text processing** — removed "undo", "clear", "new line", and all other voice commands. Removed all punctuation shortcuts ("period", "comma", etc.). Deleted `text-processing.ts` module entirely. Live streaming transcription writes text to the editor before voice commands can be detected, making them fundamentally unreliable. pi-listen now does one thing well: hold space to record, release to transcribe.

## [3.4.0] - 2026-03-14

### Removed
- **Non-functional voice commands** — Pi's extension API doesn't support triggering keybindings, slash commands, or message submission. Removed all voice commands: session management, model switching, thinking control, display toggles, dev commands, control commands, "hey pi" prefix, and submit/send/stop. Kept editor text manipulation (undo, clear, new line) and dictation shortcuts (punctuation, brackets, symbols).

## [3.3.3] - 2026-03-14

### Added
- **Banner and preview images** — terminal banner in `assets/banner.png`, refreshed docs hero in `docs/images/hero.png`, and social preview card in `docs/images/social-preview.png`

## [3.3.2] - 2026-03-14

### Added
- **Documentation images** — photorealistic terminal hero and voice command screenshots in `docs/images/hero.png` and `docs/images/voice-commands.png`

## [3.3.1] - 2026-03-14

### Added
- **External editor voice command** — say "open editor", "external editor", or "vim" to launch `/editor`

## [3.3.0] - 2026-03-14

### Added
- **Session management commands** — "new session", "compact", "fork", "resume", "tree", "reload", "settings" trigger their `/slash` equivalents
- **Model switching commands** — "switch model", "next model", "previous model", "change to X" for model picker and direct selection
- **Thinking commands** — "cycle thinking", "more thinking", "thinking level" to cycle levels; "show/hide/toggle thinking" for visibility
- **Display commands** — "expand/collapse/show/hide tools" for tool call display
- **Editor commands** — "select all", "clear all" variants
- **Control commands** — "stop", "cancel", "abort" to interrupt the agent
- **Dev commands** — "build", "install", "format", "push", "pull", "show log", "git status", "git diff"
- **7 new punctuation shortcuts** — "hash" (`#`), "at sign" (`@`), "dollar sign" (`$`), "ampersand" (`&`), "percent" (`%`), "asterisk" (`*`), "tab"

## [3.2.0] - 2026-03-14

### Changed
- **Audio capture fallback chain** — no longer requires SoX. Tries `rec` (SoX) → `ffmpeg` → `arecord` (Linux ALSA) in order, uses the first available tool
- ffmpeg uses avfoundation on macOS, pulse on Linux, dshow on Windows
- arecord available as zero-install option on Linux (built into ALSA)
- Audio tool detection result is cached for the process lifetime

## [3.1.3] - 2026-03-14

### Added
- **Pre-recording** — audio capture starts during warmup countdown, never miss the first word
- **Tail recording** — keeps recording 1.5s after release so your last word isn't clipped
- **Reactive waveform** — audio-level-driven 12-bar animation with fast attack / slow decay and center emphasis
- **Typing cooldown** — space holds within 400ms of other keypresses are ignored, preventing false activation mid-sentence
- **Sound feedback** — macOS system sounds (Tink, Pop, Basso) for recording start, stop, and error
- **Session corruption guard** — overlapping recording requests abort the stale session first
- **Recording history** — `/voice history` shows recent transcriptions with timestamps and durations
- **Stale session watchdog** — aborts if Deepgram sends no response after 15s of audio
- **Connection timeout** — aborts if Deepgram WebSocket doesn't open within 10s

### Changed
- Hold threshold increased to 1200ms (from 800ms) for more deliberate activation
- Repeat confirm count increased to 6 (from 3) for more reliable non-Kitty hold detection
- Recording grace period increased to 800ms (from 600ms) to reduce false stops

## [3.0.2] - 2026-03-14

### Added
- **First-run welcome hint** — shows keybinding guide on first session when API key is set, or setup instructions when it's not
- **Zero-config auto-activation** — if `DEEPGRAM_API_KEY` is already in environment, voice activates immediately without running `/voice setup`
- **Deepgram API key validation** — `/voice test` now hits the Deepgram API to verify the key is valid (not just checking if it's set)
- **Full diagnostics output** — `/voice test` shows pass/fail for each prerequisite with actionable setup instructions

## [3.0.0] - 2026-03-14

### Changed
- **Complete rewrite** — Deepgram streaming-only architecture (removed local daemon, 5-backend system, BTW side conversations)
- **Separated Pompom companion** — creature animation now ships as its own extension (`@codexstar/pi-pompom`)
- **Renamed package** — `@codexstar/pi-voice` → `@codexstar/pi-listen`

### Added
- **Double-escape editor clear** — press Escape twice within 500ms to clear the editor text
- **Cross-platform escape handling** — filters Kitty key-release/repeat events to prevent false triggers
- **Voice commands** — "hey pi, run tests", "undo", "submit", "new line", punctuation shortcuts
- **Continuous dictation** — `/voice dictate` for long-form input without holding keys
- **Recording history** — `/voice history` shows recent transcriptions
- **Audio-reactive UI** — braille waveform + face widget that reacts to voice levels
- **Enterprise hold detection** — Kitty protocol + non-Kitty gap-based fallback with typing cooldown

### Removed
- Local STT daemon (`daemon.py`, `transcribe.py`)
- 5-backend system (faster-whisper, moonshine, whisper-cpp, parakeet)
- BTW side conversations
- VAD pre-filtering
- Pompom/Lumo creature companion (now separate package)

[7.0.1]: https://github.com/codexstar69/pi-listen/releases/tag/v7.0.1
[7.0.0]: https://github.com/codexstar69/pi-listen/releases/tag/v7.0.0
[6.0.0]: https://github.com/codexstar69/pi-listen/releases/tag/v6.0.0
[5.1.0]: https://github.com/codexstar69/pi-listen/releases/tag/v5.1.0
[5.0.9]: https://github.com/codexstar69/pi-listen/releases/tag/v5.0.9
[5.0.8]: https://github.com/codexstar69/pi-listen/releases/tag/v5.0.8
[5.0.7]: https://github.com/codexstar69/pi-listen/releases/tag/v5.0.7
[5.0.5]: https://github.com/codexstar69/pi-listen/releases/tag/v5.0.5
[5.0.1]: https://github.com/codexstar69/pi-listen/releases/tag/v5.0.1
[5.0.4]: https://github.com/codexstar69/pi-listen/releases/tag/v5.0.4
[4.0.0]: https://github.com/codexstar69/pi-listen/releases/tag/v4.0.0
[3.4.0]: https://github.com/codexstar69/pi-listen/releases/tag/v3.4.0
[3.3.1]: https://github.com/codexstar69/pi-listen/releases/tag/v3.3.1
[3.3.3]: https://github.com/codexstar69/pi-listen/releases/tag/v3.3.3
[3.3.2]: https://github.com/codexstar69/pi-listen/releases/tag/v3.3.2
[3.3.0]: https://github.com/codexstar69/pi-listen/releases/tag/v3.3.0
[3.2.0]: https://github.com/codexstar69/pi-listen/releases/tag/v3.2.0
[3.1.3]: https://github.com/codexstar69/pi-listen/releases/tag/v3.1.3
[3.0.2]: https://github.com/codexstar69/pi-listen/releases/tag/v3.0.2
[3.0.0]: https://github.com/codexstar69/pi-listen/releases/tag/v3.0.0

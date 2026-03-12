# pi-voice QA results

This document records the interactive/manual QA evidence gathered after the onboarding overhaul and the follow-up model-aware onboarding work.

## Verification baseline

Fresh automated verification completed successfully before and after manual QA checks:

```sh
bun run check
```

That run currently covers:
- config migration / scope tests
- onboarding fallback / finalization tests
- diagnostics recommendation tests
- provisioning-plan tests
- model-detection metadata tests
- model-aware recommendation and labeling tests
- model-readiness confidence tests (`installed` / `download required` / `unknown` / `api`)
- TypeScript compilation
- Python compilation

## Manual / RPC-assisted checks completed

The checks below were run against the real Pi RPC mode and the current `extensions/voice.ts` implementation.

### 1. Fresh install prompts for onboarding on startup
**Result:** pass

Observed startup UI requests for a clean HOME / clean cwd:

```json
{
  "case": "fresh-first-run",
  "titles": [
    "setStatus",
    "Set up pi-voice now?"
  ],
  "selectTitle": "Set up pi-voice now?",
  "options": [
    "Start voice setup",
    "Remind me later"
  ]
}
```

### 2. Partial legacy config still re-enters onboarding
**Result:** pass

Seeded legacy config:

```json
{ "voice": { "enabled": true } }
```

Observed startup prompt:

```json
{
  "case": "partial-legacy",
  "selectTitle": "Set up pi-voice now?",
  "options": [
    "Start voice setup",
    "Remind me later"
  ]
}
```

This confirms partial legacy config no longer counts as fully onboarded.

### 3. Remind-me-later suppresses startup prompt and `/voice reconfigure` reopens setup
**Result:** pass

Seeded config with recent `onboarding.skippedAt` and observed no startup onboarding prompt before sending `/voice reconfigure`.

Observed UI requests:

```json
{
  "case": "skipped-then-reconfigure",
  "titles": [
    "setStatus",
    "How do you want to use speech-to-text?"
  ]
}
```

This confirms the defer window suppresses startup prompting while keeping reconfiguration available.

### 4. Reconfigure flow saves project-scoped config
**Result:** pass

Executed `/voice reconfigure` in RPC mode and selected:
- Cloud API
- Deepgram
- `nova-3`
- Project scope

Observed notify + saved config:

```json
{
  "case": "project-scope-save",
  "notify": "Voice setup saved, but action is still required.\nMode: Cloud API\nBackend: deepgram\nModel: nova-3\nScope: project\n...",
  "savedVoice": {
    "version": 2,
    "enabled": true,
    "language": "en",
    "mode": "api",
    "backend": "deepgram",
    "model": "nova-3",
    "scope": "project",
    "btwEnabled": true,
    "onboarding": {
      "completed": false,
      "schemaVersion": 2,
      "source": "repair"
    }
  }
}
```

This confirms:
- reconfigure launches the onboarding flow
- project scope writes to `.pi/settings.json`
- incomplete provisioning/validation leaves onboarding in repair-needed state rather than falsely complete

### 5. Local path still exposes backend choices with install hints
**Result:** pass

Observed local backend selection options:

```json
{
  "case": "local-fallback-options",
  "backendOptions": [
    "faster-whisper — available",
    "moonshine — pip install useful-moonshine[onnx]",
    "whisper-cpp — brew install whisper-cpp",
    "parakeet — pip install nemo_toolkit[asr]"
  ]
}
```

This confirms the local flow does not dead-end when only some backends are installed and still surfaces install guidance.

### 6. `/voice doctor` separates current repair from recommended alternative
**Result:** pass

Observed output:

```json
{
  "case": "voice-doctor",
  "notify": "Voice doctor:\n  python3: OK\n  sox/rec: missing\n  brew: OK\n  deepgram key: missing\n  available backends: faster-whisper (local)\n\nCurrent config: api/deepgram/nova-3\nRepair current setup:\n  - brew install sox\n  - Set DEEPGRAM_API_KEY before using Deepgram API mode\n\nRecommended alternative: local/faster-whisper/small\nWhy: Recommended local default with good balance of quality and setup effort.\nFixable issues:\n  - Install SoX for microphone recording\nSuggested commands for the recommendation:\n  - brew install sox"
}
```

This confirms doctor output now distinguishes:
- how to repair the current saved config
- what the system recommends instead

### 7. Model-aware backend metadata is now exposed from `transcribe.py`
**Result:** pass

Observed `--list-backends` metadata snippet:

```json
[
  {
    "name": "faster-whisper",
    "installed_models": [],
    "install_detection": "huggingface-cache"
  },
  {
    "name": "moonshine",
    "installed_models": [],
    "install_detection": "moonshine-cache-heuristic"
  },
  {
    "name": "whisper-cpp",
    "installed_models": [],
    "install_detection": "whisper-cpp-model-paths"
  },
  {
    "name": "deepgram",
    "installed_models": [],
    "install_detection": "api-key"
  },
  {
    "name": "parakeet",
    "installed_models": [],
    "install_detection": "huggingface-or-nemo-cache"
  }
]
```

This confirms the backend scan now exposes machine-readable model-readiness metadata, even when no models are currently detected on disk.

### 8. Local reconfigure path stays model-aware even when no local model is currently cached
**Result:** pass

Observed reconfigure options in the current environment:

```json
{
  "backendOptions": [
    "faster-whisper — backend ready",
    "moonshine — pip install useful-moonshine[onnx]",
    "whisper-cpp — brew install whisper-cpp",
    "parakeet — pip install nemo_toolkit[asr]"
  ],
  "modelOptions": [
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small (recommended)",
    "small.en",
    "medium",
    "medium.en",
    "large-v3",
    "large-v3-turbo",
    "distil-small.en",
    "distil-medium.en",
    "distil-large-v3"
  ]
}
```

This confirms:
- model-aware metadata does not break the local onboarding path when no cached models are found
- the user still gets a sensible backend list and model selection flow
- the recommended model remains visible even without an installed-model hit

### 9. Model-aware logic covers both installed-model and missing-model paths
**Result:** pass (automated coverage), release-machine confirmation still recommended

Automated model-aware checks currently cover:
- installed-model recommendation preference
- installed-model labeling in onboarding
- model readiness states for local vs cloud backends
- provisioning behavior when backend is available but selected model is not installed
- transcribe metadata contract including `installed_models` and `install_detection`

This gives regression safety for the model-aware phase even though the current QA machine does not have local STT model caches populated.

### 10. `/voice info` surfaces current model readiness and scope details
**Result:** pass

Observed output:

```text
Voice config:
  enabled:  true
  mode:     local
  scope:    global
  backend:  faster-whisper
  model:    small
  model status: download required
  language: en
  state:    idle
  setup:    complete (setup-command)
  socket:   /tmp/.../pi-voice-<hash>.sock
  daemon:   stopped
```

This confirms `info` now reports:
- selected mode/backend/model
- model readiness state
- scope
- config-scoped socket path
- setup state

### 11. `/voice backends` surfaces model-aware backend summaries
**Result:** pass

Observed output:

```text
Backends:
+ faster-whisper   local  no confirmed installed models
    detection: huggingface-cache
- moonshine        local  install: pip install useful-moonshine[onnx]
    detection: moonshine-cache-heuristic
- whisper-cpp      local  install: brew install whisper-cpp
    detection: whisper-cpp-model-paths
- deepgram         cloud  needs setup: Set DEEPGRAM_API_KEY env var (free: deepgram.com)
    detection: api-key
- parakeet         local  install: pip install nemo_toolkit[asr]
    detection: huggingface-or-nemo-cache
```

This confirms `backends` now distinguishes:
- installed models when present
- no confirmed installed models
- API readiness / setup-needed cloud wording
- install detection source hints

### 12. `/voice test` reports current model status plus missing-model guidance
**Result:** pass

Observed output:

```text
Voice test:
  mode: local
  backend: faster-whisper
  model: small
  model status: download required
  language: en
  onboarding: complete
  python3: OK
  sox/rec: missing
  daemon: not running

Suggested commands:
  - brew install sox

Manual steps:
  - Selected model small is not installed yet and may need to be downloaded on first use
```

This confirms `test` now combines:
- current model readiness
- recording dependency status
- targeted install/manual guidance for the selected model path

## Remaining target-machine checks

The highest-value interactive checks have been exercised. Remaining follow-up checks are mainly environment- or hardware-specific, especially for the new model-aware behavior.

### Strongly recommended on the target release machine
- real microphone capture with user audio input
- real Deepgram end-to-end API validation with a valid key
- local STT success path on a machine with SoX installed and the desired backend available
- at least one path where a local model is **already cached** so onboarding can display an explicit installed-model path (`already installed`, `ready now`, or equivalent)
- at least one path where a backend is installed but the selected model is **not** cached, to confirm the download-required messaging is clear
- `/voice info` and `/voice test` on a machine with real model caches, so the displayed model status can be compared against actual local assets

### Why these are still worth running
The current environment used for QA did not have local STT model caches populated, so the model-aware logic was verified through:
- automated tests
- backend metadata checks
- RPC-assisted onboarding checks

That is sufficient for regression safety and release evidence, but a final pass on a machine with real cached local models would provide the strongest product validation for the installed-model UX.

## Release signoff

Based on:
- `bun run check`
- the manual/RPC-assisted onboarding and repair checks above
- the model-detection metadata and model-aware onboarding checks documented here

`pi-voice` now satisfies the current release-hardening checklist for the onboarding overhaul and the initial model-aware onboarding upgrade, with the remaining checks clearly isolated to target-machine / real-audio validation.

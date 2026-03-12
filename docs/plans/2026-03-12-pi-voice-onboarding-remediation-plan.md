# pi-voice onboarding remediation plan

## Objective

Turn `pi-voice` from a promising power-user extension into a world-class Pi package with a polished, enterprise-grade first-run experience.

Because Pi packages do not appear to offer a dedicated interactive install hook, the package should treat **first interactive `session_start`** as the onboarding moment. Installation should become:

1. package installed,
2. first Pi session starts,
3. extension detects that voice is not configured,
4. extension launches a guided terminal onboarding flow,
5. extension provisions or validates dependencies,
6. extension runs a verification check,
7. extension leaves the user in a known-good state.

## Current-state gaps

### UX gaps
- No first-run onboarding exists today. `extensions/voice.ts` simply loads config and starts the daemon on `session_start`.
- `/voice setup` is a narrow backend/model picker, not a complete onboarding journey.
- The current setup flow does **not** start with the user’s real decision: **cloud API vs local download**.
- There is no capability scan summary, no privacy/cost explanation, no model recommendation, no install progress UI, and no success confirmation flow.
- The user must already know to run `/voice setup`, install SoX manually, and understand backend tradeoffs.

### Product gaps
- Configuration is too thin: `enabled`, `language`, `backend`, and `model` are not enough for a polished product.
- No onboarding versioning or migration marker exists, so the extension cannot safely re-run setup when UX changes.
- No distinction exists between:
  - onboarding completed,
  - dependencies missing,
  - misconfigured backend,
  - backend intentionally cloud/local,
  - user skipped setup.
- The package has no README, install guide, troubleshooting guide, or compatibility matrix.

### Technical gaps
- `session_start` eagerly calls `ensureDaemon(config)` before setup is confirmed.
- The daemon socket is global (`/tmp/pi-voice-daemon.sock`), which can cause collisions across projects or multiple Pi sessions.
- `transcribeAudio()` sends daemon requests without explicitly passing `backend`/`model`, so a pre-existing daemon may keep using stale settings.
- `/voice setup` writes only global settings today; it does not offer global vs project scope.
- There is no installer orchestration layer for Python packages, Homebrew packages, env vars, or model warmup.
- There is no structured diagnostics engine for “what is broken and how do we fix it automatically?”

## Target experience

On the first interactive session after install, the user should see a clean onboarding flow with these steps:

1. **Welcome**
   - What pi-voice does.
   - What will be configured.
   - Approximate time.

2. **Choose operating mode**
   - **API mode**: fastest setup, cloud transcription, API key required.
   - **Download mode**: local/private, dependency installation, model choice required.

3. **Smart recommendation**
   - Detect OS, Python, Homebrew, CPU, network, existing packages, env vars.
   - Recommend a default backend/model with a one-line reason.

4. **Guided configuration**
   - API path: provider, API key detection/input, model choice, privacy/cost note.
   - Local path: backend choice, model size, latency/accuracy tradeoff, disk/CPU expectations.

5. **Provisioning**
   - Install missing dependencies automatically where possible.
   - Show cancellable progress UI.
   - Offer fallback commands if auto-install is not allowed.

6. **Validation**
   - Mic check.
   - Backend check.
   - Optional test recording + transcription.

7. **Completion**
   - Show ready state, configured mode, shortcuts, and a “re-run setup” command.

That should feel like a polished product setup wizard, not a developer-only slash command.

## Proposed architecture

### 1) Versioned voice config schema
Expand the current `voice` config into a versioned schema with explicit onboarding state.

Suggested shape:

```json
{
  "voice": {
    "version": 2,
    "enabled": true,
    "language": "en",
    "mode": "local",
    "backend": "faster-whisper",
    "model": "small",
    "scope": "global",
    "onboarding": {
      "completed": true,
      "completedAt": "2026-03-12T...Z",
      "schemaVersion": 2,
      "source": "first-run",
      "lastValidatedAt": "2026-03-12T...Z"
    },
    "local": {
      "autoInstall": true,
      "warmDaemonOnStart": true
    },
    "cloud": {
      "provider": "deepgram",
      "apiKeyEnv": "DEEPGRAM_API_KEY"
    }
  }
}
```

This gives the extension enough structure to support migrations, validation, and enterprise-grade UX.

### 2) Separate concerns inside the extension
Refactor the single large `extensions/voice.ts` file into focused modules.

Recommended structure:

- `extensions/voice.ts` or `extensions/voice/index.ts` — thin entrypoint
- `extensions/voice/config.ts` — schema, loading, saving, migration, scope handling
- `extensions/voice/onboarding.ts` — first-run flow orchestration
- `extensions/voice/ui.ts` — reusable TUI components and option lists
- `extensions/voice/diagnostics.ts` — environment scan and validation
- `extensions/voice/install.ts` — dependency provisioning and command runners
- `extensions/voice/runtime.ts` — recording, transcription, daemon integration
- `extensions/voice/btw.ts` — BTW side-conversation feature

This is required before polishing UX; the current single-file layout is too dense for safe iteration.

### 3) Capability scan + recommendation engine
Build a diagnostic layer that returns a structured environment report before onboarding makes recommendations.

Inputs:
- `python3` availability
- SoX / `rec`
- Homebrew availability
- current backend availability from `transcribe.py --list-backends`
- env vars like `DEEPGRAM_API_KEY`
- maybe available RAM / CPU class / Apple Silicon detection

Outputs:
- recommended mode
- recommended backend
- recommended model
- blocking issues
- fixable issues
- commands required for remediation

### 4) First-run onboarding trigger
On `session_start`, the extension should:
- load and migrate config,
- skip onboarding in non-interactive mode,
- skip onboarding if already completed and still valid,
- otherwise show onboarding UI,
- only start the daemon after configuration is valid.

This replaces the current eager daemon startup behavior.

## Implementation phases

### Phase 0 — stabilize current behavior
Goal: remove correctness risks before adding more UX.

Changes:
- Pass `backend` and `model` explicitly in daemon transcribe requests.
- Ensure daemon config reloads when saved config differs from loaded daemon state.
- Stop using one hard-coded global socket path; derive a safer user/session-specific socket path.
- Prevent automatic daemon startup before onboarding/validation succeeds.

Files:
- `extensions/voice.ts`
- `daemon.py`

Priority: **P0**

### Phase 1 — config and diagnostics foundation
Goal: create the substrate required for polished onboarding.

Changes:
- Introduce versioned config schema and migration helpers.
- Add a diagnostics engine that returns structured install/readiness state.
- Add save helpers for both global and project scope.
- Record onboarding completion/version.

Files:
- new `extensions/voice/config.ts`
- new `extensions/voice/diagnostics.ts`
- trim `extensions/voice.ts`

Priority: **P0**

### Phase 2 — world-class onboarding TUI
Goal: deliver the actual guided experience.

Flow:
- welcome screen
- choose **API** or **Download**
- show recommendation card
- choose backend/provider
- choose model/profile
- choose config scope (global vs project)
- confirm summary

Implementation notes:
- Use `ctx.ui.custom()` with `SelectList`, bordered layout, and clear footer hints.
- Keep a consistent visual system across steps.
- Add timeout-free, deliberate, keyboard-first navigation.
- Support “Back”, “Cancel”, and “Skip for now”.
- If skipped, set explicit incomplete state and show a persistent lightweight reminder, not a blocking modal every turn.

Files:
- new `extensions/voice/onboarding.ts`
- new `extensions/voice/ui.ts`
- `extensions/voice.ts`

Priority: **P0**

### Phase 3 — provisioning engine
Goal: make setup actually succeed, not just save preferences.

API mode:
- detect existing `DEEPGRAM_API_KEY`
- if absent, ask for it or provide precise setup instructions
- validate with a lightweight connectivity/provider check

Download mode:
- detect/install SoX
- detect/install Python package dependencies per backend
- offer backend-specific installation plans
- optionally pre-load model or warm daemon
- show progress and explicit failures with recovery suggestions

Important principle:
- Enterprise-grade means **automatic where safe, explicit where not**.
- If auto-install fails, the extension should capture stderr and render an actionable next-step screen.

Files:
- new `extensions/voice/install.ts`
- `transcribe.py`
- `daemon.py`
- `package.json` for scripts and packaging metadata

Priority: **P1**

### Phase 4 — post-onboarding runtime polish
Goal: make the configured experience feel premium every day.

Changes:
- Better status text than `MIC` / `REC` / `STT...`; include backend or mode when helpful.
- Add `/voice doctor` for full diagnostics and auto-fix suggestions.
- Add `/voice reconfigure` as the canonical onboarding re-entry point.
- Add `/voice test` upgrade: record, transcribe, and summarize readiness in one view.
- Improve empty-editor hold-to-talk discoverability with a first-success toast and help text.
- Persist last-known-good daemon/backend status.

Files:
- `extensions/voice.ts`
- `extensions/voice/runtime.ts`
- `extensions/voice/diagnostics.ts`

Priority: **P1**

### Phase 5 — package, docs, and release hardening
Goal: make the package credible and supportable.

Changes:
- Add `README.md` with:
  - value proposition
  - first-run behavior
  - cloud vs local comparison
  - supported backends/models
  - install requirements
  - troubleshooting
- Add `docs/` pages for backend matrix and operational behavior.
- Add screenshots/GIF for package gallery metadata.
- Update `package.json` metadata and files list to include docs/README.
- Add changelog and release checklist.

Files:
- new `README.md`
- new `docs/backends.md`
- new `docs/troubleshooting.md`
- `package.json`

Priority: **P1**

## File-level change list

### `extensions/voice.ts`
- Reduce to extension wiring and command registration.
- Call config migration and onboarding gate on `session_start`.
- Defer daemon startup until readiness confirmed.
- Route `/voice setup` to the new onboarding flow.
- Add `/voice doctor` and `/voice reconfigure`.

### `daemon.py`
- Support explicit backend/model consistency checks.
- Consider richer status payload for UI/doctor flow.
- Use safer socket naming strategy.

### `transcribe.py`
- Expand backend metadata with recommendation hints, install strategy, and maybe estimated resource tiers.
- Add machine-readable diagnostics output beyond `--list-backends` if needed.

### `package.json`
- Add `README.md` to `files`.
- Add scripts for `typecheck`, Python smoke validation, and tests.
- Potentially add gallery image/video metadata.

### New docs
- `README.md`
- `docs/backends.md`
- `docs/troubleshooting.md`
- `docs/plans/...` (this file)

## Testing strategy

### Automated
TypeScript:
- config migration tests
- onboarding decision logic tests
- diagnostics parser tests
- command routing tests

Python:
- backend metadata smoke tests
- daemon command contract tests
- config-specific load/transcribe path tests

Integration:
- first-run with no config
- first-run with API key present
- first-run with local backend already installed
- skipped onboarding then reconfigure
- stale daemon with mismatched model

### Manual verification matrix
- macOS Apple Silicon with faster-whisper available
- no SoX installed
- no Python backend installed
- Deepgram API key present / absent
- project-local scope vs global scope
- interactive TUI mode vs non-interactive mode

### Acceptance criteria
- Fresh install leads to a guided flow without requiring `/voice setup`.
- User is explicitly asked **API or download** before backend/model details.
- At least one happy path completes end-to-end and verifies with a real test transcription.
- Failed provisioning produces actionable remediation, not silent degradation.
- Re-running setup is safe and deterministic.
- Docs are sufficient for a new user to understand tradeoffs before enabling voice.

## Rollout priorities

### Must ship first
1. P0 stabilization fixes
2. Config schema + diagnostics foundation
3. First-run onboarding with API vs download decision

### Should ship next
4. Automated provisioning
5. `/voice doctor` and upgraded validation flow
6. README + troubleshooting docs

### Nice-to-have after that
7. More providers than Deepgram
8. richer model recommendations based on hardware
9. onboarding screenshots/video for package gallery

## Recommended execution order

1. Refactor for modularity.
2. Fix daemon/config correctness issues.
3. Add diagnostics layer.
4. Add first-run onboarding gate.
5. Add install/provisioning actions.
6. Add validation flow.
7. Add docs and release hardening.

## Final recommendation

Do **not** start with visual polish alone. The right order is:

**correctness -> diagnostics -> onboarding -> provisioning -> docs**.

If those are shipped in that order, `pi-voice` can feel like a premium enterprise extension instead of an expert-only prototype.

# pi-voice world-class onboarding master plan

## Context

This master plan consolidates the current audit of `pi-voice` and the initial team-agent planning pass into a single end-to-end execution plan.

Goal: turn `pi-voice` into a polished Pi package with a premium first-run onboarding experience that asks the user how they want to use STT:

- **Cloud API** vs **Local download**
- if local: **which backend/model**
- then provisions, validates, and leaves the user in a known-good state

Because Pi packages do not appear to expose a dedicated interactive install hook, the product should treat **first interactive `session_start` after install** as the onboarding moment.

---

## Product outcome

A user should be able to:

1. `pi install ...`
2. start Pi normally
3. get a guided terminal onboarding wizard automatically on first run
4. choose **API** or **Local**
5. get a recommendation with tradeoff explanations
6. install or validate the required dependencies
7. run a real mic + transcription test
8. finish with a success receipt and confidence that voice is working

Re-entry should always be available via `/voice setup` or `/voice reconfigure`.

---

## Current-state problems to fix

### UX
- No first-run onboarding; setup is hidden behind `/voice setup`
- Current setup is a thin backend/model picker, not a product workflow
- No API-vs-local decision at the top of the flow
- No recommendation engine, progress UI, recovery UX, or completion receipt
- No scope selection (global vs project)

### Technical correctness
- Daemon can keep stale backend/model config across sessions
- Single hard-coded socket path risks cross-session/project collisions
- Setup writes only global settings despite reading project + global settings
- Session startup eagerly tries to start the daemon before configuration is known-good

### Packaging/productization
- No README or troubleshooting docs
- No backend matrix or install guide
- Minimal package metadata
- No onboarding state versioning or migration strategy

---

## Guiding principles

1. **Correctness before polish** — runtime behavior must match saved settings every time
2. **Onboarding is a product feature** — not an afterthought command
3. **Automatic where safe, explicit where necessary**
4. **Explain tradeoffs in user language** — privacy, speed, quality, setup effort
5. **Recovery is part of UX** — failed setup must produce actionable next steps
6. **Reconfiguration must be safe** — switching mode/model should be deterministic

---

## Target user journeys

### Journey A — fastest setup
- User installs package
- First session starts
- Wizard recommends **API mode**
- Detects API key presence or guides user to add it
- Runs validation
- Finishes in under 2 minutes

### Journey B — privacy/offline-first
- User installs package
- Wizard recommends **Local mode**
- Explains backend options and suggests a default model
- Installs SoX + Python/backend dependencies where possible
- Warms daemon/model
- Runs local transcription test

### Journey C — project-specific configuration
- User wants different voice config in one repo
- Wizard offers **Use everywhere** or **Only in this project**
- Saves to the correct settings file
- Status and runtime behavior reflect the selected scope

### Journey D — recovery
- Existing user has broken deps, stale daemon, or invalid config
- Startup detects mismatch
- User sees a non-destructive repair flow
- `/voice doctor` and `/voice reconfigure` provide guided recovery

---

## Milestone plan

## Milestone 1 — runtime correctness foundation

### Objective
Eliminate stale-config and lifecycle issues before adding a more ambitious UX layer.

### Deliverables
- Runtime always uses the selected backend/model/language
- Daemon startup is gated by valid config
- Socket strategy is safer than one global hard-coded path
- Config persistence supports both global and project scopes

### File changes

#### `extensions/voice.ts`
- Stop treating current in-memory config as sufficient truth
- Pass `backend`, `model`, and `language` in daemon transcription requests
- Add explicit runtime reload behavior when daemon config differs from desired config
- Remove unconditional eager warm-start from `session_start` until config readiness is known
- Split load/save logic into scope-aware helpers

#### `daemon.py`
- Return richer daemon status including active backend/model and config fingerprint
- Support safe reload semantics if requested config changes
- Move away from one universal socket path toward a config-aware or user-scoped socket strategy

#### `transcribe.py`
- Keep backend registry stable and machine-readable
- Ensure metadata needed by diagnostics/onboarding is available consistently

### Acceptance criteria
- Switching backend/model in config always affects the next transcription
- Existing running daemon cannot silently keep old settings
- Project-level config can override global config cleanly
- No daemon starts on first run unless onboarding/config permits it

### Validation
- `bunx tsc -p tsconfig.json`
- `python3 -m py_compile daemon.py transcribe.py`
- Manual test: start daemon on one config, switch to another config, verify transcription uses the new backend/model

---

## Milestone 2 — config schema + onboarding state

### Objective
Add a durable, versioned configuration model that supports migrations and first-run logic.

### Deliverables
- Versioned voice config schema
- Explicit onboarding state model
- Migration path from current simple config
- Scope-aware persistence API

### Proposed config shape

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
    "btwEnabled": true,
    "onboarding": {
      "completed": true,
      "schemaVersion": 2,
      "completedAt": "2026-03-12T00:00:00.000Z",
      "lastValidatedAt": "2026-03-12T00:00:00.000Z",
      "source": "first-run"
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

### New modules
- `extensions/voice/config.ts`
- `extensions/voice/types.ts`

### Responsibilities
- `loadConfigWithSource(cwd)`
- `saveConfig(config, scope, cwd)`
- `migrateLegacyConfig(config)`
- `needsOnboarding(config)`
- `needsRepair(config, diagnostics)`

### Acceptance criteria
- Legacy config continues to work
- New users get a complete state model
- Incomplete onboarding can be distinguished from valid onboarding
- Scope of settings is explicit and reversible

---

## Milestone 3 — diagnostics and recommendation engine

### Objective
Create the intelligence layer that powers onboarding, repair, and doctor flows.

### Deliverables
- Structured environment scan
- Recommendation engine
- Blocking vs fixable issue classification
- User-facing explanations for each recommendation

### New module
- `extensions/voice/diagnostics.ts`

### Inputs to scan
- `python3`
- SoX / `rec`
- Homebrew availability
- backend availability from `transcribe.py --list-backends`
- env vars like `DEEPGRAM_API_KEY`
- architecture / OS hints where useful
- daemon status

### Outputs
- recommended mode (`api` | `local`)
- recommended backend/model
- blockers
- auto-fixable issues
- manual remediation instructions

### UX requirements
The engine should answer questions such as:
- “You want privacy and faster-whisper is already available, so local is recommended.”
- “You want the fastest setup and no local stack is installed, so Deepgram is recommended.”
- “SoX is missing; local setup can continue after installing it.”

### Acceptance criteria
- Recommendation logic is deterministic
- Recommendation is explainable in one or two lines
- All blocking issues are surfaced before provisioning starts

---

## Milestone 4 — premium onboarding wizard

### Objective
Replace today’s setup picker with a full first-run guided terminal flow.

### Deliverables
- Automatic first-run onboarding on interactive session start
- Re-runnable setup via `/voice setup` and `/voice reconfigure`
- Multi-step keyboard-first TUI
- Summary + completion receipt

### New modules
- `extensions/voice/onboarding.ts`
- `extensions/voice/ui.ts`

### UX flow

#### Step 1 — Welcome
- What pi-voice does
- What the setup will configure
- Estimated time
- Option to continue or skip for now

#### Step 2 — How do you want to use STT?
- **API mode** — fastest, cloud-based, API key required
- **Download mode** — local/private, more setup, works offline
- **Recommended** — decide for me

#### Step 3 — What matters most?
- setup speed
- privacy/offline use
- best accuracy
- low resource usage

#### Step 4 — Recommendation card
- recommended mode/backend/model
- why it is recommended
- alternatives and tradeoffs

#### Step 5A — API branch
- choose provider
- detect API key or prompt for setup path
- choose model
- explain privacy/cost implications

#### Step 5B — Local branch
- choose backend
- choose model size/profile
- explain speed/accuracy/setup implications
- show expected dependencies

#### Step 6 — Scope
- **Use everywhere**
- **Only in this project**

#### Step 7 — Confirm summary
- mode
- backend/model
- scope
- dependencies to be installed or validated

#### Step 8 — Provision + progress
- install/prepare dependencies
- show progress + live status
- allow graceful cancel

#### Step 9 — Validation
- mic check
- sample recording
- sample transcription
- show actual transcript result

#### Step 10 — Completion receipt
- success state
- selected config summary
- shortcuts
- commands: `/voice doctor`, `/voice reconfigure`

### Behavior requirements
- support Back, Cancel, Skip for now
- if skipped, do not hard-block every session; use a lightweight reminder state
- non-interactive mode must skip wizard safely

### Acceptance criteria
- Fresh install triggers onboarding automatically on first interactive session
- User is explicitly asked **API or Local** before backend details
- User can complete setup without prior knowledge of STT backends
- Successful path ends with real verification, not just config save

---

## Milestone 5 — provisioning and repair engine

### Objective
Make the selected onboarding path actually succeed.

### Deliverables
- Automatic dependency provisioning where safe
- Robust failure capture and recovery guidance
- Repair mode for broken existing setups

### New module
- `extensions/voice/install.ts`

### API path requirements
- detect `DEEPGRAM_API_KEY`
- validate provider availability
- fail with clear next-step guidance if key is absent or invalid

### Local path requirements
- detect/install SoX
- detect/install backend-specific Python packages or Homebrew dependencies
- warm or load the selected model/backend
- surface stderr in a clean error screen if install fails

### Safety rules
- automatic install should be opt-in or clearly confirmed
- if auto-install is not allowed, present exact commands and explain what each does
- installation failures should not leave ambiguous config behind

### Repair flow
Add `/voice doctor` with:
- diagnostics summary
- fix suggestions
- rerun test flow
- optional entry into reconfigure flow

### Acceptance criteria
- API happy path works end-to-end
- Local happy path works end-to-end
- Failure states provide precise remediation
- Repair flow can recover from at least one broken dependency scenario

---

## Milestone 6 — daily runtime polish

### Objective
Make the post-onboarding experience feel premium, not just functional.

### Deliverables
- polished status line behavior
- better notifications
- improved test and doctor output
- clearer voice runtime observability

### Improvements
- Replace bare `MIC`, `REC`, `STT...` with more informative statuses where helpful
- Optionally show configured mode/backend in status or info flow
- Improve first-success notification after the user completes setup
- Refine `/voice info`, `/voice test`, `/voice doctor`, `/voice daemon status`
- Persist last-known-good validation timestamp

### Acceptance criteria
- User can understand current voice state quickly
- Runtime messages feel calm and professional
- Debugging common issues does not require code reading

---

## Milestone 7 — package hardening and docs

### Objective
Make the package installable, understandable, and supportable like a serious product.

### Deliverables
- `README.md`
- backend matrix doc
- troubleshooting doc
- improved package metadata
- release checklist

### Files
- `README.md`
- `docs/backends.md`
- `docs/troubleshooting.md`
- `package.json`

### README contents
- what pi-voice does
- what happens on first run
- cloud vs local comparison
- supported backends/models
- shortcuts and commands
- troubleshooting starter guide

### `package.json` improvements
- richer description
- homepage/repository/bugs metadata if available
- add docs to `files`
- add scripts for typecheck and Python smoke checks
- add peer dependency declarations for Pi runtime packages if appropriate for publishing

### Acceptance criteria
- New user can understand the product from README alone
- Package metadata is good enough for distribution and discovery
- Troubleshooting docs cover at least the common install/runtime failures

---

## Milestone 8 — QA, migration, and release

### Objective
Ship safely and prove the experience works across common scenarios.

### Deliverables
- automated smoke checks
- manual QA matrix
- onboarding migration behavior
- release readiness checklist

### Automated checks
- `bunx tsc -p tsconfig.json`
- `python3 -m py_compile daemon.py transcribe.py`
- TypeScript tests for config migration and recommendation logic
- Python tests or smoke scripts for daemon contract and backend metadata

### Manual QA matrix
1. fresh install, no config, no backends
2. fresh install, local backend already present
3. fresh install, API key present
4. fresh install, API key absent
5. local install path with missing SoX
6. project-scope config
7. global-scope config
8. stale daemon scenario
9. reconfigure from API -> Local
10. reconfigure from Local -> API
11. skipped onboarding -> later resume
12. non-interactive session start

### Definition of done
- First-run onboarding is automatic in interactive mode
- API vs Local decision is the top-level onboarding choice
- At least one API happy path and one Local happy path work end-to-end
- Existing legacy users migrate safely
- Runtime uses the saved config deterministically
- Docs and package metadata are production-ready

---

## Work breakdown by file

### `extensions/voice.ts`
- shrink to orchestration/registration layer
- startup gating
- command registration
- shortcut wiring

### `extensions/voice/config.ts`
- schema
- migrations
- scope-aware load/save
- onboarding state helpers

### `extensions/voice/types.ts`
- shared types for config, diagnostics, recommendations, onboarding steps

### `extensions/voice/diagnostics.ts`
- environment scan
- recommendation engine
- doctor report data

### `extensions/voice/onboarding.ts`
- wizard controller
- first-run flow
- reconfigure flow

### `extensions/voice/ui.ts`
- reusable TUI components and rendering helpers
- summary cards, progress screens, completion receipt

### `extensions/voice/install.ts`
- provisioning actions
- command execution wrappers
- result parsing

### `extensions/voice/runtime.ts`
- daemon communication
- recording/transcription logic
- runtime correctness helpers

### `extensions/voice/btw.ts`
- extract BTW feature for separation of concerns

### `daemon.py`
- explicit config consistency
- status improvements
- safer socket strategy

### `transcribe.py`
- richer backend metadata
- stable diagnostics contract

### `README.md`, `docs/backends.md`, `docs/troubleshooting.md`
- user-facing package documentation

---

## Execution priority

### P0 — critical
1. Runtime correctness and daemon staleness fixes
2. Scope-aware config persistence
3. Versioned config + onboarding state
4. Diagnostics foundation
5. First-run onboarding gate

### P1 — major
6. Full onboarding wizard
7. Provisioning engine
8. `/voice doctor` + repair flow
9. Package docs and metadata

### P2 — polish
10. Runtime polish and premium messaging
11. Gallery/demo assets
12. richer backend recommendation heuristics

---

## Risks and mitigations

### Risk: install automation becomes too platform-specific
**Mitigation:** keep provisioning modular, detect capabilities first, auto-install only where safe, always provide manual fallback commands.

### Risk: onboarding becomes too invasive
**Mitigation:** run only for first interactive session or invalid config, support skip/remind later, never block non-interactive usage.

### Risk: daemon complexity increases during refactor
**Mitigation:** ship Milestone 1 correctness improvements before UI expansion, and keep daemon protocol explicit.

### Risk: local backend ecosystem is inconsistent
**Mitigation:** use recommendation logic to steer users toward known-good defaults and document tradeoffs clearly.

---

## Suggested shipping order

### Release 1
- correctness fixes
- config schema + migration
- diagnostics foundation
- first-run onboarding skeleton with API vs Local choice

### Release 2
- full wizard
- provisioning engine
- validation flow
- `/voice doctor`

### Release 3
- docs hardening
- premium UI polish
- rollout tuning and edge-case cleanup

---

## Final recommendation

The right sequence is:

**correctness -> config/migration -> diagnostics -> onboarding -> provisioning -> validation -> docs -> polish**

If executed in that order, `pi-voice` can move from a promising expert-only extension to a world-class, enterprise-grade Pi package with a first-run experience that feels intentional, reliable, and premium.

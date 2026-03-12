# pi-voice model-aware onboarding execution plan

## Goal

Upgrade `pi-voice` from backend-aware onboarding to **model-aware onboarding** so the system can:

- detect when a backend is installed
- detect when a specific model is already available locally
- prefer already-installed local models when reasonable
- avoid unnecessary redownload/setup prompts
- clearly separate:
  - ready now
  - backend installed but model missing
  - download required
  - cloud/API path

This plan builds on the onboarding/runtime foundation already implemented in the repo.

---

## Current status

### Already complete in repo
- first-run onboarding on interactive `session_start`
- API vs Local decision flow
- backend/model/scope selection
- config migration and onboarding state
- daemon correctness fixes
- `/voice doctor` and `/voice reconfigure`
- runtime/info/test improvements
- docs + QA artifacts + automated tests

### Started in this phase
- added model-readiness metadata to `transcribe.py --list-backends`
- added model-aware recommendation tests and provisioning tests
- started wiring installed-model preferences into TypeScript diagnostics/onboarding

---

## Workstreams

## Workstream A — detection contract (Python)

### Objective
Expose per-model readiness metadata from backend discovery.

### Deliverables
- `installed_models` in `--list-backends` JSON output
- `install_detection` metadata for each backend
- conservative detection heuristics by backend

### Files
- `transcribe.py`

### Status
- **in progress / partially implemented**

### Backend strategy
- **faster-whisper**: Hugging Face cache repo detection
- **whisper-cpp**: direct model file path detection
- **deepgram**: API-based, no local model download
- **moonshine**: heuristic / low-confidence cache detection
- **parakeet**: Hugging Face or NeMo cache heuristic

### Acceptance criteria
- `python3 transcribe.py --list-backends` includes per-backend `installed_models`
- missing caches do not crash detection
- conservative heuristics prefer underclaiming over false positives

---

## Workstream B — diagnostics and recommendation engine (TypeScript)

### Objective
Consume the richer Python metadata and prefer already-installed models when appropriate.

### Deliverables
- extended `BackendAvailability` type
- recommendation logic that prefers ready-now local models
- support helpers for installed-model queries

### Files
- `extensions/voice/diagnostics.ts`

### Status
- **in progress / partially implemented**

### Acceptance criteria
- recommendation changes when installed-model metadata changes
- recommendation reasons mention already-installed models where relevant
- tests cover installed vs missing model cases

---

## Workstream C — onboarding UX

### Objective
Surface installed-model awareness in backend and model selection.

### Deliverables
- backend labels like `2 installed models`
- model labels like:
  - `small (recommended, installed)`
  - `medium (installed)`
  - `large-v3-turbo`
- summary copy that tells the user whether the path is ready now or still requires setup

### Files
- `extensions/voice/onboarding.ts`

### Status
- **in progress / partially implemented**

### Acceptance criteria
- onboarding makes existing assets visible
- user can still override recommendations
- local path does not feel like a forced redownload when models already exist

---

## Workstream D — provisioning short-circuiting

### Objective
Avoid unnecessary install guidance when selected model is already available.

### Deliverables
- provisioning plan becomes model-aware
- ready state only when backend/model/recording prerequisites are satisfied
- missing-model path is distinguished from missing-backend path

### Files
- `extensions/voice/install.ts`
- `extensions/voice.ts`

### Status
- **in progress / partially implemented**

### Acceptance criteria
- installed selected model => no redundant install guidance
- backend installed but model missing => targeted guidance
- doctor/test/info reflect the distinction accurately

---

## Workstream E — product surfaces and docs

### Objective
Make the new behavior visible everywhere users will look.

### Deliverables
- `/voice backends` includes installed-model context where useful
- `/voice doctor`, `/voice test`, `/voice info` show model readiness
- docs updated to explain installed-model detection behavior and caveats

### Files
- `extensions/voice.ts`
- `README.md`
- `docs/backends.md`
- `docs/troubleshooting.md`
- `docs/qa-results.md`

### Status
- **not started / partial prep done**

---

## Workstream F — QA and release hardening

### Objective
Validate model-aware onboarding end-to-end.

### Deliverables
- updated QA matrix/results
- automated test coverage for model-aware logic
- manual/RPC-assisted smoke evidence for:
  - fresh onboarding
  - partial legacy config
  - remind-me-later + reconfigure
  - project-scope save
  - installed-model path
  - missing-model path

### Files
- `tests/transcribe-metadata.test.ts`
- `tests/model-selection.test.ts`
- `tests/install.test.ts`
- `docs/qa-results.md`

### Status
- **in progress**

---

## Execution order

### Phase 1 — finish Python detection contract
1. finalize `installed_models` heuristics
2. verify `--list-backends` shape
3. keep conservative fallback behavior

### Phase 2 — finish TypeScript integration
4. complete installed-model-aware recommendation logic
5. finish onboarding labels and summary copy
6. finish provisioning short-circuit logic

### Phase 3 — surface behavior in commands/docs
7. upgrade `/voice backends`, `/voice doctor`, `/voice test`, `/voice info`
8. update docs and QA results

### Phase 4 — final verification
9. run `bun run check`
10. run manual/RPC-assisted smoke checks for installed-model-aware flows

---

## Verification commands

### Automated
- `bun run check`
- `python3 transcribe.py --list-backends`
- targeted Bun tests:
  - `bun test tests/transcribe-metadata.test.ts`
  - `bun test tests/model-selection.test.ts`
  - `bun test tests/install.test.ts`

### Manual / RPC-assisted
- fresh first-run onboarding prompt
- local path shows installed/missing model labels
- reconfigure path saves project scope correctly
- doctor separates current repair from recommendation

---

## Team usage

### Teammates already assigned
- `plan-lead` → remaining implementation plan
- `model-researcher` → backend/model detection heuristics
- `qa-lead` → QA/release checklist

### How to use teammates during implementation
- use a teammate for docs refresh while main code work continues
- use a teammate for QA artifact drafting once command outputs stabilize
- use a teammate reviewer pass before final signoff

---

## Definition of done

This phase is done when:
- onboarding explicitly shows when a model is already available
- recommendation logic prefers installed models where sensible
- provisioning skips redundant install guidance for already-ready paths
- doctor/test/info/backends reflect model-aware state accurately
- docs explain the behavior and caveats
- verification passes and QA evidence is captured

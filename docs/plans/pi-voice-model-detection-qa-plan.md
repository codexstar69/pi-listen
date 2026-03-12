# pi-voice model-detection QA / release execution checklist

## Objective

Validate the next onboarding iteration where `pi-voice` can detect already-available models, prefer ready-to-use local setups when appropriate, and clearly distinguish:
- **already installed / ready now**
- **backend installed but model missing**
- **download required**
- **cloud/API path**

This checklist focuses on release confidence for **model-detection-aware onboarding**, not just the current onboarding baseline.

---

## Release gates

A release is not ready until all four gates pass.

### Gate 1 — Static / automated checks
- [ ] `bun run check`
- [ ] model-detection unit tests pass
- [ ] onboarding recommendation tests pass with installed-model scenarios
- [ ] provisioning-plan tests pass with installed vs missing model scenarios
- [ ] any new Python smoke checks for model discovery pass

### Gate 2 — Onboarding behavior
- [ ] first-run onboarding still launches correctly
- [ ] onboarding clearly marks installed models as available immediately
- [ ] onboarding does not ask users to re-download models already present
- [ ] onboarding still offers alternative models/backends when installed assets exist
- [ ] onboarding summary accurately reflects whether validation is complete or repair is still required

### Gate 3 — Runtime correctness
- [ ] selected installed model is actually used at runtime
- [ ] model-detection state does not get out of sync with daemon/runtime state
- [ ] project scope vs global scope still works correctly after model-aware setup
- [ ] `/voice info`, `/voice test`, and `/voice doctor` reflect installed-model state accurately

### Gate 4 — Docs / support readiness
- [ ] README explains installed-model detection behavior
- [ ] backend docs explain when cached/existing models may be reused
- [ ] troubleshooting docs include “backend installed but model missing” and “model detected but validation failed” cases
- [ ] QA evidence is captured in `docs/qa-results.md`

---

## Test matrix

## A. Fresh install / no cached models

### A1. Fresh startup, no config, no local model assets
- [ ] onboarding prompt appears
- [ ] user can choose API or Local
- [ ] local path marks all local options as requiring install/download
- [ ] doctor output does not falsely claim any model is already available

### A2. Local mode with nothing installed
- [ ] onboarding still offers backend choices
- [ ] install guidance appears for backend + model path
- [ ] completion state remains `repair` / incomplete until validation succeeds

### A3. API mode with no key
- [ ] Deepgram path clearly says API key is missing
- [ ] onboarding does not mislabel API mode as “ready now”

---

## B. Existing installed local model paths

### B1. Backend installed, model already cached
Example target: `faster-whisper` backend available and chosen model already present.
- [ ] onboarding highlights the model as **already installed** or equivalent
- [ ] recommendation prefers the installed model when it matches user goals reasonably well
- [ ] provisioning does not suggest re-downloading that same model
- [ ] summary says the model is ready for immediate configuration
- [ ] runtime validation uses the installed model successfully

### B2. whisper.cpp backend installed with model file already present
- [ ] onboarding can identify existing whisper.cpp model file
- [ ] model is marked available without requiring re-download
- [ ] runtime uses the located model path successfully
- [ ] doctor output reports the model as found, not merely backend available

### B3. Multiple installed local models
- [ ] onboarding distinguishes between multiple installed models
- [ ] recommended option is sensible and clearly justified
- [ ] user can override recommendation and choose a different installed model
- [ ] saving the non-default installed model persists correctly

### B4. Installed local model but missing SoX
- [ ] onboarding correctly says model is ready but recording path still needs SoX
- [ ] summary distinguishes **model ready** vs **recording dependency missing**
- [ ] repair state is used instead of complete state until validation passes

---

## C. Backend installed, model missing

### C1. Backend available but requested model not present
Example: `faster-whisper` available, `medium` not downloaded.
- [ ] onboarding marks backend as installed but selected model as **download required**
- [ ] recommendation may prefer an already installed smaller model if appropriate
- [ ] provisioning suggests only the missing model path, not a full backend reinstall
- [ ] summary and doctor explain the difference clearly

### C2. whisper.cpp installed but no model file found
- [ ] onboarding does not say whisper.cpp is fully ready
- [ ] onboarding explains that model files are missing
- [ ] doctor separates “install backend” from “obtain model file” if backend already exists

### C3. Partial cache / corrupted model asset
- [ ] discovery does not falsely mark model as ready if validation fails
- [ ] onboarding/doctor route user to repair path
- [ ] runtime does not mark onboarding complete after failed validation

---

## D. Cloud/API branch

### D1. Cloud path with valid key
- [ ] API mode remains the fastest setup path
- [ ] onboarding does not incorrectly prefer stale local detection over an explicitly selected API choice
- [ ] completion state becomes complete after validation succeeds

### D2. Cloud path with installed local model also present
- [ ] recommendation explains the tradeoff clearly
- [ ] API path remains selectable even when local is ready
- [ ] selected API mode is respected and saved
- [ ] doctor distinguishes current API config from local recommended alternative

---

## E. Migration and persistence

### E1. Existing legacy config + installed local model
- [ ] migration does not skip onboarding incorrectly for partial legacy configs
- [ ] onboarding can suggest the already installed model immediately
- [ ] saved config includes the correct versioned onboarding state

### E2. Reconfigure from API -> installed local model
- [ ] `/voice reconfigure` detects the local installed model
- [ ] reconfigure flow offers the installed model without download guidance
- [ ] config updates correctly
- [ ] runtime/doctor/info reflect the new local state

### E3. Reconfigure from local -> API
- [ ] existing local detection does not block switching to API
- [ ] cloud config saves cleanly
- [ ] doctor still reports local assets accurately as alternatives

### E4. Scope behavior
- [ ] global save still writes global settings
- [ ] project save still writes `.pi/settings.json`
- [ ] project-level model-aware config overrides global config cleanly
- [ ] local installed-model detection still behaves correctly under either scope

---

## F. Runtime and daemon regression checks

### F1. Config-scoped socket behavior
- [ ] switching between projects/scopes/backends/models does not silently reuse stale daemon state
- [ ] already installed model in one scope does not cause another scope to falsely appear ready unless it truly is

### F2. `/voice info`
- [ ] reports mode/backend/model/scope accurately
- [ ] if model-detection metadata is added, it reports installed vs missing status accurately

### F3. `/voice test`
- [ ] reports installed-model readiness accurately
- [ ] does not imply success when model is missing
- [ ] still exercises mic sample flow correctly

### F4. `/voice doctor`
- [ ] shows current-config repair path first
- [ ] shows recommended alternative separately
- [ ] reports model-ready vs model-missing status clearly

### F5. Hold-to-talk regression
- [ ] hold `Space` still works when editor is empty
- [ ] `Ctrl+Shift+V` fallback still works
- [ ] `Ctrl+Shift+B` BTW voice path still works

---

## Suggested execution order

1. **Automated checks**
   - unit tests for model detection and recommendation changes
   - `bun run check`
2. **Fresh install / no model path**
3. **Installed local model happy path**
4. **Backend installed / model missing path**
5. **API branch with and without local alternatives**
6. **Migration + reconfigure paths**
7. **Daemon/runtime regressions**
8. **Docs verification and QA result capture**

---

## Evidence to capture

For each major scenario, capture at least one of:
- JSON/RPC output snippet
- screenshot of onboarding step
- saved config snippet
- `/voice doctor` output
- `/voice info` or `/voice test` output

Recommended artifact locations:
- `docs/qa-results.md` for pass/fail summaries
- optional raw snippets under `docs/qa-artifacts/`

---

## Signoff criteria

Model-detection-aware onboarding is release-ready when all are true:
- [ ] already-installed models are surfaced correctly in onboarding
- [ ] onboarding avoids unnecessary re-download guidance
- [ ] backend-installed/model-missing states are clearly explained
- [ ] API/local branches both remain understandable and correct
- [ ] migration + reconfigure paths remain safe
- [ ] runtime/daemon behavior matches selected config
- [ ] docs reflect the new behavior
- [ ] QA evidence is recorded

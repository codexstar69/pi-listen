# pi-voice remaining implementation plan

## Purpose

This plan covers the next stage of `pi-voice` after the onboarding overhaul already shipped in code.

The remaining product goal is to make onboarding and repair flows **cache-aware and model-aware**, so users are told when a backend/model is already available on their machine and can configure it immediately without unnecessary download or setup prompts.

This plan assumes the current repo already has:
- first-run onboarding on interactive `session_start`
- API vs Local mode selection
- scope-aware config saving
- diagnostics and recommendations
- `/voice doctor` and `/voice reconfigure`
- provisioning-plan helpers
- runtime correctness fixes
- docs and automated coverage

## Outcome

A user installing or reconfiguring `pi-voice` should get a setup flow that can say:
- "You already have `faster-whisper` installed"
- "You already have model `small` ready to use"
- "This model would require download"
- "We can configure the existing model now"
- "We recommend this already-installed path over a fresh download"

The final UX should reduce unnecessary installation friction and make setup feel smarter and more premium.

---

## Current gaps

### Diagnostics gaps
- `extensions/voice/diagnostics.ts` currently detects **backend availability**, but not **per-model availability**.
- The diagnostics contract only exposes `available`, `type`, `default_model`, and `models`.
- There is no distinction between:
  - backend package installed but model not cached
  - backend package installed and a specific model already present
  - backend package installed but model presence unknown

### Onboarding gaps
- `extensions/voice/onboarding.ts` can recommend a backend/model, but it cannot tell the user whether that model is already present.
- Backend choices show only `available` / install-hint style suffixes, not model-ready states.
- Model selection treats all models the same regardless of whether they are already downloaded.

### Provisioning gaps
- `extensions/voice/install.ts` can build install/manual steps for missing dependencies, but it cannot short-circuit when a selected model is already present.
- `/voice doctor` can distinguish repair-vs-recommendation flows, but not model-ready-vs-download-required states.

### Docs/QA gaps
- Current docs describe backend-level setup well enough, but not installed-model detection behavior.
- QA docs capture onboarding flows and repair states, but not model-aware recommendations and no-redownload paths.

---

## Scope of this plan

This plan covers:
1. installed-model detection
2. cache-aware recommendations
3. onboarding UX for already-installed models
4. provisioning short-circuiting
5. doctor/test/info updates
6. docs updates
7. QA and release validation

This plan does **not** require redesigning the entire onboarding architecture again.

---

## Guiding principles

1. **Backend-aware and model-aware are different.** The product must surface both.
2. **Prefer ready-now flows.** If a good local model already exists, recommend using it.
3. **Do not overclaim certainty.** Model detection should support `installed`, `not_installed`, and `unknown` states.
4. **Stay machine-readable first.** Expose detection in `transcribe.py` first, then consume it in TypeScript.
5. **Preserve current correctness guarantees.** No model-awareness feature should regress config/daemon behavior.

---

## Proposed milestones

## Milestone A — model detection contract

### Objective
Extend backend discovery so the system can report model-level readiness, not just backend-level readiness.

### Deliverables
- machine-readable model presence detection in `transcribe.py`
- a stable JSON contract consumed by TypeScript diagnostics
- confidence-aware detection rules per backend

### File changes

#### `transcribe.py`
Add support for richer backend metadata in `--list-backends` output, for example:

```json
{
  "name": "faster-whisper",
  "available": true,
  "type": "local",
  "default_model": "small",
  "models": [
    {
      "id": "small",
      "status": "installed",
      "source": "huggingface-cache",
      "path": "~/.cache/..."
    },
    {
      "id": "medium",
      "status": "unknown"
    }
  ]
}
```

If fully nested model objects are too disruptive, use an additive field such as:
- `model_statuses`
- `installed_models`
- `model_presence`

Recommended shape:

```json
{
  "models": ["small", "medium", "large-v3-turbo"],
  "model_presence": {
    "small": { "status": "installed", "source": "cache", "path": "/..." },
    "medium": { "status": "not_installed" },
    "large-v3-turbo": { "status": "unknown" }
  }
}
```

### Detection strategy by backend

#### faster-whisper
Detect using best-effort heuristics such as:
- Hugging Face cache directories under `~/.cache/huggingface/`, `~/Library/Caches/huggingface/`, or environment-driven locations if practical
- any CTranslate2/converter output cache locations if they are predictably discoverable

Confidence note:
- `installed` should only be returned when a concrete model directory/file can be found with confidence
- otherwise prefer `unknown` over a false `not_installed`

#### whisper-cpp
Detect by checking common model file locations already partially used by the runtime:
- `~/.cache/whisper-cpp/ggml-<model>.bin`
- `/opt/homebrew/share/whisper-cpp/models/ggml-<model>.bin`
- `/usr/local/share/whisper-cpp/models/ggml-<model>.bin`

This backend has the most reliable model-file detection path.

#### moonshine
If a predictable cache path cannot be verified safely, use:
- backend installed = true/false
- model presence = `unknown`

#### parakeet
If NeMo/Hugging Face cache layout is not easy to detect safely, use:
- backend installed = true/false
- model presence = `unknown`

#### deepgram
No local model download applies.
Use:
- `status = api`
- or omit model presence and treat cloud models separately in TypeScript

### Acceptance criteria
- `python3 transcribe.py --list-backends` returns model-level metadata
- model presence detection does not crash when caches are absent
- unknown states are used conservatively when certainty is low

### Verification
- `python3 transcribe.py --list-backends`
- `python3 -m py_compile transcribe.py daemon.py`
- add targeted Python smoke tests if practical

---

## Milestone B — TypeScript diagnostics upgrade

### Objective
Consume the richer model detection contract and expose it to onboarding, doctor, and provisioning flows.

### Deliverables
- new TypeScript types for model presence
- richer `EnvironmentDiagnostics`
- helper functions for querying installed/preferred models

### File changes

#### `extensions/voice/diagnostics.ts`
Extend the contract:
- `BackendAvailability` should support model-level presence metadata
- `EnvironmentDiagnostics` should surface a normalized structure usable by UI and provisioning logic

Suggested TypeScript additions:

```ts
export type ModelPresenceStatus = "installed" | "not_installed" | "unknown" | "api";

export interface ModelPresence {
  status: ModelPresenceStatus;
  source?: string;
  path?: string;
}

export interface BackendAvailability {
  ...
  modelPresence?: Record<string, ModelPresence>;
}
```

Add helpers such as:
- `getInstalledModels(backend)`
- `isModelInstalled(backend, model)`
- `getPreferredInstalledModel(backends, preference)`
- `hasReadyLocalPath(diagnostics)`

### Recommendation logic changes
Update `recommendVoiceSetup()` so it can prefer:
1. already-installed good local models
2. already-configured API mode
3. backend-only installed local paths
4. install-required local paths

Expected behavior examples:
- if `faster-whisper small` is installed, recommend that over a download-required `medium`
- if no local models are installed but Deepgram is configured and user chose speed, recommend API
- if local backend is present but model presence is unknown, explain uncertainty honestly

### Acceptance criteria
- recommendations change when installed-model info changes
- recommendation reasons mention ready-now states where appropriate
- tests cover installed / not-installed / unknown cases

### Verification
- new Bun tests for recommendation ranking
- `bun run test`
- `bun run typecheck`

---

## Milestone C — onboarding UX for already-installed models

### Objective
Make the onboarding flow visibly smarter when models are already present.

### Deliverables
- backend list messaging that reflects installed-model readiness
- model picker with installed/download-required labels
- summary screen that says when a model is ready immediately

### File changes

#### `extensions/voice/onboarding.ts`
Enhance:
- `buildSelectableBackends()`
- `chooseBackendAndModel()`
- summary generation

### UX changes

#### Backend selection
Display labels like:
- `faster-whisper — backend installed, 2 models ready`
- `whisper-cpp — model file found`
- `moonshine — installed, model status unknown`
- `deepgram — API available`

#### Model selection
Display labels like:
- `small (already installed, recommended)`
- `medium (download required)`
- `large-v3-turbo (status unknown)`

#### Summary screen
Include language such as:
- `Model status: already installed`
- `Setup path: ready to configure now`
- `Download required: no`

or
- `Model status: not installed`
- `Setup path: dependency/model download required`

### Product behavior
If a model is already installed:
- the onboarding flow should still ask for user selection
- but it should explicitly say: `We found this on your machine`
- and it should prefer that model in recommendations where reasonable

### Acceptance criteria
- onboarding distinguishes installed vs missing models
- user sees explicit no-redownload paths
- selection flow remains keyboard-friendly and simple

### Verification
- new Bun tests for option labeling helpers
- manual QA in RPC mode for UI branch coverage

---

## Milestone D — provisioning short-circuiting

### Objective
Avoid showing install/download steps when the selected model is already ready.

### Deliverables
- `buildProvisioningPlan()` becomes model-aware
- selected installed models produce zero unnecessary install commands
- repair flows remain correct for missing prerequisites

### File changes

#### `extensions/voice/install.ts`
Enhance provisioning logic to account for:
- backend installed?
- model installed?
- SoX missing?
- API key missing?

Expected behavior:
- local backend installed + model installed + SoX present => `ready: true`
- local backend installed + model missing => install/download guidance only for model/backend path
- API mode + key present + no local SoX requirement for transcription itself, but keep recording prereq guidance if recording is part of UX

### Acceptance criteria
- no redundant install commands when model is already present
- provisioning summary reflects current machine state accurately

### Verification
- new Bun tests for installed-model provisioning cases
- `bun run test`

---

## Milestone E — doctor/test/info upgrades

### Objective
Make runtime introspection and repair UX model-aware.

### Deliverables
- `/voice doctor` reports current backend + model readiness
- `/voice test` reports current config and whether selected model appears ready
- `/voice info` reports model-aware state cleanly

### File changes

#### `extensions/voice.ts`
Update command output to include:
- selected model presence (`installed` / `not installed` / `unknown`)
- whether current setup is ready-now or repair-needed
- whether recommendation is using an already-installed model

### Example doctor output

```text
Current config: local/faster-whisper/small
Current model status: installed
Repair current setup:
  - none

Recommended alternative: local/faster-whisper/small
Why: already installed and ready now
```

### Acceptance criteria
- doctor/test/info no longer operate at backend-only granularity
- current-config repair and recommendation remain clearly separated

### Verification
- manual/RPC-assisted QA for command output
- update QA docs with expected text fragments

---

## Milestone F — docs and release polish

### Objective
Document the smarter onboarding behavior clearly.

### Deliverables
- README updated with installed-model-aware behavior
- backend guide updated with model detection caveats
- troubleshooting updated with model-ready vs model-missing guidance
- QA docs updated with installed-model scenarios

### File changes
- `README.md`
- `docs/backends.md`
- `docs/troubleshooting.md`
- `docs/qa-matrix.md`
- `docs/qa-results.md`

### Doc requirements
Explain clearly:
- backend installed does not always mean every model is ready
- some backends can reliably detect model presence; others may report `unknown`
- onboarding may say `already installed`, `download required`, or `status unknown`

### Acceptance criteria
- docs match real behavior
- no overpromising on weak-confidence detection backends

---

## Milestone G — QA and release signoff

### Objective
Ship the model-aware onboarding safely.

### Deliverables
- automated coverage for model-aware logic
- manual/RPC-assisted evidence for the major paths
- release signoff checklist

### Automated test targets
Add or expand tests for:
- diagnostics parsing of model presence
- recommendation preferring installed models
- onboarding labeling for installed/download-required models
- provisioning short-circuit behavior
- current-config repair vs recommendation output with model-aware state

### Manual / RPC-assisted QA targets
1. fresh install, no config -> startup prompt appears
2. partial legacy config -> onboarding still appears
3. remind-me-later -> suppress startup prompt within defer window
4. `/voice reconfigure` -> launches setup flow
5. project scope save -> writes `.pi/settings.json`
6. local path with installed model -> summary reports ready-now path
7. local path with missing model -> summary reports download-required path
8. doctor output -> separates current repair from recommendation and includes model state

### Verification commands
- `bun run check`
- `python3 transcribe.py --list-backends`
- manual/RPC-assisted smoke scripts as needed

---

## Recommended implementation order

### Step 1 — detection contract
Do first because everything else depends on it.
- update `transcribe.py`
- add detection heuristics conservatively
- validate JSON shape

### Step 2 — TypeScript diagnostics
- parse the new contract
- add model presence helpers
- update recommendation logic

### Step 3 — provisioning logic
- make `buildProvisioningPlan()` model-aware
- add tests

### Step 4 — onboarding UI
- upgrade backend/model labels
- surface ready-now paths
- add summary enhancements

### Step 5 — doctor/test/info
- reflect model-aware readiness in runtime UX

### Step 6 — docs + QA
- update docs
- run verification
- capture QA evidence

---

## Risks and mitigations

### Risk: model cache detection is unreliable for some backends
**Mitigation:** support `unknown` state and document it. Never fake certainty.

### Risk: faster-whisper cache layout varies by environment
**Mitigation:** mark `installed` only when concrete evidence exists. Otherwise return `unknown`, not `not_installed`.

### Risk: onboarding becomes too verbose
**Mitigation:** keep labels compact and push detail into summary/doctor output.

### Risk: install logic gets too coupled to backend heuristics
**Mitigation:** keep detection, provisioning, and UI mapping in separate modules.

### Risk: deepgram/API path gets conflated with local model semantics
**Mitigation:** represent API models distinctly (`api` status or equivalent), not as local installed/not-installed.

---

## Time estimate

From the current repository state, this remaining work should take roughly:
- **16–24 hours** for a solid implementation
- **20–32 hours** for a fully polished, release-ready version

### Breakdown
- model detection contract: 4–8h
- diagnostics + recommendation updates: 3–6h
- onboarding UX updates: 4–8h
- provisioning/doctor/test/info updates: 3–6h
- docs + QA: 3–6h

Practical working estimate: **2–4 focused days**.

---

## Definition of done

This remaining work is done when:
- onboarding can explicitly say when a chosen local model is already available
- recommendations prefer already-installed models when appropriate
- provisioning skips unnecessary install steps for already-ready paths
- doctor/test/info reflect model-aware readiness
- docs explain the behavior accurately
- verification passes and QA evidence is captured

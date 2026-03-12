# pi-voice model-aware review

## Scope reviewed

Focused review of the current model-aware phase in:
- `transcribe.py`
- `extensions/voice/diagnostics.ts`
- `extensions/voice/onboarding.ts`
- `extensions/voice/install.ts`
- `extensions/voice.ts`

I also sanity-checked the stated behavior against the current implementation shape and looked specifically for:
- false claims
- command UX inconsistencies
- installed-model edge cases
- regression risks

---

## Summary

The model-aware phase is moving in the right direction:
- backend discovery now emits installed-model metadata
- recommendations can prefer an installed local model
- onboarding labels installed models distinctly
- provisioning distinguishes between backend-missing and model-missing cases
- doctor/test/info now surface model-aware state

However, there are still a few important issues before calling this fully product-grade.

---

## Findings

### 1. Onboarding stops offering alternative local backends as soon as any local backend is discovered
**Severity:** high

**Where:**
- `extensions/voice/onboarding.ts:81-83`

**What happens:**
`buildSelectableBackends("local", diagnostics)` returns `discoveredLocalBackends` immediately when *any* local backend is discovered. That means if the machine has only `faster-whisper` installed, onboarding will no longer offer:
- `moonshine`
- `whisper-cpp`
- `parakeet`

with install hints.

**Why it matters:**
This conflicts with the intended UX:
- “you already have this, we can configure it now”
- but still let the user pick another backend if they want

Right now the model-aware path improves the ready-now story, but narrows backend choice too aggressively.

**Recommended fix:**
Return a merged local backend list:
- discovered backends first
- then undiscovered fallback backends with install hints
- deduplicated by backend name

That preserves the “already installed” path without removing other selectable options.

---

### 2. `faster-whisper` model detection likely produces false negatives for some models
**Severity:** high

**Where:**
- `transcribe.py:86-93`

**What happens:**
`faster_whisper_repo_ids()` assumes:
- standard models map to `Systran/faster-whisper-<model>`
- distil models map to `distil-whisper/<model>`

But the actual repo mapping for some models differs. In particular:
- `large-v3-turbo` is not guaranteed to live under `Systran/faster-whisper-large-v3-turbo`
- distil model repos are not reliably `distil-whisper/<model>`

**Why it matters:**
The product will claim “download required” or fail to recognize installed models when they actually already exist.

This undermines the core model-aware onboarding promise.

**Recommended fix:**
Use the real `faster_whisper` model mapping if available from the library, or maintain an explicit repo map for the supported model IDs instead of constructing repo IDs heuristically.

---

### 3. Heuristic `installed_models` are treated like high-confidence installed models in recommendation ranking
**Severity:** medium

**Where:**
- `extensions/voice/diagnostics.ts:89-97`
- `extensions/voice/diagnostics.ts:118-132`

**What happens:**
`getPreferredLocalBackend()` prefers the first local backend with any `installed_models`, regardless of detection confidence.

That means a heuristic backend such as:
- `moonshine`
- `parakeet`

can outrank a safer backend if its heuristic reports an installed model.

**Why it matters:**
The code already treats some detectors as low confidence (`unknown` in `getModelReadiness()`), but recommendation ranking does not use that same confidence model.

So the recommendation engine can still over-trust heuristic detections.

**Recommended fix:**
In recommendation ranking, prefer installed models only for high-confidence detectors first, for example:
1. high-confidence installed model
2. high-confidence available backend
3. heuristic installed model
4. heuristic available backend

At minimum, gate “already installed and ready to configure” recommendation language behind the same confidence rules used by `getModelReadiness()`.

---

### 4. `/voice backends` output is still backend-centric when a backend is available but no installed models are confirmed
**Severity:** low

**Where:**
- `extensions/voice.ts:915-931`

**What happens:**
If a backend is available and `installed_models` is empty, the output falls back to:
- `models: <count>`

That is not wrong, but it is weaker than the new model-aware UX elsewhere.

**Why it matters:**
Users comparing `/voice backends` with onboarding/doctor/test may get less clarity here than they expect.

Example ambiguity:
- backend ready
- zero confirmed installed models
- but output only says `models: 13`

That does not tell the user whether the likely next state is:
- download required
- unknown confidence
- or ready-now via API

**Recommended fix:**
Use model-aware wording here too, for example:
- `installed: small, medium`
- `no confirmed installed models`
- `model detection: unknown confidence`
- `api ready`

---

### 5. Moonshine cache detection may under-detect because it only checks directories, not concrete model files
**Severity:** low

**Where:**
- `transcribe.py:131-140`
- `_existing_dirs()` at `transcribe.py:28-34`

**What happens:**
The moonshine fallback local candidates are passed through `_existing_dirs()`, which only returns existing directories. If a practical moonshine install stores key artifacts as files rather than dedicated directories, this logic will miss them.

**Why it matters:**
This will most likely cause false negatives rather than false positives, but it still weakens the ready-now experience for moonshine.

**Recommended fix:**
If moonshine detection remains heuristic, consider checking both:
- directories
- likely file artifacts under those roots

Or explicitly document the backend as lower-confidence and keep recommendation weight conservative.

---

## What looks good

These parts are solid and worth keeping:
- model-aware tests were added before/with behavior changes
- `getModelReadiness()` now distinguishes high-confidence local detection from heuristic/unknown paths
- provisioning wording is more conservative for heuristic backends
- doctor output separates repair current setup from recommended alternative
- onboarding model labels are much better than before

---

## Recommended priority order

1. **Fix local backend list merging in onboarding**
2. **Fix `faster-whisper` repo/model mapping**
3. **Make recommendation ranking confidence-aware**
4. **Polish `/voice backends` output to match model-aware UX**
5. **Optionally improve moonshine detection fidelity**

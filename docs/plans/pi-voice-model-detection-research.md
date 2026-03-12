# pi-voice model detection research

## Goal

Figure out how `pi-voice` can detect whether a backend/model is **already available locally** so onboarding can say things like:

- "You already have `small` installed — configure this now"
- "`medium` is not downloaded yet — first use will download it"
- "Deepgram is API-based — no local model download is needed"

This research focuses on practical heuristics that can be implemented now in this repo, with confidence notes.

---

## Bottom-line recommendation

Implement model detection as a **best-effort, backend-specific layer** that returns one of:

- `installed`
- `missing`
- `unknown`

Do **not** force every backend into a binary installed/missing answer if the storage semantics are unclear.

### Recommended rule
- Prefer **runtime/library probes** over hard-coded path guesses when the backend library exposes a reliable local-only lookup.
- Use **filesystem heuristics** where the project already has explicit path logic.
- If confidence is low, return `unknown` and let onboarding say:
  - "Package/backend is installed, but local model presence could not be confirmed."

---

## Suggested output contract

Add a model-detection layer that can return something like:

```json
{
  "backend": "faster-whisper",
  "backendAvailable": true,
  "models": [
    {
      "name": "small",
      "status": "installed",
      "confidence": "high",
      "path": "/Users/.../.cache/huggingface/hub/..."
    },
    {
      "name": "medium",
      "status": "missing",
      "confidence": "high"
    },
    {
      "name": "large-v3-turbo",
      "status": "unknown",
      "confidence": "low",
      "reason": "backend installed, but cache semantics are backend-managed"
    }
  ]
}
```

This is better than a plain boolean because onboarding can explicitly say:
- **Already installed**
- **Download required**
- **Unknown / may download on first use**

---

## Cross-backend cache/env locations to consider first

Before backend-specific logic, normalize these locations:

- `HF_HOME`
- `HUGGINGFACE_HUB_CACHE`
- `XDG_CACHE_HOME`
- `TORCH_HOME`

### Practical default resolution order

#### Hugging Face cache base
1. `HUGGINGFACE_HUB_CACHE`
2. `${HF_HOME}/hub`
3. `${XDG_CACHE_HOME}/huggingface/hub`
4. `~/.cache/huggingface/hub`

#### Torch / NeMo cache base
1. `${TORCH_HOME}/NeMo`
2. `${XDG_CACHE_HOME}/torch/NeMo`
3. `~/.cache/torch/NeMo`
4. `~/.cache/nemo`

These should be centralized in one helper so all model detection uses the same cache roots.

---

## Backend-by-backend findings

## 1. faster-whisper

### Current repo behavior
`transcribe.py` treats `faster-whisper` as available when the Python package imports:
- `is_faster_whisper_available()`

At runtime it calls:
- `WhisperModel(model_name, ...)`

### Best detection strategy
**Use the library itself**.

The installed package exposes `faster_whisper.utils.download_model()`, which supports:
- `local_files_only=True`

I verified locally that this works as a pure local probe:

```python
from faster_whisper.utils import download_model
path = download_model("small", local_files_only=True)
```

If present, it returns the local cached snapshot path.
If missing, it raises `LocalEntryNotFoundError`.

### Why this is the best option
- high confidence
- no need to reverse-engineer the exact cache layout
- automatically respects Hugging Face cache behavior
- avoids false positives from partial folders

### Repo mapping data available now
The package exposes a concrete model -> repo mapping via `faster_whisper.utils._MODELS`.

Observed locally:

```python
{
  'small': 'Systran/faster-whisper-small',
  'small.en': 'Systran/faster-whisper-small.en',
  'medium': 'Systran/faster-whisper-medium',
  'large-v3': 'Systran/faster-whisper-large-v3',
  'large-v3-turbo': 'mobiuslabsgmbh/faster-whisper-large-v3-turbo',
  'distil-small.en': 'Systran/faster-distil-whisper-small.en',
  ...
}
```

### Filesystem heuristic (fallback only)
If you need path-only detection, Hugging Face snapshots typically appear under:

```text
~/.cache/huggingface/hub/models--<org>--<repo>/snapshots/<rev>/
```

Examples:
- `models--Systran--faster-whisper-small`
- `models--mobiuslabsgmbh--faster-whisper-large-v3-turbo`

### Confidence
- **Library probe:** high
- **Filesystem-only scan:** medium

### Recommendation
Implement detection for `faster-whisper` via:
1. import `faster_whisper.utils.download_model`
2. call with `local_files_only=True`
3. mark `installed` or `missing`
4. return the resolved path when found

---

## 2. whisper-cpp

### Current repo behavior
`transcribe.py` already contains concrete model path logic:

```python
candidates = [
    ~/.cache/whisper-cpp/ggml-{model}.bin,
    /opt/homebrew/share/whisper-cpp/models/ggml-{model}.bin,
    /usr/local/share/whisper-cpp/models/ggml-{model}.bin,
]
```

It also accepts `model_name` directly if it is already a path.

### Best detection strategy
Use the **same candidate path logic already in the repo**.

For each model:
1. if `model_name` itself is an existing path -> `installed`
2. otherwise check:
   - `~/.cache/whisper-cpp/ggml-{model}.bin`
   - `/opt/homebrew/share/whisper-cpp/models/ggml-{model}.bin`
   - `/usr/local/share/whisper-cpp/models/ggml-{model}.bin`
3. if any exists -> `installed`
4. else -> `missing`

### Why this is strong
- this is already the runtime resolution behavior
- onboarding can exactly mirror real runtime behavior
- avoids mismatch between UI and transcription runtime

### Confidence
- **Filesystem/path detection:** very high

### Recommendation
Extract the candidate-path logic into a reusable helper so:
- transcription
- doctor
- onboarding
all share the same source of truth.

---

## 3. deepgram

### Current repo behavior
There is no local model download.

Availability is simply:
- `DEEPGRAM_API_KEY` present -> backend available

### Best detection strategy
Treat all models as:
- `installed` in the sense of **ready to use now** if the API key exists
- `missing` if the API key does not exist

### Suggested wording in onboarding
Instead of "already downloaded", say:
- **Ready via API**
- **Requires API key**

### Confidence
- **Env-based readiness:** very high

### Recommendation
Represent cloud models differently in UI:
- no local disk/download badge
- `Ready now` vs `Needs API key`

---

## 4. moonshine

### Current repo behavior
`transcribe.py` supports two paths:
- Python API via `moonshine_onnx`
- CLI fallback via `moonshine`

Models exposed in the repo:
- `moonshine/tiny`
- `moonshine/base`

### Problem
I do **not** have strong evidence yet for a stable public cache layout for Moonshine model files.

### Practical detection options

#### Option A — package resource scan (recommended first)
If `moonshine_onnx` is installed:
- inspect the installed package directory
- search for `.onnx` or related model artifacts that contain `tiny` / `base`

Example approach:
```python
import moonshine_onnx, inspect
from pathlib import Path
pkg_dir = Path(moonshine_onnx.__file__).resolve().parent
# scan recursively for files containing "tiny" / "base" and ".onnx"
```

This is implementable now, though still heuristic.

#### Option B — CLI/package-managed cache scan
If the CLI is installed but Python package is not:
- inspect likely package-managed cache locations under:
  - `~/.cache`
  - `${XDG_CACHE_HOME}`
- search for directories/files matching `moonshine`, `tiny`, `base`, `.onnx`

#### Option C — conservative fallback
If backend package/CLI is installed but no model artifacts can be confidently located:
- return `unknown`

### Confidence
- **Package-resource scan:** medium
- **Generic cache scan:** low
- **Binary installed => model installed:** too risky, do not assume

### Recommendation
For the first implementation:
- detect backend availability normally
- try package-resource scan
- if inconclusive, return `unknown`
- onboarding should say:
  - `moonshine/base — backend installed, local model presence not confirmed`

This is safer than lying about downloaded state.

---

## 5. parakeet / NeMo

### Current repo behavior
`transcribe.py` uses:

```python
nemo_asr.models.ASRModel.from_pretrained(model_name)
```

Model names are repo-like strings such as:
- `nvidia/parakeet-tdt-0.6b-v2`
- `nvidia/parakeet-ctc-0.6b`

### Problem
NeMo caching can vary by version/environment.
It may use:
- Hugging Face cache
- Torch/NeMo cache
- internal download helpers

### Practical detection options

#### Option A — cache path scan by model slug
Check likely cache roots:
- `${TORCH_HOME}/NeMo`
- `~/.cache/torch/NeMo`
- `~/.cache/nemo`
- Hugging Face cache under `models--nvidia--parakeet-*`

Search for folder/file names containing a normalized slug, e.g.:
- `nvidia--parakeet-tdt-0.6b-v2`
- `parakeet-tdt-0.6b-v2`

#### Option B — local-only library probe (if NeMo exposes one)
If there is a NeMo utility for local cache resolution without downloading, that would be preferable. I do not have that confirmed from the current environment.

#### Option C — conservative fallback
If `nemo.collections.asr` imports but the cache cannot be confidently located:
- return `unknown`

### Confidence
- **Torch/NeMo + HF path scan:** medium-low
- **Library-level local-only probe:** unknown until implemented/researched further

### Recommendation
For the first shipping iteration:
- backend availability from import
- model presence from cache-root scan only
- return `unknown` if no confident hit

This is enough for onboarding to say:
- `Parakeet backend is installed, but local model presence could not be confirmed`

---

## Local machine observations from this environment

Observed cache/env state on this machine:

### Cache/env
- `HF_HOME`: unset
- `HUGGINGFACE_HUB_CACHE`: unset
- `XDG_CACHE_HOME`: unset
- `TORCH_HOME`: unset
- `~/.cache/huggingface/hub`: exists
- `~/.cache/whisper-cpp`: absent
- `/opt/homebrew/share/whisper-cpp/models`: absent

### Hugging Face hub contents observed
Current HF hub cache did **not** show any obvious `faster-whisper-*` repos present in the default cache root.

### faster-whisper local-only probe results
I verified locally that these probes report `missing` via `LocalEntryNotFoundError`:
- `small`
- `large-v3-turbo`
- `distil-small.en`

That confirms the local-only probe is usable as an installed/missing signal.

---

## Recommended implementation order

## Phase 1 — high-confidence backends first
Implement model detection for:
1. `faster-whisper`
2. `whisper-cpp`
3. `deepgram`

These give the biggest product value with the least ambiguity.

## Phase 2 — conservative support for the ambiguous backends
Implement best-effort detection for:
4. `moonshine`
5. `parakeet`

Return `unknown` rather than over-claiming installed state.

---

## Suggested onboarding behavior

### For `installed`
Show:
- **Already installed**
- **Ready to configure now**

### For `missing`
Show:
- **Download required**
- backend-specific install guidance

### For `unknown`
Show:
- **Backend installed**
- **Model presence not confirmed**
- **May download or initialize on first use**

This lets the system feel smart without making brittle claims.

---

## Concrete implementation suggestion for this repo

### Add to `transcribe.py`
A machine-readable command such as:

```bash
python3 transcribe.py --list-model-status --backend faster-whisper
```

Output shape:

```json
{
  "backend": "faster-whisper",
  "models": [
    {"name": "small", "status": "installed", "confidence": "high", "path": "..."},
    {"name": "medium", "status": "missing", "confidence": "high"}
  ]
}
```

### Then wire into
- `extensions/voice/diagnostics.ts`
- `extensions/voice/onboarding.ts`
- `extensions/voice/install.ts`

### Product behavior unlocked
- prefer already-downloaded models in recommendations
- show `Already installed` badges in onboarding
- skip unnecessary download/setup prompts when model is already present

---

## Confidence summary

| Backend | Best detection method | Confidence |
|---|---|---|
| `faster-whisper` | `download_model(..., local_files_only=True)` | high |
| `whisper-cpp` | exact path checks already used in `transcribe.py` | very high |
| `deepgram` | `DEEPGRAM_API_KEY` presence | very high |
| `moonshine` | package-resource scan + fallback `unknown` | medium / low |
| `parakeet` | Torch/NeMo/HF cache scan + fallback `unknown` | medium-low |

---

## Final recommendation

Implement **installed-model-aware onboarding** in two passes:

### Pass 1 (ship fast, high confidence)
- faster-whisper
- whisper-cpp
- deepgram

### Pass 2 (ship carefully)
- moonshine
- parakeet
- use `unknown` status when confidence is not good enough

That gives `pi-voice` the product behavior you want without introducing misleading model-detection claims.

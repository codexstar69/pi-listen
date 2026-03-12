# pi-voice backend guide

`pi-voice` supports both **local** and **cloud** speech-to-text (STT) backends.

Today, the extension exposes backend selection through `/voice setup` and reads settings from:

- global: `~/.pi/agent/settings.json`
- project: `.pi/settings.json`

Project settings override global settings when both are present.

## Quick recommendation

If you want a conservative default:

- choose **faster-whisper** for a strong local default on macOS and general desktop use
- choose **Deepgram** if you want the fastest route to a working setup and you already have an API key

If onboarding tells you a model is already installed, prefer that ready-now path unless you have a strong reason to switch.

## Backend comparison

| Backend | Mode | Best for | Tradeoffs | Typical install path | Model detection confidence |
|---|---|---|---|---|---|
| `faster-whisper` | Local | Best overall local default, good quality/speed balance | Python dependency, model download time, CPU usage | `pip install faster-whisper` | High |
| `moonshine` | Local | Lower-resource local experiments | Smaller ecosystem, fewer model choices | `pip install useful-moonshine[onnx]` | Medium / heuristic |
| `whisper-cpp` | Local | CLI-oriented local setups, users already invested in whisper.cpp | Model file management is more manual | `brew install whisper-cpp` | Very high |
| `deepgram` | Cloud | Fastest setup if API key already exists | Audio leaves the machine, network required, API billing | set `DEEPGRAM_API_KEY` | API-ready vs missing key |
| `parakeet` | Local | NVIDIA/NeMo-oriented experimentation | Heavy dependency footprint, slower setup | `pip install nemo_toolkit[asr]` | Medium-low / heuristic |

## Backend details

## faster-whisper

**Use this when:**
- you want a high-confidence local default
- privacy/offline use matters
- you are okay with Python-based dependencies

**Pros**
- mature Whisper-family local option
- strong model selection range
- fits the current `pi-voice` architecture well

**Cons**
- first install can take longer than cloud setup
- larger models increase local CPU and startup costs

**Suggested starting models**
- `small` — good conservative starting point
- `small.en` — useful for English-only setups
- `medium` — move here if you want more accuracy and can accept higher cost

## moonshine

**Use this when:**
- you want a lighter local option
- you are willing to trade maturity and flexibility for smaller setup/runtime needs

**Pros**
- lightweight local direction
- fewer model decisions

**Cons**
- narrower model ecosystem
- less common than Whisper-family defaults

**Suggested starting models**
- `moonshine/base` — safer default
- `moonshine/tiny` — lowest-cost experiment path

## whisper-cpp

**Use this when:**
- you already use whisper.cpp
- you prefer CLI-driven local tooling
- you want to manage model files more explicitly

**Pros**
- straightforward local CLI model
- familiar to users already in the whisper.cpp ecosystem

**Cons**
- model file placement matters
- setup can feel more manual than Python-backed alternatives

**Suggested starting models**
- `small` — practical starting point
- `small.en` — good English-only choice

## deepgram

**Use this when:**
- you want the fastest route to a working setup
- you are comfortable with cloud transcription
- you already have or plan to add `DEEPGRAM_API_KEY`

**Pros**
- minimal local dependency burden
- fast setup when credentials are ready
- good fit for users optimizing for time-to-value

**Cons**
- requires network connectivity
- audio is sent to a cloud provider
- subject to account limits and billing

**Suggested starting model**
- `nova-3` — the current default in `transcribe.py`

## parakeet

**Use this when:**
- you specifically want to experiment with NVIDIA NeMo/Parakeet
- you are comfortable with a heavier Python stack

**Pros**
- interesting specialized option
- useful for experimentation in the NeMo ecosystem

**Cons**
- heavier setup than the other paths
- not the best default for a first-time user

## Choosing local vs cloud

## Choose local when
- privacy matters more than setup speed
- you want offline use
- you are comfortable installing dependencies
- you want per-project control without relying on network services

## Choose cloud when
- you want the fastest setup path
- you are okay with a provider API key
- you want to avoid local model/dependency installation
- you are comfortable with network and billing constraints

## Model selection guidance

The current package exposes raw backend model names plus model-aware status where it can detect it.

A conservative way to choose:

- prefer **already installed** models first when they meet your needs
- start small: `small`, `small.en`, or backend defaults
- only move up if the current model is not accurate enough
- prefer English-only variants when your use case is English-only and the backend provides them
- avoid jumping to the biggest model first unless you already know your machine can handle it

## How to read model status

During onboarding and in some diagnostics output, local models can show up as:

- **installed** — the model appears to already exist locally and should not need a fresh download
- **recommended, installed** — same as above, and also the preferred choice for the current path
- **download required** — backend is available, but the selected model does not appear to be present
- **unknown** — backend is present, but model presence could not be confirmed confidently
- **api** — cloud path; no local model download applies

For `unknown`, do not assume the system is broken. It means `pi-voice` is being conservative and avoiding a false claim.

## Configuration scope

`pi-voice` now supports both:

- **Global scope** — applies across projects
- **Project scope** — applies only in the current repository

Use **project scope** when:
- one repo needs a different voice workflow
- you want a safer rollout before using the same config everywhere
- you need local STT in one project but cloud STT in another

## Inspect current backend state

Useful commands:

- `/voice info` — show current effective config and selected model status
- `/voice backends` — list available backends, installed models, and install detection hints
- `/voice setup` — choose backend and model interactively
- `/voice test` — do a basic voice environment check plus current model readiness
- `/voice doctor` — compare current setup repair steps with a recommended alternative
- `/voice daemon status` — inspect the warm daemon state

## Current limitations

Be conservative about expectations:

- backend installation is still partly toolchain-dependent
- some setup steps may still require manual `brew` or `pip` work
- not every backend has equally strong model-detection confidence
- `unknown` model status is intentional when the system cannot verify a local cache safely
- backend available does **not** always mean every model is already ready locally

If you hit issues, see [`docs/troubleshooting.md`](./troubleshooting.md).

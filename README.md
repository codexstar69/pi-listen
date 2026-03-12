# pi-listen

Voice input and side-channel voice workflows for Pi.

`pi-listen` adds:
- **hold-to-talk voice input** for the Pi editor
- a persistent **STT daemon** to keep local models warm
- multiple STT backend options, including **local** and **cloud/API** paths
- **BTW side conversations** for quick parallel questions without interrupting the main session

## Status

This package is actively being upgraded toward a more polished, enterprise-grade onboarding flow.

### Available today
- voice recording into the editor
- fallback keyboard shortcut for non-hold-to-talk terminals
- guided **first-run onboarding** on the first interactive session after install
- top-level **API vs Local** setup decision
- model-aware local setup that can show when a backend/model appears to already be available
- `/voice setup`, `/voice reconfigure`, `/voice doctor`, `/voice info`, `/voice test`, `/voice backends`, and daemon controls
- BTW side-channel conversations
- scope-aware settings save during setup (`global` or `project`)

### In progress
- richer provisioning and auto-install flows
- deeper validation and repair automation
- expanded docs and troubleshooting polish

This README is intentionally conservative: it documents the package as it exists now, while also outlining the direction of the onboarding improvements.

## What it does

### 1. Hold-to-talk voice input
When voice is enabled, you can record audio and transcribe it into the Pi editor.

### 2. Persistent transcription daemon
For supported local backends, the package can keep a transcription model warm in memory to reduce cold-start latency.

### 3. BTW side conversations
You can record or type short side questions and keep them in a lightweight thread without interrupting the main task.

## Installation

Install as a Pi package:

```bash
pi install npm:pi-listen
```

For local development, you can also install from a path:

```bash
pi install /path/to/pi-listen
```

## Bootstrap scripts

If you want a one-command setup helper for a laptop, this repo now ships separate platform scripts:
- `scripts/setup-macos.sh`
- `scripts/setup-windows.ps1`

### macOS

```bash
bash scripts/setup-macos.sh --mode local --backend faster-whisper
```

API mode example:

```bash
bash scripts/setup-macos.sh --mode api --deepgram-key YOUR_KEY --persist-deepgram-key
```

### Windows (PowerShell)

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup-windows.ps1 -Mode local -Backend faster-whisper
```

API mode example:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup-windows.ps1 -Mode api -DeepgramKey YOUR_KEY -PersistDeepgramKey
```

What the scripts handle:
- install/check `python3` or Python 3.12
- install/check SoX / `rec`
- install a supported local backend or wire up `DEEPGRAM_API_KEY`
- run `pi install npm:pi-listen` when the `pi` command is available
- run backend + daemon smoke tests from `transcribe.py` and `daemon.py`
- write ready-to-use Pi voice config into `settings.json`

What still stays manual:
- granting microphone permission in the OS

You should not need to run `/voice setup` on the happy path.
If you want project-local config instead of global config, pass the script scope flag:
- macOS: `--scope project --project-dir /path/to/repo`
- Windows: `-Scope project -ProjectDir C:\path\to\repo`

## Setup experience

### Automatic first-run onboarding
On the first interactive session after install, `pi-listen` now prompts to start setup.

The onboarding flow asks:
- whether you want **API** or **Local** speech-to-text
- what matters most (balanced, speed, privacy, accuracy, lower resource usage)
- which backend/model to use
- whether to save settings globally or for the current project

It also shows a recommendation plus suggested install/manual steps based on your machine.

When the package can detect that a local model is already present, onboarding can surface that as:
- **already installed**
- **ready to configure now**
- **download required**
- or **status unknown** for lower-confidence backends

### Re-running setup
You can re-open the onboarding flow any time with:

```text
/voice setup
```

or:

```text
/voice reconfigure
```

## Modes

## API mode
Best when you want:
- fast setup
- minimal local dependencies
- cloud transcription

Current cloud backend in the package:
- **Deepgram** (`DEEPGRAM_API_KEY` required)

## Local mode
Best when you want:
- privacy/offline workflows
- lower dependence on external services
- warm local inference through the daemon

Current local backends in the package:
- `faster-whisper`
- `moonshine`
- `whisper-cpp`
- `parakeet`

Availability depends on what is installed on the machine.

For local paths, `pi-listen` now distinguishes between:
- **backend available** — the package/CLI exists
- **model already installed** — a specific model appears ready now
- **download required** — backend exists but the chosen model is not yet present
- **unknown** — backend is present but model presence could not be confirmed with high confidence

## Backend notes

This package currently exposes backend discovery through:

```bash
/voice backends
```

For a fuller comparison matrix, see `docs/backends.md`.

The command output now includes model-aware signals where possible, such as:
- installed models for a backend
- install detection method
- install hints when a backend or model is still missing

## Keyboard shortcuts

### Voice input
- **Hold `Space`** — record to the editor when the editor is empty
- **`Ctrl+Shift+V`** — toggle voice recording as a fallback shortcut

### BTW voice flow
- **Hold `Ctrl+Shift+B`** — record and send the result to the BTW side thread

## Commands

### Voice commands
- `/voice on` — enable voice for the current session
- `/voice off` — disable voice for the current session
- `/voice info` — show current voice configuration and runtime state
- `/voice test` — run a quick voice setup check
- `/voice setup` — run the onboarding/setup flow
- `/voice reconfigure` — alias for setup when you want to switch modes or models
- `/voice doctor` — inspect environment readiness and suggested next steps
- `/voice backends` — list detected backend availability
- `/voice daemon` or `/voice daemon start` — start the daemon
- `/voice daemon stop` — stop the daemon
- `/voice daemon status` — inspect daemon status

### BTW commands
- `/btw <message>` — ask a side question
- `/btw:new [message]` — start a fresh BTW thread
- `/btw:clear` — dismiss and clear the BTW thread
- `/btw:inject` — inject the BTW thread into the main agent context
- `/btw:summarize` — summarize the BTW thread and inject the summary

## Configuration

Voice settings are stored under the `voice` key in Pi settings.

Depending on how you save setup, configuration can live in either:
- global settings
- project-local settings

### Example shape

```json
{
  "voice": {
    "version": 2,
    "enabled": true,
    "language": "en",
    "mode": "local",
    "backend": "faster-whisper",
    "model": "small",
    "scope": "project",
    "btwEnabled": true,
    "onboarding": {
      "completed": true,
      "schemaVersion": 2,
      "completedAt": "2026-03-12T00:00:00.000Z",
      "lastValidatedAt": "2026-03-12T00:00:00.000Z",
      "source": "setup-command"
    }
  }
}
```

If setup is deferred or still needs repair, the `onboarding` block can remain incomplete until validation succeeds.

## Dependencies

Common requirements today:
- `python3`
- `SoX` / `rec` for microphone recording
- backend-specific Python or system packages depending on your STT choice

Examples:
- `brew install sox`
- `pip install faster-whisper`
- `brew install whisper-cpp`
- set `DEEPGRAM_API_KEY` for Deepgram

## Suggested first run

A practical path is:

1. Install the package
2. Accept the first-run setup prompt, or run `/voice setup`
3. Choose **API** or **Local**
4. If you choose Local, look for labels such as `installed`, `recommended, installed`, or `download required`
5. Review the suggested commands/manual steps
6. Run `/voice doctor` or `/voice test` if you want extra validation
7. Try hold-to-talk in an empty editor

## Troubleshooting

See `docs/troubleshooting.md` for deeper guidance.

### Common headings
- microphone recording issues
- SoX / `rec` not found
- Python backend missing
- backend installed but chosen model missing
- backend installed but model status unknown
- daemon not running
- stale or unexpected backend/model behavior
- Deepgram key missing or invalid
- local backend installed but not detected
- project settings vs global settings confusion

## Docs

- `docs/backends.md` — backend comparison and tradeoffs
- `docs/troubleshooting.md` — setup and runtime troubleshooting
- `docs/plans/2026-03-12-pi-voice-master-plan.md` — implementation plan for the onboarding overhaul

## Roadmap

Planned next improvements include:
- broader model-detection confidence improvements for heuristic backends
- richer provisioning and auto-install flows
- deeper validation and repair automation
- more release-machine QA for real microphones and cached-model scenarios
- stronger packaging and release hardening

## Development

Current local verification commands:

```bash
bunx tsc -p tsconfig.json
python3 -m py_compile daemon.py transcribe.py
```

## License

MIT

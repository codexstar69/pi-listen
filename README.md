# pi-listen

<p align="center">
  <img src="assets/banner.png" alt="pi-listen — Voice input for the Pi coding agent" width="100%" />
</p>

**Voice input for [Pi](https://github.com/mariozechner/pi-coding-agent) — hold SPACE to talk, Deepgram streaming STT.**

[![npm version](https://img.shields.io/npm/v/@codexstar/pi-listen.svg)](https://www.npmjs.com/package/@codexstar/pi-listen)
[![license](https://img.shields.io/npm/l/@codexstar/pi-listen.svg)](https://github.com/codexstar69/pi-listen/blob/main/LICENSE)
[![tests](https://img.shields.io/badge/tests-40%2F40%20passing-brightgreen)](#testing)

---

## Overview

pi-listen adds hands-free voice input to the [Pi coding agent](https://github.com/mariozechner/pi-coding-agent) CLI. Hold SPACE to record, release to transcribe directly into the editor with live streaming results.

### Key Features

| Feature | Description |
|---------|-------------|
| **Hold-to-talk** | Hold `SPACE` → record → release → live transcript in editor |
| **Deepgram streaming** | Real-time WebSocket STT with Nova-3 (interim + final results) |
| **Voice commands** | "Hey Pi, run tests" → auto-executes |
| **Voice shortcuts** | "new line", "period", "submit" → inline text expansion |
| **Continuous dictation** | `/voice dictate` — speak freely, no hold required |
| **Recording history** | `/voice history` — recent transcriptions |
| **Kitty protocol** | True key-release detection on supported terminals |

---

## Quick Start

### Install

```bash
pi install npm:@codexstar/pi-listen
```

### Prerequisites

```bash
brew install sox                          # Audio capture (macOS)
export DEEPGRAM_API_KEY="your-key-here"   # Free $200 credits at https://dpgr.am/pi-voice
```

### Bootstrap Script (zero-touch)

```bash
# macOS — installs SoX, extension, and writes settings
curl -fsSL https://raw.githubusercontent.com/codexstar69/pi-listen/main/scripts/setup-macos.sh | bash

# With API key
curl -fsSL https://raw.githubusercontent.com/codexstar69/pi-listen/main/scripts/setup-macos.sh | bash -s -- --deepgram-key YOUR_KEY --persist-deepgram-key
```

```powershell
# Windows — installs SoX via winget
irm https://raw.githubusercontent.com/codexstar69/pi-listen/main/scripts/setup-windows.ps1 | iex
```

You should not need to run `/voice setup` on the happy path — the bootstrap scripts handle everything.

---

## Usage

### Voice Input

| Action | Keybinding | Description |
|--------|-----------|-------------|
| Record → Editor | Hold `SPACE` | Hold spacebar to record, release to transcribe |
| Record (toggle) | `Ctrl+Shift+V` | Toggle recording on/off |
| Cancel | `Escape` | Cancel recording, restore editor text |

Quick SPACE tap types a space normally — voice only activates after 800ms hold.

### Voice Commands

Say these during recording for auto-execution:

| Command | Action |
|---------|--------|
| "run tests" / "run lint" / "run typecheck" | Executes in editor |
| "hey pi, explain this code" | Sends as agent message |
| "hey pi, search for auth" | Sets up search |
| "commit" / "commit this" | `git add -A && git commit` |
| "undo" / "clear" / "submit" | Editor actions |
| "new line" / "period" / "comma" | Inline text shortcuts |

### Commands

```bash
/voice              # Toggle voice on/off
/voice on           # Enable voice
/voice off          # Disable voice
/voice setup        # Run setup wizard (API key + scope)
/voice dictate      # Continuous dictation mode
/voice stop         # Stop dictation or recording
/voice test         # Test microphone + Deepgram connection
/voice info         # Show current config
/voice history      # Recent transcriptions
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Pi CLI (Node.js)                        │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │  voice.ts     │    │  config.ts   │    │  onboarding.ts   │  │
│  │  (extension)  │    │  (settings)  │    │  (setup wizard)  │  │
│  └──────┬───────┘    └──────────────┘    └──────────────────┘  │
│         │                                                       │
│         │  SoX → Deepgram WebSocket (streaming)                 │
│         │  Live interim + final transcripts                     │
│         ▼                                                       │
│  ┌──────────────────────────────────────────────────────────────┐│
│  │  Deepgram Nova-3 (cloud STT)                                ││
│  │  wss://api.deepgram.com/v1/listen                           ││
│  └──────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### Components

| Component | Purpose |
|-----------|---------|
| `extensions/voice.ts` | Main extension: streaming, hold-to-talk, UI widgets, commands |
| `extensions/voice/config.ts` | Config loading, saving, migration |
| `extensions/voice/onboarding.ts` | Interactive setup wizard |
| `extensions/voice/text-processing.ts` | Voice command detection, text shortcuts |
| `extensions/voice/deepgram.ts` | Deepgram URL builder, API key resolution |

---

## Configuration

Settings stored in Pi's settings files:

| Scope | Path | Use case |
|-------|------|----------|
| Global | `~/.pi/agent/settings.json` | Same setup across all projects |
| Project | `<project>/.pi/settings.json` | Project-specific settings |

### Settings Structure

```json
{
  "voice": {
    "version": 2,
    "enabled": true,
    "language": "en",
    "scope": "global",
    "deepgramApiKey": "...",
    "onboarding": {
      "completed": true,
      "schemaVersion": 2,
      "completedAt": "2026-03-13T00:00:00.000Z",
      "source": "setup-command"
    }
  }
}
```

---

## Testing

```bash
bun run test        # Run all tests (40 tests)
bun run typecheck   # TypeScript type checking
bun run check       # Full check: typecheck + tests
```

### Test Coverage

| Suite | Tests | Covers |
|-------|-------|--------|
| `config.test.ts` | 8 | Config loading, migration, persistence, onboarding |
| `onboarding.test.ts` | 2 | Onboarding finalization |
| `text-processing.test.ts` | 18 | Voice commands, text shortcuts |
| `deepgram.test.ts` | 8 | URL building, API key resolution |
| `setup-scripts.test.ts` | 4 | Bootstrap script validation |

---

## Security

- **No telemetry:** pi-listen does not collect or transmit any usage data
- **Audio streaming:** Audio is sent to Deepgram's API for transcription (cloud STT)
- **API key storage:** Deepgram key can be stored in config or env var
- **No secrets in errors:** Error responses do not expose API keys or internal paths

---

## License

[MIT](LICENSE) © 2026 codexstar69

---

## Links

- **npm:** [npmjs.com/package/@codexstar/pi-listen](https://www.npmjs.com/package/@codexstar/pi-listen)
- **GitHub:** [github.com/codexstar69/pi-listen](https://github.com/codexstar69/pi-listen)
- **Deepgram:** [Free $200 credits](https://dpgr.am/pi-voice) (no credit card)
- **Pi CLI:** [github.com/mariozechner/pi-coding-agent](https://github.com/mariozechner/pi-coding-agent)

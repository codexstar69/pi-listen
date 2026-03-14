# pi-voice

<p align="center">
  <img src="assets/banner.png" alt="pi-voice — Voice input for the Pi coding agent" width="100%" />
</p>

**Hold-to-talk voice input for [Pi](https://github.com/mariozechner/pi-coding-agent).**

[![npm version](https://img.shields.io/npm/v/@codexstar/pi-voice.svg)](https://www.npmjs.com/package/@codexstar/pi-voice)
[![license](https://img.shields.io/npm/l/@codexstar/pi-voice.svg)](https://github.com/codexstar69/pi-listen/blob/main/LICENSE)

---

## What It Does

pi-voice adds hands-free voice input to the Pi coding agent CLI. Hold `SPACE` to record, release to transcribe — text appears in the editor in real time via Deepgram streaming STT.

### Features

| Feature | Description |
|---------|-------------|
| **Hold-to-talk** | Hold `SPACE` to record, release to stop — text streams into the editor live |
| **Streaming transcription** | Deepgram Nova 3 WebSocket — interim results appear as you speak |
| **Voice commands** | "hey pi, run tests", "undo", "submit", "new line", "period" |
| **Continuous dictation** | `/voice dictate` for long-form speaking without holding keys |
| **Double-escape clear** | Press `Escape` twice to clear the editor |
| **Zero-config start** | Auto-activates if `DEEPGRAM_API_KEY` is set — no wizard needed |
| **Full diagnostics** | `/voice test` validates SoX, mic capture, and Deepgram API key in one command |
| **Cross-platform** | macOS, Windows, Linux — Kitty protocol + non-Kitty fallback |

---

## Install

```bash
pi install npm:@codexstar/pi-voice
```

### Prerequisites

- **SoX** — microphone recording (`brew install sox` / `apt install sox` / `choco install sox`)
- **Deepgram API key** — set `DEEPGRAM_API_KEY` env var ([get $200 free credit](https://dpgr.am/pi-voice))

### Quick Setup

```bash
brew install sox                           # macOS
export DEEPGRAM_API_KEY="your-key-here"    # add to ~/.zshrc
```

Then open Pi — if `DEEPGRAM_API_KEY` is in your environment, pi-voice auto-activates and shows a keybinding guide. No wizard needed.

If the key isn't set yet, you'll see a one-time hint pointing you to `/voice setup`.

---

## Usage

### Voice Input

| Action | Keybinding | Notes |
|--------|-----------|-------|
| Record to editor | Hold `SPACE` | Release to finalize transcription |
| Toggle recording | `Ctrl+Shift+V` | Works in all terminals |
| Clear editor | `Escape` × 2 | Double-tap within 500ms |

### Commands

```bash
/voice              # Toggle voice on/off
/voice on           # Enable voice
/voice off          # Disable voice
/voice setup        # Run onboarding wizard
/voice test         # Test microphone + Deepgram pipeline
/voice info         # Show current config and status
/voice dictate      # Continuous dictation mode
/voice stop         # Stop active recording or dictation
/voice history      # Show recent transcriptions
```

### Voice Commands

Say these during recording — they're detected and executed automatically:

| Trigger | Action |
|---------|--------|
| "hey pi, run tests" | Inserts `bun run test` |
| "undo" / "undo that" | Removes last word |
| "clear" / "clear all" | Clears editor |
| "submit" / "send it" | Submits editor content |
| "new line" | Inserts `\n` |
| "period" / "comma" / "question mark" | Inserts punctuation |

---

## How It Works

```
User holds SPACE
    ↓
SoX captures PCM audio from microphone
    ↓
Audio streams to Deepgram Nova 3 via WebSocket
    ↓
Interim transcripts update editor in real time
    ↓
User releases SPACE → CloseStream → final transcript
```

### Hold Detection

Two terminal modes are supported:

**Kitty protocol** (Ghostty, Kitty, WezTerm, Windows Terminal 1.22+):
True key-down/repeat/release events. First press enters warmup immediately.

**Non-Kitty** (macOS Terminal, older terminals):
Gap-based detection. Counts rapid key-repeat events to distinguish hold from tap.

Both modes require holding for ≥800ms before recording activates. Quick taps type a normal space.

### Architecture

```
extensions/voice.ts        Main extension — recording, UI, state machine
extensions/voice/config.ts Config loading, saving, migration
extensions/voice/onboarding.ts  First-run setup wizard
```

---

## Configuration

Settings stored in Pi's settings files under the `voice` key:

| Scope | Path |
|-------|------|
| Global | `~/.pi/agent/settings.json` |
| Project | `<project>/.pi/settings.json` |

```json
{
  "voice": {
    "version": 2,
    "enabled": true,
    "language": "en",
    "scope": "global",
    "onboarding": {
      "completed": true,
      "schemaVersion": 2
    }
  }
}
```

---

## Troubleshooting

```bash
/voice test     # Full diagnostics: SoX, mic capture, Deepgram API key validation
```

`/voice test` checks everything in one shot — SoX installation, microphone capture, and validates your Deepgram API key against the live API. It tells you exactly what's missing and how to fix it.

| Problem | Solution |
|---------|----------|
| "DEEPGRAM_API_KEY not set" | `export DEEPGRAM_API_KEY="your-key"` in `~/.zshrc` |
| "INVALID KEY" | Check your key at [console.deepgram.com](https://console.deepgram.com) |
| "SoX error" | `brew install sox` (macOS) / `apt install sox` (Linux) / `choco install sox` (Windows) |
| Space doesn't activate | Check `/voice info` — voice may be disabled |
| Double space in editor | Increase typing cooldown or use `Ctrl+Shift+V` |

See [docs/troubleshooting.md](docs/troubleshooting.md) for more.

---

## Security

- **Cloud STT:** Audio is sent to Deepgram for transcription. No local fallback.
- **No telemetry:** pi-voice does not collect or transmit usage data.
- **API key:** Stored in env var or Pi settings file — never logged or exposed in errors.

See [SECURITY.md](SECURITY.md) for vulnerability reporting.

---

## License

[MIT](LICENSE) © 2026 codexstar69

---

## Links

- **npm:** [npmjs.com/package/@codexstar/pi-voice](https://www.npmjs.com/package/@codexstar/pi-voice)
- **GitHub:** [github.com/codexstar69/pi-listen](https://github.com/codexstar69/pi-listen)
- **Pi CLI:** [github.com/mariozechner/pi-coding-agent](https://github.com/mariozechner/pi-coding-agent)

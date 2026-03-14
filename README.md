# pi-listen

<p align="center">
  <img src="assets/banner.png" alt="pi-listen — Voice input for the Pi coding agent" width="100%" />
</p>

**Hold-to-talk voice input for [Pi](https://github.com/mariozechner/pi-coding-agent).**

[![npm version](https://img.shields.io/npm/v/@codexstar/pi-listen.svg)](https://www.npmjs.com/package/@codexstar/pi-listen)
[![license](https://img.shields.io/npm/l/@codexstar/pi-listen.svg)](https://github.com/codexstar69/pi-listen/blob/main/LICENSE)

---

## Setup (2 minutes)

### 1. Install the extension

```bash
# In a regular terminal (not inside Pi)
pi install npm:@codexstar/pi-listen
```

### 2. Get a Deepgram API key

Sign up at [dpgr.am/pi-voice](https://dpgr.am/pi-voice) — $200 free credit, no card needed.

```bash
export DEEPGRAM_API_KEY="your-key-here"    # add to ~/.zshrc or ~/.bashrc
```

### 3. Open Pi

On first launch, pi-listen checks your setup and tells you exactly what's ready and what's missing:
- Deepgram API key — set or not
- Audio capture tool — sox, ffmpeg, or arecord (auto-detected)
- If everything is configured, voice activates immediately with a keybinding guide
- If something is missing, you get step-by-step instructions

### Audio capture

pi-listen auto-detects your audio capture tool. No manual install needed if you already have sox or ffmpeg.

| Priority | Tool | Platforms | Install |
|----------|------|-----------|---------|
| 1 | **SoX** (`rec`) | macOS, Linux, Windows | `brew install sox` / `apt install sox` / `choco install sox` |
| 2 | **ffmpeg** | macOS, Linux, Windows | `brew install ffmpeg` / `apt install ffmpeg` |
| 3 | **arecord** | Linux only | Pre-installed (ALSA) |

If none are found, `/voice test` tells you what to install.

### Verify everything works

```bash
/voice test    # Inside Pi — checks audio tool, mic, and validates API key
```

---

## Usage

### Keybindings

| Action | Key | Notes |
|--------|-----|-------|
| **Record to editor** | Hold `SPACE` (≥1.2s) | Release to finalize. Pre-records during warmup so you don't miss words. |
| **Toggle recording** | `Ctrl+Shift+V` | Works in all terminals — press to start, press again to stop. |
| **Clear editor** | `Escape` × 2 | Double-tap within 500ms to clear all text. |

### How recording works

1. **Hold SPACE** — warmup countdown appears, audio capture starts immediately (pre-recording)
2. **Keep holding** — live transcription streams into the editor as you speak
3. **Release SPACE** — recording continues for 1.5s (tail recording) to catch your last word, then finalizes
4. Text appears in the editor, ready to send

### Commands

| Command | Description |
|---------|-------------|
| `/voice-setup` | Interactive setup wizard (Deepgram key, scope) |
| `/voice test` | Full diagnostics — audio tool, mic capture, API key validation |
| `/voice-language` | Change transcription language (56+ supported, fuzzy picker) |
| `/voice-settings` | Show current voice config |
| `/voice on` / `off` | Enable or disable voice |
| `/voice dictate` | Continuous dictation (no key hold needed) |
| `/voice stop` | Stop active recording or dictation |
| `/voice info` | Show current config and status |
| `/voice history` | Recent transcriptions |
| `/voice` | Toggle on/off |

---

## Features

| Feature | Description |
|---------|-------------|
| **Audio fallback chain** | Tries sox, ffmpeg, arecord in order — works on most systems without extra installs |
| **Pre-recording** | Audio capture starts during warmup countdown — you never miss the first word |
| **Tail recording** | Keeps recording 1.5s after release so your last word isn't clipped |
| **Live streaming** | Deepgram Nova 3 WebSocket — interim transcripts appear as you speak |
| **Reactive waveform** | Audio-level-driven animation with fast attack / slow decay |
| **56+ languages** | `/voice-language` fuzzy picker — Chinese auto-switches to Nova-2 |
| **Continuous dictation** | `/voice dictate` for long-form input without holding keys |
| **Double-escape clear** | Press Escape twice to clear the editor |
| **Zero-config start** | Auto-activates if `DEEPGRAM_API_KEY` is set — no wizard needed |
| **First-run diagnostics** | Checks API key + audio tool on first launch, shows what's ready and what to install |
| **API key validation** | `/voice test` validates your key against the live Deepgram API |
| **Typing cooldown** | Space holds within 400ms of typing are ignored — voice never fires mid-sentence |
| **Sound feedback** | macOS system sounds for start, stop, and error events |
| **Session corruption guard** | Overlapping recording requests abort the stale session first |
| **Cross-platform** | macOS, Windows, Linux — Kitty protocol + non-Kitty fallback |

---

## How It Works

```
Hold SPACE → warmup countdown (pre-recording starts)
                ↓ (≥1.2s)
         Audio capture (sox → ffmpeg → arecord fallback)
                ↓
         Streams PCM to Deepgram Nova 3 via WebSocket
                ↓
         Interim transcripts update editor in real time
                ↓
Release SPACE → tail recording (1.5s) → CloseStream → final transcript
```

### Audio capture fallback chain

pi-listen tries three audio backends in order and uses the first one found:

1. **SoX** (`rec`) — purpose-built for recording, best quality
2. **ffmpeg** — widely available, captures from default mic (avfoundation on macOS, pulse on Linux, dshow on Windows)
3. **arecord** — built into Linux ALSA, zero install needed

The result is cached for the process lifetime. If none are found, you get a clear error with install instructions.

### Hold detection

**Kitty protocol** (Ghostty, Kitty, WezTerm, Windows Terminal 1.22+):
True key-down/repeat/release events. First press enters warmup immediately. Release stops recording.

**Non-Kitty** (macOS Terminal, older terminals):
Gap-based detection. Counts rapid key-repeat events to confirm hold. Gap in repeats = released.

Both modes: ≥1.2s hold to activate. Quick taps type a normal space.

### Architecture

```
extensions/voice.ts              Main extension — state machine, recording, UI
extensions/voice/config.ts       Config loading, saving, migration
extensions/voice/onboarding.ts   First-run setup wizard, language picker
extensions/voice/deepgram.ts     Deepgram URL builder, API key resolver
```

---

## Configuration

Settings in Pi's settings files under the `voice` key:

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
    "onboarding": { "completed": true, "schemaVersion": 2 }
  }
}
```

---

## Troubleshooting

Run `/voice test` inside Pi for full diagnostics. It checks your audio capture tool, mic, and validates your Deepgram API key against the live API.

| Problem | Solution |
|---------|----------|
| "DEEPGRAM_API_KEY not set" | [Get a key](https://dpgr.am/pi-voice) → `export DEEPGRAM_API_KEY="..."` in `~/.zshrc` |
| "INVALID KEY" | Check at [console.deepgram.com](https://console.deepgram.com) |
| "No audio capture tool found" | Install one of: `brew install sox`, `brew install ffmpeg`, or use `arecord` (Linux, pre-installed) |
| Space doesn't activate voice | Run `/voice info` — voice may be disabled or onboarding incomplete |
| Voice triggers in fuzzy search | Typing cooldown should prevent this — try `Ctrl+Shift+V` instead |

See [docs/troubleshooting.md](docs/troubleshooting.md) for more.

---

## Security

- **Cloud STT** — audio is sent to Deepgram for transcription
- **No telemetry** — pi-listen does not collect or transmit usage data
- **API key** — stored in env var or Pi settings, never logged or exposed in errors

See [SECURITY.md](SECURITY.md) for vulnerability reporting.

---

## License

[MIT](LICENSE) © 2026 codexstar69

---

## Links

- **npm:** [npmjs.com/package/@codexstar/pi-listen](https://www.npmjs.com/package/@codexstar/pi-listen)
- **GitHub:** [github.com/codexstar69/pi-listen](https://github.com/codexstar69/pi-listen)
- **Deepgram:** [dpgr.am/pi-voice](https://dpgr.am/pi-voice) ($200 free credit)
- **Pi CLI:** [github.com/mariozechner/pi-coding-agent](https://github.com/mariozechner/pi-coding-agent)

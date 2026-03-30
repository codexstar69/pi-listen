# pi-listen troubleshooting

## First things to check

Run these built-in commands inside Pi:

- `/voice test` — checks SoX, mic capture, and validates your Deepgram API key against the live API
- `/voice info` — shows current config, state, and whether Kitty protocol is detected

If you only do one thing, start with `/voice test`.

## Symptom: "DEEPGRAM_API_KEY not set"

### What it means
No Deepgram API key was found in environment variables or saved Pi settings.

### Fix
1. Get a free key at [dpgr.am/pi-voice](https://dpgr.am/pi-voice) ($200 free credit, no card needed)
2. Set it in your shell:
   ```sh
   export DEEPGRAM_API_KEY="your-key-here"
   ```
3. Add to `~/.zshrc` or `~/.bashrc` for persistence
4. Or run `/voice-setup` inside Pi to paste it interactively

### Important behavior
- If the key comes from your shell environment, pi-listen uses it at runtime
  only and does not copy it into `~/.pi/agent/settings.json`
- If you paste a key during onboarding, that is an explicit save and it goes to
  `~/.env.secrets` or `~/.zshrc`

## Symptom: "INVALID KEY" from `/voice test`

### What it means
The API key is set but Deepgram rejected it.

### Fix
- Check your key at [console.deepgram.com](https://console.deepgram.com)
- Make sure you're using an API key, not a project ID
- Verify your Deepgram account has available credits

## Symptom: "Voice requires SoX. Install: brew install sox"

### What it means
pi-listen could not find the `rec` command used for audio recording.

### Fix
Install SoX:

```sh
brew install sox          # macOS
sudo apt install sox      # Ubuntu/Debian
choco install sox         # Windows
```

Then restart Pi or run `/voice test` again.

## Symptom: recording starts, but transcription is empty or says "No speech detected"

### Likely causes
- Recording was too short
- Microphone input level is too low
- Background noise or device permissions interfered
- Microphone is muted or pointing at wrong input device

### Fixes
- Hold the record key longer (at least 2-3 seconds of speech)
- Run `/voice test` to validate mic capture — check the byte count
- On macOS: System Settings → Privacy & Security → Microphone — ensure your terminal app has access
- Try `Ctrl+Shift+V` instead of hold-SPACE if the hold detection isn't working for your terminal

## Symptom: "Microphone captured no audio"

### What it means
SoX's `rec` command ran but produced zero audio data. This usually means a permissions issue.

### Fix (macOS)
1. System Settings → Privacy & Security → Microphone
2. Enable microphone access for your terminal app (Ghostty, Terminal, iTerm2, etc.)
3. You may need to restart the terminal after granting access

### Fix (Linux)
- Check PulseAudio/PipeWire is running: `pactl info`
- Verify recording works: `rec -d 2 /tmp/test.wav && play /tmp/test.wav`

## Symptom: space doesn't activate voice

### Common causes
- Voice is disabled — run `/voice on`
- Onboarding not completed — run `/voice-setup`
- Typing cooldown active — if you were just typing, wait 400ms before holding SPACE
- You're in a picker, search, or other non-editor context — use `Ctrl+Shift+V` instead

### What to check
- `/voice info` — confirm `enabled: true` and `setup: complete`
- The status bar should show `󰍬 Cloud` when voice is ready (`MIC STREAM` if you switch Status back to Classic)

## Symptom: voice triggers accidentally while typing

### What it means
Your terminal may be sending key-repeat events faster than expected. The typing cooldown (400ms) should prevent this in most cases.

### Workarounds
- Use `Ctrl+Shift+V` exclusively (toggle mode, no hold detection involved)
- Kitty-protocol terminals now wait ~200ms before showing warmup, which reduces brief accidental flashes
- The hold threshold is 1200ms — a normal space press is well under this

## Symptom: "Connection lost" or "Deepgram connection timed out"

### What it means
The WebSocket connection to Deepgram failed or dropped.

### Fix
- Check your internet connection
- Verify your API key is valid: `/voice test`
- If behind a corporate firewall or VPN, ensure `wss://api.deepgram.com` is accessible
- Try again — transient network issues resolve on retry

## Symptom: "No response from Deepgram (15s)"

### What it means
Audio was streaming but Deepgram sent no response for 15 seconds. This usually indicates a network issue or invalid API key.

### Fix
- Run `/voice test` to validate your API key
- Check network connectivity to `api.deepgram.com`

## Symptom: last word gets cut off

### What it means
You released SPACE before finishing your sentence. The tail recording feature (1.5s) should catch trailing words, but if you pause longer than that before the final word, it may be missed.

### Workaround
- Finish speaking before releasing SPACE
- Or use `/voice dictate` for continuous dictation (no hold needed)

## Symptom: project config is ignored

### What it means
Either:
- The config was saved globally instead of at project scope
- The project does not have `.pi/settings.json`

### Fix
1. Re-run `/voice-setup` and select **Project only** when prompted
2. Inspect both files:
   - `~/.pi/agent/settings.json` (global)
   - `<project>/.pi/settings.json` (project)
3. Project settings override global settings

## Debug logging

Set `PI_VOICE_DEBUG=1` to enable verbose debug logging:

```sh
PI_VOICE_DEBUG=1 pi
```

Logs go to stderr and to a file at `$TMPDIR/pi-voice-debug.log`. This shows the full state machine transitions, key events, and Deepgram connection lifecycle.

## If you are still stuck

Capture these pieces of information before asking for help:

1. `/voice test` output
2. `/voice info` output
3. Your OS, terminal app, and Pi version
4. Debug log (run with `PI_VOICE_DEBUG=1`)

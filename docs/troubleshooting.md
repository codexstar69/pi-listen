# pi-voice troubleshooting

This guide focuses on the current `pi-voice` behavior and the most likely setup/runtime issues.

## First things to check

Run these built-in commands first:

- `/voice info` — shows the active config the extension believes it should use
- `/voice test` — checks SoX, daemon state, and current model readiness
- `/voice backends` — lists detected STT backends, installed models, and install hints
- `/voice doctor` — compares how to repair the current config vs a recommended alternative
- `/voice daemon status` — shows the current daemon backend/model state
- `/voice setup` — re-run backend/model selection

If you only do one thing, start with `/voice test`.

## Symptom: "Voice requires SoX. Install: brew install sox"

### What it means
`pi-voice` could not find the `rec` command used for audio recording.

### Fix
Install SoX:

```sh
brew install sox
```

Then restart Pi or run `/voice test` again.

### Why this matters
Without SoX, the extension cannot record microphone input, even if the transcription backend itself is installed correctly.

## Symptom: `/voice backends` shows everything as unavailable

### What it means
No STT backend is currently detected.

### Common fixes
Choose one path:

#### Local default path
```sh
python3 -m pip install faster-whisper
```

#### Lightweight local path
```sh
python3 -m pip install 'useful-moonshine[onnx]'
```

#### whisper.cpp path
```sh
brew install whisper-cpp
```

#### Cloud path
Set a Deepgram API key in your shell environment:

```sh
export DEEPGRAM_API_KEY=your_key_here
```

Then restart Pi so the environment is visible to the extension.

## Symptom: `/voice test` says `SoX (rec): OK` but `Daemon: not running`

### What it means
The warm daemon is not currently running. This is not always fatal because `pi-voice` can still fall back to direct transcription subprocesses.

### Fix
Start it manually:

```text
/voice daemon start
```

Then inspect it:

```text
/voice daemon status
```

### If it still will not start
Check Python availability and backend installation:

```sh
python3 --version
python3 transcribe.py --list-backends
```

## Symptom: `/voice daemon status` shows the wrong backend or model

### What it means
The running daemon does not match the config you expect.

Recent work in this repo is moving toward config-specific sockets and more explicit backend/model requests, but if you still see mismatch behavior, treat it as a runtime desynchronization issue.

### Fixes
1. Re-run setup:
   ```text
   /voice setup
   ```
2. Stop the daemon:
   ```text
   /voice daemon stop
   ```
3. Start it again:
   ```text
   /voice daemon start
   ```
4. Re-check:
   ```text
   /voice daemon status
   ```

## Symptom: recording starts, but transcription is empty or says "No speech detected"

### Likely causes
- recording was too short
- microphone input level is too low
- background noise or device permissions interfered
- the backend is installed but not functioning correctly for the chosen model

### Fixes
- hold the record key a bit longer
- try `/voice test` first to validate microphone capture
- confirm the recorded sample file is not empty
- switch to a more conservative model/backend through `/voice setup`

## Symptom: cloud setup is selected, but transcription still fails

### Likely causes
- `DEEPGRAM_API_KEY` is missing or invalid
- Pi was launched before the shell environment contained the key
- network access is blocked or failing

### Fixes
1. Verify the environment variable exists in the shell that launches Pi:
   ```sh
   echo $DEEPGRAM_API_KEY
   ```
2. Restart Pi after setting the variable.
3. Confirm the backend is detected:
   ```sh
   python3 transcribe.py --list-backends
   ```
4. Re-run `/voice setup` if needed.

## Symptom: backend is installed, but the selected model is still reported as missing

### What it means
`pi-voice` can see the backend package or CLI, but it does not see the specific model you selected as already available locally.

### Typical examples
- `faster-whisper` installed, but `medium` or `large-v3-turbo` not cached yet
- `whisper-cpp` installed, but no `ggml-<model>.bin` file found
- backend available, but onboarding marks the selected model as **download required**

### Fixes
- choose an **installed** model in onboarding if one is already available
- keep the current model and allow first use to download it if that is acceptable
- use `/voice backends` to inspect installed-model hints
- use `/voice doctor` to compare your current setup with a recommended alternative

## Symptom: backend is installed, but model status is unknown

### What it means
The backend package exists, but `pi-voice` cannot verify local model presence with high confidence for that backend.

This is a conservative result, not necessarily an error.

### Fixes
- try the chosen model anyway if you expect it to already exist
- use `/voice test` and `/voice doctor` to see whether repair is still needed
- if you want a more deterministic local path, prefer a backend with stronger model detection, such as `faster-whisper` or `whisper-cpp`

## Symptom: local backend selected, but transcription is slow

### What it means
The chosen local model may be too heavy for the current machine or use case.

### Fixes
- switch to a smaller model (`small`, `small.en`, or backend default)
- prefer an already-installed smaller model if onboarding shows one
- prefer `faster-whisper` as the conservative local default
- use cloud mode if setup speed and responsiveness matter more than privacy/offline behavior

## Symptom: project config is ignored

### What it means
Either:
- the config was saved globally instead of at project scope, or
- the project does not have `.pi/settings.json`, or
- an older config file is still being read

### Fixes
1. Re-run setup and select **Project only** when prompted.
2. Inspect both files:
   - `~/.pi/agent/settings.json`
   - `.pi/settings.json`
3. Remember that project settings are intended to override global settings.

## Symptom: the hold-to-talk shortcut does nothing

### Current behavior to remember
- hold **Space** to talk only when the editor is empty
- `Ctrl+Shift+V` is the fallback toggle shortcut
- `Ctrl+Shift+B` is the BTW voice shortcut

### Fixes
- make sure the editor is empty before using hold-Space
- try `Ctrl+Shift+V` instead
- use `/voice on` if voice was disabled
- run `/voice info` to confirm `enabled: true`

## Symptom: "Recording too short" or "No audio recorded"

### What it means
The audio file was missing, too small, or recording ended before a usable sample was captured.

### Fixes
- hold the key slightly longer
- try a direct microphone test via `/voice test`
- confirm SoX can record in your environment
- avoid tapping the shortcut too quickly

## Manual backend checks

These are useful outside Pi too:

```sh
python3 transcribe.py --list-backends
python3 daemon.py ping
python3 daemon.py status
```

If you are debugging local setup, `--list-backends` is usually the most useful first command because it now includes installed-model hints and detection metadata.

## When to re-run setup

Use `/voice setup` again when:
- switching from cloud to local or vice versa
- changing model sizes
- moving from global to project scope
- recovering from a broken dependency install

## If you are still stuck

Capture these four pieces of information before debugging further:

1. `/voice info` output
2. `/voice test` output
3. `/voice backends` output
4. `/voice daemon status` output

That is usually enough to identify whether the issue is:
- recording
- backend installation
- selected model missing vs already installed
- model status unknown
- API credentials
- daemon state
- config scope

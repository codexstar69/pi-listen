# pi-listen — API Documentation

## Extension API

pi-listen registers as a Pi extension via the `pi` field in `package.json`:

```json
{
  "pi": {
    "extensions": ["./extensions/voice.ts"]
  }
}
```

The extension exports a default function that receives the Pi `ExtensionAPI`.

---

## Daemon Protocol

The STT daemon communicates via Unix domain sockets using newline-delimited JSON.

### Socket Location

Socket path is derived from config hash:
```
/tmp/pi-voice-<sha1_12chars>.sock
```

### Commands

#### `ping`

Check if daemon is alive.

**Request:**
```json
{"cmd": "ping"}
```

**Response:**
```json
{"status": "ok", "pid": 12345}
```

---

#### `status`

Get daemon status and loaded model info.

**Request:**
```json
{"cmd": "status"}
```

**Response:**
```json
{
  "status": "running",
  "pid": 12345,
  "uptime": 120.5,
  "requests": 42,
  "idle": 5.2,
  "backend": "faster-whisper",
  "model": "small",
  "model_loaded": true
}
```

---

#### `load`

Load or switch to a specific backend and model.

**Request:**
```json
{"cmd": "load", "backend": "faster-whisper", "model": "small"}
```

**Response:**
```json
{"status": "loaded", "backend": "faster-whisper", "model": "small", "load_time": 2.34}
```

Or if already loaded:
```json
{"status": "already_loaded", "backend": "faster-whisper", "model": "small"}
```

---

#### `transcribe`

Transcribe an audio file.

**Request:**
```json
{
  "cmd": "transcribe",
  "audio": "/path/to/file.wav",
  "language": "en",
  "vad": true
}
```

**Response:**
```json
{
  "text": "hello world",
  "duration": 1.23,
  "backend": "faster-whisper",
  "model": "small",
  "language": "en"
}
```

With VAD skip (no speech detected):
```json
{
  "text": "",
  "duration": 0,
  "vad": {"has_speech": false, "vad_available": true, "segments": 0},
  "skipped": true
}
```

---

#### `vad`

Run Voice Activity Detection only (no transcription).

**Request:**
```json
{"cmd": "vad", "audio": "/path/to/file.wav"}
```

**Response:**
```json
{
  "has_speech": true,
  "vad_available": true,
  "segments": 3,
  "speech_duration_ms": 4200
}
```

---

#### `backends`

List all available backends and their status.

**Request:**
```json
{"cmd": "backends"}
```

**Response:**
```json
{
  "backends": [
    {
      "name": "faster-whisper",
      "available": true,
      "type": "local",
      "default_model": "small",
      "models": ["tiny", "base", "small", "medium", "large-v3"]
    }
  ]
}
```

---

#### `shutdown`

Gracefully stop the daemon.

**Request:**
```json
{"cmd": "shutdown"}
```

**Response:**
```json
{"status": "shutting_down"}
```

---

## CLI Usage

### transcribe.py

```bash
# Transcribe with auto-detected backend
python3 transcribe.py audio.wav

# Specify backend and model
python3 transcribe.py --backend faster-whisper --model small audio.wav

# Specify language
python3 transcribe.py --language auto audio.wav

# List available backends
python3 transcribe.py --list-backends

# List models for a backend
python3 transcribe.py --list-models --backend faster-whisper
```

### daemon.py

```bash
# Start daemon
python3 daemon.py start --backend faster-whisper --model small

# Start with custom socket path
python3 daemon.py start --socket /tmp/my-daemon.sock

# Check status
python3 daemon.py status

# Transcribe via daemon
python3 daemon.py transcribe audio.wav --language en --vad

# Load a different model
python3 daemon.py load --backend moonshine --model moonshine/base

# Stop daemon
python3 daemon.py stop

# Ping daemon
python3 daemon.py ping
```

---

## Error Responses

All errors follow this format:

```json
{"error": "Human-readable error message"}
```

Common errors:

| Error | Cause |
|-------|-------|
| `"Audio file not found: /path"` | Audio file doesn't exist |
| `"No model loaded. Send 'load' first."` | Transcribe called before loading a model |
| `"No STT backend found"` | No backend is installed or available |
| `"Unknown command: foo"` | Invalid command sent to daemon |
| `"Message exceeds maximum size"` | Request larger than 1 MB |
| `"Daemon not running"` | Socket connection failed |
| `"Daemon timeout"` | Daemon didn't respond within timeout |

---

## Audio Requirements

| Property | Value |
|----------|-------|
| Format | WAV |
| Sample rate | 16000 Hz |
| Channels | 1 (mono) |
| Bit depth | 16-bit |

Audio is recorded via SoX's `rec` command with these parameters automatically.

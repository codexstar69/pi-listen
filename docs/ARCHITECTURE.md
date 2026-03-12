# pi-listen Architecture

## System Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           Pi CLI (Node.js)                               │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                      extensions/voice.ts                            │ │
│  │                                                                     │ │
│  │  ┌───────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │ │
│  │  │ Recording  │  │ Daemon   │  │   BTW    │  │   Hold-to-Talk   │  │ │
│  │  │ (SoX rec)  │  │   IPC    │  │ Threads  │  │  (Kitty proto)   │  │ │
│  │  └─────┬─────┘  └────┬─────┘  └────┬─────┘  └────────┬─────────┘  │ │
│  │        │              │             │                  │            │ │
│  └────────┼──────────────┼─────────────┼──────────────────┼────────────┘ │
│           │              │             │                  │              │
│  ┌────────┼──────────────┼─────────────┼──────────────────┼────────────┐ │
│  │        │     voice/config.ts   voice/diagnostics.ts                 │ │
│  │        │     voice/onboarding.ts  voice/install.ts                  │ │
│  └────────┼──────────────┼─────────────────────────────────────────────┘ │
│           │              │                                               │
└───────────┼──────────────┼───────────────────────────────────────────────┘
            │              │ Unix Domain Socket
            │              │ (newline-delimited JSON)
            │              ▼
┌───────────┼──────────────────────────────────────────────────────────────┐
│           │          daemon.py (Python, persistent)                       │
│           │                                                              │
│  ┌────────▼────────┐  ┌─────────────────┐  ┌──────────────────────────┐ │
│  │   Audio Files    │  │   ModelCache    │  │    DaemonServer          │ │
│  │   (WAV, /tmp)    │  │  (thread-safe)  │  │  (socket, idle timeout)  │ │
│  └─────────────────┘  └────────┬────────┘  └──────────────────────────┘ │
│                                │                                         │
│                       ┌────────▼────────┐                                │
│                       │  transcribe.py  │                                │
│                       │   (5 backends)  │                                │
│                       └────────┬────────┘                                │
│                                │                                         │
│              ┌─────────────────┼─────────────────────┐                   │
│              ▼                 ▼                     ▼                   │
│  ┌──────────────────┐ ┌──────────────┐ ┌──────────────────┐             │
│  │  faster-whisper   │ │  whisper-cpp │ │    Deepgram      │             │
│  │  moonshine        │ │              │ │  (cloud API)     │             │
│  │  parakeet         │ │              │ │                  │             │
│  └──────────────────┘ └──────────────┘ └──────────────────┘             │
└──────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Voice Input Flow

```
1. User holds SPACE (or Ctrl+Shift+V)
2. voice.ts spawns `rec` (SoX) → writes WAV to /tmp
3. User releases key
4. voice.ts kills `rec`
5. voice.ts sends {"cmd": "transcribe", "audio": "/tmp/pi-voice-XXX.wav"} to daemon
6. daemon.py runs transcription via loaded model
7. daemon.py returns {"text": "...", "duration": 1.23}
8. voice.ts injects text into Pi editor
9. voice.ts deletes temp WAV file
```

### BTW Flow

```
1. User types /btw <message> (or holds Ctrl+Shift+B for voice)
2. voice.ts builds context from system prompt + prior BTW thread
3. voice.ts streams LLM response via pi-ai streamSimple()
4. Response displayed in BTW widget above editor
5. User can /btw:inject to push thread into main agent context
```

### Daemon Lifecycle

```
1. Extension starts → ensureDaemon() checks if daemon is running
2. If not running → spawns python3 daemon.py start --socket <path>
3. Daemon pre-loads configured backend + model
4. Daemon serves requests on Unix socket
5. After 5 minutes of inactivity → auto-shutdown
6. On session_shutdown → extension does NOT kill daemon (persists for reuse)
```

## Thread Safety

- **ModelCache**: Protected by `threading.Lock()` — only one transcription runs at a time
- **DaemonServer**: Each client connection handled in a separate daemon thread
- **voice.ts state**: Single-threaded Node.js event loop — no mutex needed, but async races are possible (documented in bug-hunter audit)

## Configuration Resolution

```
1. Check project settings: <cwd>/.pi/settings.json → voice key
2. Check global settings: ~/.pi/agent/settings.json → voice key
3. Fall back to DEFAULT_CONFIG (auto backend, small model, enabled)
```

Project settings override global settings completely (no merging).

## Security Boundaries

| Boundary | Protection |
|----------|------------|
| Unix socket | Local-only (no TCP), file permissions |
| Message size | MAX_MSG_SIZE = 1 MB |
| Audio files | Validated with os.path.exists() |
| Backend names | Validated against BACKENDS registry |
| Error responses | No stack traces (logged to stderr only) |
| Idle timeout | Daemon auto-shuts down after 5 minutes |
| Temp files | Deleted immediately after transcription |

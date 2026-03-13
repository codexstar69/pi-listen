# Architecture

> Updated for v2.x (Deepgram streaming only).

## Overview

pi-listen is a Pi extension that adds voice input via Deepgram's WebSocket
streaming API. Audio is captured with SoX (`rec`), streamed to Deepgram
Nova-3, and interim/final transcripts are shown live in the editor.

## Modules

```
extensions/
├── voice.ts                  # Main extension (state machine, UI, key handling)
└── voice/
    ├── config.ts             # Config loading, saving, migration
    ├── onboarding.ts         # Setup wizard (API key + scope)
    ├── text-processing.ts    # Voice command detection, text shortcuts
    └── deepgram.ts           # Deepgram URL builder, API key resolution
```

## State Machine

```
idle → warmup → recording → finalizing → idle
         ↑         │
         └─────────┘  (rapid re-press recovery)
```

- **idle:** Waiting for SPACE hold or Ctrl+Shift+V
- **warmup:** User is holding SPACE, countdown shown (800ms threshold)
- **recording:** SoX captures audio → Deepgram WebSocket streaming
- **finalizing:** CloseStream sent, waiting for final transcript

## Hold-to-Talk Detection

Two terminal modes:

### A) Kitty Protocol (Ghostty on Linux, Kitty, WezTerm)
True key-down/repeat/release events. On first SPACE press, enter warmup
immediately. True release event cancels or stops recording.

### B) Non-Kitty (macOS Terminal, Ghostty on macOS)
No key-release events. Holding sends rapid press events (~30-90ms apart).
Tap = one press. Hold = 3+ rapid presses → enter warmup. Release detected
by gap in presses (250ms during recording, 500ms during warmup).

## Audio Pipeline

```
SoX rec (native rate) → rate effect (16kHz) → WebSocket → Deepgram
                                             ↓
                              interim/final transcript → editor
```

SoX captures at native CoreAudio rate and applies the `rate` effect to
downsample to 16kHz. Buffer size is 4096 bytes for low latency.

## Safety Features

- **Re-entrancy guard:** Prevents overlapping `startVoiceRecording` calls
- **Session corruption guard:** Auto-aborts stale sessions before new ones
- **Connection timeout:** 10s WebSocket connection timeout
- **Error cooldown:** 3-5s block on re-activation after errors
- **Keepalive:** 8s keepalive pings to Deepgram
- **Max recording:** 120s automatic stop
- **Escape cancel:** Restores editor text to pre-recording state

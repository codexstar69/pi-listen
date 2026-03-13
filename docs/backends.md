# STT Backend

> As of v2.x, pi-listen uses **Deepgram Nova-3** exclusively via WebSocket streaming.

## Deepgram Nova-3

| Property | Value |
|----------|-------|
| Type | Cloud |
| Model | nova-3 |
| Encoding | linear16 (16-bit signed integer) |
| Sample rate | 16000 Hz |
| Channels | 1 (mono) |
| Features | Interim results, smart formatting, endpointing |
| Latency | ~200ms for interim, ~1s for final |
| Languages | 30+ (configurable via `language` setting) |

### Setup

```bash
export DEEPGRAM_API_KEY="your-key-here"
```

Get a free key with $200 credits (no credit card) at [dpgr.am/pi-voice](https://dpgr.am/pi-voice).

### How It Works

1. SoX `rec` captures audio at native rate, downsamples to 16kHz via `rate` effect
2. Audio chunks are sent to Deepgram's WebSocket endpoint in real-time
3. Deepgram returns interim (partial) and final transcripts
4. Interim results update the editor live; final results replace when complete

### Historical

v1.x supported local backends (faster-whisper, moonshine, whisper-cpp, parakeet)
via a Python daemon. These were removed in v2.0 in favor of streaming-only
Deepgram integration for lower latency and simpler architecture.

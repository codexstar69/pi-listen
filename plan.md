# Plan: Dual-Engine Architecture — Moonshine Streaming + Sherpa-onnx

## Goal
Add Moonshine native streaming engine alongside sherpa-onnx so we can use the streaming-only
Small (123M) and Medium (245M) models with **live interim transcription** — fully offline.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Local Backend (local.ts)                      │
│                                                                 │
│  transcribeInProcess() dispatches to correct engine:            │
│                                                                 │
│  ┌─────────────────────┐     ┌──────────────────────────┐      │
│  │  Sherpa Engine       │     │  Moonshine Streaming     │      │
│  │  (sherpa-engine.ts)  │     │  Engine                  │      │
│  │                      │     │  (moonshine-engine.ts)   │      │
│  │  Runtime:            │     │                          │      │
│  │   sherpa-onnx-node   │     │  Runtime:                │      │
│  │                      │     │   onnxruntime-node       │      │
│  │  Models:             │     │                          │      │
│  │  - Whisper           │     │  Models:                 │      │
│  │  - Moonshine v1/v2   │     │  - Moonshine Small Str.  │      │
│  │    (non-streaming)   │     │  - Moonshine Medium Str. │      │
│  │  - Parakeet          │     │                          │      │
│  │  - GigaAM            │     │  Mode: TRUE STREAMING    │      │
│  │  - SenseVoice        │     │  (live interim results   │      │
│  │                      │     │   as user speaks)        │      │
│  │  Mode: BATCH         │     │                          │      │
│  │  (record → transcribe│     │  Files per model:        │      │
│  │   after stop)        │     │  encoder.ort             │      │
│  │                      │     │  frontend.ort            │      │
│  └──────────────────────┘     │  decoder_kv.ort          │      │
│                               │  cross_kv.ort            │      │
│                               │  adapter.ort             │      │
│                               └──────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

## Inference Pipeline (Moonshine Streaming)

From C++ source (moonshine-streaming-model.h):
1. **Audio → Frontend**: Raw 16kHz PCM → `frontend.ort` → mel features (80-sample frames)
2. **Frontend → Encoder**: Accumulated frames → `encoder.ort` → hidden states
3. **Encoder → Adapter**: Hidden states → `adapter.ort` → memory representation
4. **Memory → Cross-KV**: Memory buffer → `cross_kv.ort` → precomputed cross-attention K/V
5. **Decode loop**: Token-by-token via `decoder_kv.ort` using self-attention KV cache + cross-KV

State between chunks: KV caches (self + cross), memory buffer, adapter position offset.
Vocab size: 32,768 tokens. Decoder dim: 320. Frame lookahead: 16 frames.

## Implementation Steps

### Step 1: Add onnxruntime-node dependency
- `bun add onnxruntime-node` (v1.24.3, N-API binding)
- Verify Bun compatibility (same N-API concern as sherpa-onnx-node)

### Step 2: Add streaming model entries to LOCAL_MODELS
- `moonshine-v2-small-streaming`: ~280 MB (est), 5 ORT files from download.moonshine.ai
- `moonshine-v2-medium-streaming`: ~560 MB (est), 5 ORT files from download.moonshine.ai
- New `sherpaModel.type` value: `"moonshine_streaming"` (signals dispatch to moonshine engine)
- Download URLs: `https://download.moonshine.ai/model/{small,medium}-streaming-en/quantized/{file}`
- Also need `tokenizer.bin` and `streaming_config.json` per model

### Step 3: Create moonshine-engine.ts
- Load 5 ORT sessions from model directory
- Parse `streaming_config.json` for model dimensions
- Implement streaming state management:
  - Sample buffer (79 samples between chunks)
  - KV caches (self-attention + cross-attention)
  - Memory accumulation buffer
  - Adapter position tracking
- Export functions matching sherpa-engine pattern:
  - `initMoonshineStreaming(): Promise<boolean>`
  - `createStreamingSession(modelDir, config): MoonshineStreamingSession`
  - `feedAudioChunk(session, pcmFloat32): string[]` — returns new tokens
  - `finalizeSession(session): string` — flush remaining audio
  - `destroySession(session): void`

### Step 4: Update model-download.ts
- Support `download.moonshine.ai` URLs (not just HuggingFace)
- Add `tokenizer.bin` and `streaming_config.json` to file role handling

### Step 5: Update local.ts transcription dispatch
- In `transcribeInProcess()`: check if model type is `moonshine_streaming`
  - If yes → use moonshine-engine (streaming mode with interim callbacks)
  - If no → use sherpa-engine (batch mode, existing behavior)
- For streaming models, audio chunks stream through in real-time instead of buffering

### Step 6: Wire streaming into voice.ts session management
- Streaming moonshine models behave like Deepgram (live interim results)
- New session type or extend LocalSession with streaming capability
- SoX stdout → feedAudioChunk() → onTranscript(interim, finals)
- This gives local backend the same UX as Deepgram (live partial transcripts)

### Step 7: Update device.ts model fitness
- Streaming models need slightly more RAM (5 sessions + KV caches)
- Adjust `runtimeRamMB` estimates for streaming overhead

### Step 8: Tests
- Unit tests for moonshine-engine tensor operations
- Integration test: verify model catalog entries (download URLs, file roles)
- Device fitness tests for new streaming models

## Key Design Decisions

1. **Separate engine file** — `moonshine-engine.ts` is independent from `sherpa-engine.ts`
   (different runtime, different inference pipeline, different state model)

2. **Dispatch by model type** — `sherpaModel.type === "moonshine_streaming"` routes to
   moonshine engine; all other types go to sherpa engine. Clean separation.

3. **Streaming models get live interim results** — unlike batch sherpa models, streaming
   moonshine feeds audio chunks continuously and emits partial transcripts. This gives
   the local backend Deepgram-like UX without any cloud dependency.

4. **download.moonshine.ai as source** — these models aren't on HuggingFace in ORT format,
   so we use the official moonshine CDN directly.

## Files to Create/Modify

| File | Action |
|------|--------|
| `extensions/voice/moonshine-engine.ts` | **CREATE** — streaming inference engine |
| `extensions/voice/local.ts` | **MODIFY** — add 2 streaming models, dispatch logic |
| `extensions/voice/model-download.ts` | **MODIFY** — support moonshine.ai URLs |
| `extensions/voice/device.ts` | **MODIFY** — streaming RAM estimates |
| `extensions/voice/config.ts` | **MODIFY** — no changes needed (backend stays "local") |
| `tests/device.test.ts` | **MODIFY** — add streaming model tests |
| `tests/model-download.test.ts` | **MODIFY** — add moonshine.ai URL validation |
| `package.json` | **MODIFY** — add onnxruntime-node dependency |

## Risk Assessment

- **Bun + onnxruntime-node N-API**: Same risk as sherpa-onnx-node. Test first.
- **Tensor shape mismatches**: C++ header gives us shapes but we need to verify at runtime.
  The streaming_config.json should provide definitive dimensions.
- **KV cache memory**: Streaming models hold state in memory. Need proper cleanup on abort.
- **tokenizer.bin format**: Need to understand BPE tokenizer binary format for token→text.
  May need to bundle a tokenizer decoder or use the tokens.txt if available.

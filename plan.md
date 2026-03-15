# Plan: Seamless Local Model Setup — Auto-Download, Device Detection, Server Management

## Problem Statement

The local transcription backend requires users to manually:
1. Clone and build whisper.cpp (or another server)
2. Download model files from HuggingFace
3. Start the server on the correct port
4. Know which models fit their device

This makes local mode unusable for most users. We need a seamless "pick a model → it works" experience.

## Architecture Decision: Why whisper.cpp Server

pi-listen talks to local servers via OpenAI-compatible HTTP API (`POST /v1/audio/transcriptions`). We need a server binary. Options considered:

| Option | Pros | Cons |
|--------|------|------|
| whisper.cpp server | Pre-built binaries on GitHub Releases, supports GGML models, widely used, CPU-optimized | Only supports Whisper models |
| faster-whisper-server | Python, good accuracy | Requires Python + pip, heavy |
| transcribe-rs (Handy) | Supports ALL 20 models (Moonshine, SenseVoice, etc.) | No standalone server binary published |
| ollama-style approach | Familiar UX | Would need to build from scratch |

**Decision**: Support **whisper.cpp server** as the auto-managed backend (covers Whisper models with pre-built binaries), with the existing manual server option preserved for users who want Moonshine/SenseVoice/etc. via their own server. This is pragmatic — whisper.cpp has GitHub Releases with pre-built binaries for every platform.

For non-Whisper models, we'll detect if the user picks one and guide them to the right server setup (transcribe-rs/Handy), rather than trying to auto-manage every possible server.

## Implementation Plan

### Step 1: Device Detection Module (`extensions/voice/device.ts`)

Add hardware detection to recommend appropriate models and warn about incompatible ones.

```typescript
export interface DeviceProfile {
  platform: "darwin" | "linux" | "win32";
  arch: "arm64" | "x64" | "arm";         // process.arch
  totalRamMB: number;                      // os.totalmem()
  cpuCores: number;                        // os.cpus().length
  cpuModel: string;                        // os.cpus()[0].model
  isRaspberryPi: boolean;                  // detected from cpuModel or /proc/device-tree/model
  hasGpu: boolean;                         // heuristic: check for nvidia-smi or metal support
}

export function detectDevice(): DeviceProfile { ... }
```

**Model fitness scoring**:
- Add `minRamMB` and `recRamMB` fields to `LocalModelInfo`
- Add `tier: "edge" | "standard" | "heavy"` field
- Filter/sort models during onboarding based on device profile
- Show warnings like "⚠ This model needs ~1.1 GB RAM, your device has 512 MB"

Concrete thresholds (based on model sizes + runtime overhead):

| Model | Size | Min RAM | Tier |
|-------|------|---------|------|
| moonshine-v2-tiny | ~31 MB | 256 MB | edge |
| moonshine-tiny | ~60 MB | 512 MB | edge |
| moonshine-v2-small | ~100 MB | 512 MB | edge |
| moonshine-base | ~130 MB | 512 MB | edge |
| moonshine-v2-medium | ~192 MB | 1 GB | standard |
| sensevoice-small | ~228 MB | 1 GB | standard |
| gigaam-v3 | ~225 MB | 1 GB | standard |
| parakeet-v2 | 473 MB | 2 GB | standard |
| whisper-small | 487 MB | 2 GB | standard |
| whisper-medium | 492 MB | 2 GB | standard |
| whisper-large | 1.1 GB | 4 GB | heavy |
| whisper-turbo | 1.6 GB | 4 GB | heavy |

**Raspberry Pi detection**: Read `/proc/device-tree/model` (Linux) or check `cpuModel` for BCM/Cortex patterns. If RPi detected, auto-filter to edge-tier models and show a recommendation.

### Step 2: Model Download Manager (`extensions/voice/model-download.ts`)

Auto-download GGML model files for Whisper models (the ones whisper.cpp supports).

**Storage location**: `~/.pi/models/<model-id>/` (consistent with pi's config directory)

```typescript
export interface ModelDownloadInfo {
  modelId: string;
  url: string;           // HuggingFace direct download URL
  filename: string;      // e.g. "ggml-small.bin"
  expectedSizeMB: number;
  sha256?: string;       // optional integrity check
}

export async function downloadModel(info: ModelDownloadInfo, onProgress: (pct: number) => void): Promise<string>
export function isModelDownloaded(modelId: string): boolean
export function getModelPath(modelId: string): string | null
export function deleteModel(modelId: string): boolean
```

**Download sources** (GGML format for whisper.cpp):
- `https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin`
- `https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin`
- `https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin`
- `https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin`

**Progress display**: Use the existing `ctx.ui.notify()` with periodic updates, or a simple "Downloading whisper-small (487 MB)... 45%" status line.

**Resume support**: Use HTTP Range headers to resume interrupted downloads.

**Integrity**: Verify file size matches expected; optionally check SHA-256.

### Step 3: Server Binary Manager (`extensions/voice/server-manager.ts`)

Auto-download and manage the whisper.cpp server binary.

**Binary source**: whisper.cpp GitHub Releases publish pre-built binaries:
- `https://github.com/ggerganov/whisper.cpp/releases/latest`
- Platform-specific: `whisper-server-linux-x86_64`, `whisper-server-darwin-arm64`, etc.

```typescript
export async function ensureServerBinary(): Promise<string>  // returns path to binary
export async function startServer(modelPath: string, port: number): Promise<ChildProcess>
export async function stopServer(): Promise<void>
export function isServerRunning(port: number): Promise<boolean>
```

**Lifecycle**:
- Server starts when user begins a local voice session
- Server stops when extension deactivates or user switches to Deepgram
- Server auto-restarts if it crashes during a session
- Port: use configured `localEndpoint` port, default 8080

**Binary storage**: `~/.pi/bin/whisper-server` (alongside models directory)

**Fallback**: If auto-download fails, show the existing manual setup instructions.

### Step 4: Update Onboarding Flow (`extensions/voice/onboarding.ts`)

Redesign the local setup experience:

1. **Detect device** → build DeviceProfile
2. **Show filtered model list** → sorted by recommendation, incompatible models grayed out or moved to bottom with warning
3. **Auto-download model** → show progress, handle errors
4. **Auto-start server** → verify transcription works with a test request
5. **Confirm setup** → "Local transcription is ready! Using whisper-small on localhost:8080"

For non-Whisper models (Moonshine, SenseVoice, etc.):
- Show message: "This model requires a compatible server (transcribe-rs/Handy). See setup guide."
- Link to documentation
- Still allow selection — just skip auto-download/server-start

### Step 5: Update `local.ts` — Auto-Server Integration

Modify `transcribeWithServer()` and session lifecycle:

- Before transcribing, check if server is running → if not, auto-start it
- Add `ensureServerRunning()` that starts server with the configured model
- Handle server startup delay (poll until `/v1/models` responds, max 30s timeout)
- On first use after model change, restart server with new model

### Step 6: Add `/voice-download` Command

A utility command for manual model management:
- List downloaded models and their sizes
- Download a specific model
- Delete cached models to free disk space
- Show disk usage: "Models: 1.2 GB used in ~/.pi/models/"

### Step 7: Tests

- `device.test.ts` — mock `os.totalmem()`, `os.cpus()`, filesystem reads for RPi detection
- `model-download.test.ts` — mock fetch, test progress tracking, resume logic, path resolution
- `server-manager.test.ts` — mock spawn, test lifecycle, port checking
- Update `onboarding.test.ts` — test filtered model list based on device profile

## File Changes Summary

| File | Action | Description |
|------|--------|-------------|
| `extensions/voice/device.ts` | **New** | Device detection and model fitness |
| `extensions/voice/model-download.ts` | **New** | Model downloading with progress |
| `extensions/voice/server-manager.ts` | **New** | whisper.cpp server binary management |
| `extensions/voice/local.ts` | **Edit** | Add `minRamMB`, `tier` to LocalModelInfo; integrate auto-server |
| `extensions/voice/onboarding.ts` | **Edit** | Filtered model list, auto-download flow |
| `extensions/voice.ts` | **Edit** | Hook server lifecycle into extension activate/deactivate |
| `tests/device.test.ts` | **New** | Device detection tests |
| `tests/model-download.test.ts` | **New** | Download manager tests |
| `tests/server-manager.test.ts` | **New** | Server manager tests |
| `tests/onboarding.test.ts` | **Edit** | Update for new onboarding flow |

## Execution Order

1. **Step 1** (device.ts) — no dependencies, foundational
2. **Step 2** (model-download.ts) — no dependencies, foundational
3. **Step 3** (server-manager.ts) — depends on step 2 for model paths
4. **Step 4** (onboarding.ts updates) — depends on steps 1-3
5. **Step 5** (local.ts integration) — depends on step 3
6. **Step 6** (/voice-download command) — depends on step 2
7. **Step 7** (tests) — done incrementally with each step

Steps 1 and 2 can be done in parallel.

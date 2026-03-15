# Plan: Seamless Local Model Setup — Auto-Download, Device Detection, In-Process Inference

## Problem Statement

The local transcription backend requires users to manually:
1. Clone and build whisper.cpp (or another server)
2. Download model files from HuggingFace
3. Start the server on the correct port
4. Know which models fit their device

Goal: **"Pick a model → it works"** — minimal user effort, best output.

## Research Evidence (Verified)

### whisper.cpp Binary Distribution — No Linux/macOS Binaries
Official releases (v1.8.3, `ggml-org/whisper.cpp`) only have Windows binaries + Apple XCFramework.
No pre-built server binaries for Linux or macOS. Auto-download of whisper.cpp server is NOT viable.

### sherpa-onnx-node v1.12.29 (Verified — Best Option)
- Pre-built native binaries for ALL platforms (Linux x64/arm64, macOS x64/arm64, Windows x64)
- **Supports ALL 20 models**: Whisper, Moonshine v1, Moonshine v2, SenseVoice, GigaAM, Parakeet
- In-process inference — no external server needed
- ~30-80 MB installed (platform-dependent)
- Apache 2.0 license
- **Caveat**: requires `LD_LIBRARY_PATH`/`DYLD_LIBRARY_PATH` at runtime
- **Risk**: Bun N-API compatibility untested

### int8 Quantized Models (Significant Download Savings)
sherpa-onnx provides int8 quantized ONNX models with minimal quality loss:

| Model | FP32 | int8 | Savings |
|-------|------|------|---------|
| whisper-tiny | ~75 MB | ~40 MB | 47% |
| whisper-small | ~488 MB | ~180 MB | 63% |
| whisper-medium | ~1.53 GB | ~500 MB | 67% |
| whisper-large-v3 | ~3.1 GB | ~1.0 GB | 68% |

**Decision: Default to int8 models** — faster downloads, less RAM, less disk, near-identical accuracy.

### Runtime RAM (from whisper.cpp docs + research)
Formula: **~2-3x model file size** (weights + KV cache + audio buffer + runtime).

| Model | int8 Size | Runtime RAM | Min System RAM |
|-------|-----------|-------------|----------------|
| moonshine-v2-tiny | ~31 MB | ~80 MB | 256 MB |
| moonshine-tiny | ~40 MB | ~100 MB | 512 MB |
| whisper-tiny int8 | ~40 MB | ~120 MB | 512 MB |
| moonshine-base | ~60 MB | ~150 MB | 512 MB |
| moonshine-v2-small | ~100 MB | ~250 MB | 1 GB |
| moonshine-v2-medium | ~192 MB | ~450 MB | 1 GB |
| sensevoice-small | ~228 MB | ~550 MB | 2 GB |
| gigaam-v3 | ~225 MB | ~550 MB | 2 GB |
| whisper-small int8 | ~180 MB | ~450 MB | 2 GB |
| parakeet-v2 | ~473 MB | ~1.0 GB | 4 GB |
| whisper-medium int8 | ~500 MB | ~1.2 GB | 4 GB |
| whisper-large-v3 int8 | ~1.0 GB | ~2.5 GB | 8 GB |
| whisper-turbo int8 | ~574 MB | ~1.4 GB | 4 GB |

### Bun Fallback Strategy
If Bun N-API fails with sherpa-onnx-node:
- **Fallback A**: `sherpa-onnx` WASM package (~13 MB, no N-API, works in any JS runtime, but single-threaded/slower)
- **Fallback B**: Spawn Node.js subprocess with sherpa-onnx-node, communicate via stdin/stdout
- **Fallback C**: External server mode (existing code, user manages their own server)

### Device Detection (Verified Node.js APIs)
- RAM: `os.totalmem()` with cgroup fallback for containers
- RPi: `/proc/device-tree/model` (most reliable)
- GPU: `nvidia-smi` for NVIDIA, `darwin+arm64` for Metal
- Container: `/.dockerenv` or `/proc/1/cgroup`
- Locale: `Intl.DateTimeFormat().resolvedOptions().locale` for language auto-detection

### Fitness Thresholds (from LM Studio/Ollama/GPT4All patterns)
- `runtimeRAM > 80% totalRAM` → **incompatible** (system will thrash)
- `runtimeRAM > freeRAM` → **warning** (may work if apps close)
- `runtimeRAM = 50-80% totalRAM` → **compatible**
- `runtimeRAM < 50% totalRAM` → **recommended**

## Architecture: Single-Tier In-Process

Since sherpa-onnx-node supports ALL 20 models, no two-tier split is needed:

```
User picks model → auto-download ONNX files → sherpa loads in-process → transcribe
```

External server mode (`localEndpoint`) is preserved as an **advanced escape hatch** only — not part of the default onboarding flow.

### Why this is better than the previous two-tier plan:
- Simpler code (one engine, not two code paths)
- Simpler UX (no "this model needs a server" distinction)
- Simpler config (no `localEndpoint` question in default flow)

## Implementation Plan

### Step 0: Bun Compatibility Test + Fix Catalog Bugs

**0a. Test sherpa-onnx-node with Bun:**
```bash
bun add sherpa-onnx-node
```
Test: create `OfflineRecognizer` with a tiny model, transcribe a silent WAV.
If N-API fails, test `sherpa-onnx` (WASM package) as fallback.

**0b. Fix model catalog sizes:**
- `whisper-medium`: "492 MB" → "1.53 GB" (FP32), "~500 MB" (int8)
- `whisper-large`: "1.1 GB" → "3.09 GB" (FP32), "~1.0 GB" (int8)

**0c. Set LD_LIBRARY_PATH programmatically:**
Before `require('sherpa-onnx-node')`, resolve the platform package path and set:
```typescript
process.env.LD_LIBRARY_PATH = `${platformPkgDir}:${process.env.LD_LIBRARY_PATH || ''}`;
```

### Step 1: Device Detection Module (`extensions/voice/device.ts`)

```typescript
export interface DeviceProfile {
  platform: NodeJS.Platform;
  arch: string;
  totalRamMB: number;        // container-aware
  freeRamMB: number;
  cpuCores: number;
  cpuModel: string;
  isRaspberryPi: boolean;
  piModel?: string;
  gpu: {
    hasNvidia: boolean;
    hasMetal: boolean;
    vramMB?: number;
    gpuName?: string;
  };
  isContainer: boolean;
  systemLocale: string;       // e.g. "en-US", "ja-JP"
}

export function detectDevice(): DeviceProfile
export function getModelFitness(model: LocalModelInfo, device: DeviceProfile):
  'recommended' | 'compatible' | 'warning' | 'incompatible'
export function autoRecommendModel(device: DeviceProfile, language: string): LocalModelInfo
```

**`autoRecommendModel` logic** (new — reduces user effort):
1. Filter models by language support
2. Filter to 'recommended' or 'compatible' fitness
3. Sort by: best accuracy (larger) within recommended tier
4. Return the best model that fits comfortably

Example: RPi 5 (4 GB) + English → recommends `moonshine-v2-small` (100 MB, streaming, fits easily).
Example: MacBook M2 (16 GB) + English → recommends `whisper-small` (180 MB int8, best accuracy that fits).
Example: Any device + Russian → recommends `gigaam-v3` (only Russian model).

### Step 2: Model Download Manager (`extensions/voice/model-download.ts`)

```typescript
export async function downloadModel(
  modelId: string,
  onProgress: (downloadedMB: number, totalMB: number) => void,
): Promise<string>  // returns model directory path

export function isModelDownloaded(modelId: string): boolean
export function getModelPath(modelId: string): string | null
export function getModelsDir(): string                        // ~/.pi/models/
export function deleteModel(modelId: string): boolean
export function getDownloadedModels(): { id: string; sizeMB: number }[]
```

Storage: `~/.pi/models/{modelId}/` (e.g., `~/.pi/models/whisper-small/encoder.int8.onnx`)

**Default to int8**: Download int8 quantized models unless user explicitly requests FP32.

Download flow:
1. Check if model dir exists with expected files → return early
2. Download individual ONNX files from HuggingFace (not tar.bz2 — avoids needing tar extraction)
3. Stream to disk with progress reporting
4. Support resume via HTTP Range headers
5. Verify file sizes match expected
6. Return model directory path

Updated `LocalModelInfo`:
```typescript
export interface LocalModelInfo {
  id: string;
  name: string;
  size: string;                    // human-readable display (int8 size)
  sizeBytes: number;               // int8 download size
  runtimeRamMB: number;            // peak RAM during inference (~2.5x int8 size)
  notes: string;
  langSupport: "whisper" | "english-only" | "parakeet-multi" | "sensevoice"
    | "russian-only" | "single-ar" | "single-zh" | "single-ja"
    | "single-ko" | "single-uk" | "single-vi" | "single-es";
  tier: "edge" | "standard" | "heavy";
  sherpaModel: {                   // sherpa-onnx model config
    type: "whisper" | "moonshine" | "sense_voice" | "nemo_ctc" | "transducer";
    files: Record<string, string>; // role -> filename within model dir
    downloadUrls: Record<string, string>; // role -> HuggingFace URL
  };
}
```

Note: `engine: "sherpa" | "server"` field removed since all models use sherpa now.
External server mode is a config-level override (`localEndpoint` set → use HTTP), not per-model.

### Step 3: In-Process Transcription Engine (`extensions/voice/sherpa-engine.ts`)

```typescript
export function initSherpa(): void  // set LD_LIBRARY_PATH, require module
export function createRecognizer(modelId: string, modelDir: string, language: string): OfflineRecognizer
export function transcribeBuffer(pcmData: Buffer, recognizer: OfflineRecognizer): string
export function destroyRecognizer(recognizer: OfflineRecognizer): void

// Cached recognizer management
export function getOrCreateRecognizer(modelId: string, modelDir: string, language: string): OfflineRecognizer
export function clearRecognizerCache(): void
```

Recognizer is cached across transcriptions (model loading is expensive).
Destroyed on: model change, language change, extension deactivation.

### Step 4: Update `local.ts` — Unified Engine

```typescript
export async function transcribeLocal(pcmData: Buffer, config: VoiceConfig): Promise<string> {
  // External server override (advanced users only)
  if (config.localEndpoint) {
    return transcribeWithServer(createWavBuffer(pcmData), config);
  }

  // Default: in-process via sherpa-onnx
  const model = LOCAL_MODELS.find(m => m.id === config.localModel);
  if (!model) throw new Error(`Unknown model: ${config.localModel}`);

  const modelDir = await ensureModelDownloaded(model.id, (dl, total) => {
    // Progress callback — update UI
  });
  const recognizer = getOrCreateRecognizer(model.id, modelDir, config.language);
  return transcribeBuffer(pcmData, recognizer);
}
```

Key change from previous plan: no `engine` routing — sherpa is the only engine.
`localEndpoint` in config = override to external server (escape hatch).

### Step 5: Update Onboarding Flow (`extensions/voice/onboarding.ts`)

**Redesigned for minimal effort** — 2 interactions instead of 5:

```
┌─ New local onboarding flow ───────────────────────────────────┐
│                                                               │
│  1. Auto-detect: device profile + system locale               │
│     "Detected: 4 GB RAM, ARM64, Raspberry Pi 5"              │
│                                                               │
│  2. Show smart recommendation:                                │
│     "Recommended: Moonshine v2 Small (100 MB) — English"      │
│     "Fits your device well. Best accuracy for your hardware." │
│     [Enter] Install recommended                               │
│     [c] Choose different model                                │
│     [a] Advanced (external server)                            │
│                                                               │
│  3. If Enter → download with progress → test → done           │
│     If 'c' → show full model list filtered by device          │
│     If 'a' → existing server URL flow                         │
│                                                               │
│  4. "Local transcription ready! Using moonshine-v2-small"     │
└───────────────────────────────────────────────────────────────┘
```

**Versus current flow** (5+ interactions):
1. "Choose backend" → 2. "Choose model" (20 options!) → 3. Read server setup instructions → 4. "Enter server URL" → 5. "Choose language" → 6. "Choose scope"

**New flow** (2-3 interactions):
1. "Choose backend" (deepgram / local)
2. [Enter] to accept recommendation OR [c] to browse
3. "Choose scope"

Language auto-detected from system locale (overridable via `/voice-language`).
Server URL skipped entirely (in-process inference).

### Step 6: Add `/voice-models` Command

Model management utility:
- List downloaded models + disk usage
- Download/delete models
- Show device profile + all model fitness ratings
- Switch active model (with auto-download)
- Force FP32 download for specific model

### Step 7: Tests

| Test file | Tests |
|-----------|-------|
| `tests/device.test.ts` | RAM detection, RPi detection, fitness scoring, container detection, autoRecommendModel |
| `tests/model-download.test.ts` | Download flow, resume, path resolution, int8 defaults, cleanup |
| `tests/sherpa-engine.test.ts` | Recognizer creation, transcription, cache management, LD_LIBRARY_PATH |
| `tests/onboarding.test.ts` | Auto-recommendation, smart defaults, device-aware model filtering |

## File Changes Summary

| File | Action | Description |
|------|--------|-------------|
| `extensions/voice/local.ts` | **Edit** | Fix sizes, add sizeBytes/runtimeRamMB/tier/sherpaModel to LocalModelInfo, replace transcribeWithServer routing with sherpa-first logic |
| `extensions/voice/device.ts` | **New** | Device detection + model fitness + auto-recommendation |
| `extensions/voice/model-download.ts` | **New** | ONNX model download (int8 default) with progress + resume |
| `extensions/voice/sherpa-engine.ts` | **New** | In-process transcription via sherpa-onnx-node |
| `extensions/voice/onboarding.ts` | **Edit** | Smart recommendation flow, device-aware, minimal interactions |
| `extensions/voice.ts` | **Edit** | Hook recognizer lifecycle, add /voice-models, call sherpa init |
| `extensions/voice/config.ts` | **Edit** | localEndpoint becomes optional advanced override (no behavior change, just docs) |
| `package.json` | **Edit** | Add sherpa-onnx-node dependency |
| `tests/device.test.ts` | **New** | Device detection tests |
| `tests/model-download.test.ts` | **New** | Download manager tests |
| `tests/sherpa-engine.test.ts` | **New** | Sherpa engine tests |
| `tests/onboarding.test.ts` | **Edit** | Update for smart recommendation flow |

## Execution Order

1. **Step 0a** — Bun compatibility test (BLOCKER)
2. **Step 0b+0c** — Fix catalog bugs + LD_LIBRARY_PATH setup (parallel with 0a)
3. **Step 1** — device.ts (no dependencies)
4. **Step 2** — model-download.ts (no dependencies)
5. **Step 3** — sherpa-engine.ts (depends on 0a result + 0c)
6. **Step 4** — local.ts unified engine (depends on 2+3)
7. **Step 5** — onboarding redesign (depends on 1+4)
8. **Step 6** — /voice-models command (depends on 1+2)
9. **Step 7** — tests (incremental)

Steps 1, 2, and 0b+0c can all run in parallel.

## UX Comparison: Before vs After

| Aspect | Before (current) | After (this plan) |
|--------|------------------|-------------------|
| Model selection | Scroll through 20 models, no guidance | Auto-recommend best model for device+language |
| Model download | Manual (clone repo, run scripts) | Auto-download with progress bar |
| Server setup | Clone whisper.cpp, build, run server | None — in-process inference |
| Device detection | None — user guesses | Auto-detect RAM, CPU, GPU, RPi |
| First-use flow | 5+ steps, 3 terminal commands | 2-3 clicks: backend → [Enter] → done |
| Incompatible model | Server crashes, cryptic error | Warning before download, filtered list |
| Language setup | Manual selection from 57 options | Auto-detect from system locale |

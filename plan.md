# Plan: Seamless Local Model Setup — Auto-Download, Device Detection, Server Management

## Problem Statement

The local transcription backend requires users to manually:
1. Clone and build whisper.cpp (or another server)
2. Download model files from HuggingFace
3. Start the server on the correct port
4. Know which models fit their device

This makes local mode unusable for most users. We need a seamless "pick a model → it works" experience.

## Research Evidence (Verified)

### whisper.cpp Binary Distribution — CORRECTED
**Official releases (v1.8.3, repo now at `ggml-org/whisper.cpp`) only have:**
- Windows x86/x64 binaries (with CPU, OpenBLAS, CUDA variants)
- Apple XCFramework (library, not standalone binary)
- **NO Linux binaries. NO macOS standalone binaries. NO ARM64 binaries.**

This means auto-downloading a whisper.cpp server binary is NOT viable for Linux/macOS.

The `/v1/audio/transcriptions` endpoint is also NOT built-in — requires `--inference-path "/v1/audio/transcriptions"` flag.

### sherpa-onnx-node — Best Alternative (Verified)
- **npm package**: `sherpa-onnx-node` v1.11.3, ~500 weekly downloads, actively maintained
- **Pre-built native binaries** for ALL platforms via optionalDependencies:
  - `sherpa-onnx-linux-x64`, `sherpa-onnx-linux-arm64` (covers RPi)
  - `sherpa-onnx-darwin-x64`, `sherpa-onnx-darwin-arm64` (covers Apple Silicon)
  - `sherpa-onnx-win-x64`
- **No compilation needed** — N-API bindings, ~30-50 MB installed
- **Supports**: Whisper (all sizes), Moonshine v1 (tiny/base), SenseVoice — via ONNX format
- **Does NOT support**: GigaAM, Parakeet (these keep manual server option)
- **Moonshine v2**: likely supported but needs verification
- **License**: Apache 2.0
- **Bun compatibility**: UNTESTED — uses N-API, which Bun partially supports. Critical risk.
- **Model format**: ONNX (not GGML). Pre-packaged at GitHub releases + HuggingFace
- **API**: `OfflineRecognizer` for batch, `OnlineRecognizer` for streaming

### Model Sources (ONNX format for sherpa-onnx)
From `k2-fsa/sherpa-onnx` GitHub releases:
- `sherpa-onnx-whisper-tiny.tar.bz2` (~60 MB, int8: ~25 MB)
- `sherpa-onnx-whisper-small.tar.bz2` (~488 MB)
- `sherpa-onnx-whisper-medium.tar.bz2` (~1.5 GB)
- `sherpa-onnx-whisper-large-v3.tar.bz2` (~3.1 GB)
- Moonshine: from HuggingFace `UsefulSensors/` repos (ONNX files)
- SenseVoice: from HuggingFace `csukuangfj/` repos

### GGML Model Sizes — Catalog Bugs Found
| Filename | Actual Size | Our catalog says | Status |
|----------|------------|-----------------|--------|
| ggml-small.bin | 488 MB | 487 MB | OK |
| ggml-medium.bin | **1.53 GB** | 492 MB | **BUG** |
| ggml-large-v3.bin | **3.09 GB** | 1.1 GB | **BUG** |
| ggml-large-v3-turbo.bin | 1.62 GB | 1.6 GB | OK |

### Runtime RAM (from whisper.cpp docs)
| Model | File Size | Runtime RAM | Source |
|-------|-----------|-------------|--------|
| tiny | 77 MB | ~390 MB | whisper.cpp README |
| base | 148 MB | ~500 MB | whisper.cpp README |
| small | 488 MB | ~1.0 GB | whisper.cpp README |
| medium | 1.53 GB | ~2.6 GB | whisper.cpp README |
| large | 3.09 GB | ~4.7 GB | whisper.cpp README |

Formula: **~1.7x model file size** for peak runtime RAM.

### Device Detection (Verified Node.js APIs)
- `os.totalmem()`: physical RAM (reports HOST RAM in containers — need cgroup fallback)
- Container RAM: `/sys/fs/cgroup/memory.max` (v2) or `memory.limit_in_bytes` (v1)
- RPi: `/proc/device-tree/model` → `"Raspberry Pi 5 Model B Rev 1.0"` (most reliable)
- NVIDIA GPU: `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits`
- Apple Metal: `process.platform === 'darwin' && process.arch === 'arm64'`
- Container: `/.dockerenv` file or `/proc/1/cgroup` docker string

### UX Patterns (from Ollama, LM Studio, MacWhisper)
- Download on first use (Ollama)
- Warn but don't block model selection (Ollama, LM Studio)
- Color-code fitness: recommended/compatible/warning/incompatible (LM Studio)
- `peakRAM > 80% totalRAM` → incompatible; `< 50%` → recommended

## Architecture Decision

### Two-tier approach:

**Tier 1 — In-process via sherpa-onnx-node (seamless, zero-config)**
- Covers: Whisper, Moonshine v1, SenseVoice (~15 models)
- No external server needed — runs inference directly in the extension process
- Pre-built binaries for all platforms including RPi
- Models auto-downloaded from GitHub/HuggingFace on first use

**Tier 2 — External server (manual, power-user)**
- Covers: GigaAM, Parakeet, Moonshine v2, and any custom server
- Preserves existing `POST /v1/audio/transcriptions` flow
- User manages their own server (whisper.cpp, transcribe-rs, faster-whisper)
- Required for models sherpa-onnx doesn't support

### Why this over alternatives:

| Option | Verdict | Reason |
|--------|---------|--------|
| whisper.cpp pre-built binary | ❌ | Doesn't exist for Linux/macOS |
| Auto-build from source | ⚠️ Heavy | Needs cmake+gcc, 3-10 min on RPi |
| Docker | ❌ | Too heavy for CLI extension |
| faster-whisper-server (pip) | ⚠️ | Requires Python ecosystem |
| WASM in-process | ❌ | 5-8x slower than native |
| **sherpa-onnx-node** | ✅ | Pre-built, in-process, multi-model |

### Critical Risk: Bun Compatibility
sherpa-onnx-node uses N-API. Bun has partial N-API support. If it fails:
- **Fallback A**: Spawn a Node.js subprocess that loads sherpa-onnx-node, communicate via IPC
- **Fallback B**: Fall back to external server mode with setup instructions

**Step 0 of implementation must be a Bun compatibility test.**

## Implementation Plan

### Step 0: Bun Compatibility Test + Fix Catalog Bugs

**0a. Test sherpa-onnx-node with Bun:**
```bash
bun add sherpa-onnx-node
```
Write a minimal test that creates an `OfflineRecognizer` and transcribes a silent WAV.
If it works → proceed with Tier 1 architecture.
If it fails → evaluate Fallback A (Node subprocess) or switch to external-server-only approach.

**0b. Fix model catalog sizes:**
- `whisper-medium`: "492 MB" → "1.53 GB"
- `whisper-large`: "1.1 GB" → "3.09 GB"

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
}

export function detectDevice(): DeviceProfile
export function getModelFitness(model: LocalModelInfo, device: DeviceProfile):
  'recommended' | 'compatible' | 'warning' | 'incompatible'
```

Detection chain:
1. RAM: `os.totalmem()` → cgroup override if container detected
2. RPi: `/proc/device-tree/model` → `/proc/cpuinfo` BCM → arch heuristic
3. GPU: `nvidia-smi` (5s timeout) or `darwin+arm64` for Metal
4. Container: `/.dockerenv` or `/proc/1/cgroup`

Fitness thresholds (from LM Studio/Ollama patterns):
- `runtimeRamMB > 80% totalRAM` → incompatible
- `runtimeRamMB > freeRAM` → warning
- `runtimeRamMB = 50-80% totalRAM` → compatible
- `runtimeRamMB < 50% totalRAM` → recommended

### Step 2: Model Download Manager (`extensions/voice/model-download.ts`)

Downloads ONNX model packages from sherpa-onnx GitHub releases / HuggingFace.

```typescript
export async function downloadModel(
  modelId: string,
  onProgress: (downloadedMB: number, totalMB: number) => void,
): Promise<string>  // returns path to model directory

export function isModelDownloaded(modelId: string): boolean
export function getModelPath(modelId: string): string | null  // returns dir with ONNX files
export function getModelsDir(): string                        // ~/.pi/models/
export function deleteModel(modelId: string): boolean
```

Storage: `~/.pi/models/{modelId}/` (e.g., `~/.pi/models/whisper-small/encoder.onnx`)

Download flow:
1. Check if model dir exists with expected files → return early
2. Download `.tar.bz2` from GitHub releases (or individual ONNX files from HuggingFace)
3. Extract to model directory
4. Verify file sizes
5. Return model directory path

Add to `LocalModelInfo`:
```typescript
export interface LocalModelInfo {
  id: string;
  name: string;
  size: string;                    // human-readable display size
  sizeBytes: number;               // download size in bytes
  runtimeRamMB: number;            // peak RAM during inference
  notes: string;
  langSupport: "whisper" | "english-only" | ...;
  tier: "edge" | "standard" | "heavy";
  engine: "sherpa" | "server";     // sherpa = in-process, server = external
  modelFiles?: {                   // sherpa model file mapping
    type: "whisper" | "moonshine" | "sense_voice";
    files: Record<string, string>; // role -> filename
  };
}
```

### Step 3: In-Process Transcription Engine (`extensions/voice/sherpa-engine.ts`)

Replace HTTP-based transcription with in-process sherpa-onnx for supported models.

```typescript
export function createRecognizer(modelId: string, modelDir: string): OfflineRecognizer
export async function transcribeInProcess(
  pcmData: Buffer,
  recognizer: OfflineRecognizer,
): Promise<string>
export function destroyRecognizer(recognizer: OfflineRecognizer): void
```

The recognizer is heavy to create (loads model into memory), so we cache it and reuse across transcriptions. Destroy on model change or extension deactivation.

### Step 4: Update `local.ts` — Dual Engine Support

Modify transcription flow to route based on `engine` field:

```typescript
export async function transcribeLocal(pcmData: Buffer, config: VoiceConfig): Promise<string> {
  const model = LOCAL_MODELS.find(m => m.id === config.localModel);

  if (model?.engine === "sherpa") {
    // In-process: ensure model downloaded, create/reuse recognizer, transcribe
    const modelDir = await ensureModelDownloaded(model.id);
    const recognizer = getOrCreateRecognizer(model.id, modelDir);
    return transcribeInProcess(pcmData, recognizer);
  } else {
    // External server: existing HTTP flow (unchanged)
    return transcribeWithServer(createWavBuffer(pcmData), config);
  }
}
```

### Step 5: Update Onboarding Flow (`extensions/voice/onboarding.ts`)

New local setup:
1. **Detect device** → show summary ("4 GB RAM, ARM64, Raspberry Pi 5")
2. **Show model list** → sorted by fitness, sherpa models marked "auto-download":
   - Recommended first (fits device well)
   - Compatible next
   - Incompatible at bottom with warning — still selectable
3. **User picks model** →
   - If sherpa engine: "Download whisper-small (488 MB)?" → progress → load → test → done
   - If server engine: show setup instructions → test connection → done
4. **Confirm** → "Local transcription ready!"

### Step 6: Add `/voice-models` Command

Model management:
- List downloaded models + disk usage
- Download/delete models
- Show device profile + recommendations
- Test transcription with current model

### Step 7: Tests

| Test file | Tests |
|-----------|-------|
| `tests/device.test.ts` | RAM detection, RPi detection, fitness scoring, container detection |
| `tests/model-download.test.ts` | Download flow, resume, path resolution, cleanup |
| `tests/sherpa-engine.test.ts` | Recognizer creation, transcription, model type routing |
| `tests/onboarding.test.ts` | Device-aware model list, download flow |

## File Changes Summary

| File | Action | Description |
|------|--------|-------------|
| `extensions/voice/local.ts` | **Edit** | Fix sizes, add sizeBytes/runtimeRamMB/tier/engine/modelFiles to LocalModelInfo, add dual-engine routing |
| `extensions/voice/device.ts` | **New** | Device detection + model fitness scoring |
| `extensions/voice/model-download.ts` | **New** | ONNX model download from GitHub/HuggingFace |
| `extensions/voice/sherpa-engine.ts` | **New** | In-process transcription via sherpa-onnx-node |
| `extensions/voice/onboarding.ts` | **Edit** | Device-aware model list, auto-download flow |
| `extensions/voice.ts` | **Edit** | Hook recognizer lifecycle, add /voice-models command |
| `package.json` | **Edit** | Add sherpa-onnx-node dependency |
| `tests/device.test.ts` | **New** | Device detection tests |
| `tests/model-download.test.ts` | **New** | Download manager tests |
| `tests/sherpa-engine.test.ts` | **New** | Sherpa engine tests |
| `tests/onboarding.test.ts` | **Edit** | Update for new flow |

## Execution Order

1. **Step 0a** — Bun compatibility test (BLOCKER — determines entire architecture)
2. **Step 0b** — Fix catalog bugs (parallel with 0a)
3. **Step 1** — device.ts (no dependencies)
4. **Step 2** — model-download.ts (no dependencies)
5. **Step 3** — sherpa-engine.ts (depends on step 0a result)
6. **Step 4** — local.ts dual engine (depends on steps 2+3)
7. **Step 5** — onboarding updates (depends on steps 1-4)
8. **Step 6** — /voice-models command (depends on steps 1+2)
9. **Step 7** — tests (incremental with each step)

Steps 1 and 2 can run in parallel. Step 0b can run in parallel with everything.

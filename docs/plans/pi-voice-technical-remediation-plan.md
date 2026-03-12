# pi-voice technical remediation plan

## Goal

Turn `pi-voice` into a technically robust package that can support a premium onboarding flow without accumulating more correctness debt. This plan focuses on the architecture behind onboarding, runtime correctness, dependency provisioning, backend metadata, portability, and package structure.

## Current technical issues

### 1. Config is too thin
The current config model only captures:

```json
{
  "voice": {
    "enabled": true,
    "language": "en",
    "backend": "auto",
    "model": "small"
  }
}
```

That is not enough to represent:
- whether onboarding was completed
- whether the user chose cloud vs local intentionally
- which scope owns the config (global vs project)
- whether auto-install is allowed
- whether the backend was validated successfully
- which daemon/runtime profile should be active

### 2. Global daemon state can drift from config
Today the extension uses one hard-coded socket path:
- `extensions/voice.ts` -> `SOCKET_PATH`
- `daemon.py` -> `DEFAULT_SOCKET`

That creates shared mutable state across:
- projects
- sessions
- potentially different backend/model selections

`ensureDaemon()` returns early if any daemon is already running, even if that daemon was started with a different backend/model. `transcribeAudio()` currently does not pass `backend` and `model` to the daemon request, so the active daemon can silently continue with stale settings.

### 3. Setup persistence is inconsistent
`loadConfig()` merges global and project settings, but `/voice setup` writes only to global settings. That means the package already behaves as if scoped config matters, but it cannot save project-level intent.

### 4. Backend metadata is insufficient for recommendation and provisioning
`transcribe.py` exposes:
- `name`
- `available`
- `type`
- `default_model`
- `install`
- `models`

That is enough for a raw picker, but not for a recommendation engine or enterprise onboarding. The package lacks structured metadata for:
- privacy posture
- network requirement
- expected latency
- expected disk use
- install complexity
- whether warm model caching is supported
- whether the backend can be auto-installed safely
- recommended hardware tier

### 5. No provisioning abstraction exists
Installation and dependency repair are currently implicit/manual. There is no dedicated subsystem for:
- checking prerequisites
- producing an install plan
- executing safe install actions
- collecting stdout/stderr
- classifying failure modes
- retrying or falling back

### 6. Package shape is too minimal
The package is technically valid, but not operationally mature. It lacks:
- docs in the publish list
- scripts for validation
- package metadata for discoverability/supportability
- a modular code layout that can safely absorb onboarding and diagnostics growth

## Technical design principles

1. **Config is explicit, versioned, and migratable.**
2. **Runtime profile selection must be deterministic.** The active daemon must always match saved intent.
3. **Onboarding state is first-class.** “Not configured”, “configured”, and “broken” must be distinct states.
4. **Backend metadata must drive both UX and install logic.**
5. **Provisioning must be declarative first, imperative second.** Build an install plan before executing commands.
6. **Project-local and global scope must both be supported cleanly.**
7. **Non-interactive and interactive modes must degrade safely.**
8. **Package structure should be modular before adding more product surface.**

## Target architecture

## 1. Versioned config model

Replace the flat config with a versioned schema.

### Proposed shape

```json
{
  "voice": {
    "version": 2,
    "enabled": true,
    "mode": "local",
    "language": "en",
    "backend": "faster-whisper",
    "model": "small",
    "scope": "global",
    "hotkeys": {
      "editorHoldToTalk": true,
      "toggleShortcut": "ctrl+shift+v",
      "btwShortcut": "ctrl+shift+b"
    },
    "runtime": {
      "warmDaemonOnStart": true,
      "vadEnabled": true,
      "daemonProfile": "global-local-faster-whisper-small-en"
    },
    "onboarding": {
      "state": "completed",
      "schemaVersion": 2,
      "source": "first-run",
      "completedAt": "2026-03-12T00:00:00.000Z",
      "lastValidatedAt": "2026-03-12T00:02:00.000Z",
      "dismissedAt": null
    },
    "provisioning": {
      "autoInstall": true,
      "allowBrew": true,
      "allowPipUser": true,
      "lastPlanId": "install-plan-123"
    },
    "cloud": {
      "provider": null,
      "apiKeyEnv": null
    },
    "health": {
      "status": "ready",
      "lastError": null,
      "lastBackendCheckAt": "2026-03-12T00:01:00.000Z"
    }
  }
}
```

### Required config helpers

Create a dedicated config module with:
- `loadVoiceConfig(cwd)`
- `saveVoiceConfig(config, scope, cwd)`
- `migrateVoiceConfig(input)`
- `getVoiceSettingsPaths(cwd)`
- `resolveEffectiveConfig(globalConfig, projectConfig)`
- `getConfigOwnership(config)`

### Scope model

Support both:
- global: `~/.pi/agent/settings.json`
- project: `<cwd>/.pi/settings.json`

Save path must be explicit, not implied.

## 2. Onboarding state machine

Treat onboarding as a state machine, not a boolean.

### Proposed states
- `unconfigured`
- `recommended`
- `provisioning`
- `validation_pending`
- `completed`
- `degraded`
- `dismissed`

### Why this matters
This separates:
- first install
- partially configured local install
- failed cloud auth
- a previously good setup that later broke

### Trigger logic on `session_start`

1. Load and migrate config.
2. If no UI, skip onboarding and runtime prompts.
3. If state is `completed` and health is `ready`, proceed.
4. If state is `unconfigured` or `degraded`, launch guided setup or show repair CTA.
5. Only start the daemon after the config has been validated for the active runtime profile.

## 3. Backend catalog abstraction

Move backend registry concerns out of ad-hoc strings and into a structured catalog consumable by both TypeScript and Python.

### Proposed backend metadata fields

For each backend, expose:
- `id`
- `label`
- `mode` (`local` or `cloud`)
- `available`
- `warmModelCapable`
- `installMethod` (`pip`, `brew`, `env`, `manual`, `mixed`)
- `installCommands`
- `models`
- `defaultModel`
- `recommendedModelByProfile`
- `privacy` (`offline`, `cloud`, `hybrid`)
- `networkRequired`
- `resourceTier` (`low`, `medium`, `high`)
- `latencyTier` (`fast`, `balanced`, `slow`)
- `qualityTier` (`good`, `better`, `best`)
- `diskEstimate`
- `hardwareNotes`
- `apiKeyEnv`
- `validationStrategy`
- `autoInstallSupported`

### Implementation options

Preferred:
- Add a shared `backends.json` or `backends.ts` manifest at the package level.
- Keep Python and TypeScript aligned by generating one side from the other, or by treating JSON as the source of truth.

Minimum acceptable:
- keep the Python registry but enrich `--list-backends` output with structured fields
- consume that richer JSON from the extension

## 4. Diagnostics engine

Add a diagnostics subsystem that produces a machine-readable readiness report.

### Inputs
- platform / architecture
- `python3` availability
- `brew` availability
- `rec` / SoX availability
- backend availability from `transcribe.py --list-backends`
- required env vars
- optional daemon status
- optional sample audio validation

### Output shape

```json
{
  "status": "ready",
  "recommendedMode": "local",
  "recommendedBackend": "faster-whisper",
  "recommendedModel": "small",
  "issues": [
    {
      "id": "missing-sox",
      "severity": "blocking",
      "message": "SoX is required for microphone recording.",
      "fix": {
        "auto": true,
        "commands": ["brew install sox"]
      }
    }
  ],
  "capabilities": {
    "python": true,
    "brew": true,
    "rec": false,
    "cloudAuth": false
  }
}
```

### Responsibilities
- recommend mode/backend/model
- classify blockers vs warnings
- provide install plans
- power `/voice doctor`
- feed onboarding defaults

## 5. Provisioning subsystem

Create a dedicated provisioning layer instead of embedding installation logic inside command handlers.

### New module responsibilities
- build install plan from diagnostics
- execute allowed steps
- stream progress and capture logs
- classify failures
- emit structured result objects

### Proposed install action model

```ts
interface InstallAction {
  id: string;
  type: "brew" | "pip" | "env-check" | "python-import" | "warm-model" | "validate-cloud";
  label: string;
  command?: string[];
  safeToAutoRun: boolean;
  retryable: boolean;
}
```

### Local provisioning flows
- install SoX if missing
- install backend runtime if supported
- optionally warm the selected model
- verify transcription path

### Cloud provisioning flows
- verify required env var exists
- optionally validate API access with a lightweight provider request
- verify selected model is accepted

### Failure handling
Each failed step should produce:
- step id
- command
- exit code
- stderr excerpt
- recommended remediation

That should be stored in memory during onboarding and surfaced by `/voice doctor`.

## 6. Deterministic daemon lifecycle

This is the most important runtime fix.

### Current problem
A global daemon can continue using a stale backend/model because the extension does not fully bind requests to the desired config.

### Required design change
Introduce a `RuntimeProfile` abstraction.

```ts
interface RuntimeProfile {
  mode: "local" | "cloud" | "auto";
  backend: string;
  model: string;
  language: string;
  vadEnabled: boolean;
  scope: "global" | "project";
  profileId: string;
  configHash: string;
  socketPath: string;
}
```

### Rules
1. The extension computes a runtime profile from effective config.
2. The daemon status endpoint returns enough data to identify the active profile.
3. If daemon profile != desired profile, the extension reloads or replaces it.
4. Transcribe requests must include `backend`, `model`, and `language` even when a daemon is already running.

### Socket strategy

Preferred:
- derive socket path from `profileId` or `configHash`
- example: `/tmp/pi-voice-<uid>-<hash>.sock`

Benefits:
- avoids collisions across projects/config scopes
- allows multiple compatible daemons if needed
- makes debugging easier

Minimum acceptable:
- keep one socket but always send explicit runtime parameters and force reload on mismatch

### Daemon status contract changes
Add fields to `status`:
- `backend`
- `model`
- `language`
- `profile_id`
- `config_hash`
- `warm_capable`
- `last_error`

### Daemon control changes
Add or refine commands:
- `status`
- `load`
- `reload`
- `shutdown`
- optional `backends`

### Important note
Cloud backends do not need persistent warm models, but they still need a stable profile and health contract.

## 7. Runtime module split

Refactor the monolithic `extensions/voice.ts` into smaller modules.

### Proposed file layout

```text
extensions/
  voice/
    index.ts
    config.ts
    onboarding.ts
    diagnostics.ts
    provisioning.ts
    runtime.ts
    daemon.ts
    backends.ts
    commands.ts
    btw.ts
    types.ts
```

### Responsibilities
- `index.ts`: extension registration and lifecycle wiring
- `config.ts`: load/save/migrate/effective config logic
- `onboarding.ts`: first-run and reconfigure flow orchestration
- `diagnostics.ts`: environment scan and health reporting
- `provisioning.ts`: install actions and remediation execution
- `runtime.ts`: audio recording and transcription dispatch
- `daemon.ts`: profile-aware daemon communication
- `backends.ts`: backend metadata and recommendation helpers
- `commands.ts`: `/voice`, `/voice doctor`, `/voice reconfigure`, `/voice info`
- `btw.ts`: BTW-specific logic only
- `types.ts`: shared TS interfaces

This should happen before adding more complex UI or provisioning logic.

## 8. Portability strategy

The package currently assumes a macOS-ish environment in practice.

### Current hidden assumptions
- `rec` exists and comes from SoX
- `brew` may be used to install native dependencies
- `python3` is available
- local model stacks work on CPU

### Portability tiers

#### Tier 1: macOS Apple Silicon
Primary target for polished local onboarding.
- optimize recommendations for `faster-whisper`
- provide `brew` + `pip` install paths
- use this as the first-class supported path

#### Tier 2: Linux desktop
Supported with diagnostics-led setup.
- no Homebrew assumption
- more manual/local distro variance
- provisioning may need to fall back to manual commands

#### Tier 3: non-interactive / RPC / CI
Do not auto-onboard.
- commands should return structured errors
- diagnostics should still work
- setup should be re-runnable when UI exists

### Portability policy
Document explicitly which flows are:
- fully automated
- partially automated
- manual only

## 9. Package structure and metadata

### `package.json` improvements
Add:
- richer description
- `repository`
- `homepage`
- `bugs`
- `keywords`
- scripts
- peerDependencies for Pi runtime packages
- `README.md` and `docs/` in `files`

### Suggested scripts

```json
{
  "scripts": {
    "typecheck": "bunx tsc -p tsconfig.json",
    "check:python": "python3 -m py_compile daemon.py transcribe.py",
    "check": "bun run typecheck && bun run check:python"
  }
}
```

### Publish list
Include:
- `extensions`
- `daemon.py`
- `transcribe.py`
- `README.md`
- `docs`
- any shared backend metadata assets

## 10. Recommended implementation sequence

### Phase A — correctness foundation
Files:
- `extensions/voice.ts`
- `daemon.py`

Tasks:
- always pass `backend`/`model`/`language` to daemon transcribe requests
- make daemon status profile-aware
- stop eager daemon startup until config is validated
- add safer socket/profile strategy

### Phase B — config and metadata foundation
Files:
- new `extensions/voice/config.ts`
- new `extensions/voice/types.ts`
- `transcribe.py` or shared backend manifest

Tasks:
- versioned config schema
- scope-aware save/load helpers
- onboarding state machine
- richer backend metadata contract

### Phase C — diagnostics and provisioning foundation
Files:
- new `extensions/voice/diagnostics.ts`
- new `extensions/voice/provisioning.ts`
- `transcribe.py`

Tasks:
- machine-readable readiness checks
- install plan generation
- execution wrappers and failure classification

### Phase D — extension modularization
Files:
- split current `extensions/voice.ts` into focused modules

Tasks:
- isolate BTW logic
- isolate runtime recording/transcription logic
- isolate command registration

### Phase E — onboarding and doctor flows
Files:
- `extensions/voice/onboarding.ts`
- `extensions/voice/commands.ts`

Tasks:
- first-run wizard
- `/voice doctor`
- `/voice reconfigure`
- summary/health reporting

### Phase F — package hardening
Files:
- `package.json`
- `README.md`
- `docs/*`

Tasks:
- metadata
- docs
- validation scripts
- release supportability

## Risks and decisions to resolve early

### 1. Shared metadata source of truth
Decide whether backend metadata lives in:
- Python only,
- TypeScript only,
- or shared JSON.

Recommendation: **shared JSON or TS-generated JSON**.

### 2. Auto-install safety policy
Decide what the extension may execute automatically:
- just diagnostics
- diagnostics + optional package install
- diagnostics + fully automatic remediation by default

Recommendation: **opt-in auto-install during onboarding** with clear user confirmation.

### 3. Daemon multiplicity
Decide whether to support:
- one daemon per config profile
- or one daemon that reloads in place

Recommendation: **profile-derived socket path**, with reload support as a secondary optimization.

### 4. Cloud provider expansion
Today only Deepgram is implemented for cloud. The config model should be provider-ready even if the first polished flow only supports one cloud backend.

## Definition of done for the technical architecture

The architecture work is complete when:
- config has a versioned schema and migration path
- onboarding state is explicit and persisted
- project/global scope is saved intentionally
- backend metadata supports recommendation and install logic
- diagnostics returns structured readiness data
- provisioning executes from an install plan, not ad-hoc shelling out
- daemon/runtime profile selection is deterministic
- stale daemon config cannot silently override saved user intent
- code is split into maintainable modules
- package metadata and validation scripts support shipping the package confidently

## Final recommendation

Do the technical remediation in this order:

**runtime correctness -> config/schema -> backend metadata -> diagnostics -> provisioning -> modularization -> package hardening**

That order prevents the team from building a polished onboarding flow on top of an unstable runtime contract.
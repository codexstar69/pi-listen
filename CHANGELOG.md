# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.10] - 2026-03-12

### Added
- **`/voice-setup` command** — dedicated, discoverable command for first-time voice configuration (shows up in pi's `/` command list)

## [1.0.9] - 2026-03-12

### Changed
- **No more auto-popup on startup** — removed the first-run onboarding prompt that appeared every time pi launched
- Voice features now only activate if setup was previously completed via `/voice setup`
- Users run `/voice setup` (or `/voice reconfigure`) to configure voice input on their own terms

## [1.0.8] - 2026-03-12

### Fixed
- **Extension load crash:** Changed `import { Theme }` to `import type { Theme }` — value import caused undefined reference during jiti module resolution

## [1.0.7] - 2026-03-12

### Fixed
- **Critical: Extension crash on load** — Replaced deep internal imports (`dist/core/messages.js`, `dist/modes/interactive/theme/theme.js`) with top-level `@mariozechner/pi-coding-agent` imports that resolve correctly through pi's jiti virtual module system
- Removed unused `CustomMessage` type import; used generic `registerMessageRenderer<T>` instead

## [1.0.6] - 2026-03-12

### Fixed
- Bootstrap script (`setup-macos.sh`) now works when piped from curl (`curl | bash`)
- Fixed `BASH_SOURCE[0]: unbound variable` error in piped execution
- Smoke test gracefully skips when daemon.py isn't available locally

## [1.0.5] - 2026-03-12

### Changed
- Improved Quick Start section with bootstrap script front and center
- Removed migration note from README (clean install path only)

## [1.0.4] - 2026-03-12

### Changed
- **Package scoped:** Renamed from `pi-listen` to `@codexstar/pi-listen`
- Updated all install commands, documentation, and references to use scoped name
- Users migrating from `pi-listen` should run: `pi uninstall pi-listen && pi install npm:@codexstar/pi-listen`

## [1.0.3] - 2026-03-12

### Fixed
- **Traceback leak:** Daemon no longer sends Python stack traces to socket clients (CWE-200)
- **Orphan process:** ensureDaemon kills spawned process on timeout instead of leaking orphans
- **Auto-backend reload:** Skip redundant daemon reload when `config.backend='auto'` (default)
- **isHolding state:** Reset hold state when recording fails to start (e.g. SoX not installed)
- **Shutdown race:** Capture temp file path in local variable before async transcription
- **BTW fallback API:** Use `sendUserMessage` instead of `sendMessage` in BTW error path

### Security
- Full adversarial bug-hunter audit completed (13 findings, 8 confirmed, 6 fixed)

## [1.0.2] - 2026-03-12

### Fixed
- Normalize `auto` voice backend resolution in transcribe.py
- Unbounded buffer vulnerability in daemon socket handler (MAX_MSG_SIZE guard)
- Replace environ-based backend transport with CLI arguments (`sys.argv[1]`)

### Added
- macOS bootstrap script (`scripts/setup-macos.sh`)
- Windows bootstrap script (`scripts/setup-windows.ps1`)
- Setup script integration tests
- Enterprise-grade zero-touch installation automation

## [1.0.1] - 2026-03-12

### Changed
- Package renamed from `pi-voice` to `pi-listen`
- Updated package metadata and documentation

## [1.0.0] - 2026-03-12

### Added
- Hold-to-talk voice input (Kitty protocol key release detection)
- Fallback `Ctrl+Shift+V` toggle for non-Kitty terminals
- Persistent STT daemon (`daemon.py`) with warm model cache
- Multi-backend transcription engine (`transcribe.py`)
  - faster-whisper (CTranslate2)
  - moonshine (ONNX)
  - whisper-cpp (CLI)
  - Deepgram Nova 3 (cloud API)
  - Parakeet (NVIDIA NeMo)
- BTW side conversations (`/btw`, `/btw:new`, `/btw:clear`, `/btw:inject`, `/btw:summarize`)
- Voice → BTW glue: `Ctrl+Shift+B` = hold to record → auto-send as BTW
- Guided first-run onboarding wizard
- Model-aware setup with installed model detection
- `/voice` command suite: `on`, `off`, `test`, `info`, `setup`, `reconfigure`, `doctor`, `backends`, `daemon`
- Global and project-scoped configuration
- VAD (Voice Activity Detection) pre-filtering via Silero
- Auto-idle shutdown for daemon (5-minute timeout)

[1.0.10]: https://github.com/codexstar69/pi-listen/compare/v1.0.9...v1.0.10
[1.0.9]: https://github.com/codexstar69/pi-listen/compare/v1.0.8...v1.0.9
[1.0.8]: https://github.com/codexstar69/pi-listen/compare/v1.0.7...v1.0.8
[1.0.7]: https://github.com/codexstar69/pi-listen/compare/v1.0.6...v1.0.7
[1.0.6]: https://github.com/codexstar69/pi-listen/compare/v1.0.5...v1.0.6
[1.0.5]: https://github.com/codexstar69/pi-listen/compare/v1.0.4...v1.0.5
[1.0.4]: https://github.com/codexstar69/pi-listen/compare/v1.0.3...v1.0.4
[1.0.3]: https://github.com/codexstar69/pi-listen/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/codexstar69/pi-listen/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/codexstar69/pi-listen/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/codexstar69/pi-listen/releases/tag/v1.0.0

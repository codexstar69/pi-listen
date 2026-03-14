# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.2] - 2026-03-14

### Added
- **First-run welcome hint** — shows keybinding guide on first session when API key is set, or setup instructions when it's not
- **Zero-config auto-activation** — if `DEEPGRAM_API_KEY` is already in environment, voice activates immediately without running `/voice setup`
- **Deepgram API key validation** — `/voice test` now hits the Deepgram API to verify the key is valid (not just checking if it's set)
- **Full diagnostics output** — `/voice test` shows pass/fail for each prerequisite with actionable setup instructions

## [3.0.0] - 2026-03-14

### Changed
- **Complete rewrite** — Deepgram streaming-only architecture (removed local daemon, 5-backend system, BTW side conversations)
- **Separated Pompom companion** — creature animation now ships as its own extension (`@codexstar/pi-pompom`)
- **Renamed package** — `@codexstar/pi-listen` → `@codexstar/pi-voice`

### Added
- **Double-escape editor clear** — press Escape twice within 500ms to clear the editor text
- **Cross-platform escape handling** — filters Kitty key-release/repeat events to prevent false triggers
- **Voice commands** — "hey pi, run tests", "undo", "submit", "new line", punctuation shortcuts
- **Continuous dictation** — `/voice dictate` for long-form input without holding keys
- **Recording history** — `/voice history` shows recent transcriptions
- **Audio-reactive UI** — braille waveform + face widget that reacts to voice levels
- **Enterprise hold detection** — Kitty protocol + non-Kitty gap-based fallback with typing cooldown

### Removed
- Local STT daemon (`daemon.py`, `transcribe.py`)
- 5-backend system (faster-whisper, moonshine, whisper-cpp, parakeet)
- BTW side conversations
- VAD pre-filtering
- Pompom/Lumo creature companion (now separate package)

[3.0.0]: https://github.com/codexstar69/pi-listen/releases/tag/v3.0.0

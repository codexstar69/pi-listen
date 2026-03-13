# Security Policy

## Reporting a Vulnerability

Email security concerns to the maintainer via GitHub Issues (mark as confidential if needed).

## Scope

- **Audio data** is streamed to Deepgram's API for transcription. Audio is not stored locally.
- **API keys** can be stored in `~/.env.secrets`, `~/.zshrc`, or Pi settings files.
- **No telemetry** — pi-listen does not collect or transmit usage data.
- **Error messages** do not expose API keys, file paths, or stack traces to the UI.

## Supported Versions

| Version | Supported |
|---------|-----------|
| 2.x     | ✅        |
| 1.x     | ❌        |

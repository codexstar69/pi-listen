# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 1.0.x   | ✅ Active support   |
| < 1.0   | ❌ Not supported    |

## Reporting a Vulnerability

If you discover a security vulnerability in pi-listen, **please report it responsibly.**

### How to Report

1. **Do NOT open a public GitHub issue** for security vulnerabilities
2. Email: Open a private security advisory on GitHub via the [Security tab](https://github.com/codexstar69/pi-listen/security/advisories/new)
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact assessment
   - Suggested fix (if any)

### Response Timeline

- **Acknowledgment:** Within 48 hours
- **Assessment:** Within 7 days
- **Fix release:** Within 30 days for confirmed vulnerabilities

### Scope

The following are **in scope** for security reports:

| Area | Examples |
|------|----------|
| Daemon socket | Unauthorized command execution, information disclosure |
| Audio handling | Path traversal in audio file paths, temp file races |
| Config files | Credential exposure, unsafe deserialization |
| Subprocess spawning | Command injection via backend/model parameters |
| Dependencies | Known CVEs in direct dependencies |

The following are **out of scope:**

| Area | Reason |
|------|--------|
| Deepgram API key exposure | User-managed credential, documented in setup |
| Local Unix socket permissions | OS-level configuration, not application code |
| Denial of service via large audio files | Local-only, self-inflicted |
| AI/LLM prompt injection in BTW | BTW conversations are user-initiated |

## Security Design Principles

pi-listen follows these security principles:

### 1. Local-First Processing
Audio is processed locally by default. Cloud backends (Deepgram) are opt-in and clearly labeled during onboarding.

### 2. No Telemetry
pi-listen does not collect, transmit, or store any usage data, analytics, or telemetry.

### 3. Minimal Attack Surface
- Unix domain sockets (not TCP — no network exposure)
- Daemon auto-shuts down after 5 minutes of inactivity
- No persistent storage of audio recordings
- Temp files are immediately deleted after transcription

### 4. Defense in Depth
- Socket message size limit (1 MB) prevents buffer exhaustion
- Error responses do not expose stack traces or internal paths
- Backend names are validated before use
- Audio file paths are validated before transcription

### 5. Principle of Least Privilege
- Daemon runs as the current user (no root required)
- No network listeners (Unix socket only)
- No filesystem access beyond temp directory and config files

## Recent Security Audit

**Date:** 2026-03-12  
**Method:** Adversarial bug-hunter pipeline (Recon → Hunter → Skeptic → Referee)  
**Scope:** Full codebase (7 source files, 2948 lines)  
**Results:**

| Metric | Count |
|--------|-------|
| Findings reported | 13 |
| Confirmed bugs | 8 |
| Security-relevant | 1 (CWE-200, fixed) |
| Fixed | 6 |
| Remaining (low severity) | 2 |

See `.bug-hunter/report.md` for the full audit report.

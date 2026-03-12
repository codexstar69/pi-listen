# pi-voice QA matrix

This checklist separates what has already been verified from what is still worth running on a target release machine.

## Automated checks completed

- [x] TypeScript compiles via `bun run typecheck`
- [x] Python files compile via `python3 -m py_compile daemon.py transcribe.py`
- [x] Config migration and scope tests pass
- [x] Diagnostics recommendation tests pass
- [x] Provisioning-plan helper tests pass
- [x] Onboarding fallback/finalization tests pass
- [x] Model-detection metadata tests pass
- [x] Model-aware recommendation and labeling tests pass
- [x] Aggregate verification passes via `bun run check`

## RPC-assisted / smoke checks completed

### First-run onboarding
- [x] Fresh install with no config triggers onboarding prompt on interactive startup
- [x] Choosing `Remind me later` suppresses the prompt during the defer window
- [x] `/voice reconfigure` reopens the onboarding flow
- [x] Partial legacy config re-enters onboarding instead of being treated as complete
- [x] API-vs-Local choice appears correctly in the terminal UI

### Local mode and model-aware behavior
- [x] Local path still offers backend choices even when no cached models are found
- [x] Suggested commands are sensible for missing SoX and local backend installs
- [x] Missing-model path is represented cleanly through model-aware provisioning/manual-step logic
- [x] `/voice doctor` distinguishes current-config repair from recommended alternative
- [x] Backend scan exposes `installed_models` and `install_detection` metadata
- [x] `/voice backends` surfaces model-aware backend summaries and detection hints
- [x] `/voice info` surfaces current model readiness and scope details
- [x] `/voice test` surfaces current model readiness plus targeted guidance

### Scope and reconfiguration
- [x] Project scope writes to `.pi/settings.json`
- [x] Reconfigure path saves updated choices
- [x] Repair-needed states stay incomplete instead of being falsely marked complete

## Remaining target-machine checks

These are the highest-value checks still worth running on the actual release machine.

### Real audio / device validation
- [ ] Hold-to-talk with a real microphone transcribes successfully after setup
- [ ] `Ctrl+Shift+V` fallback recording works in the target terminal
- [ ] `Ctrl+Shift+B` BTW voice path still works end-to-end

### Real provider / ready-now validation
- [ ] Cloud setup completes successfully when `DEEPGRAM_API_KEY` is present
- [ ] Local setup completes successfully on a machine with SoX installed and a supported backend available
- [ ] Runtime uses the selected backend/model after reconfiguration
- [ ] Config-scoped socket behavior avoids silent stale-daemon reuse under real session switching

### Installed-model specific validation
- [ ] At least one machine with an already cached local model shows an explicit ready-now / already-installed path in onboarding
- [ ] At least one machine with backend installed but selected model missing shows clear download-required messaging
- [ ] `/voice info` and `/voice test` reflect the selected model's readiness accurately on a machine with real caches

## Suggested execution order

1. Fresh install / first-run prompt
2. Remind-me-later and reconfigure path
3. Project-scope save path
4. Model-aware local path with no cached models
5. Installed-model ready-now path on a machine with real caches
6. Cloud/API happy path with valid credentials
7. Real microphone / hold-to-talk sanity pass

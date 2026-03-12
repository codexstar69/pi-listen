# pi-voice release and validation plan

## Goal

Ship the pi-voice onboarding overhaul with enough automated coverage, manual verification, migration safety, and release hygiene that a fresh install feels reliable and polished for both local and cloud STT users.

This plan assumes the product work adds a first-run onboarding wizard, API-vs-local setup flow, richer config state, and daemon/config correctness fixes.

## Baseline findings

Current repo state, verified during planning:

- TypeScript currently compiles: `bunx tsc -p tsconfig.json`
- Python files currently compile: `python3 -m py_compile daemon.py transcribe.py`
- Backend scan currently works: `python3 transcribe.py --list-backends`
- `faster-whisper` is available in the current environment; other backends are not.
- The package currently lacks release-facing docs and package metadata hardening:
  - no `README.md`
  - no test scripts in `package.json`
  - no peer dependency declarations
  - no gallery preview metadata
- Setup is currently command-driven (`/voice setup`) rather than first-run guided onboarding.

## Release risks to manage

### Product risks
- First-run onboarding becomes annoying instead of helpful.
- Users cannot tell the difference between cloud and local tradeoffs.
- Project-scoped and global-scoped configuration behave inconsistently.
- Recovery paths for partial setup failures are unclear.

### Technical risks
- Daemon keeps stale backend/model state across sessions.
- Config migrations break existing users or re-trigger onboarding incorrectly.
- Auto-install/provisioning paths behave differently across machines.
- Validation passes in development but fails for fresh installs.

### Release risks
- Package ships without docs, screenshots, or troubleshooting.
- No repeatable QA checklist exists.
- Regressions in recording, daemon startup, or transcription are discovered after release.

## Test strategy

Use four layers of validation.

### 1. Static and smoke checks
These run on every change and before release.

Required commands:

```sh
bunx tsc -p tsconfig.json
python3 -m py_compile daemon.py transcribe.py
python3 transcribe.py --list-backends
```

Add package scripts so release validation is one command away:

```json
{
  "scripts": {
    "typecheck": "bunx tsc -p tsconfig.json",
    "check:python": "python3 -m py_compile daemon.py transcribe.py",
    "check:backends": "python3 transcribe.py --list-backends",
    "check": "bun run typecheck && bun run check:python && bun run check:backends"
  }
}
```

### 2. Unit and module tests
Add tests around deterministic logic that should not require live audio.

Priority test targets:
- config load/merge precedence
- config migration from old shape to new onboarding-aware shape
- onboarding state gating logic
- scope-aware save behavior (global vs project)
- recommendation logic for API vs local mode
- backend metadata normalization
- daemon reconciliation logic when config differs from loaded daemon state
- doctor/diagnostics result formatting

Suggested structure:
- `extensions/voice/config.test.ts`
- `extensions/voice/onboarding-state.test.ts`
- `extensions/voice/diagnostics.test.ts`
- `extensions/voice/runtime.test.ts`
- `test_python/test_daemon_contract.py` or lightweight Python script-based tests if no framework is added

### 3. Integration tests
Add smoke tests that exercise multi-step workflows with mocked or controlled dependencies.

High-value integration scenarios:
- first session with no voice config launches onboarding
- completed onboarding does not relaunch every session
- `/voice setup` reopens the same wizard
- saving to project scope persists under `.pi/settings.json`
- saving to global scope persists under `~/.pi/agent/settings.json`
- daemon reloads when backend/model changes
- `/voice doctor` reports missing SoX and missing API key correctly
- backend scan failure renders a recovery state instead of crashing

### 4. Manual end-to-end QA
Because this extension depends on terminal UI, mic input, Python libraries, and host tools, manual QA remains mandatory before release.

## Validation matrix

### Environment matrix

| Environment | Why it matters | Must pass |
|---|---|---|
| macOS Apple Silicon, fresh machine profile | primary target, likely most users | Yes |
| macOS Apple Silicon, existing faster-whisper installed | migration and detection path | Yes |
| macOS Apple Silicon, no SoX installed | local onboarding provisioning | Yes |
| macOS Apple Silicon, Deepgram key present | cloud onboarding fast path | Yes |
| macOS Apple Silicon, Deepgram key absent | cloud error/recovery path | Yes |
| Existing user with prior `voice` config | migration and no-regression path | Yes |
| Project-local config present overriding global | precedence and scope correctness | Yes |
| Multiple Pi sessions with daemon already running | stale daemon/state isolation | Yes |

### Feature matrix

| Area | Validation focus |
|---|---|
| First-run onboarding | launches once, clear choices, cancellable, resumable |
| API mode | key detection, validation, copy, recovery |
| Local mode | dependency install path, model selection, warmup |
| Scope | global/project save behavior and re-load behavior |
| Diagnostics | accurate pass/fail and actionable remediation |
| Runtime | hold-to-talk, toggle shortcut, BTW voice path |
| Daemon | correct backend/model, reload, shutdown, status |
| Docs | install guide, troubleshooting, shortcuts, privacy notes |
| Packaging | files included, metadata correct, package installs cleanly |

## Manual QA scenarios

Each scenario should end with a short written outcome and screenshots or terminal captures where useful.

### Fresh install scenarios
1. Install package into a clean environment with no voice config.
2. Start Pi interactively.
3. Confirm onboarding appears automatically.
4. Verify the user is first asked how they want to use STT: API or local download.
5. Complete setup and verify a successful test transcription.
6. Restart Pi and verify onboarding does not reappear.

### Local mode scenarios
1. Start from no SoX and no backend dependency.
2. Choose local/download mode.
3. Verify recommendation copy is understandable.
4. Verify missing dependencies are detected.
5. Verify auto-install or guided-install copy is correct.
6. Verify model choice reflects tradeoffs.
7. Verify daemon warms the selected model.
8. Record and transcribe successfully.

### Cloud mode scenarios
1. Start with no `DEEPGRAM_API_KEY`.
2. Choose API mode.
3. Verify missing key is explained clearly.
4. Add key and retry without restarting the world if possible.
5. Verify transcription succeeds.

### Migration scenarios
1. Seed existing config with the old shape (`enabled`, `backend`, `model`, `language`).
2. Start Pi.
3. Confirm config migrates safely.
4. Verify onboarding does not clobber a valid prior setup.
5. Verify invalid prior setups are detected and routed into repair flow.

### Failure and recovery scenarios
1. Break backend scan intentionally.
2. Break daemon startup intentionally.
3. Simulate missing mic audio.
4. Verify `/voice doctor` and onboarding recovery screens show next steps, not generic failures.

### Scope scenarios
1. Save global config.
2. Open a project with no project-level override.
3. Confirm global config is honored.
4. Save project-level config.
5. Confirm project config overrides global.
6. Switch to another project and confirm the project-specific override does not leak.

### Daemon correctness scenarios
1. Run session A with backend/model X.
2. Start session B with backend/model Y.
3. Verify the daemon uses the requested config and does not silently keep X.
4. Validate `/voice info` and `/voice daemon status` reflect reality.

## Automated release gates

Before tagging any release, require all of the following:

### Gate 1: code health
- TypeScript check passes.
- Python compile check passes.
- Backend list check passes.
- New unit/integration tests pass.

### Gate 2: packaging health
- `package.json` includes required metadata.
- published file list includes extension, Python files, README, docs if intended
- no accidental junk files in package
- install from local path works in a clean test directory

### Gate 3: UX acceptance
- first-run onboarding flow verified manually
- API and local paths both verified manually
- error copy reviewed for clarity
- success screen reviewed for polish and completeness

### Gate 4: documentation acceptance
- README explains first-run onboarding
- README includes cloud vs local comparison
- README includes troubleshooting section
- docs include shortcut summary and doctor/setup commands

## Migration and versioning strategy

### Config schema versioning
Introduce an explicit config version under `voice.version`.

Recommended migration rules:
- `version: 1` = legacy shape
- `version: 2` = onboarding-aware shape with scope, onboarding metadata, and validation markers

Migration requirements:
- old config is upgraded in memory first
- migration is idempotent
- migrated config is only written back after successful validation or explicit confirmation
- invalid legacy config routes to repair/onboarding instead of failing silently

### Onboarding versioning
Track onboarding independently from base config shape if needed:

```json
{
  "voice": {
    "version": 2,
    "onboarding": {
      "schemaVersion": 2,
      "completed": true,
      "completedAt": "...",
      "lastValidatedAt": "..."
    }
  }
}
```

Re-run onboarding when:
- config version is too old
- onboarding schema version is too old
- required dependency state is no longer valid
- user explicitly runs `/voice setup` or `/voice reconfigure`

### Release versioning
Use semver aligned to impact:
- patch: bugfixes, copy, docs, non-breaking diagnostics improvements
- minor: new onboarding flow, new commands, new backend options, migration-safe config additions
- major: breaking config changes, command removals, behavior changes that require user adaptation

For the onboarding overhaul, a **minor release** is appropriate if migration is safe. Use a **major release** only if existing settings or workflows break.

## Docs and package metadata plan

### Required new docs
- `README.md`
- `docs/backends.md`
- `docs/troubleshooting.md`
- optional `docs/release-checklist.md`

### README requirements
Include:
- what pi-voice does
- what happens on first run after install
- API vs local decision guide
- supported backends and who each is for
- installation requirements
- shortcuts and commands
- troubleshooting basics
- privacy and cost notes

### Package metadata requirements
Update `package.json` with:
- `homepage`
- `repository`
- `bugs`
- richer description
- `peerDependencies` for Pi runtime packages
- scripts for checks
- `files` list including README and docs if shipped
- optional package gallery `image` or `video`

## Rollout sequencing

### Phase 1: stabilization
Ship only after daemon/config correctness issues are fixed.

Release focus:
- config-vs-daemon consistency
- scope-aware persistence
- no first-run UX yet if correctness is still shaky

### Phase 2: onboarding beta
Add the first-run onboarding flow behind a conservative release.

Release focus:
- clean new-user path
- migration-safe old-user path
- basic doctor/setup recovery

### Phase 3: provisioning and polish
Improve dependency installation, doctor flow, docs, and package presentation.

Release focus:
- fewer manual steps
- clearer remediation
- publication-grade package page

### Phase 4: broad confidence release
After at least one full QA pass across the matrix, publish the polished release broadly.

## Release checklist

### Pre-release
- [ ] Merge product, technical, and QA plans
- [ ] Add all required package scripts
- [ ] Add/refresh docs
- [ ] Verify package contents
- [ ] Run automated checks
- [ ] Run manual QA matrix
- [ ] Review migration behavior from old config
- [ ] Review stale-daemon scenarios

### Release candidate signoff
- [ ] New-user onboarding approved
- [ ] Existing-user migration approved
- [ ] Local mode approved
- [ ] API mode approved
- [ ] Diagnostics approved
- [ ] README/package metadata approved

### Post-release
- [ ] Validate install from published source
- [ ] Confirm first-run flow on a clean environment
- [ ] Capture any support issues or friction reports
- [ ] Patch critical regressions before expanding scope

## Definition of done

The onboarding overhaul is done only when all of the following are true:

### Functional
- Fresh installs trigger guided onboarding automatically in interactive sessions.
- Users are explicitly asked whether they want API or local STT before backend details.
- Local setup can provision or clearly guide missing dependencies.
- Cloud setup can detect and validate required API credentials.
- Config scope works correctly for global and project settings.
- Runtime always uses the selected backend/model, even with an already-running daemon.

### Quality
- Static checks and tests pass from a single release command.
- Manual QA matrix is completed and documented.
- Failure states provide actionable recovery guidance.
- Existing users migrate safely.

### Documentation and packaging
- README is complete and accurate.
- Troubleshooting docs exist.
- Package metadata is production-quality.
- Published package contains everything needed and nothing accidental.

### Product polish
- First-run flow feels deliberate and polished, not like an internal setup tool.
- Success state clearly tells the user what was configured and what to do next.
- Reconfiguration and doctor flows are discoverable.

## Recommended next actions

1. Add the release scripts and test scaffolding to `package.json`.
2. Implement config versioning and migration tests before building more UX.
3. Create a reusable manual QA checklist document from this plan.
4. Treat stale daemon correctness as a release blocker.
5. Do not publish the onboarding overhaul until both fresh-install and migration paths are signed off.

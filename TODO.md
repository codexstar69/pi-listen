# pi-voice world-class onboarding

## Milestones
- [x] Milestone 1: Runtime correctness foundation
  - [x] Make daemon requests explicit about backend/model/language
  - [x] Fix stale daemon config behavior and reload semantics
  - [x] Introduce safer socket strategy
  - [x] Add scope-aware config save/load helpers
  - [x] Stop eager daemon startup before config is ready
- [x] Milestone 2: Versioned config and onboarding state
  - [x] Add structured voice config types and migration path
  - [x] Track onboarding completion/version/validation state
  - [x] Separate config logic into dedicated module(s)
- [x] Milestone 3: Diagnostics and recommendation engine
  - [x] Implement environment scan (python, sox, backend availability, env vars)
  - [x] Implement API-vs-local recommendation logic
  - [x] Surface fixable vs blocking issues
- [x] Milestone 4: Premium onboarding wizard
  - [x] Launch first-run onboarding on interactive session_start
  - [x] Build API-vs-local flow
  - [x] Add backend/model/scope selection screens
  - [x] Add confirmation and completion receipt screens
- [x] Milestone 5: Provisioning and repair
  - [x] Add dependency provisioning helpers
  - [x] Add /voice doctor
  - [x] Add /voice reconfigure and repair flows
- [x] Milestone 6: Runtime polish
  - [x] Improve status line and notifications
  - [x] Improve /voice test and /voice info UX
- [x] Milestone 7: Package/docs hardening
  - [x] Add README.md
  - [x] Add docs/backends.md
  - [x] Add docs/troubleshooting.md
  - [x] Improve package metadata/scripts
- [x] Milestone 8: QA and release hardening
  - [x] Add automated verification coverage where practical
  - [x] Run manual QA matrix
  - [x] Confirm migration and reconfiguration paths

## Completion
- [x] ALL_TASKS_COMPLETE

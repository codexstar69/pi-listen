# pi-voice review findings

## Scope reviewed

Reviewed the current implementation changes around:
- config migration and persistence
- diagnostics and recommendation logic
- onboarding flow wiring
- provisioning-plan helpers
- package/docs updates

I also sanity-checked that the repo still passes its current verification command (`bun run check`).

## Summary

The recent changes are a strong step forward: config handling is now modular, there is automated coverage for config/diagnostics/provisioning logic, first-run onboarding is wired, and docs/package metadata exist.

However, there are still a few meaningful correctness and product-integration gaps that should be fixed before calling the onboarding flow production-grade.

---

## Findings

### 1. Partial legacy configs are migrated as "fully onboarded" even when they are incomplete
**Severity:** high

**Where:**
- `extensions/voice/config.ts:109-122`
- `extensions/voice/config.ts:183-186`

**What happens:**
`migrateConfig()` sets `fallbackCompleted = hasExplicitLegacyData`, and `hasExplicitLegacyData` is true for any non-empty `voice` object. That means a legacy config like:

```json
{
  "voice": {
    "enabled": true
  }
}
```

will be treated as onboarding-complete after migration, even though it has no explicit backend choice, no scope choice, and no validated setup state.

Because `needsOnboarding()` only checks `source === "default"` or `!config.onboarding.completed`, these partially configured users will skip first-run onboarding entirely.

**Why it matters:**
This can strand existing users in a half-configured state while the new first-run setup never appears.

**Recommended fix:**
Only mark migrated configs as onboarding-complete when the legacy data contains enough meaningful setup information, for example:
- explicit backend + model, or
- an explicit onboarding block, or
- a post-migration validation pass succeeds.

At minimum, replace the current `Object.keys(rawVoice).length > 0` heuristic with a validity check.

---

### 2. Local onboarding silently dead-ends when backend scan returns no local entries
**Severity:** high

**Where:**
- `extensions/voice/onboarding.ts:70-72`
- `extensions/voice/onboarding.ts:162-167`

**What happens:**
For the local path, `runVoiceOnboarding()` filters `diagnostics.backends` down to local backends and passes the result to `chooseBackendAndModel()`. If the backend scan failed, Python is missing, or the list is empty for any reason, `chooseBackendAndModel()` returns `undefined` immediately.

That bubbles up as a cancelled setup, and the caller shows:
- `Voice setup skipped. Run /voice setup anytime.`

**Why it matters:**
This is exactly the scenario where onboarding should be most helpful. Instead, users who need guided local installation get bounced out of setup without a backend choice or a concrete recovery screen.

**Recommended fix:**
When no local backends are discoverable, synthesize fallback onboarding options instead of aborting. For example:
- show `faster-whisper`, `moonshine`, `whisper-cpp`, `parakeet` as installable options
- keep the install/provisioning hints visible
- only block if the user explicitly cancels

In other words: discovery failure should still lead to a guided install path, not a silent exit.

---

### 3. Onboarding is marked complete before provisioning or validation succeeds
**Severity:** high

**Where:**
- `extensions/voice/onboarding.ts:123-135`
- `extensions/voice.ts:666-677`
- `extensions/voice.ts:867-882`

**What happens:**
`runVoiceOnboarding()` returns a config with:
- `onboarding.completed = true`
- `completedAt = now`

This happens before any provisioning, daemon warmup, or transcription validation succeeds.

Then `voice.ts` saves the config and immediately shows `Voice setup complete.`. If `ensureDaemon(config)` fails, the success receipt has already been emitted.

**Why it matters:**
Users can end up with:
- onboarding marked complete
- startup onboarding suppressed in future sessions
- a broken local/cloud setup that never actually worked

This is the biggest remaining integration gap between onboarding, provisioning, and validation.

**Recommended fix:**
Split onboarding state into at least two phases, for example:
- `selected` / `configured`
- `validated`

Only mark `onboarding.completed = true` after:
- required provisioning checks pass, and
- daemon startup / provider validation succeeds, and ideally
- a real mic/transcription smoke test passes.

If validation fails, persist a repair-needed state instead of a completed state.

---

### 4. `/voice doctor` suggests fixes for the recommendation, not for the user’s current config
**Severity:** medium

**Where:**
- `extensions/voice.ts:900-931`

**What happens:**
`/voice doctor` computes a recommendation, then builds the provisioning plan from that recommendation:

- `mode: recommendation.mode`
- `backend: recommendation.backend`
- `model: recommendation.model`

That means the suggested commands can describe how to install the recommended path rather than how to repair the user’s actual current setup.

Example failure mode:
- user has selected `whisper-cpp`
- doctor recommends `faster-whisper`
- doctor outputs `python3 -m pip install faster-whisper`
- but the actual thing the user needs may be `brew install whisper-cpp` or model-file remediation

**Why it matters:**
Doctor output should first help the user fix the configuration they are using now. Recommendations are useful, but they should be presented separately.

**Recommended fix:**
Show two sections:
1. **Repair current setup** — based on the current saved config
2. **Recommended alternative** — based on the recommendation engine

Do not conflate them into one provisioning plan.

---

### 5. README configuration example is now stale relative to the code
**Severity:** low

**Where:**
- `README.md:152-160`

**What happens:**
The README still shows the old minimal config shape:
- `enabled`
- `language`
- `backend`
- `model`

But the implementation now uses a richer versioned shape with:
- `version`
- `mode`
- `scope`
- `onboarding`
- `btwEnabled`

**Why it matters:**
The new docs correctly talk about scope-aware onboarding and versioned config, so the example block is now misleading and under-documents the actual stored state.

**Recommended fix:**
Update the example to either:
- show the current real shape, or
- explicitly label the short example as “minimal conceptual example” and add a second “actual saved shape” example.

---

### 6. "Remind me later" is not persisted, so the prompt will repeat every fresh session
**Severity:** low

**Where:**
- `extensions/voice/onboarding.ts:20-27`
- `extensions/voice.ts:661-682`

**What happens:**
When the user chooses `Remind me later`, no state is written. The extension simply notifies and returns.

Because `needsOnboarding()` will still be true on the next session, the same prompt will appear again.

**Why it matters:**
This is not a runtime correctness bug, but it is a noticeable polish gap for a “world-class” onboarding experience.

**Recommended fix:**
Persist a lightweight defer marker such as:
- `onboarding.skippedAt`
- `onboarding.deferUntil`

Then suppress the startup prompt until the defer window expires or the user explicitly runs `/voice setup`.

---

## What looks good

These changes are solid and worth keeping:
- modular config extraction in `extensions/voice/config.ts`
- deterministic tests for config, diagnostics, and provisioning helper logic
- config-scoped socket derivation via `getSocketPath()`
- explicit backend/model on daemon transcription requests
- first-run onboarding hook on interactive `session_start`
- package scripts + docs now exist

## Recommended priority order for fixes

1. **Fix migrated partial configs being treated as completed**
2. **Fix onboarding completion semantics so success requires validation**
3. **Fix local onboarding dead-end when discovery returns zero local backends**
4. **Split `/voice doctor` into current-config repair vs recommended alternative**
5. **Persist “remind me later” and refresh README config example**

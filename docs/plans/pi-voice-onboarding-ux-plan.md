# pi-voice onboarding UX/product plan

## Objective

Turn pi-voice from a technically capable extension into a polished, enterprise-grade product that feels guided from the first interactive session after installation. The extension should help users choose **cloud API vs local/offline STT**, recommend the right backend and model, validate the setup end to end, and leave the user confident that voice input is ready.

This plan assumes Pi packages do **not** provide a dedicated interactive install hook. The primary onboarding moment is therefore the **first interactive `session_start`** after package installation, with the same flow re-openable via `/voice setup`.

## Current UX gaps

- Install ends with no guidance; the user must discover `/voice setup` manually.
- `session_start` only loads config and starts the daemon if enabled, so first-run users get no onboarding help.
- `/voice setup` is a thin backend/model picker rather than a guided setup flow.
- Unavailable backends only show raw install hints, not guided next steps.
- Setup writes only to global settings even though the extension reads both global and project settings.
- The current language is implementation-centric (`backend`, `model`) rather than user-centric (`privacy`, `speed`, `works offline`).

## UX principles

1. **Lead with outcomes, not implementation.** Ask how the user wants to use voice, not which backend they want.
2. **Make the recommended path obvious.** Every decision screen should identify one recommended option.
3. **Let novices stay simple; let experts go deep.** Keep advanced knobs behind an “Advanced options” branch.
4. **No dead ends.** Every failure state must offer retry, change choice, or exit-with-instructions.
5. **Always end with proof.** Setup is not complete until a mic check and transcription test succeeds or the user intentionally skips validation.
6. **Respect scope.** Users should choose whether setup applies globally or only to the current project.

## Primary user journeys

### 1. First-time user, wants fastest path
- Installs package.
- Opens Pi interactively.
- Sees a welcome wizard.
- Chooses “Fastest setup”.
- System recommends cloud/API if no local stack is present, or a ready local backend if already installed.
- User validates with a short voice sample.
- Completion screen shows shortcuts and how to reopen setup.

### 2. Privacy-focused user, wants local/offline
- Opens wizard.
- Chooses “Keep audio on this machine”.
- System recommends the best local backend available, with model suggestions by latency/quality.
- If dependencies are missing, onboarding moves into guided install/remediation.
- Validation confirms offline transcription works.

### 3. Team/project-specific user
- Opens setup from a repo with shared preferences.
- Chooses “Only for this project”.
- Onboarding writes to `.pi/settings.json`.
- Completion screen explains that the project now has voice defaults separate from the user’s global defaults.

### 4. Returning user with broken setup
- `session_start` detects incomplete onboarding or failed validation.
- User sees a lightweight recovery banner:
  - Re-run setup
  - View diagnostics
  - Continue without voice

## Proposed onboarding architecture

## Triggering rules

Show the full onboarding wizard when all are true:
- `ctx.hasUI === true`
- this is the first run for the current onboarding version, or config is incomplete
- voice onboarding has not been explicitly dismissed for the current version

Show a smaller recovery prompt when:
- voice is enabled but dependencies or validation are missing
- previously selected backend/model is no longer available

Do not auto-launch onboarding in:
- print mode
- non-interactive mode
- sessions where the user already completed onboarding for the current version

## State model required for UX

The product layer should persist more than the current `enabled/language/backend/model` fields. UX needs:

- `onboardingVersion`
- `onboardingCompletedAt`
- `onboardingDismissedAt`
- `installMode` (`cloud`, `local`, `auto`)
- `preferenceProfile` (`fastest`, `privacy`, `accuracy`, `balanced`)
- `settingsScope` (`global`, `project`)
- `lastValidatedAt`
- `lastValidationResult`
- `selectedBackend`
- `selectedModel`
- `language`
- `btwEnabled`
- `hotkeyMode`

This state enables re-entry, migration, recovery prompts, and a completion receipt.

## End-to-end onboarding flow

### Step 0: Welcome
Purpose: explain value, reassure the user, and set expectation.

Screen content:
- Title: `Set up voice input`
- One-line value prop: “Talk to Pi instead of typing. Voice can run through a cloud API or locally on your machine.”
- Estimated time: “About 2–3 minutes”
- Actions:
  - `Start setup` (recommended)
  - `Remind me later`
  - `Skip and disable voice`

### Step 1: How do you want to use speech-to-text?
This is the key product question.

Options:
- `Use a cloud API` — best for fastest setup, usually strong accuracy, requires network and API credentials
- `Run locally on this machine` — best for privacy/offline use, may require local dependencies and model downloads
- `Help me choose` — guided recommendation path

This screen should use plain language, not backend names.

### Step 2: What matters most?
Shown either after “Help me choose” or as a tuning screen for the other choices.

Options:
- `Fastest setup`
- `Best privacy`
- `Best accuracy`
- `Balanced default` (recommended)

This preference drives recommendation logic and default model choice.

### Step 3: Recommendation screen
The extension should translate user intent into a recommended path.

Example output card:
- `Recommended: faster-whisper / small`
- Why: “Runs locally, is already available on this machine, and offers a good speed/accuracy tradeoff.”
- Secondary choices:
  - `Deepgram / nova-3` for fast cloud setup
  - `faster-whisper / medium` for higher accuracy

This screen should always include:
- recommendation
- one-sentence reasoning
- alternate choices
- an `Advanced options` action for explicit backend/model picking

### Step 4: Scope selection
Ask where to save settings.

Options:
- `Use in all projects` → writes to `~/.pi/agent/settings.json`
- `Only use in this project` → writes to `<cwd>/.pi/settings.json`

Copy should explain the difference in plain English.

### Step 5: Backend/model confirmation
This step is simple for the default path and richer for advanced users.

Default path:
- Show selected backend/model pair with brief tradeoff copy.
- Confirm or edit.

Advanced path:
- Show backend list grouped by `Cloud` and `Local`.
- Each option should include:
  - availability
  - whether install is required
  - quality/speed/privacy hint
- Model picker should include badges such as:
  - `fast`
  - `recommended`
  - `higher accuracy`
  - `larger download`

### Step 6: Readiness and install guidance
This screen should summarize what the extension needs to finish setup.

Examples:
- `Microphone capture tool: ready`
- `Python runtime: ready`
- `Selected backend: needs install`
- `API key: missing`

Actions:
- `Install now` when safe/automatable
- `Show commands` when manual install is required
- `Choose a different option`
- `Continue after I’ve installed it`

This step should use a loader/spinner UI during scans and install attempts. Avoid dropping the user straight into notifications.

### Step 7: Live validation
This is the confidence moment.

Flow:
1. prompt user to say a short sentence
2. record 2–4 seconds
3. transcribe with the chosen backend/model
4. show transcript and latency
5. ask if the result looks correct

Actions:
- `Looks good`
- `Try again`
- `Change model`
- `Open advanced settings`

If validation is skipped, completion should clearly say “Configured but not validated”.

### Step 8: Completion receipt
The final screen should feel premium and operational.

Show:
- selected mode (`cloud` or `local`)
- backend and model
- where settings were saved
- key shortcuts:
  - `Hold SPACE` to talk when the editor is empty
  - `Ctrl+Shift+V` to toggle recording
  - `Ctrl+Shift+B` for voice → BTW
- commands:
  - `/voice setup`
  - `/voice doctor`
  - `/voice info`

Actions:
- `Finish`
- `Open advanced settings`
- `Run another test`

## Recommendation logic

## Input signals
Use these inputs to drive recommendation UX:
- installed backends from `transcribe.py --list-backends`
- presence of API key(s)
- SoX / `rec` availability
- Python availability
- current OS assumptions (Apple Silicon-friendly defaults)
- user’s stated preference (`fastest`, `privacy`, `accuracy`, `balanced`)

## Recommendation rules

### If user chooses `cloud API`
- Recommend `deepgram / nova-3` when API credentials exist.
- If credentials are missing, show it as available-but-needs-auth.
- If the user wants fastest setup and local dependencies are absent, cloud can still be recommended with a clear note about the missing API key step.

### If user chooses `local`
- Prefer a backend that is already installed.
- If `faster-whisper` is installed, make it the default recommendation.
- If nothing local is installed, recommend the easiest reliable local path and explain required setup.
- Use `small` as the default balanced model, `base`/`tiny` for speed-first, `medium` or better for accuracy-first where supported.

### If user chooses `help me choose`
- `privacy` → local first
- `fastest setup` → whichever of cloud-with-key or already-installed local backend minimizes friction
- `accuracy` → best available model/backend with clear tradeoff warning
- `balanced` → installed local backend first, else cloud if credentialed, else easiest local path

## Failure and recovery UX

### Failure class 1: missing dependency before selection
Example: `rec` missing.

UX:
- show a checklist screen
- explain why the dependency matters
- offer install/remediation path
- let the user continue exploring options without losing progress

### Failure class 2: selected backend unavailable
Example: user picks Deepgram without an API key.

UX:
- inline error card: `Deepgram needs an API key before it can be used.`
- actions:
  - `Set up credentials`
  - `Choose another backend`
  - `Go back`

### Failure class 3: test recording fails
Example: no mic data recorded.

UX:
- say exactly what failed: no audio recorded vs device unavailable vs capture tool missing
- offer `Try again`, `Open diagnostics`, `Choose another setup path`

### Failure class 4: transcription succeeds but quality is poor
UX:
- present transcript with confidence framing: “We got a result, but it may not be accurate enough yet.”
- actions:
  - `Use a larger model`
  - `Switch to cloud`
  - `Keep current setup`

### Failure class 5: onboarding abandoned mid-flow
UX:
- save draft onboarding state
- on next session start, show `Continue voice setup` instead of starting from zero

## Settings scope UX

Settings scope is important enough to get its own explicit decision.

### Global scope
Copy: “Use these voice settings in every project on this machine.”

Use when:
- the user is setting up a personal default workflow
- the chosen backend is tied to user-level credentials or machine-level installs

### Project scope
Copy: “Use these voice settings only in this project. This is useful for team-shared defaults or repo-specific preferences.”

Use when:
- the project has team norms
- a repo wants a specific STT mode
- the user wants to avoid changing other projects

### UX requirement
Every confirmation/receipt screen must repeat where settings were saved.

## TUI implementation guidance

Use Pi’s richer custom TUI patterns instead of plain `ctx.ui.select()` for the main wizard.

Recommended building blocks:
- `ctx.ui.custom()` for the multi-step wizard shell
- `SelectList` for decision screens
- `BorderedLoader` for scans, installs, and test transcription
- `SettingsList` for advanced toggles
- `setStatus` for persistent `MIC`, `SETUP`, `READY`, `ERROR` state
- `setWidget` for non-modal reminders such as `Voice setup incomplete — /voice setup`

## Suggested UX structure in code

- `showVoiceOnboardingWizard(ctx)`
- `showVoiceRecoveryPrompt(ctx)`
- `showVoiceValidationFlow(ctx, draftConfig)`
- `showVoiceCompletionReceipt(ctx, summary)`
- `showVoiceAdvancedSettings(ctx)`

The same wizard should power both:
- automatic first-run onboarding
- manual `/voice setup`

## Success metrics / acceptance criteria

The UX is ready when:
- a first-time user can complete setup without reading README instructions
- the user is asked API vs local in plain language
- the user gets a recommendation with reasoning, not just a backend list
- setup clearly distinguishes global vs project scope
- the user can validate mic + transcription before completion
- any failed step offers retry, back, or fallback without losing progress
- `/voice setup` reopens the same polished flow
- the completion screen leaves the user knowing exactly how to start using voice

## Phase-by-phase UX delivery plan

### Phase 1: Product foundation
- Expand config/onboarding state model.
- Add first-run detection and recovery-banner logic.
- Define backend metadata needed for recommendation copy.

### Phase 2: Wizard MVP
- Build welcome, mode choice, preference choice, recommendation, scope, and completion screens.
- Keep advanced settings behind a secondary branch.
- Reuse existing backend scan data.

### Phase 3: Validation and recovery
- Add live mic/transcription test flow.
- Add resumable onboarding state.
- Add incomplete-setup widget/banner on session start.

### Phase 4: Premium polish
- Better copy, badges, progress states, completion receipt, and clearer failure messages.
- Add `/voice doctor` and route broken states there.
- Tighten “recommended” logic based on observed machine readiness.

## Deliverables from the UX track

1. A multi-step first-run onboarding wizard spec.
2. Plain-language decision tree for cloud vs local STT.
3. Recommendation/copy matrix for backend and model suggestions.
4. Recovery UX for missing dependencies, missing auth, failed recording, and failed transcription.
5. Settings-scope UX that clearly supports both global and project-local configuration.

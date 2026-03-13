# Autoresearch: pi-voice Enterprise Quality

## Goal
Optimize pi-voice for stability, reliability, and code quality.
Deepgram streaming is the only backend. The system works — the goal
is to find bugs, harden edge cases, eliminate dead code, and polish.

## Primary Metric
**Test pass count + typecheck clean** — measured as: all tests pass (exit 0)
and `tsc --noEmit` passes (exit 0). Metric = total issues found & fixed per run.

## Experiment Loop
1. Audit a specific area (e.g., error handling, memory leaks, race conditions)
2. Identify concrete issues with evidence
3. Fix them surgically
4. Verify: typecheck + tests + manual code review
5. Log the result

## Rules
- Never break existing functionality
- Every fix must be evidence-based (show the bug before fixing)
- Run typecheck + tests after every change
- One focused area per experiment

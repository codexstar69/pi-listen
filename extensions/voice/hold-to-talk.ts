export type VoiceHoldState = "idle" | "warmup" | "recording" | "finalizing";

export function getKittyHoldTiming({
	heldMs,
	intentDelayMs,
	holdThresholdMs,
}: {
	heldMs: number;
	intentDelayMs: number;
	holdThresholdMs: number;
}): { warmupDelayMs: number; activationDelayMs: number } {
	const normalizedHeldMs = Math.max(0, heldMs);
	return {
		warmupDelayMs: Math.max(0, intentDelayMs - normalizedHeldMs),
		activationDelayMs: Math.max(0, holdThresholdMs - normalizedHeldMs),
	};
}

export function shouldInsertSpaceOnKittyReleaseBeforeRecording(
	state: VoiceHoldState,
	spaceConsumed: boolean,
): boolean {
	return !spaceConsumed && (state === "idle" || state === "warmup");
}

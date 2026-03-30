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
		activationDelayMs: holdThresholdMs,
	};
}

export function stripInsertedSpaceOnWarmupStart({
	textBeforePress,
	currentText,
}: {
	textBeforePress: string;
	currentText: string;
}): string {
	return currentText === `${textBeforePress} ` ? textBeforePress : currentText;
}

export function shouldStartRecordingFromWarmup({
	state,
	releasePending,
}: {
	state: VoiceHoldState;
	releasePending: boolean;
}): boolean {
	return state === "warmup" && !releasePending;
}

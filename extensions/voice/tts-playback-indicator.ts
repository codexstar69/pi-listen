/**
 * Honest playback state indicator — §6 of v7.1 plan.
 *
 * v1 of the plan proposed a fake amplitude meter; both Codex and Gemini
 * flagged it as misleading (a bouncing meter would imply audio is
 * playing even when the player is muted, stuck, or failed). v7.1 ships
 * a state spinner only:
 *
 *   ◓ Synthesizing · 1s
 *   ◑ Playing · 4s · [esc] stop
 *   (mounted only while state ∈ {synthesizing, playing}; dismissed on idle)
 *
 * Rotating spinner phase comes from the shared `RenderTicker`, so it
 * stays in phase with any concurrently-mounted install bar — no
 * tearing.
 *
 * v7.2 may add a real PCM amplitude bar ALONGSIDE the state word once
 * the streaming-playback path emits real levels; until then the
 * label-only design keeps every signal honest.
 */

import { BaseDisposableWidget, type WidgetRegistry, WIDGET_KEY } from "./ui-widget-base";
import type { RenderTicker } from "./ui-render-ticker";
import { spinnerFrame } from "./ui-icons";
import type { InstallWidgetUI } from "./tts-install-progress";

export type PlaybackState = "idle" | "synthesizing" | "playing";

export interface TtsPlaybackIndicatorOpts {
	readonly ui: InstallWidgetUI;
	readonly registry: WidgetRegistry;
	readonly ticker: RenderTicker;
	readonly onStop?: () => void;
}

export class TtsPlaybackIndicator extends BaseDisposableWidget {
	readonly key = WIDGET_KEY.ttsPlayback;
	private readonly ui: InstallWidgetUI;
	private readonly onStop?: () => void;
	private state: PlaybackState = "idle";
	private startedAt: number | null = null;
	private spinnerTick = 0;

	constructor(opts: TtsPlaybackIndicatorOpts) {
		super(opts.registry, () => opts.ui.setWidget(WIDGET_KEY.ttsPlayback, undefined));
		this.ui = opts.ui;
		this.onStop = opts.onStop;
		this.unsubTicker = opts.ticker.subscribe({
			tick: () => this.onTick(),
			dispose: () => this.dispose(),
			label: "tts-playback",
		});
		opts.registry.register(this);
	}

	/** Caller flips state through (synthesizing → playing → idle). */
	setState(s: PlaybackState): void {
		if (this.disposed) return;
		if (s === this.state) return;
		this.state = s;
		if (s === "idle") {
			// Idle dismisses the widget — render an empty slot, then
			// dispose so the registry/ticker drop us cleanly.
			this.dispose();
			return;
		}
		// Re-arm timestamp on each non-idle transition so the elapsed
		// counter resets between synthesize and play.
		this.startedAt = Date.now();
		this.renderFrame();
	}

	/** Caller invokes this from [esc] when the indicator owns escape. */
	stop(): void {
		try { this.onStop?.(); } catch { /* never fail caller */ }
		this.dispose();
	}

	private onTick(): void {
		if (this.disposed) return;
		this.spinnerTick++;
		this.renderFrame();
	}

	private renderFrame(): void {
		const state = this.state;
		const startedAt = this.startedAt ?? Date.now();
		const tick = this.spinnerTick;
		this.ui.setWidget(
			this.key,
			(_tui: any, theme: any) => ({
				invalidate() {},
				render: (width: number): string[] => renderPlaybackLine({ theme, width, state, startedAt, tick }),
			}),
			{ placement: "belowEditor" },
		);
	}
}

interface RenderInput {
	theme: any;
	width: number;
	state: PlaybackState;
	startedAt: number;
	tick: number;
}

/** Pure render — exported for tests. */
export function renderPlaybackLine(input: RenderInput): string[] {
	const fg = (role: string, s: string): string => (input.theme?.fg ? input.theme.fg(role, s) : s);
	const dim = (s: string) => fg("dim", s);
	const accent = (s: string) => fg("accent", s);

	if (input.state === "idle") return [""];

	const spinner = spinnerFrame(input.tick);
	const word = input.state === "synthesizing" ? "Synthesizing" : "Playing";
	const elapsed = Math.max(0, Math.round((Date.now() - input.startedAt) / 1000));
	const elapsedStr = `${elapsed}s`;

	// Speaking-state animation: a 3-bar Siri-style wave whose height is
	// modulated by the *real* RMS of the most recently written PCM
	// chunk (published by `speak.publishPlaybackLevel`). Clock-driven
	// per-bar phase so it looks alive even between chunk updates;
	// amplitude scales with the published level so a long pause goes
	// quiet instead of dancing dishonestly.
	const wave = input.state === "playing"
		? renderSpeakingWave(input.tick, dim, accent)
		: "";

	const showHint = input.width >= 60;
	const hint = showHint ? ` ${dim("·")} ${dim("[esc] stop")}` : "";
	const wavePart = wave ? `  ${wave}` : "";
	return [` ${accent(spinner)} ${accent(word)} ${dim("·")} ${dim(elapsedStr)}${wavePart}${hint}`];
}

/**
 * 3-cell speaking wave. Each cell picks one of the standard block
 * heights `▁▂▃▄▅▆▇` driven by `sin(t * speed + phase[i])`. Amplitude
 * is gated by the real published playback RMS so the wave goes still
 * during silence and animates during speech.
 *
 * Decays to 0 over 400ms after the last chunk write — sherpa generates
 * full chunks at once so writes are bursty (~once per sentence). The
 * decay keeps the wave moving smoothly between writes without claiming
 * amplitude that doesn't exist long after the chunk has played out.
 */
function renderSpeakingWave(tick: number, dim: (s: string) => string, accent: (s: string) => string): string {
	const bars = ["▁", "▂", "▃", "▄", "▅", "▆", "▇"];
	const shared = (globalThis as any).__piListen as { playbackLevel?: number; playbackLevelAt?: number } | undefined;
	const rawLevel = shared?.playbackLevel ?? 0;
	const lastAt = shared?.playbackLevelAt ?? 0;
	const ageMs = Date.now() - lastAt;
	const decay = Math.max(0, 1 - ageMs / 400);
	const level = rawLevel * decay;

	// Tick advances every RenderTicker frame. Convert to a continuous
	// "time" so the sine motion is smooth across frames.
	const t = tick * 0.18;
	let out = "";
	for (let i = 0; i < 3; i++) {
		const phase = i * 1.7;
		const oscillation = (Math.sin(t + phase) + 1) * 0.5; // 0..1
		const v = Math.max(0, Math.min(0.999, level * (0.5 + 0.5 * oscillation)));
		const idx = Math.floor(v * bars.length);
		const ch = bars[Math.min(bars.length - 1, idx)] ?? "▁";
		// Tallest bar in accent, shorter ones dim — feels like a wave
		// rolling across the cells without strobing color changes.
		out += idx >= 4 ? accent(ch) : dim(ch);
	}
	return out;
}

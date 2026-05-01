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
	const showHint = input.width >= 60;
	const hint = showHint ? ` ${dim("·")} ${dim("[esc] stop")}` : "";
	return [` ${accent(spinner)} ${accent(word)} ${dim("·")} ${dim(elapsedStr)}${hint}`];
}

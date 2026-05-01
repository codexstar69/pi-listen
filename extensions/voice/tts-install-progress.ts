/**
 * Sticky download progress widget — §5 of v7.1 plan.
 *
 * Visual: one-line widget under the editor showing model name, a
 * `█`-filled / `░`-empty progress bar, percentage, transfer speed, and
 * ETA. After download completes, the bar morphs into a spinner-led
 * status line for the extract / verify phases. On `phase: "done"` (or
 * abort) the widget disposes itself.
 *
 * Lifecycle (§1 contract, v5):
 *   - Per-instance widget key via `installWidgetKey(modelId)` so two
 *     concurrent installs (different model ids) coexist.
 *   - Subscribes to the shared `RenderTicker` as a `TickerSubscriber`
 *     `{ tick, dispose, label }` so an auto-eviction (3 throws in a
 *     row) cleanly tears down the slot.
 *   - `dispose()` runs the BaseDisposableWidget order: idempotency
 *     guard → set `disposed` → unsubscribe ticker → `onDispose()` →
 *     clear slot → owner-checked `registry.unregister()`.
 *   - `onProgress()` is guarded by `disposed` so a fetched-tar chunk
 *     landing 200 ms after dispose is a no-op.
 *
 * Cancellation (§4 contract):
 *   - Owns an `AbortController`; `cancel()` aborts it and disposes.
 *   - `cancel()` is what voice.ts wires to the [esc] handler when the
 *     install widget is at the top of the escape priority order.
 */

import { BaseDisposableWidget, type WidgetRegistry, installWidgetKey } from "./ui-widget-base";
import type { RenderTicker } from "./ui-render-ticker";
import type { TtsInstallProgress } from "./tts-local-models";
import { ICON, spinnerFrame } from "./ui-icons";
import { truncateToVisualWidth, visualWidth } from "./ui-width";

// Pi extension UI surface — typed as the minimal subset we use, kept
// loose intentionally so the widget can be instantiated from
// production code (real ExtensionContext) and tests (a stub).
export interface InstallWidgetUI {
	setWidget(
		key: string,
		content: ((tui: any, theme: any) => { invalidate(): void; render(width: number): string[] }) | undefined,
		options?: { placement?: "aboveEditor" | "belowEditor" },
	): void;
}

export interface TtsInstallProgressWidgetOpts {
	readonly ui: InstallWidgetUI;
	readonly modelId: string;
	readonly modelName: string;
	readonly totalBytesEstimate: number;
	readonly registry: WidgetRegistry;
	readonly ticker: RenderTicker;
	readonly controller: AbortController;
}

const SAMPLE_WINDOW = 30; // ~3 s of byte samples at 10 Hz tick

export class TtsInstallProgressWidget extends BaseDisposableWidget {
	readonly key: string;
	private readonly ui: InstallWidgetUI;
	private readonly modelId: string;
	private readonly modelName: string;
	private readonly controller: AbortController;
	private phase: TtsInstallProgress["phase"] = "download";
	private bytes = 0;
	private totalBytes: number;
	private spinnerTick = 0;
	private readonly samples: { t: number; bytes: number }[] = [];

	constructor(opts: TtsInstallProgressWidgetOpts) {
		super(opts.registry, () => opts.ui.setWidget(installWidgetKey(opts.modelId), undefined));
		this.key = installWidgetKey(opts.modelId);
		this.ui = opts.ui;
		this.modelId = opts.modelId;
		this.modelName = opts.modelName;
		this.totalBytes = opts.totalBytesEstimate;
		this.controller = opts.controller;

		// Explicit ownership: ticker calls dispose() on auto-eviction.
		this.unsubTicker = opts.ticker.subscribe({
			tick: () => this.onTick(),
			dispose: () => this.dispose(),
			label: `install:${opts.modelId}`,
		});
		opts.registry.register(this);
		// Immediate first frame so the widget appears before the first tick.
		this.renderFrame();
	}

	/** Receives `ensureTtsModelInstalled`'s onProgress events. */
	onProgress(info: TtsInstallProgress): void {
		if (this.disposed) return;
		this.phase = info.phase;
		if (typeof info.bytes === "number") this.bytes = info.bytes;
		if (typeof info.totalBytes === "number") this.totalBytes = info.totalBytes;
		if (info.phase === "done") this.dispose();
	}

	/** Abort the install (calls controller.abort) and dispose the widget. */
	cancel(): void {
		if (this.disposed) return;
		try { this.controller.abort(); } catch { /* abort never throws but be defensive */ }
		this.dispose();
	}

	protected override onDispose(): void {
		this.samples.length = 0;
	}

	private onTick(): void {
		if (this.disposed) return;
		this.spinnerTick++;
		// Sample byte counter for speed/ETA smoothing.
		const now = Date.now();
		this.samples.push({ t: now, bytes: this.bytes });
		if (this.samples.length > SAMPLE_WINDOW) this.samples.shift();
		this.renderFrame();
	}

	private speedBytesPerSec(): number | null {
		if (this.samples.length < 2) return null;
		const first = this.samples[0]!;
		const last = this.samples[this.samples.length - 1]!;
		const dt = (last.t - first.t) / 1000;
		if (dt <= 0) return null;
		const dbytes = last.bytes - first.bytes;
		if (dbytes <= 0) return null;
		return dbytes / dt;
	}

	private etaSeconds(): number | null {
		const speed = this.speedBytesPerSec();
		if (speed == null || this.totalBytes <= 0) return null;
		const remaining = Math.max(0, this.totalBytes - this.bytes);
		if (remaining === 0) return 0;
		return Math.round(remaining / speed);
	}

	private formatBytes(n: number): string {
		if (n < 1024) return `${n} B`;
		if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
		if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
		return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`;
	}

	private formatEta(seconds: number): string {
		if (seconds < 60) return `${seconds}s`;
		const m = Math.floor(seconds / 60);
		const s = seconds % 60;
		return `${m}m${String(s).padStart(2, "0")}s`;
	}

	private renderFrame(): void {
		// Captured snapshot — render closure reads these, not `this.*`,
		// so a dispose-mid-render leaves the closure with last values.
		const phase = this.phase;
		const bytes = this.bytes;
		const totalBytes = this.totalBytes;
		const spinner = spinnerFrame(this.spinnerTick);
		const speed = this.speedBytesPerSec();
		const eta = this.etaSeconds();
		const modelName = this.modelName;

		this.ui.setWidget(
			this.key,
			(_tui: any, theme: any) => ({
				invalidate() {},
				render: (width: number): string[] => {
					return renderInstallLine({ theme, width, phase, bytes, totalBytes, spinner, speed, eta, modelName, formatBytes: (n: number) => this.formatBytes(n), formatEta: (s: number) => this.formatEta(s) });
				},
			}),
			{ placement: "belowEditor" },
		);
	}
}

interface RenderInput {
	theme: any;
	width: number;
	phase: TtsInstallProgress["phase"];
	bytes: number;
	totalBytes: number;
	spinner: string;
	speed: number | null;
	eta: number | null;
	modelName: string;
	formatBytes: (n: number) => string;
	formatEta: (s: number) => string;
}

/** Pure render — exported for tests and small-screen tier dispatch. */
export function renderInstallLine(input: RenderInput): string[] {
	const { theme, width, phase, bytes, totalBytes, spinner, speed, eta, modelName, formatBytes, formatEta } = input;
	const fg = (role: string, s: string): string => (theme?.fg ? theme.fg(role, s) : s);
	const dim = (s: string) => fg("dim", s);
	const accent = (s: string) => fg("accent", s);

	if (phase === "extract" || phase === "verify") {
		const status = phase === "extract" ? "Extracting" : "Verifying";
		const label = ` ${accent(spinner)} ${accent(modelName)} ${dim("·")} ${status}…`;
		return [label];
	}

	// download phase — bar + percentage + speed + ETA
	const percent = totalBytes > 0 ? Math.min(100, Math.floor((bytes * 100) / totalBytes)) : 0;
	const showEta = width >= 80;
	const showSpeed = width >= 70;

	// Reserve room for: " <name>  <pct>%  <bytes/total>  <speed>  <eta>"
	const nameMax = Math.min(visualWidth(modelName), Math.floor(width * 0.28));
	const truncatedName = truncateToVisualWidth(modelName, nameMax);
	const sizeStr = totalBytes > 0
		? `${formatBytes(bytes)} / ${formatBytes(totalBytes)}`
		: formatBytes(bytes);
	const speedStr = speed != null ? `${formatBytes(speed)}/s` : "";
	const etaStr = eta != null ? `ETA ${formatEta(eta)}` : "";

	const parts: string[] = [];
	parts.push(` ${accent(truncatedName)} `);
	parts.push(`${dim(`${percent}%`.padStart(4))}  `);
	parts.push(`${dim(sizeStr)}`);
	if (showSpeed && speedStr) parts.push(`  ${dim(ICON.middot)}  ${dim(speedStr)}`);
	if (showEta && etaStr) parts.push(`  ${dim(ICON.middot)}  ${dim(etaStr)}`);

	const prefix = parts.join("");
	const used = visualWidth(prefix);
	const barCols = Math.max(0, width - used - 1);

	if (barCols < 4) {
		// Not enough room for a meaningful bar — fall back to a percent-only line.
		return [prefix + ` ${dim(`(${percent}%)`)}`];
	}
	// v7.2 — subpixel-smooth progress bar (Charm/lipgloss convention).
	// Each cell can be 0/8, 1/8, 2/8 ... 8/8 filled via the U+2581-2588
	// block elements. Drops the chunky █░ "8-bit pixel" look in favor
	// of a gradient fill that updates smoothly at 10Hz.
	const totalEighths = Math.round((bytes / Math.max(1, totalBytes)) * barCols * 8);
	const fullCells = Math.floor(totalEighths / 8);
	const partialEighths = totalEighths % 8; // 0..7
	const partialCell = partialEighths === 0 ? "" : ICON.barPartial[partialEighths - 1]!;
	const emptyCells = Math.max(0, barCols - fullCells - (partialCell ? 1 : 0));
	const bar = ICON.barFilled.repeat(fullCells) + partialCell + ICON.barEmpty.repeat(emptyCells);
	return [prefix + ` ${accent(bar)}`];
}

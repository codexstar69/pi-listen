/**
 * Single shared 10 Hz frame coalescer for animated widgets — §2 of the
 * v7.1 plan.
 *
 * Why this exists: three independent `setInterval(…, 100ms)` widgets
 * drift relative to each other and each call `ctx.ui.setWidget()`
 * separately, causing 20-30 partial re-renders per second instead of 10
 * cohesive ones. With this ticker, every animated widget subscribes to
 * a single `setInterval`; all `setWidget` calls land in the same JS
 * turn so the Pi TUI sees one batch of slot updates per tick.
 *
 * Failure isolation (Codex v4 #2 + Gemini v3 #2):
 *   - Subscriber `tick()` is wrapped in try/catch — one throwing widget
 *     cannot crash the loop.
 *   - 3 consecutive throws auto-unsubscribe the widget. The ticker
 *     then calls `sub.dispose?.()` to evict its slot — but the
 *     `dispose()` invocation is ITSELF wrapped in try/catch, so a
 *     widget whose state is corrupted enough to throw on render
 *     (likely also throws on dispose) does not crash the ticker.
 *
 * Lifetime:
 *   - Lazy start: `setInterval` is created on first subscriber.
 *   - Lazy stop: `setInterval` is cleared on last unsubscribe.
 *   - `dispose()` clears all subscribers + the interval. Idempotent.
 */

import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

const VOICE_DEBUG = !!process.env.PI_VOICE_DEBUG;
const VOICE_LOG_FILE = path.join(os.tmpdir(), "pi-voice-debug.log");

function debug(...args: unknown[]) {
	if (!VOICE_DEBUG) return;
	const ts = new Date().toISOString().split("T")[1];
	const line = `[voice-ticker ${ts}] ${args.map(a => (typeof a === "object" ? JSON.stringify(a) : String(a))).join(" ")}\n`;
	try { fs.appendFileSync(VOICE_LOG_FILE, line); } catch { /* best-effort */ }
}

/** Tick frequency — 10 Hz keeps animation smooth without burning CPU. */
const TICK_INTERVAL_MS = 100;

/** Throws-in-a-row threshold before auto-eviction. */
const THROW_THRESHOLD = 3;

/**
 * Subscriber descriptor — explicit ownership (Codex v3 #2). Pass
 * `dispose` only when the ticker should evict this subscriber's slot
 * after auto-unsubscribe; ad-hoc test subscribers can omit it.
 */
export interface TickerSubscriber {
	/** Called once per tick (10 Hz). May throw; the ticker isolates. */
	readonly tick: () => void;
	/**
	 * Optional ownership hook. If present and the subscriber is
	 * auto-unsubscribed after `THROW_THRESHOLD` consecutive throws,
	 * the ticker calls `dispose()` to tear down the broken widget's
	 * slot. The call is wrapped in `try/catch` so a corrupted widget
	 * cannot crash the ticker.
	 */
	readonly dispose?: () => void;
	/** Optional debug label, surfaced in `voiceDebug()` output. */
	readonly label?: string;
}

export interface RenderTicker {
	/** Subscribe. Returns an unsubscribe fn — calling it is idempotent. */
	subscribe(subscriber: TickerSubscriber): () => void;
	/** Active subscriber count. Test/debug. */
	refCount(): number;
	/** Tear down the ticker — clears interval + subscribers. Idempotent. */
	dispose(): void;
}

interface SubEntry {
	readonly sub: TickerSubscriber;
	throwsInARow: number;
	unsubscribed: boolean;
}

class RenderTickerImpl implements RenderTicker {
	private readonly entries = new Set<SubEntry>();
	private timer: ReturnType<typeof setInterval> | null = null;
	private disposed = false;

	subscribe(subscriber: TickerSubscriber): () => void {
		if (this.disposed) {
			// Late subscriber after dispose — give them a no-op
			// unsubscriber and don't start the timer.
			return () => {};
		}
		const entry: SubEntry = { sub: subscriber, throwsInARow: 0, unsubscribed: false };
		this.entries.add(entry);
		if (this.timer == null) this.startTimer();
		return () => {
			if (entry.unsubscribed) return;
			entry.unsubscribed = true;
			this.entries.delete(entry);
			if (this.entries.size === 0) this.stopTimer();
		};
	}

	refCount(): number {
		return this.entries.size;
	}

	dispose(): void {
		if (this.disposed) return;
		this.disposed = true;
		this.stopTimer();
		// Mark all entries unsubscribed so any retained closure that
		// still calls its unsubscribe fn is a no-op.
		for (const entry of this.entries) entry.unsubscribed = true;
		this.entries.clear();
	}

	private startTimer(): void {
		this.timer = setInterval(() => this.runTick(), TICK_INTERVAL_MS);
		// Don't keep the event loop alive just for animation.
		(this.timer as any)?.unref?.();
	}

	private stopTimer(): void {
		if (this.timer != null) {
			clearInterval(this.timer);
			this.timer = null;
		}
	}

	private runTick(): void {
		// Snapshot entries before iterating — auto-eviction during the
		// loop would otherwise mutate the live Set. Snapshot also lets
		// us safely call dispose() on evicted entries after the loop.
		const snapshot = Array.from(this.entries);
		const evict: SubEntry[] = [];
		for (const entry of snapshot) {
			if (entry.unsubscribed) continue;
			try {
				entry.sub.tick();
				entry.throwsInARow = 0;
			} catch (err) {
				entry.throwsInARow++;
				debug("tick threw", entry.sub.label ?? "(unlabeled)", entry.throwsInARow, "/", THROW_THRESHOLD, String(err));
				if (entry.throwsInARow >= THROW_THRESHOLD) evict.push(entry);
			}
		}
		for (const entry of evict) {
			if (entry.unsubscribed) continue;
			entry.unsubscribed = true;
			this.entries.delete(entry);
			// Eviction dispose() — Gemini v3 #2: try/catch around
			// dispose itself so a broken widget can't crash the ticker.
			try { entry.sub.dispose?.(); } catch (err) { debug("eviction dispose threw", entry.sub.label ?? "(unlabeled)", String(err)); }
		}
		if (this.entries.size === 0) this.stopTimer();
	}
}

/** Construct a fresh ticker. One per session is the expected shape. */
export function makeRenderTicker(): RenderTicker {
	return new RenderTickerImpl();
}

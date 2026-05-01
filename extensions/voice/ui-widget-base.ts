/**
 * Widget ownership model for the v7.1 Settings UI redesign — §1 of the
 * approved plan, hardened across v3/v4/v5 reviews.
 *
 * Why this exists: `ctx.ui.setWidget(key, content)` writes to a key-slotted
 * store. Two widgets with different keys coexist; two widgets with the
 * same key overwrite. v7.1 introduces install + playback widgets that
 * must coexist with the recording widget, so we need a registry that
 * (a) prevents same-key collision, (b) lets completed widgets evict
 * themselves so the registry doesn't leak across long sessions, and
 * (c) survives misbehaving widgets that throw on disposal.
 *
 * The contract was reviewed and SHIP'd by both Codex and Gemini at v5.
 * Six load-bearing properties of the implementation:
 *
 *   1. `register(w)` synchronously calls `existing.dispose()` before
 *      swapping in `w`, so a same-key handover never races.
 *   2. `unregister(key, owner)` is OWNER-CHECKED — only deletes the Map
 *      entry if it currently points to `owner`. A stale dispose path
 *      (e.g. an old install widget's queued cleanup running after a
 *      newer same-key widget was registered) cannot evict the
 *      successor.
 *   3. `disposeAll()` iterates a CLONED snapshot of values, because
 *      each widget's `dispose()` will synchronously call back into
 *      `unregister(key, this)`, mutating the underlying Map.
 *   4. Each `dispose()` call inside `disposeAll()` is wrapped in
 *      `try/catch` so one throwing widget cannot abort cleanup for
 *      the rest.
 *   5. `installWidgetKey(modelId)` produces per-model-id keys so two
 *      installs for different models occupy different slots.
 *   6. The base class's `dispose()` ordering — set `disposed` true
 *      before any cleanup work — is enforced via `BaseDisposableWidget`,
 *      so subclasses cannot accidentally clear timers / slots before
 *      flipping the flag.
 */

import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

const VOICE_DEBUG = !!process.env.PI_VOICE_DEBUG;
const VOICE_LOG_FILE = path.join(os.tmpdir(), "pi-voice-debug.log");

function debug(...args: unknown[]) {
	if (!VOICE_DEBUG) return;
	const ts = new Date().toISOString().split("T")[1];
	const line = `[voice-ui ${ts}] ${args.map(a => (typeof a === "object" ? JSON.stringify(a) : String(a))).join(" ")}\n`;
	try { fs.appendFileSync(VOICE_LOG_FILE, line); } catch { /* best-effort */ }
}

/** A widget that owns one slot key and can be torn down. */
export interface DisposableWidget {
	/**
	 * Stable widget-key the slot writes to. Install widgets use
	 * `installWidgetKey(modelId)`; recording uses `"voice-recording"`;
	 * playback uses `"voice-tts-playback"`.
	 */
	readonly key: string;
	/** Tear down the widget. MUST be idempotent (re-entry returns early). */
	dispose(): void;
}

/**
 * Registry that owns lifetime of every active widget. One per session,
 * created in `session_start` and drained in `voiceCleanup`.
 */
export interface WidgetRegistry {
	/**
	 * Register a widget under its `key`. If a widget with the same
	 * key is already registered, that incumbent's `dispose()` is
	 * called synchronously BEFORE `w` takes the slot — preventing
	 * same-key collision.
	 */
	register(w: DisposableWidget): void;
	/**
	 * Owner-checked eviction (Codex v4 #3). Removes the Map entry for
	 * `key` ONLY if the current entry is `owner` (identity-equal). If
	 * a stale dispose path runs after a NEW widget has taken the same
	 * key, the unregister is a no-op and the new widget stays bound.
	 * Calls with absent `key` or mismatched owner are no-ops.
	 */
	unregister(key: string, owner: DisposableWidget): void;
	/**
	 * Drains the registry. Iterates a CLONED snapshot of values
	 * (Gemini v4 implementation note) because each `dispose()`
	 * re-enters `unregister`. Each `dispose()` is wrapped in
	 * `try/catch` so one throwing widget cannot abort cleanup for
	 * the rest. Idempotent — safe to call twice.
	 */
	disposeAll(): void;
	/** Active widget count. Test/debug only. */
	size(): number;
}

class WidgetRegistryImpl implements WidgetRegistry {
	private readonly entries = new Map<string, DisposableWidget>();

	register(w: DisposableWidget): void {
		const existing = this.entries.get(w.key);
		if (existing && existing !== w) {
			// Synchronous dispose of incumbent before swap — same-key
			// handover never races. Wrap in try/catch so a throwing
			// incumbent cannot prevent the new widget from registering.
			try { existing.dispose(); } catch (err) { debug("register: incumbent dispose threw", w.key, String(err)); }
		}
		this.entries.set(w.key, w);
	}

	unregister(key: string, owner: DisposableWidget): void {
		const cur = this.entries.get(key);
		if (cur === owner) this.entries.delete(key);
		// else: stale call — successor already took the slot, do nothing.
	}

	disposeAll(): void {
		// Cloned snapshot — each dispose() re-enters unregister() and
		// mutates the underlying Map. Iterating a snapshot guarantees
		// no widget is skipped during the cascade.
		const snapshot = Array.from(this.entries.values());
		for (const w of snapshot) {
			try { w.dispose(); } catch (err) { debug("disposeAll: dispose threw", w.key, String(err)); }
		}
		// Defensive: any widget that didn't unregister itself (buggy
		// dispose) is force-evicted so a later disposeAll() is truly idempotent.
		this.entries.clear();
	}

	size(): number {
		return this.entries.size;
	}
}

/** Construct a fresh registry. One per session is the expected shape. */
export function makeWidgetRegistry(): WidgetRegistry {
	return new WidgetRegistryImpl();
}

/**
 * Per-instance install widget key. Concurrent installs of different
 * model ids occupy different slots (`voice-tts-install:kitten-…` vs
 * `voice-tts-install:kokoro-…`). Same-id installs are already
 * serialized by `ensureTtsModelInstalled`'s in-flight Map.
 */
export function installWidgetKey(modelId: string): string {
	return `voice-tts-install:${modelId}`;
}

/** Stable widget keys for non-install widgets. */
export const WIDGET_KEY = {
	recording: "voice-recording",
	ttsPlayback: "voice-tts-playback",
} as const;

/**
 * Optional base class enforcing the §1 dispose ordering. Subclasses
 * implement `onDispose()` for their type-specific cleanup; the base
 * runs the universal sequence around it.
 *
 * Order (Codex v4 #2 + Codex v5 nit):
 *   (a) check `if (this.disposed) return;` — idempotency on PRIOR state
 *   (b) set `this.disposed = true` BEFORE any cleanup work
 *   (c) call `this.unsubTicker?.()` so ticker refcount drops
 *   (d) `onDispose()` for subclass-specific cleanup (timers, etc.)
 *   (e) clear the slot via the injected `clearSlot()` callback
 *   (f) `registry.unregister(this.key, this)` — owner-checked eviction
 */
export abstract class BaseDisposableWidget implements DisposableWidget {
	abstract readonly key: string;
	protected disposed = false;
	protected unsubTicker: (() => void) | null = null;
	protected readonly registry: WidgetRegistry;
	protected readonly clearSlot: () => void;

	constructor(registry: WidgetRegistry, clearSlot: () => void) {
		this.registry = registry;
		this.clearSlot = clearSlot;
	}

	/** Subclass cleanup hook. Called between unsubTicker and clearSlot. */
	protected onDispose(): void {}

	dispose(): void {
		if (this.disposed) return;
		this.disposed = true;
		try { this.unsubTicker?.(); } catch (err) { debug("dispose: unsubTicker threw", this.key, String(err)); }
		this.unsubTicker = null;
		try { this.onDispose(); } catch (err) { debug("dispose: onDispose threw", this.key, String(err)); }
		try { this.clearSlot(); } catch (err) { debug("dispose: clearSlot threw", this.key, String(err)); }
		try { this.registry.unregister(this.key, this); } catch (err) { debug("dispose: unregister threw", this.key, String(err)); }
	}
}

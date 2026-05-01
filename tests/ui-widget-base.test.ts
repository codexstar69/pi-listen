import { describe, expect, test } from "bun:test";
import {
	makeWidgetRegistry,
	installWidgetKey,
	BaseDisposableWidget,
	type DisposableWidget,
	type WidgetRegistry,
} from "../extensions/voice/ui-widget-base";

class CountingWidget implements DisposableWidget {
	disposeCount = 0;
	constructor(public readonly key: string, private readonly onDispose?: () => void) {}
	dispose(): void {
		this.disposeCount++;
		this.onDispose?.();
	}
}

class ThrowingWidget implements DisposableWidget {
	disposed = false;
	constructor(public readonly key: string) {}
	dispose(): void {
		this.disposed = true;
		throw new Error("synthetic dispose failure");
	}
}

describe("installWidgetKey", () => {
	test("encodes model id into the key", () => {
		expect(installWidgetKey("kitten-nano-en-v0_2")).toBe("voice-tts-install:kitten-nano-en-v0_2");
		expect(installWidgetKey("kokoro-int8-multi-lang-v1_0")).toBe("voice-tts-install:kokoro-int8-multi-lang-v1_0");
	});
});

describe("WidgetRegistry — basic register / unregister", () => {
	test("size starts at 0", () => {
		const r = makeWidgetRegistry();
		expect(r.size()).toBe(0);
	});
	test("register adds an entry", () => {
		const r = makeWidgetRegistry();
		r.register(new CountingWidget("a"));
		expect(r.size()).toBe(1);
	});
	test("owner-checked unregister removes the widget", () => {
		const r = makeWidgetRegistry();
		const w = new CountingWidget("a");
		r.register(w);
		r.unregister("a", w);
		expect(r.size()).toBe(0);
	});
	test("unregister with absent key is a no-op", () => {
		const r = makeWidgetRegistry();
		r.unregister("never-was-here", { key: "never-was-here", dispose() {} });
		expect(r.size()).toBe(0);
	});
});

describe("WidgetRegistry — same-key collision (Codex v3 #1)", () => {
	test("registering a same-key incumbent runs its dispose() synchronously first", () => {
		const r = makeWidgetRegistry();
		const order: string[] = [];
		const a = new CountingWidget("k", () => order.push("a-disposed"));
		const b = new CountingWidget("k", () => order.push("b-disposed"));
		r.register(a);
		expect(a.disposeCount).toBe(0);
		r.register(b);
		expect(a.disposeCount).toBe(1);
		expect(order).toEqual(["a-disposed"]);
		expect(r.size()).toBe(1); // b only
	});
	test("an incumbent that throws on dispose still gets evicted", () => {
		const r = makeWidgetRegistry();
		const a = new ThrowingWidget("k");
		const b = new CountingWidget("k");
		r.register(a);
		// Should not throw — registry isolates
		expect(() => r.register(b)).not.toThrow();
		expect(a.disposed).toBe(true);
		expect(r.size()).toBe(1);
	});
});

describe("WidgetRegistry — owner-checked unregister (Codex v4 #3)", () => {
	test("stale unregister does not evict a successor under the same key", () => {
		const r = makeWidgetRegistry();
		const a = new CountingWidget("k");
		const b = new CountingWidget("k");
		r.register(a);
		r.register(b); // a.dispose() runs but a is now de-facto stale
		// Now simulate a's late cleanup running AFTER b took the slot:
		r.unregister("k", a);
		expect(r.size()).toBe(1); // b still bound
	});
});

describe("WidgetRegistry — disposeAll (Gemini v3 + v4)", () => {
	test("drains all widgets", () => {
		const r = makeWidgetRegistry();
		const a = new CountingWidget("a");
		const b = new CountingWidget("b");
		const c = new CountingWidget("c");
		r.register(a);
		r.register(b);
		r.register(c);
		r.disposeAll();
		expect(a.disposeCount).toBe(1);
		expect(b.disposeCount).toBe(1);
		expect(c.disposeCount).toBe(1);
		expect(r.size()).toBe(0);
	});
	test("a throwing widget does not block teardown of the rest", () => {
		const r = makeWidgetRegistry();
		const t = new ThrowingWidget("t");
		const c = new CountingWidget("c");
		r.register(t);
		r.register(c);
		expect(() => r.disposeAll()).not.toThrow();
		expect(t.disposed).toBe(true);
		expect(c.disposeCount).toBe(1);
		expect(r.size()).toBe(0);
	});
	test("widgets that re-enter via unregister() during disposeAll do not skip siblings", () => {
		// Reproduces the Gemini v4 implementation note. Each widget's
		// dispose() synchronously calls registry.unregister(key, this)
		// while disposeAll is mid-iteration. Cloned snapshot iteration
		// is what makes this safe.
		const r: WidgetRegistry = makeWidgetRegistry();
		class SelfUnregister implements DisposableWidget {
			disposed = false;
			constructor(public readonly key: string, private readonly reg: WidgetRegistry) {}
			dispose(): void {
				this.disposed = true;
				this.reg.unregister(this.key, this);
			}
		}
		const widgets = ["a", "b", "c", "d", "e"].map(k => new SelfUnregister(k, r));
		for (const w of widgets) r.register(w);
		expect(r.size()).toBe(5);
		r.disposeAll();
		for (const w of widgets) expect(w.disposed).toBe(true);
		expect(r.size()).toBe(0);
	});
	test("disposeAll is idempotent — a second call is a no-op", () => {
		const r = makeWidgetRegistry();
		const a = new CountingWidget("a");
		r.register(a);
		r.disposeAll();
		r.disposeAll();
		expect(a.disposeCount).toBe(1);
		expect(r.size()).toBe(0);
	});
});

describe("BaseDisposableWidget — §1 ordering", () => {
	test("dispose() is idempotent", () => {
		const r = makeWidgetRegistry();
		const events: string[] = [];
		class W extends BaseDisposableWidget {
			readonly key = "w";
			protected onDispose() { events.push("onDispose"); }
		}
		const w = new W(r, () => events.push("clearSlot"));
		r.register(w);
		w.dispose();
		w.dispose(); // no-op
		expect(events.filter(e => e === "onDispose").length).toBe(1);
		expect(events.filter(e => e === "clearSlot").length).toBe(1);
	});
	test("ordering: idempotency check → flag → unsubTicker → onDispose → clearSlot → unregister", () => {
		const r = makeWidgetRegistry();
		const log: string[] = [];
		class W extends BaseDisposableWidget {
			readonly key = "w";
			protected onDispose() { log.push("onDispose"); }
			callMe() { this.unsubTicker = () => log.push("unsubTicker"); }
		}
		const w = new W(r, () => log.push("clearSlot"));
		w.callMe();
		r.register(w);
		w.dispose();
		expect(log).toEqual(["unsubTicker", "onDispose", "clearSlot"]);
		expect(r.size()).toBe(0); // unregister ran last
	});
	test("late onProgress / re-entrant call after dispose is a no-op", () => {
		const r = makeWidgetRegistry();
		const events: string[] = [];
		class W extends BaseDisposableWidget {
			readonly key = "w";
			protected onDispose() { events.push("onDispose"); }
			lateProgress() {
				if (this.disposed) {
					events.push("lateProgress-skipped");
					return;
				}
				events.push("lateProgress-rendered");
			}
		}
		const w = new W(r, () => events.push("clearSlot"));
		r.register(w);
		w.lateProgress();
		w.dispose();
		w.lateProgress(); // arrives after dispose
		expect(events).toEqual([
			"lateProgress-rendered",
			"onDispose",
			"clearSlot",
			"lateProgress-skipped",
		]);
	});
});

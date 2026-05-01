import { describe, expect, test } from "bun:test";
import { makeRenderTicker, type TickerSubscriber } from "../extensions/voice/ui-render-ticker";

/** Run real time forward in 100 ms steps via fake clock (Bun timer mocking). */
async function awaitTicks(n: number): Promise<void> {
	// Real timers used; ticker runs at 100 ms. Wait n*100 + a small grace.
	await new Promise(resolve => setTimeout(resolve, n * 100 + 30));
}

describe("RenderTicker — lazy lifecycle", () => {
	test("starts with refCount 0", () => {
		const t = makeRenderTicker();
		expect(t.refCount()).toBe(0);
		t.dispose();
	});
	test("subscribe returns an unsubscribe fn that drops refCount", () => {
		const t = makeRenderTicker();
		const unsub = t.subscribe({ tick: () => {} });
		expect(t.refCount()).toBe(1);
		unsub();
		expect(t.refCount()).toBe(0);
		t.dispose();
	});
	test("unsubscribe is idempotent", () => {
		const t = makeRenderTicker();
		const unsub = t.subscribe({ tick: () => {} });
		unsub();
		expect(() => unsub()).not.toThrow();
		expect(t.refCount()).toBe(0);
		t.dispose();
	});
});

describe("RenderTicker — tick fires registered callbacks", () => {
	test("tick is called approximately every 100 ms", async () => {
		const t = makeRenderTicker();
		let count = 0;
		t.subscribe({ tick: () => { count++; } });
		await awaitTicks(3);
		expect(count).toBeGreaterThanOrEqual(2);
		t.dispose();
	});
	test("multiple subscribers all tick on the same interval", async () => {
		const t = makeRenderTicker();
		let a = 0;
		let b = 0;
		t.subscribe({ tick: () => { a++; } });
		t.subscribe({ tick: () => { b++; } });
		await awaitTicks(3);
		expect(a).toBeGreaterThanOrEqual(2);
		expect(b).toBeGreaterThanOrEqual(2);
		// Both subscribers tick the same number of times → coalescer working.
		expect(Math.abs(a - b)).toBeLessThanOrEqual(1);
		t.dispose();
	});
});

describe("RenderTicker — failure isolation (Codex v4 #2)", () => {
	test("a throwing subscriber does not block the rest", async () => {
		const t = makeRenderTicker();
		let healthy = 0;
		t.subscribe({ tick: () => { throw new Error("boom"); }, label: "thrower" });
		t.subscribe({ tick: () => { healthy++; }, label: "healthy" });
		await awaitTicks(3);
		expect(healthy).toBeGreaterThanOrEqual(2);
		t.dispose();
	});
});

describe("RenderTicker — auto-eviction after 3 throws (Codex v3 #2)", () => {
	test("subscriber that throws 3x in a row is unsubscribed and dispose() runs", async () => {
		const t = makeRenderTicker();
		let disposeCalled = 0;
		const sub: TickerSubscriber = {
			tick: () => { throw new Error("perma-broken"); },
			dispose: () => { disposeCalled++; },
			label: "broken-widget",
		};
		t.subscribe(sub);
		await awaitTicks(5); // 5 ticks > 3-throw threshold
		expect(disposeCalled).toBe(1); // exactly once
		expect(t.refCount()).toBe(0);
		t.dispose();
	});
	test("auto-eviction dispose() that throws does NOT crash the ticker (Gemini v3 #2)", async () => {
		const t = makeRenderTicker();
		let healthyTicks = 0;
		const broken: TickerSubscriber = {
			tick: () => { throw new Error("tick-throws"); },
			dispose: () => { throw new Error("dispose-also-throws"); },
			label: "broken-cascade",
		};
		t.subscribe(broken);
		t.subscribe({ tick: () => { healthyTicks++; }, label: "healthy" });
		await awaitTicks(6);
		// Healthy widget should keep ticking even though broken's
		// dispose() threw during eviction.
		expect(healthyTicks).toBeGreaterThanOrEqual(3);
		t.dispose();
	});
	test("subscriber without dispose() is still auto-unsubscribed cleanly", async () => {
		const t = makeRenderTicker();
		t.subscribe({ tick: () => { throw new Error("boom"); }, label: "no-dispose" });
		await awaitTicks(5);
		expect(t.refCount()).toBe(0); // evicted even without dispose hook
		t.dispose();
	});
});

describe("RenderTicker — dispose()", () => {
	test("dispose clears all subscribers and is idempotent", () => {
		const t = makeRenderTicker();
		t.subscribe({ tick: () => {} });
		t.subscribe({ tick: () => {} });
		expect(t.refCount()).toBe(2);
		t.dispose();
		expect(t.refCount()).toBe(0);
		expect(() => t.dispose()).not.toThrow();
	});
	test("subscribing after dispose returns a no-op unsub", () => {
		const t = makeRenderTicker();
		t.dispose();
		const unsub = t.subscribe({ tick: () => {} });
		expect(t.refCount()).toBe(0);
		expect(() => unsub()).not.toThrow();
	});
});

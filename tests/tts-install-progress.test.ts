import { describe, expect, test } from "bun:test";
import { TtsInstallProgressWidget, renderInstallLine, type InstallWidgetUI } from "../extensions/voice/tts-install-progress";
import { makeWidgetRegistry, installWidgetKey } from "../extensions/voice/ui-widget-base";
import { makeRenderTicker } from "../extensions/voice/ui-render-ticker";

function makeStubUI(): InstallWidgetUI & { lastKey: string | null; lastFactory: any; clearedKeys: string[] } {
	const stub = {
		lastKey: null as string | null,
		lastFactory: null as any,
		clearedKeys: [] as string[],
		setWidget(key: string, content: any, _options?: any): void {
			if (content === undefined) {
				stub.clearedKeys.push(key);
				stub.lastFactory = null;
			} else {
				stub.lastKey = key;
				stub.lastFactory = content;
			}
		},
	};
	return stub;
}

describe("TtsInstallProgressWidget — registration + initial frame", () => {
	test("registers with installWidgetKey(modelId) and renders an initial frame", () => {
		const ui = makeStubUI();
		const reg = makeWidgetRegistry();
		const ticker = makeRenderTicker();
		const ctrl = new AbortController();
		const w = new TtsInstallProgressWidget({
			ui, modelId: "kitten-x", modelName: "Kitten Nano", totalBytesEstimate: 1_000_000,
			registry: reg, ticker, controller: ctrl,
		});
		expect(w.key).toBe(installWidgetKey("kitten-x"));
		expect(reg.size()).toBe(1);
		expect(ticker.refCount()).toBe(1);
		expect(ui.lastKey).toBe(w.key);
		expect(typeof ui.lastFactory).toBe("function");
		w.dispose();
		ticker.dispose();
	});
});

describe("TtsInstallProgressWidget — onProgress + dispose lifecycle", () => {
	test("phase=done triggers self-dispose (slot cleared, registry evicted, ticker drops)", () => {
		const ui = makeStubUI();
		const reg = makeWidgetRegistry();
		const ticker = makeRenderTicker();
		const ctrl = new AbortController();
		const w = new TtsInstallProgressWidget({
			ui, modelId: "k", modelName: "K", totalBytesEstimate: 100,
			registry: reg, ticker, controller: ctrl,
		});
		w.onProgress({ phase: "download", bytes: 50, totalBytes: 100 });
		w.onProgress({ phase: "done" });
		expect(reg.size()).toBe(0);
		expect(ticker.refCount()).toBe(0);
		expect(ui.clearedKeys).toContain(w.key);
		ticker.dispose();
	});
	test("late onProgress after dispose is a no-op", () => {
		const ui = makeStubUI();
		const reg = makeWidgetRegistry();
		const ticker = makeRenderTicker();
		const w = new TtsInstallProgressWidget({
			ui, modelId: "k", modelName: "K", totalBytesEstimate: 100,
			registry: reg, ticker, controller: new AbortController(),
		});
		w.dispose();
		const before = ui.clearedKeys.length;
		expect(() => w.onProgress({ phase: "download", bytes: 90, totalBytes: 100 })).not.toThrow();
		// no additional clears, no setWidget calls
		expect(ui.clearedKeys.length).toBe(before);
		ticker.dispose();
	});
	test("dispose() is idempotent", () => {
		const ui = makeStubUI();
		const reg = makeWidgetRegistry();
		const ticker = makeRenderTicker();
		const w = new TtsInstallProgressWidget({
			ui, modelId: "k", modelName: "K", totalBytesEstimate: 100,
			registry: reg, ticker, controller: new AbortController(),
		});
		w.dispose();
		w.dispose();
		expect(reg.size()).toBe(0);
		expect(ticker.refCount()).toBe(0);
		ticker.dispose();
	});
});

describe("TtsInstallProgressWidget — cancel()", () => {
	test("cancel() aborts the controller and disposes", () => {
		const ui = makeStubUI();
		const reg = makeWidgetRegistry();
		const ticker = makeRenderTicker();
		const ctrl = new AbortController();
		const w = new TtsInstallProgressWidget({
			ui, modelId: "k", modelName: "K", totalBytesEstimate: 100,
			registry: reg, ticker, controller: ctrl,
		});
		w.cancel();
		expect(ctrl.signal.aborted).toBe(true);
		expect(reg.size()).toBe(0);
		expect(ticker.refCount()).toBe(0);
		ticker.dispose();
	});
	test("cancel() after dispose is a no-op (does not re-abort)", () => {
		const ui = makeStubUI();
		const reg = makeWidgetRegistry();
		const ticker = makeRenderTicker();
		const ctrl = new AbortController();
		const w = new TtsInstallProgressWidget({
			ui, modelId: "k", modelName: "K", totalBytesEstimate: 100,
			registry: reg, ticker, controller: ctrl,
		});
		w.dispose();
		w.cancel();
		expect(ctrl.signal.aborted).toBe(false); // never aborted because already disposed
		ticker.dispose();
	});
});

describe("renderInstallLine — pure render", () => {
	const fmtBytes = (n: number) => (n < 1024 * 1024 ? `${(n / 1024).toFixed(1)} KB` : `${(n / 1024 / 1024).toFixed(1)} MB`);
	const fmtEta = (s: number) => (s < 60 ? `${s}s` : `${Math.floor(s / 60)}m${String(s % 60).padStart(2, "0")}s`);
	const noTheme = { fg: (_role: string, s: string) => s };

	test("download phase renders bar + percent + size", () => {
		const lines = renderInstallLine({
			theme: noTheme, width: 80, phase: "download", bytes: 500_000, totalBytes: 1_000_000,
			spinner: "◐", speed: 200_000, eta: 3, modelName: "Kitten Nano",
			formatBytes: fmtBytes, formatEta: fmtEta,
		});
		expect(lines).toHaveLength(1);
		expect(lines[0]).toContain("Kitten Nano");
		expect(lines[0]).toContain("50%");
		expect(lines[0]).toContain("488.3 KB / 976.6 KB");
		expect(lines[0]).toContain("█");
		expect(lines[0]).toContain("░");
	});
	test("extract phase shows spinner + status word, no bar", () => {
		const lines = renderInstallLine({
			theme: noTheme, width: 80, phase: "extract", bytes: 0, totalBytes: 0,
			spinner: "◓", speed: null, eta: null, modelName: "Kokoro",
			formatBytes: fmtBytes, formatEta: fmtEta,
		});
		expect(lines).toHaveLength(1);
		expect(lines[0]).toContain("◓");
		expect(lines[0]).toContain("Kokoro");
		expect(lines[0]).toContain("Extracting");
		expect(lines[0]).not.toContain("█");
	});
	test("narrow widths drop ETA / speed gracefully", () => {
		const lines = renderInstallLine({
			theme: noTheme, width: 60, phase: "download", bytes: 100, totalBytes: 1000,
			spinner: "◐", speed: 50, eta: 18, modelName: "M",
			formatBytes: fmtBytes, formatEta: fmtEta,
		});
		expect(lines[0]).not.toContain("ETA");
	});
});

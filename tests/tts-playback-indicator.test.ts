import { describe, expect, test } from "bun:test";
import { TtsPlaybackIndicator, renderPlaybackLine } from "../extensions/voice/tts-playback-indicator";
import { makeWidgetRegistry, WIDGET_KEY } from "../extensions/voice/ui-widget-base";
import { makeRenderTicker } from "../extensions/voice/ui-render-ticker";
import type { InstallWidgetUI } from "../extensions/voice/tts-install-progress";

function stubUI(): InstallWidgetUI & { lastFactory: any; clearedKeys: string[] } {
	return {
		lastFactory: null,
		clearedKeys: [] as string[],
		setWidget(key: string, content: any) {
			if (content === undefined) this.clearedKeys.push(key);
			else this.lastFactory = content;
		},
	} as any;
}

describe("TtsPlaybackIndicator — state machine", () => {
	test("registers under WIDGET_KEY.ttsPlayback and starts on synth", () => {
		const ui = stubUI();
		const reg = makeWidgetRegistry();
		const ticker = makeRenderTicker();
		const w = new TtsPlaybackIndicator({ ui, registry: reg, ticker });
		expect(w.key).toBe(WIDGET_KEY.ttsPlayback);
		expect(reg.size()).toBe(1);
		w.setState("synthesizing");
		expect(typeof ui.lastFactory).toBe("function");
		w.dispose();
		ticker.dispose();
	});
	test("transition to idle disposes the widget", () => {
		const ui = stubUI();
		const reg = makeWidgetRegistry();
		const ticker = makeRenderTicker();
		const w = new TtsPlaybackIndicator({ ui, registry: reg, ticker });
		w.setState("synthesizing");
		w.setState("playing");
		w.setState("idle");
		expect(reg.size()).toBe(0);
		expect(ticker.refCount()).toBe(0);
		expect(ui.clearedKeys).toContain(WIDGET_KEY.ttsPlayback);
		ticker.dispose();
	});
	test("stop() calls onStop and disposes", () => {
		const ui = stubUI();
		const reg = makeWidgetRegistry();
		const ticker = makeRenderTicker();
		let stopped = 0;
		const w = new TtsPlaybackIndicator({ ui, registry: reg, ticker, onStop: () => { stopped++; } });
		w.setState("playing");
		w.stop();
		expect(stopped).toBe(1);
		expect(reg.size()).toBe(0);
		ticker.dispose();
	});
	test("setState after dispose is a no-op", () => {
		const ui = stubUI();
		const reg = makeWidgetRegistry();
		const ticker = makeRenderTicker();
		const w = new TtsPlaybackIndicator({ ui, registry: reg, ticker });
		w.dispose();
		expect(() => w.setState("playing")).not.toThrow();
		expect(reg.size()).toBe(0);
		ticker.dispose();
	});
});

describe("renderPlaybackLine — pure render", () => {
	const noTheme = { fg: (_r: string, s: string) => s };
	test("synthesizing renders spinner + 'Synthesizing'", () => {
		const lines = renderPlaybackLine({ theme: noTheme, width: 80, state: "synthesizing", startedAt: Date.now(), tick: 0 });
		expect(lines).toHaveLength(1);
		// v7.2: spinner is now the 10-frame braille rotation (Charm
		// convention). Frame 0 is ⠋. Test asserts it's one of the
		// braille frames rather than the legacy ◐ quarter-circle.
		expect(lines[0]).toMatch(/[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏]/);
		expect(lines[0]).toContain("Synthesizing");
	});
	test("playing renders 'Playing' + esc hint at wide widths", () => {
		const lines = renderPlaybackLine({ theme: noTheme, width: 80, state: "playing", startedAt: Date.now(), tick: 1 });
		expect(lines[0]).toContain("Playing");
		expect(lines[0]).toContain("[esc] stop");
	});
	test("narrow widths drop the esc hint", () => {
		const lines = renderPlaybackLine({ theme: noTheme, width: 50, state: "playing", startedAt: Date.now(), tick: 1 });
		expect(lines[0]).toContain("Playing");
		expect(lines[0]).not.toContain("[esc] stop");
	});
	test("idle renders empty line", () => {
		const lines = renderPlaybackLine({ theme: noTheme, width: 80, state: "idle", startedAt: Date.now(), tick: 0 });
		expect(lines).toEqual([""]);
	});
	test("no fake amplitude — no bar characters in rendered output", () => {
		const lines = renderPlaybackLine({ theme: noTheme, width: 80, state: "playing", startedAt: Date.now(), tick: 0 });
		expect(lines[0]).not.toContain("█");
		expect(lines[0]).not.toContain("░");
	});
});

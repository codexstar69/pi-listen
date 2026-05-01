import { describe, expect, test } from "bun:test";
import { TtsOnboardingOverlay } from "../extensions/voice/tts-onboarding-overlay";
import { HelpOverlay } from "../extensions/voice/ui-help-overlay";

const noTheme = undefined; // overlays gracefully fall back to plain text without theme

describe("TtsOnboardingOverlay — render", () => {
	test("wide width renders chrome + recommendation + hint row", () => {
		let resolved: any = null;
		const o = new TtsOnboardingOverlay({ systemLocale: "en", theme: noTheme }, r => { resolved = r; });
		const lines = o.render(80);
		expect(lines.length).toBeGreaterThan(5);
		// Should mention the recommended model and the action keys
		const blob = lines.join("\n");
		expect(blob).toContain("pi-listen TTS");
		expect(blob).toContain("Recommended");
		expect(blob).toContain("[↵]");
		expect(blob).toContain("[m]");
		expect(blob).toContain("[esc]");
		expect(resolved).toBeNull();
	});
	test("narrow width renders the compact 3-line fallback", () => {
		let resolved: any = null;
		const o = new TtsOnboardingOverlay({ systemLocale: "en", theme: noTheme }, r => { resolved = r; });
		const lines = o.render(50);
		expect(lines.length).toBe(3);
		expect(lines.join("\n")).toContain("pi-listen TTS");
		expect(resolved).toBeNull();
	});
});

describe("TtsOnboardingOverlay — handleInput resolves once", () => {
	test("[enter] resolves with kind 'test'", () => {
		let resolved: any = null;
		const o = new TtsOnboardingOverlay({ systemLocale: "en" }, r => { resolved = r; });
		o.handleInput("\r");
		expect(resolved).toEqual({ kind: "test" });
	});
	test("[esc] resolves with kind 'skip'", () => {
		let resolved: any = null;
		const o = new TtsOnboardingOverlay({ systemLocale: "en" }, r => { resolved = r; });
		o.handleInput("\x1b");
		expect(resolved).toEqual({ kind: "skip" });
	});
	test("'m' resolves with kind 'pickModel'", () => {
		let resolved: any = null;
		const o = new TtsOnboardingOverlay({ systemLocale: "en" }, r => { resolved = r; });
		o.handleInput("m");
		expect(resolved).toEqual({ kind: "pickModel" });
	});
	test("subsequent input is ignored after resolve", () => {
		let count = 0;
		const o = new TtsOnboardingOverlay({ systemLocale: "en" }, _ => { count++; });
		o.handleInput("\r");
		o.handleInput("m");
		o.handleInput("\x1b");
		expect(count).toBe(1);
	});
});

describe("HelpOverlay — render", () => {
	test("wide width renders all sections", () => {
		let closed = false;
		const h = new HelpOverlay({ theme: noTheme }, () => { closed = true; });
		const lines = h.render(80);
		const blob = lines.join("\n");
		expect(blob).toContain("Settings panel");
		expect(blob).toContain("Voice (TTS)");
		expect(blob).toContain("Voice (STT)");
		expect(blob).toContain("Help");
		expect(blob).toContain("F1");
		expect(blob).toContain("/voice-speak");
		expect(closed).toBe(false);
	});
	test("narrow width renders flat list (no chrome)", () => {
		const h = new HelpOverlay({ theme: noTheme }, () => {});
		const lines = h.render(50);
		const blob = lines.join("\n");
		expect(blob).toContain("pi-listen Help");
		expect(blob).not.toContain("─"); // no horizontal rule chrome
	});
});

describe("HelpOverlay — handleInput", () => {
	test("[esc] resolves the closure", () => {
		let closed = false;
		const h = new HelpOverlay({ theme: noTheme }, () => { closed = true; });
		h.handleInput("\x1b");
		expect(closed).toBe(true);
	});
	test("[enter] also closes (treats as ack)", () => {
		let closed = false;
		const h = new HelpOverlay({ theme: noTheme }, () => { closed = true; });
		h.handleInput("\r");
		expect(closed).toBe(true);
	});
	test("non-close keys are ignored", () => {
		let closed = false;
		const h = new HelpOverlay({ theme: noTheme }, () => { closed = true; });
		h.handleInput("a");
		h.handleInput("?");
		expect(closed).toBe(false);
	});
});

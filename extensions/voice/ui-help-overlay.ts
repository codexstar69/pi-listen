/**
 * Keyboard / command reference overlay — §11 of the v7.1 plan.
 *
 * Stacks ABOVE any picker/panel via `ctx.ui.custom()`. Resolves with
 * no result on `[esc]` so the caller can simply discard. Width-tier
 * aware: at <60 cols it falls back to a flat list with no chrome.
 *
 * Hotkey routing (§11):
 *   - `?`  opens help ONLY when no picker is open / no overlay in
 *          front. Inside a picker it's a literal search character.
 *          Caller is responsible for context-checked routing.
 *   - `F1` always opens help.
 *   - `h`  is intentionally NOT bound (vim users would trigger it).
 */

import { matchesKey, Key } from "@earendil-works/pi-tui";
import type { Theme } from "@earendil-works/pi-coding-agent";
import { ICON } from "./ui-icons";
import { isPanelTooNarrow } from "./ui-width";

export interface HelpOverlayDeps {
	readonly theme?: Theme;
}

interface HelpEntry { readonly key: string; readonly desc: string }
interface HelpSection { readonly heading: string; readonly entries: ReadonlyArray<HelpEntry> }

const HELP_SECTIONS: ReadonlyArray<HelpSection> = [
	{
		heading: "Settings panel",
		entries: [
			{ key: "← →", desc: "switch tab" },
			{ key: "↑ ↓", desc: "navigate row (skips group headings)" },
			{ key: "↵", desc: "select / activate" },
			{ key: "esc", desc: "back to main / close panel" },
			{ key: "type", desc: "filter (search)" },
			{ key: "bksp", desc: "clear last search char" },
		],
	},
	{
		heading: "Voice (TTS)",
		entries: [
			{ key: "/voice-speak <text>", desc: "speak text out loud" },
			{ key: "/voice-speak-test", desc: "speak a sample sentence" },
			{ key: "/voice-speak-toggle", desc: "enable / disable TTS" },
			{ key: "/voice-speak-models", desc: "open model picker" },
			{ key: "/voice-speak-info", desc: "diagnose TTS state" },
		],
	},
	{
		heading: "Voice (STT)",
		entries: [
			{ key: "hold space", desc: "push-to-talk recording" },
			{ key: "/voice-toggle", desc: "enable / disable STT" },
			{ key: "/voice-settings", desc: "open settings panel" },
			{ key: "/voice-models", desc: "browse / install local STT models" },
		],
	},
	{
		heading: "Active widget controls",
		entries: [
			{ key: "esc", desc: `cancel active install (when ${ICON.bulletActive} install widget mounted)` },
			{ key: "esc", desc: "stop active playback (when no install in front)" },
		],
	},
	{
		heading: "Help",
		entries: [
			{ key: "F1", desc: "open this help (always)" },
			{ key: "/voice-help", desc: "open this help via slash command" },
			{ key: "esc / ↵", desc: "close help" },
		],
	},
];

/**
 * Pi `Component`-shaped class — pass into `ctx.ui.custom()` directly.
 */
export class HelpOverlay {
	private readonly deps: HelpOverlayDeps;
	private readonly done: () => void;
	private resolved = false;

	constructor(deps: HelpOverlayDeps, done: () => void) {
		this.deps = deps;
		this.done = done;
	}

	render(width: number): string[] {
		if (isPanelTooNarrow(width)) return this.renderNarrow();

		const t = this.deps.theme;
		const dim = (s: string) => (t ? t.fg("dim", s) : s);
		const accent = (s: string) => (t ? t.fg("accent", s) : s);
		const bold = (s: string) => (t ? t.fg("accent", s) : s);

		const w = Math.max(60, Math.min(width - 2, 90));
		const lines: string[] = [];
		lines.push(`  ${bold("pi-listen")} ${dim(ICON.middot)} ${bold("Help")} ${dim(`${ICON.middot} press [esc] to close`)}`);
		lines.push(`  ${dim(ICON.boxH.repeat(Math.min(w, 60)))}`);
		for (const sec of HELP_SECTIONS) {
			lines.push("");
			lines.push(`  ${accent(sec.heading)}`);
			const keyW = Math.max(...sec.entries.map(e => e.key.length));
			for (const e of sec.entries) {
				const k = e.key.padEnd(keyW);
				lines.push(`    ${accent(k)}  ${dim(ICON.middot)}  ${dim(e.desc)}`);
			}
		}
		lines.push("");
		lines.push(`  ${dim("Note: Hindi (Devanagari) and Arabic voices fall back to romanized labels — see /voice-settings → Speak tab for the full voice list.")}`);
		return lines;
	}

	private renderNarrow(): string[] {
		const t = this.deps.theme;
		const dim = (s: string) => (t ? t.fg("dim", s) : s);
		const accent = (s: string) => (t ? t.fg("accent", s) : s);
		const lines: string[] = [];
		lines.push(` ${accent("pi-listen Help")}`);
		for (const sec of HELP_SECTIONS) {
			lines.push("");
			lines.push(` ${accent(sec.heading)}`);
			for (const e of sec.entries) {
				lines.push(`  ${accent(e.key)} ${dim(e.desc)}`);
			}
		}
		lines.push("");
		lines.push(` ${dim("[esc] close")}`);
		return lines;
	}

	handleInput(data: string): void {
		if (this.resolved) return;
		if (matchesKey(data, Key.escape) || matchesKey(data, Key.enter)) {
			this.resolved = true;
			try { this.done(); } catch { /* never fail closure */ }
			return;
		}
		// Any other key (including ? and F1 again) is ignored — overlay
		// is read-only.
	}

	invalidate(): void { /* render is uncached */ }
}

/**
 * "Auroral" visual language — v7.2 world-class polish.
 *
 * Three primitives that elevate pi-listen's visual identity from
 * "minimal CLI" to "premium application":
 *
 *  1. Liquid Braille waveform — sub-cell vertical bars at 4-level
 *     resolution per column, packed two-per-cell via braille glyphs.
 *     Eight effective vertical levels per CELL vs the current ▁▂▃▄▅▆▇█
 *     8-step block scale, and TWO samples per cell width (so a 16-cell
 *     waveform resolves 32 audio samples — 2× the current density).
 *
 *  2. Aurora truecolor gradient — Catppuccin-inspired ramp from
 *     cool lavender at the edges through warm peach at the peak. RGB
 *     interpolated across 64 stops; per-cell color reflects local
 *     amplitude so loud peaks "burn" toward red while soft tails stay
 *     ethereal blue. Pure ANSI 24-bit escape (`\x1b[38;2;R;G;Bm`),
 *     no theme dependency, falls back gracefully on legacy terminals
 *     (ANSI escape just renders as nothing on non-truecolor TTYs).
 *
 *  3. Floating Island chrome — `╭─╮│╰╯` bordered 3-line "card" with
 *     title in bold accent, content row, and dim keybind footer.
 *     Establishes "voice mode" spatial authority on screen rather
 *     than blending in as a single inline status line.
 *
 * References (Gemini world-class design recommendation, derived from):
 *   - Charm Bracelet lipgloss/gum nested borders + truecolor:
 *     https://github.com/charmbracelet/lipgloss
 *   - Atuin command palette layout: https://github.com/atuinsh/atuin
 *   - Lazygit pane focus + dim non-active: https://github.com/jesseduffield/lazygit
 *   - K9s status density: https://github.com/derailed/k9s
 */

import { ICON } from "./ui-icons";
import { visualWidth, padRightVisual } from "./ui-width";

// ─── Liquid Braille ───────────────────────────────────────────────────────────

/**
 * Encode (leftLevel, rightLevel) ∈ [0..4]² as a single braille glyph.
 * Each cell visually holds TWO column-bars (left + right), each
 * up to 4 dots tall — so one row of N cells = 2N audio samples.
 *
 * Braille bit layout (Unicode U+2800 + 8-bit pattern):
 *   1 4
 *   2 5
 *   3 6
 *   7 8     (rows top-down; 7,8 are bottom row)
 *
 * Vertical bars rise from the bottom up. Left column dots in order
 * 7→3→2→1 = 0x40→0x44→0x46→0x47. Right column 8→6→5→4 = 0x80→0xA0
 * →0xB0→0xB8.
 */
const BRAILLE_LEFT_BITS  = [0, 0x40, 0x44, 0x46, 0x47] as const;
const BRAILLE_RIGHT_BITS = [0, 0x80, 0xA0, 0xB0, 0xB8] as const;

export function brailleBar(left: number, right: number): string {
	const l = Math.max(0, Math.min(4, Math.round(left)));
	const r = Math.max(0, Math.min(4, Math.round(right)));
	return String.fromCodePoint(0x2800 + BRAILLE_LEFT_BITS[l]! + BRAILLE_RIGHT_BITS[r]!);
}

/**
 * Render a "Liquid Braille" waveform from a sample array. Each cell
 * encodes the LEFT and RIGHT sample as 0..4 bar levels, so `cells = N/2`.
 *
 * Pass an array of samples ∈ [0..1]. Sample 0 → leftmost cell's left
 * bar; sample 1 → leftmost cell's right bar; etc.
 *
 * Apply `colorFn(level)` per CELL using the cell's max(left, right)
 * to pick a gradient stop — produces the auroral effect where
 * peaks "burn" warmer.
 */
export function liquidBraille(samples: number[], colorFn?: (level: number) => string): string {
	const cells = Math.ceil(samples.length / 2);
	let out = "";
	for (let i = 0; i < cells; i++) {
		const leftSample = samples[i * 2] ?? 0;
		const rightSample = samples[i * 2 + 1] ?? 0;
		const left = leftSample * 4;   // 0..1 → 0..4
		const right = rightSample * 4;
		const glyph = brailleBar(left, right);
		if (colorFn) {
			const peak = Math.max(leftSample, rightSample);
			out += colorFn(peak) + glyph;
		} else {
			out += glyph;
		}
	}
	if (colorFn) out += "\x1b[0m"; // reset at end of run
	return out;
}

// ─── Aurora Truecolor Gradient ────────────────────────────────────────────────

/** Catppuccin-Mocha-inspired ramp: cool lavender → warm peach. */
const AURORA_STOPS: ReadonlyArray<readonly [number, number, number]> = [
	[180, 190, 254], // lavender (low)
	[137, 180, 250], // blue
	[203, 166, 247], // mauve
	[245, 194, 231], // pink
	[250, 179, 135], // peach
	[243, 139, 168], // red (peak)
];

/**
 * Return a 24-bit ANSI foreground escape for `level` ∈ [0..1].
 * Linearly interpolates across the AURORA_STOPS palette. The escape
 * is `\x1b[38;2;R;G;Bm` — no reset (caller must emit `\x1b[0m`
 * at end of the colored run).
 */
export function auroraColor(level: number): string {
	const t = Math.max(0, Math.min(1, level));
	const segments = AURORA_STOPS.length - 1;
	const pos = t * segments;
	const idx = Math.min(segments - 1, Math.floor(pos));
	const f = pos - idx;
	const a = AURORA_STOPS[idx]!;
	const b = AURORA_STOPS[idx + 1]!;
	const r = Math.round(a[0] + (b[0] - a[0]) * f);
	const g = Math.round(a[1] + (b[1] - a[1]) * f);
	const blue = Math.round(a[2] + (b[2] - a[2]) * f);
	return `\x1b[38;2;${r};${g};${blue}m`;
}

/**
 * Slow "breathing" color for static titles. Returns a truecolor ANSI
 * escape that smoothly cycles through a narrow band of the aurora
 * palette (mauve ↔ pink) on a ~4s cycle. Time `tickMs` should be
 * `Date.now()` from the caller so all widgets breathe in phase.
 *
 * The cycle is intentionally narrow (3 stops, indices 2–4) so the
 * title stays accent-coloured at all times; only the saturation
 * subtly shifts. Combined with bold, reads as a premium "alive but
 * not distracting" indicator.
 */
export function titleBreathe(tickMs: number): string {
	// 4-second cycle: phase ∈ [0, 1).
	const phase = ((tickMs / 4000) % 1 + 1) % 1;
	// sin-shaped modulation: 0 → mauve (idx 2), 1 → pink (idx 3).
	const sinT = (Math.sin(phase * Math.PI * 2) + 1) / 2; // 0..1
	// Map to aurora stops 2..4 (mauve → pink → peach edge).
	const t = 0.4 + sinT * 0.25; // narrow band centered around mauve/pink
	return auroraColor(t);
}

/**
 * Live audio-activity badge — a small chip-style indicator that
 * reflects the current RMS level. Replaces silent timers with a
 * one-glance "is anything happening?" signal.
 *
 *   level < 0.05 : ▁ quiet  (dim)
 *   level < 0.20 : ▃ voice  (cool aurora)
 *   level < 0.50 : ▅ active (mid aurora)
 *   level ≥ 0.50 : ▇ loud   (hot aurora)
 *
 * Returns the chip pre-styled with truecolor ANSI escapes; no
 * theme dependency. Caller wraps in any spacing they like.
 */
export function activityTag(level: number, dim: (s: string) => string): string {
	if (level < 0.05) return dim("▁ quiet");
	if (level < 0.20) return auroraColor(0.10) + "▃ voice"  + "\x1b[0m";
	if (level < 0.50) return auroraColor(0.45) + "▅ active" + "\x1b[0m";
	return                  auroraColor(0.85) + "▇ loud"   + "\x1b[0m";
}

/** Hex-color string version (for callers that want to mix into rgb fonts). */
export function auroraHex(level: number): string {
	const t = Math.max(0, Math.min(1, level));
	const segments = AURORA_STOPS.length - 1;
	const pos = t * segments;
	const idx = Math.min(segments - 1, Math.floor(pos));
	const f = pos - idx;
	const a = AURORA_STOPS[idx]!;
	const b = AURORA_STOPS[idx + 1]!;
	const r = Math.round(a[0] + (b[0] - a[0]) * f);
	const g = Math.round(a[1] + (b[1] - a[1]) * f);
	const blue = Math.round(a[2] + (b[2] - a[2]) * f);
	const hex = (n: number) => n.toString(16).padStart(2, "0");
	return `#${hex(r)}${hex(g)}${hex(blue)}`;
}

// ─── Floating Island chrome ────────────────────────────────────────────────────

/** ANSI escape stripper — required before any visual-width math on
 * styled strings, since `visualWidth` counts each ESC byte as 1 col. */
const ANSI_RE = /\x1b\[[\d;]*[A-Za-z]/g;
const stripAnsi = (s: string): string => s.replace(ANSI_RE, "");
/** Visual width of a pre-styled string (ANSI escapes excluded). */
function trueWidth(s: string): number {
	return visualWidth(stripAnsi(s));
}

/**
 * Render a 3-line "floating island" with rounded borders.
 *
 *   ╭─ {title} ─────────────────────╮
 *   │  {content}                    │
 *   ╰─ {footer} ────────────────────╯
 *
 * The title sits inline on the top border, footer inline on the
 * bottom border — the box reads as a "card" rather than a block.
 * All three lines use rounded corners (Material 3 sheet aesthetic).
 *
 * Pass pre-styled (already ANSI-coloured) strings; this helper
 * doesn't apply theming itself, just chrome.
 *
 * `width` is the OUTER width including the two `│` borders. All
 * width math is ANSI-stripped — the right edge always lands on the
 * `╮` / `╯` even when content is full of color escapes.
 */
export function island(opts: {
	width: number;
	title: string;       // pre-styled or plain
	content: string;     // pre-styled or plain
	footer?: string;     // pre-styled or plain (optional 3rd line)
	dim: (s: string) => string;   // theme dim wrapper
	bold?: (s: string) => string; // unused (retained for API stability)
}): string[] {
	const { width, title, content, footer, dim } = opts;
	const innerW = Math.max(4, width - 2);

	// Top border with title inlaid: `╭─ TITLE ────╮`
	// Layout: corner + boxH + space + title + space + filler + corner.
	// All widths stripped of ANSI before counting.
	const titleW = trueWidth(title);
	const topFillerCount = Math.max(0, innerW - 1 - 1 - titleW - 1); // -1 leading boxH, -1+1 spaces, -1 trailing boxH... fixed:
	// Layout cells:
	//   [boxH][ ][title][ ][boxH × topFillerCount]   = innerW cells total
	//   1     1  titleW 1   topFillerCount = innerW
	// → topFillerCount = innerW - titleW - 3
	const topFill = Math.max(0, innerW - titleW - 3);
	const top = dim(ICON.boxRoundedTL)
		+ dim(ICON.boxH)
		+ " " + title + " "
		+ dim(ICON.boxH.repeat(topFill))
		+ dim(ICON.boxRoundedTR);

	// Content row — pad to fill innerW visually (with ANSI-stripped width)
	const contentW = trueWidth(content);
	const contentPadded = contentW < innerW
		? content + " ".repeat(Math.max(0, innerW - contentW))
		: content;
	const middle = dim(ICON.boxV) + contentPadded + dim(ICON.boxV);

	// Bottom border with optional footer inlaid (same math as top).
	let bottom: string;
	if (footer) {
		const fW = trueWidth(footer);
		const botFill = Math.max(0, innerW - fW - 3);
		bottom = dim(ICON.boxRoundedBL)
			+ dim(ICON.boxH)
			+ " " + footer + " "
			+ dim(ICON.boxH.repeat(botFill))
			+ dim(ICON.boxRoundedBR);
	} else {
		bottom = dim(ICON.boxRoundedBL) + dim(ICON.boxH.repeat(innerW)) + dim(ICON.boxRoundedBR);
	}

	return [top, middle, bottom];
}

/** Pad-right helper that respects pre-styled (ANSI-escape-laden) strings. */
export function padToWidth(s: string, width: number): string {
	const w = trueWidth(s);
	if (w >= width) return s;
	return s + " ".repeat(width - w);
}

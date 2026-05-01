/**
 * Visual-width and width-tier helpers for the v7.1 Settings UI.
 *
 * `visualWidth(s)` returns terminal column count for a string, treating
 * East-Asian Wide / Fullwidth codepoints as 2 columns and everything else
 * as 1. Iteration is done in UTF-32 code points (via `for…of`) so surrogate
 * pairs (`String.length 2`, single code point) count as one wide unit, not
 * two halves of nothing.
 *
 * Hand-curated EAW Wide/Fullwidth ranges from Unicode 15.1 EastAsianWidth.txt
 * — only the blocks pi-listen actually ships labels for (CJK, Hangul,
 * Hiragana/Katakana, fullwidth ASCII). Hindi/Devanagari and Arabic
 * intentionally NOT covered: per the v7.1 plan their voices are rendered
 * with romanized labels, so a precise width here is unnecessary and a
 * partial width table would silently mis-align their rows. Keeping the
 * table small also keeps the zero-dependency promise.
 *
 * Width tiers (§10): three buckets — "wide" (≥80), "mid" (60..79), "narrow"
 * (<60 — hard block at the panel level).
 */

// Hand-curated EAW Wide / Fullwidth ranges, sorted ascending by `lo` so the
// scanner can early-exit on `cp < lo`. Comments label the Unicode block.
const EAW_WIDE_RANGES: ReadonlyArray<readonly [number, number]> = [
	[0x1100, 0x11ff],   // Hangul Jamo
	[0x3000, 0x30ff],   // CJK Symbols & Punctuation, Hiragana, Katakana
	[0x3100, 0x312f],   // Bopomofo
	[0x3130, 0x318f],   // Hangul Compatibility Jamo
	[0x31a0, 0x31bf],   // Bopomofo Extended
	[0x31c0, 0x31ef],   // CJK Strokes
	[0x31f0, 0x31ff],   // Katakana Phonetic Extensions
	[0x3400, 0x4dbf],   // CJK Unified Ideographs Extension A
	[0x4e00, 0x9fff],   // CJK Unified Ideographs (main block)
	[0xac00, 0xd7a3],   // Hangul Syllables (Korean)
	[0xf900, 0xfaff],   // CJK Compatibility Ideographs
	[0xff01, 0xff60],   // Fullwidth ASCII
	[0xffe0, 0xffe6],   // Fullwidth signs
	[0x20000, 0x2fffd], // CJK Unified Ideographs Extension B–F
	[0x30000, 0x3fffd], // CJK Unified Ideographs Extension G
];

function isWide(cp: number): boolean {
	for (const [lo, hi] of EAW_WIDE_RANGES) {
		if (cp >= lo && cp <= hi) return true;
		if (cp < lo) return false; // ranges are sorted
	}
	return false;
}

/**
 * Visual width of `s` in terminal columns. Surrogate pairs count as one
 * code point. EAW Wide/Fullwidth code points count as 2; everything else
 * as 1. Combining marks are NOT subtracted (Devanagari/Arabic are out of
 * scope per §8) — pass romanized labels for those scripts.
 */
export function visualWidth(s: string): number {
	let w = 0;
	for (const ch of s) {
		const cp = ch.codePointAt(0) ?? 0;
		w += isWide(cp) ? 2 : 1;
	}
	return w;
}

/**
 * Truncate `s` to fit within `max` columns, appending `…` (1 column) when
 * truncation occurs. Uses code-point iteration so surrogate pairs and
 * wide characters are never sliced mid-glyph.
 */
export function truncateToVisualWidth(s: string, max: number): string {
	if (max <= 0) return "";
	if (visualWidth(s) <= max) return s;
	if (max === 1) return "…";
	let w = 0;
	let out = "";
	for (const ch of s) {
		const cw = isWide(ch.codePointAt(0) ?? 0) ? 2 : 1;
		if (w + cw > max - 1) break; // reserve 1 col for ellipsis
		out += ch;
		w += cw;
	}
	return out + "…";
}

/** Pad `s` on the right (with spaces) to occupy exactly `width` columns. */
export function padRightVisual(s: string, width: number): string {
	const w = visualWidth(s);
	if (w >= width) return s;
	return s + " ".repeat(width - w);
}

/** Pad `s` on the left (with spaces) to occupy exactly `width` columns. */
export function padLeftVisual(s: string, width: number): string {
	const w = visualWidth(s);
	if (w >= width) return s;
	return " ".repeat(width - w) + s;
}

/** Width tier labels — see §10 of the v7.1 plan. */
export type WidthTier = "wide" | "mid" | "narrow";

/** Map a column count to a tier. ≥80 wide, 60..79 mid, <60 narrow (block). */
export function widthTier(cols: number): WidthTier {
	if (cols >= 80) return "wide";
	if (cols >= 60) return "mid";
	return "narrow";
}

/** True when the panel should hard-block rendering (§10). */
export function isPanelTooNarrow(cols: number): boolean {
	return widthTier(cols) === "narrow";
}

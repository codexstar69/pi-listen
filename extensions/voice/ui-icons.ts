/**
 * Shared glyph table — v7.2 visual polish refresh.
 *
 * Design language synthesized from established minimal-TUI references
 * (Charm Bracelet's lipgloss/bubbletea, charm/gum, lazygit, k9s, fzf,
 * and Apple HIG / Material 3 principles translated to terminal):
 *
 *   - Spinner: braille 10-frame rotation (Charm convention) — smoother
 *     and more modern than 4-frame quarter-circle.
 *   - Borders: rounded `╭─╮│╰╯` for modals/overlays (Material modal feel,
 *     softer than sharp), sharp for inline tables.
 *   - Progress: 8-step subpixel fill via `▏▎▍▌▋▊▉█` for smooth gradient
 *     (Charm/lipgloss progress).
 *   - Cursor: thin left bar `│` + accent text on selected row, dim on
 *     non-selected (HIG "deference": chrome stays subtle).
 *   - Status: colored dot + label (e.g. `● ready`, `○ download`).
 *
 * Hard rule unchanged from v7.1: NO emoji. Every glyph is geometric
 * Unicode (U+2500-25FF + braille U+2800-28FF) or one of the small
 * allowlist marks (✓ ✗ ☐ ☑ • · …).
 *
 * Roles are semantic, not visual — call sites use `ICON.activeMarker`
 * and not the literal `"›"` so a future theme swap is one edit.
 */
export const ICON = {
	// State / status
	checkOk: "✓",
	checkFail: "✗",
	bulletActive: "●",
	bulletInactive: "○",
	bulletDim: "·",
	checkboxOff: "☐",
	checkboxOn: "☑",

	// Cursors / selection
	activeMarker: "›",
	chevronRight: "›",
	chevronLeft: "‹",
	cursorBar: "│",         // v7.2: thin left bar for selected picker rows

	// Arrows
	arrowRight: "→",
	arrowLeft: "←",
	arrowUp: "↑",
	arrowDown: "↓",
	doubleArrowRight: "⇒",

	// v7.2 — Spinner frames. Two profiles, both phase-aligned at 10 Hz:
	//   - braille (10 frames): the Charm convention. Smooth, minimal.
	//     Each glyph is 1 cell wide, suitable for inline status lines.
	//   - arc (4 frames): retained as fallback for terminals without
	//     good braille font coverage (e.g. Windows default cmd.exe).
	spinnerFrames: ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"] as const,
	spinnerFramesArc: ["◐", "◓", "◑", "◒"] as const,
	// Pulse for "alive but waiting" states (e.g. install paused).
	pulseFrames: ["·", "∙", "●", "∙"] as const,

	// Progress bar
	barFilled: "█",
	barEmpty: "░",
	barPartial: ["▏", "▎", "▍", "▌", "▋", "▊", "▉"] as const, // 1/8 .. 7/8
	// v7.2: thin-line progress for compact widgets — modern alternative
	// to the chunky █░ block bar. Pair with a leading-edge cap for
	// "moving" feel during indeterminate progress.
	barThinFilled: "━",
	barThinEmpty: "─",
	barThinCap: "╾",        // leading-edge cap for moving progress

	// Box drawing (U+2500-257F) for borders / separators
	boxH: "─",
	boxV: "│",
	boxTL: "┌",
	boxTR: "┐",
	boxBL: "└",
	boxBR: "┘",
	boxTeeL: "├",
	boxTeeR: "┤",
	boxTeeT: "┬",
	boxTeeB: "┴",
	boxCross: "┼",

	// v7.2 — Rounded corners for modal/overlay chrome. Softer "pill"
	// feel that matches Apple HIG sheet/modal aesthetics.
	boxRoundedTL: "╭",
	boxRoundedTR: "╮",
	boxRoundedBL: "╰",
	boxRoundedBR: "╯",

	// Heavy / double for emphasis
	boxHHeavy: "━",
	boxVHeavy: "┃",
	boxHDouble: "═",
	boxVDouble: "║",

	// Section dividers
	bullet: "•",
	middot: "·",
	ellipsis: "…",
} as const;

/** Semantic icon role — pick one in widget code, never hardcode the glyph. */
export type IconRole = keyof typeof ICON;

/**
 * Return spinner frame for tick `t` (any non-negative integer). Rotates
 * through `ICON.spinnerFrames` in order. Used by every animated widget
 * driven by §2's RenderTicker, so all spinners stay in phase per frame.
 *
 * v7.2: defaults to the 10-frame braille rotation (Charm convention).
 * Pass `arc` for the legacy 4-frame quarter-circle if a terminal has
 * limited braille font coverage.
 */
export function spinnerFrame(t: number, profile: "braille" | "arc" = "braille"): string {
	const frames = profile === "arc" ? ICON.spinnerFramesArc : ICON.spinnerFrames;
	return frames[((t % frames.length) + frames.length) % frames.length];
}

/** Pulse frame — for "alive but waiting" states (no rotation, breathing). */
export function pulseFrame(t: number): string {
	const frames = ICON.pulseFrames;
	// Half-speed pulse: each frame holds for 2 ticks.
	const idx = Math.floor(t / 2) % frames.length;
	return frames[idx]!;
}

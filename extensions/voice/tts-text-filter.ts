/**
 * Text preprocessing for TTS — strips formats that read aloud poorly,
 * enforces length limits, and normalizes whitespace.
 *
 * Used by:
 *   - Auto-speak path (`speak.ts` → after_assistant_message): the agent's
 *     full response goes through `prepareForSpeech()` before synthesis.
 *     Critical because raw assistant output contains code fences,
 *     markdown links, ANSI escapes from prior tool output, and other
 *     forms that read as gibberish.
 *   - Manual `/voice-speak <text>` path: light normalization only —
 *     trim + collapse whitespace. Users typing explicit text don't want
 *     us second-guessing their input.
 *
 * Pure functions, no I/O, no global state. Easy to test against the
 * regression cases locked in tests/tts-text-filter.test.ts.
 *
 * Design choices:
 *   - Code blocks are dropped entirely, not paraphrased. "function foo
 *     opens brace const x equals one closes brace" is worse than silence.
 *     Surface "[code block omitted]" once per response so users know
 *     content was skipped.
 *   - Markdown link syntax `[text](url)` collapses to `text` — URLs read
 *     as gibberish ("h-t-t-p-s-colon-slash-slash-...") and the link text
 *     is what the speaker meant.
 *   - ANSI escapes (color codes, cursor moves) are stripped — they leak
 *     in from quoted tool output and synthesize as noise.
 *   - Inline code spans (single backticks) are kept inline — "use the
 *     `useState` hook" reads naturally. Triple-backtick fences are the
 *     hard skip.
 *   - Length cap is enforced AFTER stripping, so a 5000-char response
 *     that's mostly code blocks may pass.
 */

// ─── Public API ───────────────────────────────────────────────────────────────

export interface PrepareForSpeechOpts {
	/**
	 * Maximum characters in the output. If the cleaned text exceeds this,
	 * `prepareForSpeech` returns `{ skipped: true, reason: "too long" }`.
	 * Auto-speak callers default to 2000; manual /voice-speak passes
	 * Infinity.
	 */
	maxChars?: number;
	/**
	 * If true, drop fenced code blocks entirely. If false, keep them but
	 * unwrap the fences (rare — code reads poorly aloud).
	 */
	stripCodeBlocks?: boolean;
	/**
	 * If true, replace markdown link syntax with link text only. If false,
	 * keep the URL appended (only useful for debugging — no real users
	 * want to hear "https colon slash slash ..." aloud).
	 */
	collapseLinks?: boolean;
}

export interface PrepareForSpeechResult {
	/** True if the text was rejected (length cap, empty after stripping, etc). */
	skipped: boolean;
	/** Cleaned text ready for synthesis. Empty string when skipped. */
	text: string;
	/** Human-readable reason when skipped. */
	reason?: string;
	/** Diagnostic counts so callers can show "[N code blocks omitted]" hints. */
	stats: {
		codeBlocksRemoved: number;
		linksCollapsed: number;
		ansiEscapesRemoved: number;
		originalChars: number;
		finalChars: number;
	};
}

const DEFAULT_OPTS: Required<PrepareForSpeechOpts> = {
	maxChars: 2000,
	stripCodeBlocks: true,
	collapseLinks: true,
};

/**
 * Prepare assistant text for TTS synthesis. See module-level doc for the
 * design rationale on each transform.
 */
export function prepareForSpeech(input: string, opts: PrepareForSpeechOpts = {}): PrepareForSpeechResult {
	const config = { ...DEFAULT_OPTS, ...opts };
	const stats = {
		codeBlocksRemoved: 0,
		linksCollapsed: 0,
		ansiEscapesRemoved: 0,
		originalChars: typeof input === "string" ? input.length : 0,
		finalChars: 0,
	};

	if (typeof input !== "string" || !input) {
		return { skipped: true, text: "", reason: "empty input", stats };
	}

	let text = input;

	// 1. Strip ANSI escape sequences. CSI patterns from tool output:
	//    - `\x1b[<digits>;<digits>m` (color/style)
	//    - `\x1b[<digits>;<digits>H` (cursor moves)
	//    - `\x1b]...\x07` (OSC sequences for window titles, hyperlinks)
	const ansiPattern = /\x1b\[[\d;?]*[A-Za-z]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)/g;
	const ansiMatches = text.match(ansiPattern);
	stats.ansiEscapesRemoved = ansiMatches?.length ?? 0;
	text = text.replace(ansiPattern, "");

	// 2. Drop fenced code blocks. Match ``` or ~~~ fences with an optional
	//    language tag. The middle content can include any characters
	//    including newlines and other backticks. Greedy on opening, lazy
	//    on closing.
	if (config.stripCodeBlocks) {
		const codeBlockPattern = /```[\w-]*\r?\n[\s\S]*?\r?\n```|~~~[\w-]*\r?\n[\s\S]*?\r?\n~~~/g;
		const codeMatches = text.match(codeBlockPattern);
		stats.codeBlocksRemoved = codeMatches?.length ?? 0;
		text = text.replace(codeBlockPattern, " [code block omitted] ");
	}

	// 3. Collapse markdown link syntax `[text](url)` → `text`. We DO NOT
	//    resolve image syntax `![alt](url)` to alt text — image alt
	//    contents are usually decorative and rarely meaningful aloud.
	//    Drop image syntax entirely.
	if (config.collapseLinks) {
		// Image alt: drop entire `![alt](url)` form — alt text is usually
		// decorative ("a screenshot showing...") and rarely worth speaking.
		text = text.replace(/!\[[^\]]*\]\([^)]+\)/g, "");
		// Regular links: keep the visible text only.
		text = text.replace(/\[([^\]]+)\]\([^)]+\)/g, (_full, linkText: string) => {
			stats.linksCollapsed++;
			return linkText;
		});
	}

	// 4. Strip HTML tags that occasionally leak in from doc comments.
	//    Defensive — most assistant output is plain markdown.
	text = text.replace(/<\/?[a-zA-Z][^>]*>/g, " ");

	// 5. Strip raw URLs that aren't inside markdown link syntax. These
	//    read as gibberish aloud. Stop at whitespace; closing-paren is
	//    included as a stop char so URLs inside parenthetical asides
	//    like "(see https://x.dev/p)" don't swallow the closing `)`.
	text = text.replace(/https?:\/\/[^\s)]+/g, " [link omitted] ");

	// 6. Normalize markdown emphasis markers. "**bold** text *italic* text"
	//    should read as "bold text italic text" — TTS doesn't emphasize
	//    on punctuation. Preserve the inner text only.
	text = text.replace(/\*\*([^*]+)\*\*/g, "$1");
	text = text.replace(/__([^_]+)__/g, "$1");
	text = text.replace(/\*([^*\n]+)\*/g, "$1");
	text = text.replace(/_([^_\n]+)_/g, "$1");

	// 7. Normalize headings — drop the `#` markers but keep the heading
	//    text as a sentence. "# Hello\n" → "Hello. ".
	text = text.replace(/^#{1,6}\s+(.+?)$/gm, "$1.");

	// 8. Strip blockquote markers ("> quoted text" → "quoted text").
	text = text.replace(/^>\s+/gm, "");

	// 9. Strip horizontal rules.
	text = text.replace(/^[-*_]{3,}$/gm, "");

	// 10. Strip leading bullet markers from list items so "- foo" reads
	//     as "foo". Each item retains its trailing newline so the
	//     sentence segmenter (Intl.Segmenter in speak.ts) treats them
	//     as separate sentences with natural pause boundaries — a more
	//     natural speech cadence than collapsing to a comma list, which
	//     would be one long run-on with no breath points.
	text = text.replace(/^[ \t]*[-*+][ \t]+/gm, "");

	// 11. Inline code spans: keep the inner text but drop backticks.
	//     "use `useState`" → "use useState".
	text = text.replace(/`([^`\n]+)`/g, "$1");

	// 12. Collapse whitespace runs. Keep paragraph breaks (double newline)
	//     because the segmenter uses them; everything else becomes a
	//     single space.
	text = text.replace(/[ \t]+/g, " ");
	text = text.replace(/\n{3,}/g, "\n\n");
	text = text.trim();

	stats.finalChars = text.length;

	if (!text) {
		return { skipped: true, text: "", reason: "empty after stripping", stats };
	}

	if (text.length > config.maxChars) {
		return {
			skipped: true,
			text: "",
			reason: `text length (${text.length}) exceeds maxChars (${config.maxChars})`,
			stats,
		};
	}

	return { skipped: false, text, stats };
}

/**
 * Lightweight version for the manual `/voice-speak <text>` path. Trims
 * and collapses whitespace runs, but leaves code/links/ANSI alone — the
 * user typed exactly what they want spoken.
 */
export function lightNormalize(input: string): string {
	if (typeof input !== "string") return "";
	return input.replace(/[ \t]+/g, " ").trim();
}

// ─── BCP-47 normalization ─────────────────────────────────────────────────────

/**
 * Canonicalize a BCP-47-ish language tag.
 *
 * Subtag handling per RFC 5646 casing convention:
 *   - language (2-3 letters): lowercase           — `en`, `zh`
 *   - script (4 letters):     Title-case          — `Hant`, `Latn`
 *   - region (2 letters or 3 digits): UPPERCASE   — `US`, `BR`, `419`
 *   - variant (5+ letters or starts with digit): kept as-is, lowercase
 *
 * Inputs we accept: `en`, `en-US`, `en_US`, `EN-us`, `pt-br`, `zh_CN`,
 * `zh-Hant-TW`, `sl-rozaj`. Output preserves all subtags in canonical
 * casing; we never drop information.
 *
 * This is the single canonical form used by every TTS code path —
 * comparing two tags after passing both through this function is the
 * only safe way to check equality.
 */
export function normalizeBCP47(tag: string): string {
	if (typeof tag !== "string" || !tag) return "";
	const parts = tag.replace(/_/g, "-").split("-").filter(Boolean);
	if (parts.length === 0) return "";
	const out: string[] = [];
	for (let i = 0; i < parts.length; i++) {
		const sub = parts[i]!;
		if (i === 0) {
			// Primary language: 2-3 letter code, lowercase
			out.push(sub.toLowerCase());
		} else if (sub.length === 4 && /^[A-Za-z]{4}$/.test(sub)) {
			// Script subtag: Title case (e.g. Hant, Latn, Cyrl)
			out.push(sub.charAt(0).toUpperCase() + sub.slice(1).toLowerCase());
		} else if (/^[A-Za-z]{2}$/.test(sub) || /^\d{3}$/.test(sub)) {
			// Region subtag: 2-letter alpha or 3-digit UN M.49 → uppercase
			out.push(sub.toUpperCase());
		} else {
			// Variant / extension subtag: keep lowercase
			out.push(sub.toLowerCase());
		}
	}
	return out.join("-");
}

/** Extract the base language code from a BCP-47 tag. `en-US` → `en`. */
export function baseLanguage(tag: string): string {
	const norm = normalizeBCP47(tag);
	const idx = norm.indexOf("-");
	return idx === -1 ? norm : norm.slice(0, idx);
}

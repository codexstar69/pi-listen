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

	// 11a. v7.1.3 — text normalization (TN). Compact local TTS engines
	//      (Kitten/Piper) read raw "Dr." as letters and bare numbers
	//      digit-by-digit. Expand the highest-value patterns:
	//      - common English abbreviations / titles
	//      - bare cardinal numbers up to a few digits
	//      Locale-aware long-form (Microsoft Recognizers-Text style)
	//      is out of scope; this is the deterministic ~30-pattern pass
	//      that catches 80% of CLI assistant output gripes.
	text = expandAbbreviations(text);
	text = expandSimpleNumbers(text);

	// 12. v7.1.3 — strip emojis and pictographs. The TTS engines either
	//     read them as literal "smiling face with smiling eyes" (espeak
	//     fallback) or skip+space+resume in a way that breaks prosody.
	//     Using Unicode property `Extended_Pictographic` covers the
	//     full emoji set including skin-tone variants. Variation
	//     Selector-16 (U+FE0F) often follows pictographs to force emoji
	//     presentation; remove it too.
	text = text.replace(/[\p{Extended_Pictographic}\u{FE0F}\u{200D}]/gu, "");

	// 13. Strip leftover decorative chars that have no spoken equivalent:
	//     - U+2500-257F box drawing
	//     - U+2580-259F block elements
	//     - U+25A0-25FF geometric shapes
	//     - U+2600-26FF misc symbols (✓✗★, weather, etc.)
	//     - U+2700-27BF dingbats (✂✈✏…)
	//     - U+2190-21FF arrows (→←↑↓⇒)
	//     - U+2300-23FF technical (⌘⌥⏎)
	text = text.replace(/[←-⇿⌀-⏿─-╿▀-▟■-◿☀-⛿✀-➿]/g, "");

	// 14. Collapse runs of repeated punctuation. "!!!" / "..." / "???"
	//     read as comically long pauses. Reduce to a single mark.
	text = text.replace(/([!?.])\1{2,}/g, "$1");
	text = text.replace(/-{3,}/g, " ");

	// 15. Collapse whitespace runs. Keep paragraph breaks (double newline)
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

// ─── Text normalization helpers (v7.1.3) ──────────────────────────────────────

/**
 * Common English abbreviations, in regex+replacement form. Word-boundary
 * matched so "Dr." → "Doctor" but "Dropout" stays unchanged. Order matters:
 * longer patterns must precede prefixes that would partially match them.
 */
const ABBREV_RULES: ReadonlyArray<readonly [RegExp, string]> = [
	[/\bDr\./g, "Doctor"],
	[/\bMr\./g, "Mister"],
	[/\bMrs\./g, "Misses"],
	[/\bMs\./g, "Miss"],
	[/\bSt\./g, "Saint"],
	[/\bProf\./g, "Professor"],
	[/\bSr\./g, "Senior"],
	[/\bJr\./g, "Junior"],
	[/\bvs\./g, "versus"],
	[/\bi\.e\./gi, "that is"],
	[/\be\.g\./gi, "for example"],
	[/\betc\./gi, "et cetera"],
	[/\bapprox\./gi, "approximately"],
	[/\bvol\./gi, "volume"],
	[/\bch\./gi, "chapter"],
	[/\bp\.s\./gi, "P S"],
	[/\bU\.S\./g, "U S"],
	[/\bU\.K\./g, "U K"],
	[/\bE\.U\./g, "E U"],
	// CLI / dev terms commonly read poorly
	[/\bAPI\b/g, "A P I"],
	[/\bCLI\b/g, "C L I"],
	[/\bURL\b/g, "U R L"],
	[/\bHTTP\b/g, "H T T P"],
	[/\bHTTPS\b/g, "H T T P S"],
	[/\bSQL\b/g, "S Q L"],
	[/\bJSON\b/g, "JAY-son"],
	[/\bYAML\b/g, "YAH-mul"],
	[/\bCSS\b/g, "C S S"],
	[/\bHTML\b/g, "H T M L"],
];

export function expandAbbreviations(text: string): string {
	let out = text;
	for (const [re, rep] of ABBREV_RULES) out = out.replace(re, rep);
	return out;
}

const ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"];
const TEENS = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"];
const TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];

function numberToWords(n: number): string {
	if (n < 0 || n > 9_999_999 || !Number.isInteger(n)) return String(n);
	if (n === 0) return "zero";
	const parts: string[] = [];
	if (n >= 1_000_000) { parts.push(numberToWords(Math.floor(n / 1_000_000)), "million"); n %= 1_000_000; }
	if (n >= 1_000) { parts.push(numberToWords(Math.floor(n / 1_000)), "thousand"); n %= 1_000; }
	if (n >= 100) { parts.push(ONES[Math.floor(n / 100)]!, "hundred"); n %= 100; }
	if (n >= 20) { parts.push(TENS[Math.floor(n / 10)]!); n %= 10; if (n > 0) parts[parts.length - 1] += "-" + ONES[n]!; n = 0; }
	if (n >= 10) parts.push(TEENS[n - 10]!);
	else if (n > 0) parts.push(ONES[n]!);
	return parts.join(" ");
}

/**
 * Expand bare cardinal numbers (1..9_999_999) to words. Skips:
 *   - numbers attached to letters (`v2`, `5k`, `100ms`) — version/unit
 *     suffixes read fine as-is
 *   - decimals (`3.14`) — engines handle these reasonably
 *   - years between 1900-2099 (left as digits — engines say "twenty
 *     twenty-six" naturally for `2026`)
 *   - hex-like tokens (`0xFF`)
 */
export function expandSimpleNumbers(text: string): string {
	// Negative lookbehind: reject digits that are part of a version/decimal
	// (`v1.2`, `3.14`) or a word (`v2`, `5k`).
	// Negative lookahead: reject digits followed by a word char (`100ms`)
	// or by `.<digit>` (decimal/version) but ALLOW sentence-ending `.`.
	return text.replace(/(?<![\w.])(\d{1,7})(?![\w]|\.\d)/g, (match, digits: string) => {
		const n = parseInt(digits, 10);
		// Years pass through unchanged — TTS engines handle them well.
		if (digits.length === 4 && n >= 1900 && n <= 2099) return match;
		return numberToWords(n);
	});
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

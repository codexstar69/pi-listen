import { describe, expect, test } from "bun:test";
import {
	visualWidth,
	truncateToVisualWidth,
	padRightVisual,
	padLeftVisual,
	widthTier,
	isPanelTooNarrow,
} from "../extensions/voice/ui-width";

describe("visualWidth — ASCII", () => {
	test("empty string is 0", () => {
		expect(visualWidth("")).toBe(0);
	});
	test("plain ASCII is 1 column per char", () => {
		expect(visualWidth("hello")).toBe(5);
		expect(visualWidth("Voice picker")).toBe(12);
	});
});

describe("visualWidth — East-Asian Wide", () => {
	test("CJK ideographs count as 2 columns each", () => {
		expect(visualWidth("中文")).toBe(4);
		expect(visualWidth("日本語")).toBe(6);
		expect(visualWidth("한국어")).toBe(6);
	});
	test("mixed Latin + CJK adds correctly", () => {
		expect(visualWidth("Hello 中文")).toBe(5 + 1 + 4); // "Hello " = 6, "中文" = 4
	});
	test("hiragana + katakana count as 2", () => {
		expect(visualWidth("こんにちは")).toBe(10);
		expect(visualWidth("カタカナ")).toBe(8);
	});
	test("fullwidth ASCII counts as 2", () => {
		expect(visualWidth("Ａ")).toBe(2);
	});
});

describe("visualWidth — surrogate pairs", () => {
	test("astral CJK extension B is one wide code point", () => {
		// U+20000 (𠀀) is in CJK Ext B — encoded as a surrogate pair.
		const ch = String.fromCodePoint(0x20000);
		expect(ch.length).toBe(2); // two UTF-16 code units
		expect(visualWidth(ch)).toBe(2); // one wide column count
	});
});

describe("truncateToVisualWidth", () => {
	test("returns input when short enough", () => {
		expect(truncateToVisualWidth("hi", 10)).toBe("hi");
	});
	test("truncates with ellipsis when too long", () => {
		expect(truncateToVisualWidth("hello world", 8)).toBe("hello w…");
	});
	test("truncates wide chars without splitting them", () => {
		// "中文 abc" is 7 cols; clip to 5 cols → "中文…" = 4 + 1 = 5 cols
		const out = truncateToVisualWidth("中文 abc", 5);
		expect(visualWidth(out)).toBeLessThanOrEqual(5);
		expect(out).toBe("中文…");
	});
	test("max=0 returns empty", () => {
		expect(truncateToVisualWidth("anything", 0)).toBe("");
	});
	test("max=1 returns lone ellipsis", () => {
		expect(truncateToVisualWidth("anything", 1)).toBe("…");
	});
});

describe("padRightVisual / padLeftVisual", () => {
	test("pads ASCII to exact column count", () => {
		expect(padRightVisual("hi", 6)).toBe("hi    ");
		expect(padLeftVisual("hi", 6)).toBe("    hi");
	});
	test("pads wide-char strings to exact column count", () => {
		// "中文" is 4 cols → pad to 6 needs 2 spaces.
		expect(padRightVisual("中文", 6)).toBe("中文  ");
		expect(visualWidth(padRightVisual("中文", 6))).toBe(6);
	});
	test("does not over-pad strings already wider than target", () => {
		expect(padRightVisual("hello world", 5)).toBe("hello world");
	});
});

describe("widthTier", () => {
	test("≥80 cols → wide", () => {
		expect(widthTier(80)).toBe("wide");
		expect(widthTier(120)).toBe("wide");
	});
	test("60..79 → mid", () => {
		expect(widthTier(60)).toBe("mid");
		expect(widthTier(79)).toBe("mid");
	});
	test("<60 → narrow", () => {
		expect(widthTier(59)).toBe("narrow");
		expect(widthTier(40)).toBe("narrow");
		expect(widthTier(0)).toBe("narrow");
	});
	test("isPanelTooNarrow lines up with narrow tier", () => {
		expect(isPanelTooNarrow(79)).toBe(false);
		expect(isPanelTooNarrow(60)).toBe(false);
		expect(isPanelTooNarrow(59)).toBe(true);
	});
});

import { describe, expect, test } from "bun:test";
import { localeLabel, formatRomanizedLabel } from "../extensions/voice/ui-locale-labels";

describe("localeLabel — covered scripts", () => {
	test("zh returns 中文", () => {
		expect(localeLabel("zh")?.nativeName).toBe("中文");
	});
	test("ja returns 日本語", () => {
		expect(localeLabel("ja")?.nativeName).toBe("日本語");
	});
	test("ko returns 한국어", () => {
		expect(localeLabel("ko")?.nativeName).toBe("한국어");
	});
	test("uppercase tag still resolves", () => {
		expect(localeLabel("EN")?.nativeName).toBe("English");
		expect(localeLabel("FR")?.nativeName).toBe("Français");
	});
	test("supports masc / fem when known", () => {
		const fr = localeLabel("fr");
		expect(fr?.masc).toBe("Masculin");
		expect(fr?.fem).toBe("Féminin");
	});
});

describe("localeLabel — intentional omissions", () => {
	test("Arabic returns null (RTL out of scope)", () => {
		expect(localeLabel("ar")).toBeNull();
	});
	test("Hindi returns null (Devanagari combining marks out of scope)", () => {
		expect(localeLabel("hi")).toBeNull();
	});
	test("Unknown tags return null", () => {
		expect(localeLabel("xyz")).toBeNull();
		expect(localeLabel("")).toBeNull();
	});
});

describe("formatRomanizedLabel", () => {
	test("Hindi falls back to romanized 'Hindi · M' / 'Hindi · F'", () => {
		expect(formatRomanizedLabel("hi", "M")).toBe("Hindi · M");
		expect(formatRomanizedLabel("hi", "F")).toBe("Hindi · F");
	});
	test("Arabic falls back to romanized", () => {
		expect(formatRomanizedLabel("ar", "M")).toBe("Arabic · M");
	});
	test("missing gender omits suffix", () => {
		expect(formatRomanizedLabel("en", undefined)).toBe("English");
	});
	test("unknown tag uses uppercased tag as fallback name", () => {
		expect(formatRomanizedLabel("xyz", "M")).toBe("XYZ · M");
	});
});

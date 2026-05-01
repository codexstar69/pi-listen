/**
 * Native-script + gender labels for the Voice picker (§8 of v7.1 plan).
 *
 * Hand-curated for the languages pi-listen ships voices for, using the
 * BCP-47 base-language tag (not script/region) as the lookup key. Three
 * intentional omissions:
 *
 *   - `ar` (Arabic): RTL embedding inside fixed-width LTR terminal
 *     columns breaks cursor positioning on enough terminals (especially
 *     stripped-down SSH PTYs) to make it a stability hazard. Arabic
 *     voices fall back to the romanized label.
 *   - `hi` (Hindi): Devanagari uses combining marks and zero-width
 *     vowels. Without a real grapheme segmenter (which would violate
 *     the zero-dependency rule), `visualWidth` would overcount their
 *     width and break right-aligned columns. Hindi voices fall back to
 *     romanized labels.
 *   - Anything else not in this table — fall back to English
 *     `(Language · M/F)` via `formatRomanizedLabel()`.
 *
 * Gender-word translations come from the dictionary forms used in
 * mainstream OS locale pickers (macOS Languages & Region, Windows
 * Settings → Language). They're labels, not full phrases — terse on
 * purpose so they fit in a narrow voice-picker column.
 */

export interface LocaleLabel {
	/** Native-script language name (e.g. 中文, 日本語, 한국어). */
	readonly nativeName: string;
	/** Native masculine gender word, when available. */
	readonly masc?: string;
	/** Native feminine gender word, when available. */
	readonly fem?: string;
}

/**
 * Keyed by BCP-47 base language. Look up via `localeLabel(baseLang)`
 * — DO NOT index this map directly; the helper handles missing-key
 * fallback uniformly. Twelve scripts intentional; ar/hi intentionally
 * omitted (see file header).
 */
const LOCALE_LABELS: Record<string, LocaleLabel> = {
	en: { nativeName: "English", masc: "Male", fem: "Female" },
	zh: { nativeName: "中文", masc: "男声", fem: "女声" },
	ja: { nativeName: "日本語", masc: "男性", fem: "女性" },
	ko: { nativeName: "한국어", masc: "남성", fem: "여성" },
	es: { nativeName: "Español", masc: "Masculino", fem: "Femenino" },
	fr: { nativeName: "Français", masc: "Masculin", fem: "Féminin" },
	de: { nativeName: "Deutsch", masc: "Männlich", fem: "Weiblich" },
	it: { nativeName: "Italiano", masc: "Maschile", fem: "Femminile" },
	pt: { nativeName: "Português", masc: "Masculino", fem: "Feminino" },
	ru: { nativeName: "Русский", masc: "Мужской", fem: "Женский" },
	nl: { nativeName: "Nederlands", masc: "Mannelijk", fem: "Vrouwelijk" },
	tr: { nativeName: "Türkçe", masc: "Erkek", fem: "Kadın" },
};

/**
 * Look up native label by BCP-47 base language tag. Pass the base only
 * (e.g. "zh", not "zh-Hant-TW") — variants share native names. Returns
 * `null` for omitted scripts (ar, hi) and unknown tags so the caller can
 * fall back to romanized via `formatRomanizedLabel()`.
 */
export function localeLabel(baseLang: string): LocaleLabel | null {
	const key = baseLang.toLowerCase();
	return LOCALE_LABELS[key] ?? null;
}

/** English-language base-name map for romanized fallback labels. */
const ROMANIZED_NAMES: Record<string, string> = {
	en: "English",
	zh: "Chinese",
	ja: "Japanese",
	ko: "Korean",
	es: "Spanish",
	fr: "French",
	de: "German",
	it: "Italian",
	pt: "Portuguese",
	ru: "Russian",
	nl: "Dutch",
	tr: "Turkish",
	ar: "Arabic",
	hi: "Hindi",
	pl: "Polish",
	sv: "Swedish",
	da: "Danish",
	no: "Norwegian",
	fi: "Finnish",
	cs: "Czech",
	uk: "Ukrainian",
	hu: "Hungarian",
	ro: "Romanian",
	el: "Greek",
	he: "Hebrew",
	vi: "Vietnamese",
	th: "Thai",
	id: "Indonesian",
};

/**
 * Romanized fallback label "<Language> · M" or "<Language> · F" — used
 * for ar, hi, and any unknown language tag. Gender is rendered with the
 * same single-letter abbreviation everywhere, so the label width is
 * predictable across all fallback rows.
 */
export function formatRomanizedLabel(baseLang: string, gender: "M" | "F" | undefined): string {
	const key = baseLang.toLowerCase();
	const name = ROMANIZED_NAMES[key] ?? baseLang.toUpperCase();
	if (!gender) return name;
	return `${name} · ${gender}`;
}

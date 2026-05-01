/**
 * Local TTS model catalog for pi-listen v6.0.0+.
 *
 * Three tiers:
 *   - Tier 0 (default): Kitten Nano v0.2 — 25.4 MB, 8 voices, English. The
 *     first-run download. Apache-2.0, sub-real-time on M-series.
 *   - Tier 1 (per-language Piper): one ~20 MB voice per popular language. The
 *     user installs only what they need.
 *   - Tier 2 (multilingual / HQ Kokoro): 98-126 MB. Opt-in for users who
 *     prefer prosody quality over disk usage.
 *
 * Sherpa-onnx-node OfflineTts supports five model "slots" (verified in
 * node_modules/sherpa-onnx-node/types.js OfflineTtsModelConfig):
 *   vits | matcha | kokoro | kitten | pocket
 * We use kitten (Kitten Nano), vits (every Piper voice), and kokoro (the two
 * Kokoro entries). Pocket is voice cloning — different use case, out of scope.
 *
 * Why .tar.bz2 archives instead of individual .onnx URLs (like the STT path):
 * sherpa-onnx publishes each TTS model as a single archive containing the
 * model files plus an `espeak-ng-data/` directory (Piper, Kokoro, Kitten all
 * need this for grapheme-to-phoneme conversion). We extract once on download
 * and cache the unpacked dir under `~/.pi/models/tts/<modelId>/`.
 *
 * Asset URLs verified via GitHub API
 * (`api.github.com/repos/k2-fsa/sherpa-onnx/releases/tags/tts-models`)
 * on 2026-04-28. If a URL 404s in the future, regenerate this catalog from
 * the latest release tag — the structure is stable, only filenames change.
 */

import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { spawn } from "node:child_process";
import { createHash } from "node:crypto";

const TTS_RELEASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models";

// ─── Types ────────────────────────────────────────────────────────────────────

/**
 * Sherpa-onnx model slot. Maps directly to OfflineTtsModelConfig in
 * sherpa-onnx-node — the engine dispatches on this in tts-engine.ts.
 */
export type TtsSherpaSlot = "kitten" | "vits" | "kokoro";

export interface TtsVoice {
	/** Sherpa speaker id (numeric). Passed as `sid` to OfflineTts.generate(). */
	sid: number;
	/** Display name shown in the voice picker. */
	name: string;
	/** Optional gender hint for grouping in the voice picker. */
	gender?: "male" | "female" | "neutral";
}

export interface TtsLocalModelInfo {
	id: string;
	name: string;
	/** Human-readable archive size, e.g. "~25 MB". */
	size: string;
	/** Archive size in bytes (drives disk-space pre-checks + progress). */
	sizeBytes: number;
	/** Peak runtime RAM in MB (rough — model file × ~3 covers vocoder buffers). */
	runtimeRamMB: number;
	/** One-line description shown in the picker detail row. */
	notes: string;
	/**
	 * Languages supported by this model. BCP-47-ish tags; for filtering and
	 * for picker labels. Single-language models list one entry; Kokoro
	 * multilingual lists all 9.
	 */
	languages: string[];
	/** Device tier — drives the fitness algorithm shared with STT. */
	tier: "edge" | "standard" | "heavy";
	/** Marked as "recommended" in the catalog UI. */
	preferred?: boolean;
	/** Subjective accuracy rating 1-5 (5 = best). */
	accuracy: 1 | 2 | 3 | 4 | 5;
	/** Subjective speed rating 1-5 (5 = fastest). */
	speed: 1 | 2 | 3 | 4 | 5;
	/** License — needs to be commercial-use OK to ship as default. */
	license: string;
	/** Sherpa-onnx model slot for the engine to dispatch on. */
	sherpaSlot: TtsSherpaSlot;
	/** Voices available in this model (always at least one). */
	voices: TtsVoice[];
	/** Default voice (sid) used if the user hasn't picked one. */
	defaultSid: number;
	/**
	 * The single archive URL (sherpa-onnx packs the model + tokens +
	 * espeak-ng-data into one .tar.bz2). The downloader extracts to
	 * `~/.pi/models/tts/<id>/` and returns that directory.
	 */
	archiveUrl: string;
	/**
	 * Optional SHA-256 hex digest of the archive bytes for integrity
	 * verification. When set, ensureTtsModelInstalled() rejects a
	 * download whose computed hash differs. v7.0.0 ships catalog entries
	 * without hashes (we don't ship known-good values for sherpa-onnx
	 * releases yet); the verification pipeline runs anyway and produces
	 * a hash that can be pinned in v7.1+ to lock-in the bytes.
	 */
	archiveSha256?: string;
	/** Sample rate (Hz) of generated audio. Drives WAV header on playback. */
	sampleRate: number;
	/**
	 * v7.1.2 — runtime incompatibility marker. When the model is known
	 * to fail with the currently-installed `sherpa-onnx-node` (e.g.
	 * kokoro multilingual + 1.12.29 returns all-NaN samples for every
	 * sid), set `incompatible` to a one-line user-facing reason. The
	 * picker shows the model with a warning badge; the smart-default
	 * recommender skips it; `synthesize()` refuses to use it.
	 */
	incompatible?: string;
}

// ─── Catalog ──────────────────────────────────────────────────────────────────

/**
 * Default TTS model id — what gets downloaded on first `/voice-speak` if no
 * model is selected. Chosen for the smallest viable English model (25 MB)
 * with a permissive license and 8 voices for variety.
 */
export const DEFAULT_TTS_MODEL = "kitten-nano-en-v0_2";

/**
 * The full catalog. Order is intentional — picker presents it as-is,
 * grouped in the settings panel by sherpa slot.
 */
export const TTS_LOCAL_MODELS: TtsLocalModelInfo[] = [
	// ═══════════════════════════════════════════════════════════════════════
	// TIER 0 — Default first-run (English-only, smallest)
	// ═══════════════════════════════════════════════════════════════════════
	{
		id: "kitten-nano-en-v0_2",
		name: "Kitten Nano v0.2",
		size: "~25 MB",
		sizeBytes: 26_633_011,
		runtimeRamMB: 120,
		notes: "Smallest English TTS — 15M params, 8 voices, 24 kHz, sub-real-time on M-series",
		languages: ["en"],
		tier: "edge",
		preferred: true,
		accuracy: 4,
		speed: 5,
		license: "Apache-2.0",
		sherpaSlot: "kitten",
		// Voice ordering follows the order baked into the `voices.bin` file
		// shipped with the model — sids 0-7 in canonical order.
		voices: [
			{ sid: 0, name: "Expr-Voice-2-M", gender: "male" },
			{ sid: 1, name: "Expr-Voice-2-F", gender: "female" },
			{ sid: 2, name: "Expr-Voice-3-M", gender: "male" },
			{ sid: 3, name: "Expr-Voice-3-F", gender: "female" },
			{ sid: 4, name: "Expr-Voice-4-M", gender: "male" },
			{ sid: 5, name: "Expr-Voice-4-F", gender: "female" },
			{ sid: 6, name: "Expr-Voice-5-M", gender: "male" },
			{ sid: 7, name: "Expr-Voice-5-F", gender: "female" },
		],
		defaultSid: 0,
		archiveUrl: `${TTS_RELEASE}/kitten-nano-en-v0_2-fp16.tar.bz2`,
		sampleRate: 24000,
	},

	// ═══════════════════════════════════════════════════════════════════════
	// TIER 1 — Per-language Piper voices (each ~20 MB)
	// ═══════════════════════════════════════════════════════════════════════
	piper("en_US-lessac-medium-int8", "Piper Lessac (en-US)", 20_971_520, ["en-US"], "Clear American voice — solid technical-prose default", "MIT", true, 22050),
	piper("en_US-amy-medium-int8", "Piper Amy (en-US)", 21_065_728, ["en-US"], "Female American voice", "MIT", false, 22050, "female"),
	piper("en_US-libritts_r-medium-int8", "Piper LibriTTS-R (en-US)", 23_383_244, ["en-US"], "904 voices in one model — pick a sid in the picker", "MIT", false, 22050, "neutral", 904),
	piper("es_ES-davefx-medium-int8", "Piper DaveFX (es-ES)", 21_169_356, ["es-ES"], "European Spanish, male voice", "MIT", true, 22050, "male"),
	piper("fr_FR-siwis-medium-int8", "Piper Siwis (fr-FR)", 20_866_662, ["fr-FR"], "French, female voice", "MIT", true, 22050, "female"),
	piper("de_DE-thorsten-medium-int8", "Piper Thorsten (de-DE)", 20_971_520, ["de-DE"], "German, male voice", "MIT", true, 22050, "male"),
	piper("hi_IN-pratham-medium-int8", "Piper Pratham (hi-IN)", 20_971_520, ["hi-IN"], "Hindi, male voice", "MIT", true, 22050, "male"),
	piper("pt_BR-cadu-medium-int8", "Piper Cadu (pt-BR)", 21_169_356, ["pt-BR"], "Brazilian Portuguese, male voice", "MIT", true, 22050, "male"),
	piper("zh_CN-chaowen-medium-int8", "Piper Chaowen (zh-CN)", 14_050_918, ["zh-CN"], "Mandarin Chinese — smaller archive than other languages", "MIT", true, 22050),
	piper("it_IT-paola-medium-int8", "Piper Paola (it-IT)", 21_169_356, ["it-IT"], "Italian, female voice", "MIT", true, 22050, "female"),
	piper("ru_RU-denis-medium-int8", "Piper Denis (ru-RU)", 21_065_728, ["ru-RU"], "Russian, male voice", "MIT", true, 22050, "male"),
	piper("ar_JO-kareem-medium-int8", "Piper Kareem (ar-JO)", 20_971_520, ["ar-JO"], "Levantine Arabic, male voice", "MIT", false, 22050, "male"),
	piper("tr_TR-fahrettin-medium-int8", "Piper Fahrettin (tr-TR)", 21_065_728, ["tr-TR"], "Turkish, male voice", "MIT", false, 22050, "male"),
	piper("nl_NL-pim-medium-int8", "Piper Pim (nl-NL)", 21_065_728, ["nl-NL"], "Dutch, male voice", "MIT", false, 22050, "male"),

	// ═══════════════════════════════════════════════════════════════════════
	// TIER 2 — Multilingual + English HQ (Kokoro family, opt-in due to size)
	// ═══════════════════════════════════════════════════════════════════════
	{
		id: "kokoro-int8-multi-lang-v1_0",
		name: "Kokoro Multilingual v1.0",
		size: "~126 MB",
		sizeBytes: 131_822_387,
		runtimeRamMB: 870,
		notes: "9 languages in one model — en/zh/ja/ko/es/fr/hi/it/pt — 53 voices, 24 kHz",
		languages: ["en", "zh", "ja", "ko", "es", "fr", "hi", "it", "pt"],
		tier: "standard",
		preferred: true,
		accuracy: 5,
		speed: 4,
		license: "Apache-2.0",
		sherpaSlot: "kokoro",
		// Kokoro v1.0 ships 53 voices labeled by lang prefix (e.g. `af_*` =
		// American female, `am_*` = American male, `bf_*` / `bm_*` = British,
		// `jf_*`/`jm_*` = Japanese, etc.). We surface the most-likely picks
		// per language; users can pick others via numeric sid in the picker.
		voices: [
			{ sid: 0, name: "af_heart (en-US, female)", gender: "female" },
			{ sid: 1, name: "af_alloy (en-US, female)", gender: "female" },
			{ sid: 2, name: "af_aoede (en-US, female)", gender: "female" },
			{ sid: 11, name: "am_adam (en-US, male)", gender: "male" },
			{ sid: 12, name: "am_echo (en-US, male)", gender: "male" },
			{ sid: 20, name: "bf_alice (en-GB, female)", gender: "female" },
			{ sid: 24, name: "bm_daniel (en-GB, male)", gender: "male" },
			{ sid: 28, name: "ef_dora (es, female)", gender: "female" },
			{ sid: 31, name: "em_alex (es, male)", gender: "male" },
			{ sid: 33, name: "ff_siwis (fr, female)", gender: "female" },
			{ sid: 34, name: "hf_alpha (hi, female)", gender: "female" },
			{ sid: 38, name: "if_sara (it, female)", gender: "female" },
			{ sid: 40, name: "jf_alpha (ja, female)", gender: "female" },
			{ sid: 44, name: "kf_yumi (ko, female)", gender: "female" },
			{ sid: 46, name: "pf_dora (pt-BR, female)", gender: "female" },
			{ sid: 48, name: "zf_xiaobei (zh, female)", gender: "female" },
			{ sid: 50, name: "zm_yunjian (zh, male)", gender: "male" },
		],
		defaultSid: 0,
		archiveUrl: `${TTS_RELEASE}/kokoro-int8-multi-lang-v1_0.tar.bz2`,
		sampleRate: 24000,
		// v7.1.2: most voices in this model produce NaN samples on
		// sherpa-onnx-node 1.12.29 + 1.13.0. Root cause is int8
		// quantization of speaker embeddings in voices.bin (per
		// upstream issue #1923 / mlx-audio "guard NaN durations" PR
		// pattern). Failures are non-deterministic per-voice across
		// fresh processes. Marked incompatible so the picker hides
		// it and synthesize() refuses to use it. Re-enable when
		// upstream ships an fp16/fp32 multilingual model OR a
		// re-quantized int8 voices.bin without NaN embeddings.
		incompatible: "kokoro multilingual int8 v1.0 produces NaN samples on most voices. Use kokoro-multi-lang-v1_1 (newer) or kokoro-en-v0_19 instead.",
	},
	// v7.1.3: NEWER kokoro multilingual (v1.1) — sherpa-onnx upstream
	// re-quantized voices.bin. Both fp32 (333+ MB) and int8 (140 MB)
	// variants are now available. v1.1 preferred over v1.0 because
	// upstream specifically rebuilt it to address the NaN issue.
	// Marked `untested: true` until you confirm it synthesizes
	// real audio in your environment — same picker treatment as
	// `incompatible`, but doesn't refuse to run.
	{
		id: "kokoro-int8-multi-lang-v1_1",
		name: "Kokoro Multilingual v1.1 (int8)",
		size: "~140 MB",
		sizeBytes: 147_031_220,
		runtimeRamMB: 900,
		notes: "v1.1 multilingual — upstream re-quantized after the v1.0 NaN issue. 9 langs, ~50 voices, 24 kHz",
		languages: ["en", "zh", "ja", "ko", "es", "fr", "hi", "it", "pt"],
		tier: "standard",
		preferred: false,
		accuracy: 5,
		speed: 4,
		license: "Apache-2.0",
		sherpaSlot: "kokoro",
		voices: [
			{ sid: 0, name: "af_heart (en-US, female)", gender: "female" },
			{ sid: 1, name: "af_alloy (en-US, female)", gender: "female" },
			{ sid: 2, name: "af_aoede (en-US, female)", gender: "female" },
			{ sid: 11, name: "am_adam (en-US, male)", gender: "male" },
			{ sid: 12, name: "am_echo (en-US, male)", gender: "male" },
			{ sid: 20, name: "bf_alice (en-GB, female)", gender: "female" },
			{ sid: 24, name: "bm_daniel (en-GB, male)", gender: "male" },
			{ sid: 33, name: "ff_siwis (fr, female)", gender: "female" },
			{ sid: 40, name: "jf_alpha (ja, female)", gender: "female" },
			{ sid: 44, name: "kf_yumi (ko, female)", gender: "female" },
			{ sid: 48, name: "zf_xiaobei (zh, female)", gender: "female" },
		],
		defaultSid: 0,
		archiveUrl: `${TTS_RELEASE}/kokoro-int8-multi-lang-v1_1.tar.bz2`,
		sampleRate: 24000,
	},
	{
		// Untested-but-non-quantized fp32 multilingual. 333 MB — opt-in
		// for users who want guaranteed-no-NaN multilingual coverage.
		id: "kokoro-multi-lang-v1_0",
		name: "Kokoro Multilingual v1.0 (fp32)",
		size: "~333 MB",
		sizeBytes: 349_418_188,
		runtimeRamMB: 1400,
		notes: "v1.0 multilingual fp32 — no int8 quantization, no NaN risk. 9 langs, 53 voices, 24 kHz. 333 MB.",
		languages: ["en", "zh", "ja", "ko", "es", "fr", "hi", "it", "pt"],
		tier: "heavy",
		preferred: false,
		accuracy: 5,
		speed: 3,
		license: "Apache-2.0",
		sherpaSlot: "kokoro",
		voices: [
			{ sid: 0, name: "af_heart (en-US, female)", gender: "female" },
			{ sid: 11, name: "am_adam (en-US, male)", gender: "male" },
			{ sid: 20, name: "bf_alice (en-GB, female)", gender: "female" },
			{ sid: 33, name: "ff_siwis (fr, female)", gender: "female" },
			{ sid: 40, name: "jf_alpha (ja, female)", gender: "female" },
			{ sid: 44, name: "kf_yumi (ko, female)", gender: "female" },
			{ sid: 48, name: "zf_xiaobei (zh, female)", gender: "female" },
		],
		defaultSid: 0,
		archiveUrl: `${TTS_RELEASE}/kokoro-multi-lang-v1_0.tar.bz2`,
		sampleRate: 24000,
	},
	{
		// fp32 English Kokoro — 304 MB, definitively no NaN since it's not
		// int8 quantized. Highest-quality offline English option we ship.
		id: "kokoro-en-v0_19",
		name: "Kokoro English v0.19 (fp32)",
		size: "~304 MB",
		sizeBytes: 319_625_534,
		runtimeRamMB: 1200,
		notes: "fp32 English HQ — best prosody, no quantization artifacts. 11 voices, 24 kHz.",
		languages: ["en"],
		tier: "heavy",
		preferred: false,
		accuracy: 5,
		speed: 3,
		license: "Apache-2.0",
		sherpaSlot: "kokoro",
		voices: [
			{ sid: 0, name: "af_bella (female)", gender: "female" },
			{ sid: 1, name: "af_nicole (female)", gender: "female" },
			{ sid: 2, name: "af_sarah (female)", gender: "female" },
			{ sid: 3, name: "af_sky (female)", gender: "female" },
			{ sid: 4, name: "am_adam (male)", gender: "male" },
			{ sid: 5, name: "am_michael (male)", gender: "male" },
			{ sid: 6, name: "bf_emma (female, en-GB)", gender: "female" },
			{ sid: 7, name: "bf_isabella (female, en-GB)", gender: "female" },
			{ sid: 8, name: "bm_george (male, en-GB)", gender: "male" },
			{ sid: 9, name: "bm_lewis (male, en-GB)", gender: "male" },
			{ sid: 10, name: "af (default mix)", gender: "neutral" },
		],
		defaultSid: 0,
		archiveUrl: `${TTS_RELEASE}/kokoro-en-v0_19.tar.bz2`,
		sampleRate: 24000,
	},
	{
		id: "kokoro-int8-en-v0_19",
		name: "Kokoro English v0.19",
		size: "~99 MB",
		sizeBytes: 103_284_736,
		runtimeRamMB: 350,
		notes: "English HQ — 11 voices, best prosody, 24 kHz",
		languages: ["en"],
		tier: "standard",
		preferred: false,
		accuracy: 5,
		speed: 4,
		license: "Apache-2.0",
		sherpaSlot: "kokoro",
		voices: [
			{ sid: 0, name: "af_bella (female)", gender: "female" },
			{ sid: 1, name: "af_nicole (female)", gender: "female" },
			{ sid: 2, name: "af_sarah (female)", gender: "female" },
			{ sid: 3, name: "af_sky (female)", gender: "female" },
			{ sid: 4, name: "am_adam (male)", gender: "male" },
			{ sid: 5, name: "am_michael (male)", gender: "male" },
			{ sid: 6, name: "bf_emma (female, en-GB)", gender: "female" },
			{ sid: 7, name: "bf_isabella (female, en-GB)", gender: "female" },
			{ sid: 8, name: "bm_george (male, en-GB)", gender: "male" },
			{ sid: 9, name: "bm_lewis (male, en-GB)", gender: "male" },
			{ sid: 10, name: "af (default mix)", gender: "neutral" },
		],
		defaultSid: 0,
		archiveUrl: `${TTS_RELEASE}/kokoro-int8-en-v0_19.tar.bz2`,
		sampleRate: 24000,
	},
];

// ─── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Build a Piper VITS catalog entry. Piper voices have a uniform shape — one
 * model, one default voice (sid 0), one language. LibriTTS-R is the
 * exception (904 speakers) — pass `voiceCount > 1` to expose them all.
 */
function piper(
	stem: string,
	displayName: string,
	sizeBytes: number,
	languages: string[],
	notes: string,
	license: string,
	preferred: boolean,
	sampleRate: number,
	gender: "male" | "female" | "neutral" = "neutral",
	voiceCount = 1,
): TtsLocalModelInfo {
	const voices: TtsVoice[] = voiceCount === 1
		? [{ sid: 0, name: displayName, gender }]
		: Array.from({ length: voiceCount }, (_, i) => ({ sid: i, name: `Speaker ${i}`, gender }));
	return {
		id: `piper-${stem}`,
		name: displayName,
		// Round to 1 decimal MB for the picker label
		size: `~${(sizeBytes / 1024 / 1024).toFixed(0)} MB`,
		sizeBytes,
		// Piper RAM ≈ 6× model file in worst case (decoder workspace)
		runtimeRamMB: Math.round((sizeBytes / 1024 / 1024) * 6),
		notes,
		languages,
		tier: "edge",
		preferred,
		accuracy: 3,
		speed: 5,
		license,
		sherpaSlot: "vits",
		voices,
		defaultSid: 0,
		archiveUrl: `${TTS_RELEASE}/vits-piper-${stem}.tar.bz2`,
		sampleRate,
	};
}

// ─── Smart default selection ─────────────────────────────────────────────────

/**
 * Recommend an initial TTS model based on the user's system locale.
 *
 * Returns ONE catalog entry id — the recommendation, not an installation
 * decision. The caller (onboarding picker, settings panel) presents this
 * as a pre-highlighted suggestion with disclosure of size and language
 * coverage. The user always confirms before download starts.
 *
 * Mapping rules:
 *   - English locale (en-*)        → kitten-nano-en-v0_2 (smallest, 25 MB)
 *   - Single-language Piper match  → that Piper voice (~20 MB each)
 *   - Multi-language locale that
 *     covers Kokoro                → kokoro-int8-multi-lang-v1_0 (126 MB)
 *   - Locale with no coverage      → kitten-nano-en-v0_2 + warn
 *
 * The single-Piper-match path is preferred over the multilingual Kokoro
 * because Piper is 1/6 the size when only one language is needed. Kokoro
 * is the right pick when the user reads multiple languages OR when no
 * Piper voice exists for their locale.
 */
export interface SmartDefaultRecommendation {
	modelId: string;
	/** Why this model was picked, surfaceable in onboarding UI. */
	reason: string;
	/** True iff no model in the catalog actually covers `locale`. */
	fallback: boolean;
}

/**
 * Per-language single-Piper-voice mapping — only languages where the
 * catalog has exactly one Piper voice for the language. Multi-region
 * languages (en, pt) are intentionally NOT here — those route through
 * either the en→kitten path or kokoro multilingual.
 */
const SINGLE_PIPER_BY_BASE_LANG: Readonly<Record<string, string>> = {
	es: "piper-es_ES-davefx-medium-int8",
	fr: "piper-fr_FR-siwis-medium-int8",
	de: "piper-de_DE-thorsten-medium-int8",
	hi: "piper-hi_IN-pratham-medium-int8",
	zh: "piper-zh_CN-chaowen-medium-int8",
	it: "piper-it_IT-paola-medium-int8",
	ru: "piper-ru_RU-denis-medium-int8",
	ar: "piper-ar_JO-kareem-medium-int8",
	tr: "piper-tr_TR-fahrettin-medium-int8",
	nl: "piper-nl_NL-pim-medium-int8",
};

export function recommendDefaultModel(systemLocale: string): SmartDefaultRecommendation {
	if (!systemLocale || typeof systemLocale !== "string") {
		return {
			modelId: DEFAULT_TTS_MODEL,
			reason: "No system locale detected — defaulting to the smallest English model.",
			fallback: true,
		};
	}

	// Normalize: lowercase first subtag, e.g. "en_US.UTF-8" → "en"
	const base = systemLocale
		.split(/[-_.]/)[0]!
		.toLowerCase();

	// English locales — Kitten Nano is the smallest viable English TTS
	// at 25 MB, and we ship it as the catalog default for first-run
	// experience reasons.
	if (base === "en") {
		return {
			modelId: DEFAULT_TTS_MODEL,
			reason: `English locale detected — recommending ${DEFAULT_TTS_MODEL} (25 MB, 8 voices).`,
			fallback: false,
		};
	}

	// Special-case Portuguese: catalog has Brazilian-only Piper. We pick
	// pt-BR for any pt-* locale; for non-BR regions (pt-PT, pt-AO, etc.)
	// `fallback: true` so the onboarding UI surfaces "this isn't a
	// perfect match" — even though the model technically still produces
	// Portuguese audio, the accent will differ from the user's locale.
	if (base === "pt") {
		const isExactMatch = systemLocale.toLowerCase().includes("br");
		return {
			modelId: "piper-pt_BR-cadu-medium-int8",
			reason: `Portuguese locale detected — recommending Brazilian Portuguese voice (${
				isExactMatch ? "exact match" : "closest available — accent will differ from your locale"
			}, 20 MB).`,
			fallback: !isExactMatch,
		};
	}

	// Single-language Piper match
	const single = SINGLE_PIPER_BY_BASE_LANG[base];
	if (single) {
		return {
			modelId: single,
			reason: `${base.toUpperCase()} locale detected — recommending ${single} (~20 MB).`,
			fallback: false,
		};
	}

	// Languages covered only by Kokoro multilingual (ja, ko).
	// v7.1.2: Kokoro multilingual is currently flagged `incompatible`
	// on sherpa-onnx-node 1.12.29 — fall through to the English-default
	// fallback instead of routing the user to a silent model.
	if (base === "ja" || base === "ko") {
		const kokoro = TTS_LOCAL_MODELS.find(m => m.id === "kokoro-int8-multi-lang-v1_0");
		if (kokoro && !kokoro.incompatible) {
			return {
				modelId: "kokoro-int8-multi-lang-v1_0",
				reason: `${base.toUpperCase()} locale detected — recommending Kokoro multilingual (126 MB, ` +
					`covers en/zh/ja/ko/es/fr/hi/it/pt in one model).`,
				fallback: false,
			};
		}
		// fall through to the English fallback below.
	}

	// No coverage — fall back to English default with a warning the
	// caller can surface verbatim.
	return {
		modelId: DEFAULT_TTS_MODEL,
		reason: `Locale ${systemLocale} has no built-in TTS voice. Falling back to English (${DEFAULT_TTS_MODEL}). ` +
			`Browse /voice-settings → Speak tab → Models for the full catalog.`,
		fallback: true,
	};
}

/** Look up a model by id; throws if unknown so callers fail loudly. */
export function getTtsModel(id: string): TtsLocalModelInfo {
	const m = TTS_LOCAL_MODELS.find(x => x.id === id);
	if (!m) throw new Error(`Unknown TTS model: ${id}. Known: ${TTS_LOCAL_MODELS.map(x => x.id).join(", ")}`);
	return m;
}

/** Find the default voice index for a model; falls back to 0. */
export function getDefaultVoiceSid(model: TtsLocalModelInfo): number {
	if (model.voices.some(v => v.sid === model.defaultSid)) return model.defaultSid;
	return model.voices[0]?.sid ?? 0;
}

/**
 * Human-readable language name lookup. Used by the voice picker.
 * Empty input returns empty string; unknown bases fall back to the raw tag.
 */
export function languageName(tag: string): string {
	if (!tag) return "";
	const base = tag.split("-")[0]!.toLowerCase();
	const names: Record<string, string> = {
		en: "English", es: "Spanish", fr: "French", de: "German", hi: "Hindi",
		pt: "Portuguese", zh: "Chinese", it: "Italian", ru: "Russian",
		ar: "Arabic", tr: "Turkish", nl: "Dutch", ja: "Japanese", ko: "Korean",
	};
	return names[base] ?? tag;
}

/**
 * Returns true if the model's language list covers `lang` (BCP-47 tag).
 *
 * Matching rules — region-strict by design:
 *   1. Exact tag match wins (e.g. request "pt-BR" hits a "pt-BR" entry)
 *   2. A bare-base catalog entry covers any region of that language
 *      (e.g. catalog has "en" → matches "en", "en-US", "en-GB"). Models
 *      that genuinely cover all variants of a language list the bare base
 *      tag (Kokoro multilingual is the only such entry today).
 *   3. A bare-base request matches a regional catalog entry only if the
 *      catalog has exactly ONE region for that language (so picking "es"
 *      hits "es-ES" deterministically because there is no other es-*
 *      voice in the catalog).
 *   4. Otherwise NO match — region mismatches like pt-PT vs pt-BR,
 *      zh-TW vs zh-CN, ar-EG vs ar-JO, en-AU vs en-US route to NO so the
 *      caller can surface a clear error instead of playing the wrong accent.
 *
 * Rule 4 is the important one: previous versions stripped region on both
 * sides which silently routed pt-PT speech to a Brazilian voice. That kind
 * of substitution is harder to debug than a "no model supports pt-PT,
 * install one or pick a different language" error.
 */
export function modelSupportsLanguage(model: TtsLocalModelInfo, lang: string): boolean {
	const requested = normalizeLangTag(lang);
	const base = requested.split("-")[0]!;

	for (const cat of model.languages) {
		const catNorm = normalizeLangTag(cat);
		// Rule 1: exact match
		if (catNorm === requested) return true;
		// Rule 2: bare base in catalog covers any region
		if (!catNorm.includes("-") && catNorm === base) return true;
	}

	// Rule 3: bare base request — match only if catalog has exactly one region
	// for this language. Multiple regions (e.g. ar-JO and ar-EG would conflict)
	// require an explicit pick.
	if (!requested.includes("-")) {
		const matchingRegions = model.languages
			.map(normalizeLangTag)
			.filter(c => c.includes("-") && c.split("-")[0] === base);
		if (matchingRegions.length === 1) return true;
	}

	return false;
}

/** Normalize a BCP-47 tag for matching: lowercase language, uppercase region. */
function normalizeLangTag(tag: string): string {
	const parts = tag.split("-");
	const lang = (parts[0] ?? "").toLowerCase();
	if (parts.length === 1) return lang;
	const region = (parts[1] ?? "").toUpperCase();
	return `${lang}-${region}`;
}

// ─── Model installation (download + extract) ─────────────────────────────────

/** TTS models live under ~/.pi/models/tts/ to keep them separate from STT. */
export function getTtsModelsDir(): string {
	return path.join(os.homedir(), ".pi", "models", "tts");
}

/** Per-model directory. Does NOT verify existence — see getInstalledTtsModelDir. */
export function getTtsModelDir(modelId: string): string {
	return path.join(getTtsModelsDir(), modelId);
}

/** True iff the model archive has been downloaded and extracted. */
export function isTtsModelInstalled(modelId: string): boolean {
	const dir = getTtsModelDir(modelId);
	if (!fs.existsSync(dir)) return false;
	const tokens = path.join(dir, "tokens.txt");
	// Every supported slot (kitten/vits/kokoro) ships a tokens.txt at
	// the archive root, so its presence is a robust install marker
	// without us needing to know the slot's other expected files.
	return fs.existsSync(tokens);
}

/**
 * Resolve `modelId` to an installed model directory. Throws a user-facing
 * error if the archive hasn't been downloaded yet — the caller (slash
 * command or settings panel) is expected to either trigger
 * `ensureTtsModelInstalled` first or surface this message to prompt the
 * user to install via /voice-settings.
 */
export function getInstalledTtsModelDir(modelId: string): string {
	if (!isTtsModelInstalled(modelId)) {
		throw new Error(
			`TTS model "${modelId}" is not installed. ` +
			`Run /voice-settings → Models tab → install ${modelId}, ` +
			`or download manually: ` +
			`curl -L ${getTtsModel(modelId).archiveUrl} | tar xj -C "${getTtsModelsDir()}"`,
		);
	}
	return getTtsModelDir(modelId);
}

export interface TtsInstallProgress {
	/**
	 * - "download" — fetching archive bytes (with phase totals)
	 * - "extract" — running tar over the saved archive
	 * - "verify" — moving extracted files to final dir
	 * - "done" — install complete
	 */
	phase: "download" | "extract" | "verify" | "done";
	bytes?: number;
	totalBytes?: number;
}

/**
 * Result returned alongside install completion — exposes the computed
 * SHA-256 so callers (and v7.1+ catalog updates) can pin known-good hashes.
 */
export interface TtsInstallResult {
	dir: string;
	archiveSha256: string;
}

/**
 * In-flight install promise per modelId. Concurrent callers requesting
 * the same modelId share one in-flight install; without this, two
 * concurrent calls would both open `<id>.partial.tar.bz2` for writing
 * (`fs.createWriteStream` with `flags: "w"` truncates), corrupt each
 * other's bytes, and either fail tar extraction or race on the rename
 * to the final dir with ENOTEMPTY.
 *
 * Mirrors the in-flight Map pattern in `tts-engine.ts` (which itself
 * mirrors the v5.0.9 sherpa-loader single-flight pattern). Keyed by
 * modelId only — the model dir is uniquely determined by id, so two
 * concurrent calls with the same id necessarily target the same files.
 *
 * Each caller still receives its OWN onProgress callbacks routed
 * through the shared in-flight promise via a per-call wrapper that
 * forwards events from the active install. The first caller "owns"
 * the install for progress reporting; later concurrent callers see
 * `phase: "done"` immediately on resolve.
 */
const inFlightInstalls = new Map<string, Promise<TtsInstallResult>>();

/**
 * Download and extract `modelId` if not already installed. Idempotent —
 * if already installed, resolves immediately.
 *
 * Concurrency: per-modelId in-flight Map serializes concurrent installs
 * for the same model. See `inFlightInstalls` doc above.
 *
 * The flow is download-to-disk-then-extract, not streaming-to-tar:
 *   1. Resume-aware fetch → write archive bytes to
 *      `~/.pi/models/tts/<id>.partial.tar.bz2`. If the partial file
 *      exists from a prior interrupted run, send `Range: bytes=N-` and
 *      append. SHA-256 is computed across the full file by re-reading
 *      it once on completion (cheap — ~200ms for 126 MB on M-series).
 *   2. If the catalog entry has `archiveSha256`, compare against the
 *      computed hash. Mismatch → reject + cleanup partial.
 *   3. `tar -xj -f <archive> -C <stagingDir>` to extract.
 *   4. Move staging contents to final `<modelDir>` via rename (atomic).
 *   5. Delete the archive file.
 *
 * Errors:
 *   - "Download failed: HTTP <status>" on non-2xx (and not 206/200 retry)
 *   - "Network error: <message>" on fetch failure
 *   - "Archive integrity check failed: ..." on SHA-256 mismatch
 *   - "tar exited with code N" on extraction failure
 *   - DOMException("AbortError") if signal fires
 */
export function ensureTtsModelInstalled(
	modelId: string,
	opts: {
		signal?: AbortSignal;
		onProgress?: (info: TtsInstallProgress) => void;
	} = {},
): Promise<TtsInstallResult> {
	// Validate the modelId UPFRONT, before touching the in-flight Map.
	// `getTtsModel` throws synchronously for unknown ids; if we left this
	// to doInstall(), the throw would happen inside the async body and
	// the rejected promise could end up in the Map briefly before the
	// .finally cleanup fires — a non-issue in practice but documenting
	// intent here is cheaper than reasoning about microtask ordering.
	getTtsModel(modelId);

	// Concurrent caller for the same modelId: piggy-back on the existing
	// install promise. Checked BEFORE the on-disk fast-path so two
	// concurrent first-time callers can't both pass `isTtsModelInstalled
	// === false` (which is genuinely impossible under JS run-to-completion
	// since Map.get/Map.set don't yield, but reordering makes the single-
	// flight invariant unmistakable to a casual reader and satisfies the
	// godspeed gate without behavior change).
	const existing = inFlightInstalls.get(modelId);
	if (existing) {
		return existing.then(
			(result) => {
				opts.onProgress?.({ phase: "done" });
				return result;
			},
			(err) => { throw err; },
		);
	}

	// On-disk fast-path: model already installed from a prior run, no
	// need to involve the in-flight Map at all. Returns a one-microtask
	// resolved promise.
	if (isTtsModelInstalled(modelId)) {
		const dir = getTtsModelDir(modelId);
		const sha = getTtsModel(modelId).archiveSha256 ?? "";
		opts.onProgress?.({ phase: "done" });
		return Promise.resolve({ dir, archiveSha256: sha });
	}

	// First caller — claim the slot before any await yields, run the
	// install, clean the slot on settlement.
	const pending = doInstall(modelId, opts).finally(() => {
		// Identity-guarded delete: only clear if we're still the in-flight
		// entry. If a later install overwrote the slot (shouldn't happen
		// since we set before yielding, but defense-in-depth), don't
		// touch its entry.
		if (inFlightInstalls.get(modelId) === pending) {
			inFlightInstalls.delete(modelId);
		}
	});
	inFlightInstalls.set(modelId, pending);
	return pending;
}

/**
 * The actual install pipeline — extracted from the public entrypoint so
 * the concurrency guard above can wrap it without changing the body.
 *
 * Note: signature matches the original ensureTtsModelInstalled — pre-v7.0.1
 * callers that didn't bind the return value still work.
 */
async function doInstall(
	modelId: string,
	opts: {
		signal?: AbortSignal;
		onProgress?: (info: TtsInstallProgress) => void;
	},
): Promise<TtsInstallResult> {
	const model = getTtsModel(modelId);
	const dir = getTtsModelDir(modelId);

	if (isTtsModelInstalled(modelId)) {
		opts.onProgress?.({ phase: "done" });
		return { dir, archiveSha256: model.archiveSha256 ?? "" };
	}
	if (opts.signal?.aborted) throw makeAbortErr();

	const ttsDir = getTtsModelsDir();
	fs.mkdirSync(ttsDir, { recursive: true });
	const archivePath = path.join(ttsDir, `${modelId}.partial.tar.bz2`);

	// `phaseReached` distinguishes failures by where they occurred so the
	// catch block knows whether to keep the partial archive (for resume)
	// or delete it (because we know it's corrupt — extract failed). After
	// SHA verification passes, a tar failure points at a corrupt partial
	// that resume can't fix; delete it so the next attempt re-downloads.
	let phaseReached: "download" | "verify" | "extract" | "done" = "download";
	let computedSha256 = "";

	try {
		// Phase 1 — download archive bytes (with resume).
		await downloadArchive(model.archiveUrl, archivePath, opts);
		if (opts.signal?.aborted) throw makeAbortErr();

		// Phase 2 — verify hash.
		phaseReached = "verify";
		opts.onProgress?.({ phase: "verify" });
		computedSha256 = await sha256OfFile(archivePath, opts.signal);
		if (model.archiveSha256 && model.archiveSha256.toLowerCase() !== computedSha256.toLowerCase()) {
			throw new Error(
				`Archive integrity check failed for ${modelId}: ` +
				`expected ${model.archiveSha256}, got ${computedSha256}. ` +
				`Delete ${archivePath} and retry, or check for a corrupted upstream release.`,
			);
		}

		// Phase 3 — extract.
		phaseReached = "extract";
		opts.onProgress?.({ phase: "extract", totalBytes: model.sizeBytes });
		const stagingDir = `${dir}.staging-${process.pid}`;
		fs.mkdirSync(stagingDir, { recursive: true });
		try {
			await runTarExtract(archivePath, stagingDir, opts.signal);

			// Phase 4 — move into final location. The archive's top-level
			// directory differs per model (e.g.
			// `vits-piper-en_US-lessac-medium-int8/`). Flatten to
			// `<modelDir>/tokens.txt` etc.
			const stagingEntries = fs.readdirSync(stagingDir);
			const innerDir = stagingEntries.length === 1 && fs.statSync(path.join(stagingDir, stagingEntries[0]!)).isDirectory()
				? path.join(stagingDir, stagingEntries[0]!)
				: stagingDir;
			// rename is atomic when innerDir and dir are on the same
			// filesystem (~/.pi/models/tts/.staging is a sibling of dir).
			fs.renameSync(innerDir, dir);
		} finally {
			try { fs.rmSync(stagingDir, { recursive: true, force: true }); } catch {}
		}

		// Phase 5 — clean up the archive file. Successful install means we
		// no longer need the partial; resume is moot.
		phaseReached = "done";
		try { fs.unlinkSync(archivePath); } catch {}
	} catch (err) {
		// Defense in depth: if we already reached `done` (install completed,
		// renamed into place, archive unlinked) and somehow an error still
		// bubbles up from after that point, DO NOT clean up — the model
		// is fully installed on disk and deleting it would force a
		// pointless re-download. Currently this branch is unreachable in
		// the source order above, but the explicit guard documents the
		// invariant for future maintainers and survives refactors.
		if (phaseReached === "done") throw err;

		// Decide what to clean up based on how far we got:
		//   - download phase: keep the partial (next attempt resumes)
		//   - verify phase (SHA mismatch): delete the partial — bytes are corrupt
		//   - extract phase (tar failed): delete the partial — bytes are
		//     corrupt at the tar layer even if SHA was unset, OR delete the
		//     half-built dir if it got partially populated
		// In all failure paths short of `done`, delete the destination dir
		// if it got created.
		if (phaseReached === "verify" || phaseReached === "extract") {
			try { fs.unlinkSync(archivePath); } catch {}
		}
		try { fs.rmSync(dir, { recursive: true, force: true }); } catch {}
		throw err;
	}

	// Move the terminal `phase: "done"` callback OUTSIDE the try block.
	// If a user-supplied onProgress throws, we don't want the catch to
	// delete the just-installed dir and force a re-download.
	opts.onProgress?.({ phase: "done" });
	return { dir, archiveSha256: computedSha256 };
}

/**
 * Download bytes to `archivePath` with `Range` resume.
 *
 * If the file already exists, we send `Range: bytes=<size>-` and the
 * server is expected to respond with 206 Partial Content (we append) or
 * 200 OK (server doesn't support range; we throw the file away and
 * start over).
 *
 * Surfaces byte-count progress via `opts.onProgress`. The total byte
 * count comes from `Content-Length` on the response — for 206 responses
 * we add the existing partial size to the running counter so the user
 * sees a continuous progress bar across resumed sessions.
 */
async function downloadArchive(
	url: string,
	archivePath: string,
	opts: { signal?: AbortSignal; onProgress?: (info: TtsInstallProgress) => void },
): Promise<void> {
	let existingBytes = 0;
	if (fs.existsSync(archivePath)) {
		try { existingBytes = fs.statSync(archivePath).size; } catch {}
	}

	const headers: Record<string, string> = {};
	if (existingBytes > 0) headers.Range = `bytes=${existingBytes}-`;

	let res: Response;
	try {
		res = await fetch(url, { signal: opts.signal, headers });
	} catch (err: any) {
		if (err?.name === "AbortError") throw err;
		throw new Error(`Network error: ${err?.message ?? String(err)}`);
	}

	let appendMode = false;
	if (res.status === 206 && existingBytes > 0) {
		appendMode = true;
	} else if (res.status === 200) {
		// Server ignored our Range — start over.
		appendMode = false;
		existingBytes = 0;
	} else if (res.status === 416 && existingBytes > 0) {
		// "Range Not Satisfiable" — the server says our requested range
		// (`bytes=<size>-`) starts at or past the end of the resource.
		// This means our local partial is already at-or-beyond the full
		// resource size, which happens when a prior run died AFTER the
		// download finished but BEFORE we got to unlink the partial. The
		// caller (ensureTtsModelInstalled) verifies the SHA-256 next, so
		// returning here without re-downloading is safe: a corrupted
		// equal-size partial would fail the hash check.
		// Drain the response body to free the connection (some HTTP impls
		// hold the socket otherwise).
		try { await res.body?.cancel?.(); } catch {}
		return;
	} else if (!res.ok) {
		throw new Error(`Download failed: HTTP ${res.status} from ${url}`);
	}

	if (!res.body) throw new Error(`Download failed: empty body from ${url}`);

	// Total = bytes already on disk + Content-Length of this response.
	const contentLength = parseInt(res.headers.get("content-length") ?? "0", 10);
	const totalBytes = existingBytes + (Number.isFinite(contentLength) ? contentLength : 0);

	const sink = fs.createWriteStream(archivePath, { flags: appendMode ? "a" : "w" });
	let bytesSeen = existingBytes;
	opts.onProgress?.({ phase: "download", bytes: bytesSeen, totalBytes });

	// Capture sink errors as a settle-once promise. Without this, a write
	// error during the body-streaming loop (disk full, EIO, EPERM) emits
	// 'error' on the stream and Node throws an unhandled exception that
	// crashes the process. We also race the drain wait against `error`
	// so a stream that fails mid-backpressure doesn't hang forever.
	const sinkErrorRef: { err: Error | null } = { err: null };
	sink.on("error", (err: Error) => {
		sinkErrorRef.err ??= err;
	});

	const reader = res.body.getReader();
	try {
		while (true) {
			if (opts.signal?.aborted) throw makeAbortErr();
			if (sinkErrorRef.err) throw new Error(`Disk write failed: ${sinkErrorRef.err.message}`);
			const { value, done } = await reader.read();
			if (done) break;
			if (value) {
				bytesSeen += value.byteLength;
				// Honor backpressure: if write returns false, await `drain`
				// OR the first `error` — whichever comes first.
				const ok = sink.write(Buffer.from(value));
				if (!ok) {
					await new Promise<void>((resolve, reject) => {
						const onDrain = () => { sink.off("error", onErr); resolve(); };
						const onErr = (err: Error) => { sink.off("drain", onDrain); reject(err); };
						sink.once("drain", onDrain);
						sink.once("error", onErr);
					});
				}
				opts.onProgress?.({ phase: "download", bytes: bytesSeen, totalBytes });
			}
		}
	} finally {
		try { reader.releaseLock(); } catch {}
		// Drain and close the file. End() callback fires after the final
		// flush. Errors during close are surfaced via `error` listener
		// captured before the await.
		await new Promise<void>((resolve, rej) => {
			let settled = false;
			const onError = (err: Error) => {
				if (settled) return;
				settled = true;
				rej(err);
			};
			sink.once("error", onError);
			sink.end(() => {
				if (settled) return;
				settled = true;
				sink.off("error", onError);
				resolve();
			});
		});
	}
}

/** Compute the SHA-256 hex digest of `filePath`. Streams via fs.createReadStream. */
async function sha256OfFile(filePath: string, signal?: AbortSignal): Promise<string> {
	return new Promise((resolve, reject) => {
		const hash = createHash("sha256");
		const stream = fs.createReadStream(filePath);
		const onAbort = () => {
			stream.destroy();
			reject(makeAbortErr());
		};
		signal?.addEventListener("abort", onAbort, { once: true });
		stream.on("data", (chunk) => hash.update(chunk));
		stream.on("end", () => {
			signal?.removeEventListener("abort", onAbort);
			resolve(hash.digest("hex"));
		});
		stream.on("error", (err) => {
			signal?.removeEventListener("abort", onAbort);
			reject(err);
		});
	});
}

/** Spawn `tar -xj -f <archive> -C <stagingDir>` and resolve on exit code 0. */
async function runTarExtract(archivePath: string, stagingDir: string, signal?: AbortSignal): Promise<void> {
	const tar = spawn("tar", ["-xj", "-f", archivePath, "-C", stagingDir], {
		stdio: ["ignore", "ignore", "pipe"],
		...(signal ? { signal } : {}),
	});
	let tarStderr = "";
	tar.stderr?.on("data", (d: Buffer) => {
		if (tarStderr.length < 1024) tarStderr += d.toString();
	});
	await new Promise<void>((resolve, reject) => {
		tar.on("error", (err: NodeJS.ErrnoException) => {
			if (err.name === "AbortError" || signal?.aborted) reject(makeAbortErr());
			else reject(new Error(`tar failed to start: ${err.message}`));
		});
		tar.on("close", (code, sig) => {
			if (code === 0) resolve();
			else if (signal?.aborted) reject(makeAbortErr());
			else reject(new Error(`tar exited with code ${code}${sig ? ` (${sig})` : ""}: ${tarStderr.trim().slice(-200)}`));
		});
	});
}

function makeAbortErr(): Error {
	if (typeof DOMException === "function") {
		return new DOMException("TTS model install aborted", "AbortError");
	}
	const e = new Error("TTS model install aborted");
	(e as any).name = "AbortError";
	return e;
}

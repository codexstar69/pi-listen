import type { ExtensionCommandContext, ExtensionContext } from "@mariozechner/pi-coding-agent";
import type { VoiceBackend, VoiceConfig, VoiceSettingsScope } from "./config";
import {
	LOCAL_MODELS, DEFAULT_LOCAL_MODEL, DEFAULT_LOCAL_ENDPOINT,
	checkLocalServer, getLanguagesForLocalModel,
	type LocalLangEntry, type LocalModelInfo,
} from "./local";
import { detectDevice, autoRecommendModel, getModelFitness, formatDeviceSummary, localeToLanguageCode, type ModelFitness } from "./device";

type VoiceUiContext = ExtensionContext | ExtensionCommandContext;

export interface OnboardingResult {
	config: VoiceConfig;
	selectedScope: VoiceSettingsScope;
	summaryLines: string[];
}

export interface FirstRunDecision {
	action: "start" | "later";
}

// ─── Nova-3 supported languages for live streaming ──────────────────────
// All verified for streaming support. "multi" removed — not supported for live.

interface LangEntry { name: string; code: string; popular?: boolean; model?: string; }

const LANGUAGES: LangEntry[] = [
	// Top popular — shown first in picker
	{ name: "English", code: "en", popular: true },
	{ name: "Hindi", code: "hi", popular: true },
	{ name: "Spanish", code: "es", popular: true },
	{ name: "French", code: "fr", popular: true },
	{ name: "German", code: "de", popular: true },
	{ name: "Portuguese — Brazil", code: "pt-BR", popular: true },
	{ name: "Japanese", code: "ja", popular: true },
	{ name: "Korean", code: "ko", popular: true },
	{ name: "Arabic", code: "ar", popular: true },
	{ name: "Russian", code: "ru", popular: true },
	// Nova-2 only (auto-switches model — shown in picker)
	{ name: "Chinese — Mandarin [Nova-2]", code: "zh", popular: true, model: "nova-2" },
	{ name: "Chinese — Mandarin Simplified [Nova-2]", code: "zh-CN", model: "nova-2" },
	{ name: "Chinese — Mandarin Traditional [Nova-2]", code: "zh-TW", model: "nova-2" },
	{ name: "Chinese — Cantonese [Nova-2]", code: "zh-HK", model: "nova-2" },
	// All others alphabetically (Nova-3)
	{ name: "Belarusian", code: "be" },
	{ name: "Bengali", code: "bn" },
	{ name: "Bosnian", code: "bs" },
	{ name: "Bulgarian", code: "bg" },
	{ name: "Catalan", code: "ca" },
	{ name: "Croatian", code: "hr" },
	{ name: "Czech", code: "cs" },
	{ name: "Danish", code: "da" },
	{ name: "Dutch", code: "nl" },
	{ name: "English — Australia", code: "en-AU" },
	{ name: "English — India", code: "en-IN" },
	{ name: "English — New Zealand", code: "en-NZ" },
	{ name: "English — UK", code: "en-GB" },
	{ name: "English — US", code: "en-US" },
	{ name: "Estonian", code: "et" },
	{ name: "Finnish", code: "fi" },
	{ name: "Flemish", code: "nl-BE" },
	{ name: "French — Canada", code: "fr-CA" },
	{ name: "German — Switzerland", code: "de-CH" },
	{ name: "Greek", code: "el" },
	{ name: "Hebrew", code: "he" },
	{ name: "Hungarian", code: "hu" },
	{ name: "Indonesian", code: "id" },
	{ name: "Italian", code: "it" },
	{ name: "Kannada", code: "kn" },
	{ name: "Korean — KR", code: "ko-KR" },
	{ name: "Latvian", code: "lv" },
	{ name: "Lithuanian", code: "lt" },
	{ name: "Macedonian", code: "mk" },
	{ name: "Malay", code: "ms" },
	{ name: "Marathi", code: "mr" },
	{ name: "Norwegian", code: "no" },
	{ name: "Persian", code: "fa" },
	{ name: "Polish", code: "pl" },
	{ name: "Portuguese", code: "pt" },
	{ name: "Portuguese — Portugal", code: "pt-PT" },
	{ name: "Romanian", code: "ro" },
	{ name: "Serbian", code: "sr" },
	{ name: "Slovak", code: "sk" },
	{ name: "Slovenian", code: "sl" },
	{ name: "Spanish — Latin America", code: "es-419" },
	{ name: "Swedish", code: "sv" },
	{ name: "Tagalog", code: "tl" },
	{ name: "Tamil", code: "ta" },
	{ name: "Telugu", code: "te" },
	{ name: "Turkish", code: "tr" },
	{ name: "Ukrainian", code: "uk" },
	{ name: "Urdu", code: "ur" },
	{ name: "Vietnamese", code: "vi" },
];

function formatLangOption(l: LangEntry): string {
	return `${l.name} (${l.code})`;
}

/** Get the best model for a language code. Nova-2 for Chinese, Nova-3 for everything else. */
export function modelForLanguage(code: string): string {
	const entry = LANGUAGES.find(l => l.code === code);
	return entry?.model || "nova-3";
}

/** Extract language code from "Language Name (code)" format */
export function extractLanguageCode(selection: string): string {
	const match = selection.match(/\(([^)]+)\)$/);
	return match ? match[1] : "en";
}

/** Find display name for a language code */
export function languageDisplayName(code: string): string {
	const entry = LANGUAGES.find(l => l.code === code);
	return entry ? `${entry.name} (${entry.code})` : code;
}

/** Show language picker with fuzzy search — uses ctx.ui.custom() for real-time filtering.
 *  Pass `overrideLanguages` to show a model-specific language list (e.g. for local Whisper/Parakeet). */
export async function pickLanguage(
	ctx: VoiceUiContext,
	currentCode: string,
	overrideLanguages?: LocalLangEntry[],
): Promise<string | undefined> {
	const { Container, Input, Spacer, Text, fuzzyFilter, getEditorKeybindings } = await import("@mariozechner/pi-tui");

	const langList: LangEntry[] = overrideLanguages
		? overrideLanguages.map(l => ({ name: l.name, code: l.code, popular: l.popular }))
		: LANGUAGES;
	const current = overrideLanguages
		? (langList.find(l => l.code === currentCode)?.name ?? currentCode)
		: languageDisplayName(currentCode);
	const popular = langList.filter(l => l.popular);
	const allItems = langList.map(l => ({ ...l, label: formatLangOption(l) }));

	return ctx.ui.custom<string | undefined>((tui, theme, _keybindings, done) => {
		const container = new Container();
		const searchInput = new Input();
		const listContainer = new Container();

		let filtered = allItems;
		let selectedIndex = 0;

		function updateList() {
			listContainer.clear();
			const maxVisible = 12;
			const start = Math.max(0, Math.min(selectedIndex - Math.floor(maxVisible / 2), filtered.length - maxVisible));
			const end = Math.min(start + maxVisible, filtered.length);

			// Popular header when no search query
			if (!searchInput.getValue()) {
				listContainer.addChild(new Text(theme.fg("muted", "  Popular:"), 0, 0));
				for (let i = 0; i < popular.length && i < end; i++) {
					const item = filtered[i];
					if (!item) continue;
					const isSelected = i === selectedIndex;
					const isCurrent = item.code === currentCode;
					const prefix = isSelected ? theme.fg("accent", "→ ") : "  ";
					const text = isSelected ? theme.fg("accent", item.label) : item.label;
					const badge = item.model ? theme.fg("warning", ` [${item.model}]`) : "";
					const check = isCurrent ? theme.fg("success", " ✓") : "";
					listContainer.addChild(new Text(`${prefix}${text}${badge}${check}`, 0, 0));
				}
				if (filtered.length > popular.length) {
					listContainer.addChild(new Text(theme.fg("muted", "  ───────────────────"), 0, 0));
					for (let i = popular.length; i < end; i++) {
						const item = filtered[i];
						if (!item) continue;
						const isSelected = i === selectedIndex;
						const isCurrent = item.code === currentCode;
						const prefix = isSelected ? theme.fg("accent", "→ ") : "  ";
						const text = isSelected ? theme.fg("accent", item.label) : item.label;
						const badge = item.model ? theme.fg("warning", ` [${item.model}]`) : "";
						const check = isCurrent ? theme.fg("success", " ✓") : "";
						listContainer.addChild(new Text(`${prefix}${text}${badge}${check}`, 0, 0));
					}
				}
			} else {
				// Search results
				for (let i = start; i < end; i++) {
					const item = filtered[i];
					if (!item) continue;
					const isSelected = i === selectedIndex;
					const isCurrent = item.code === currentCode;
					const prefix = isSelected ? theme.fg("accent", "→ ") : "  ";
					const text = isSelected ? theme.fg("accent", item.label) : item.label;
					const badge = item.model ? theme.fg("warning", ` [${item.model}]`) : "";
					const check = isCurrent ? theme.fg("success", " ✓") : "";
					listContainer.addChild(new Text(`${prefix}${text}${badge}${check}`, 0, 0));
				}
			}

			if (filtered.length === 0) {
				listContainer.addChild(new Text(theme.fg("muted", "  No matching languages"), 0, 0));
			} else if (start > 0 || end < filtered.length) {
				listContainer.addChild(new Text(theme.fg("muted", `  (${selectedIndex + 1}/${filtered.length})`), 0, 0));
			}

			tui.requestRender();
		}

		function filterList(query: string) {
			if (!query) {
				filtered = allItems;
			} else {
				filtered = fuzzyFilter(allItems, query, (item) => `${item.name} ${item.code}`);
			}
			selectedIndex = Math.min(selectedIndex, Math.max(0, filtered.length - 1));
			updateList();
		}

		// Build UI
		container.addChild(new Spacer(1));
		container.addChild(new Text(theme.fg("accent", `Voice language (current: ${current})`), 1, 0));
		container.addChild(new Text(theme.fg("muted", "Type to search, ↑↓ to navigate, Enter to select, Esc to cancel"), 1, 0));
		container.addChild(new Spacer(1));
		container.addChild(searchInput);
		container.addChild(new Spacer(1));
		container.addChild(listContainer);
		container.addChild(new Spacer(1));

		updateList();

		const kb = getEditorKeybindings();
		(container as any).handleInput = (keyData: string) => {
			if (kb.matches(keyData, "selectUp")) {
				if (filtered.length === 0) return;
				selectedIndex = selectedIndex === 0 ? filtered.length - 1 : selectedIndex - 1;
				updateList();
			} else if (kb.matches(keyData, "selectDown")) {
				if (filtered.length === 0) return;
				selectedIndex = selectedIndex === filtered.length - 1 ? 0 : selectedIndex + 1;
				updateList();
			} else if (kb.matches(keyData, "selectConfirm") || keyData === "\n") {
				const item = filtered[selectedIndex];
				done(item ? item.code : undefined);
			} else if (kb.matches(keyData, "selectCancel")) {
				done(undefined);
			} else {
				searchInput.handleInput(keyData);
				filterList(searchInput.getValue());
			}
		};

		// Focusable for IME
		Object.defineProperty(container, "focused", {
			get: () => (searchInput as any).focused,
			set: (v: boolean) => { (searchInput as any).focused = v; },
		});

		return container;
	});
}

export function finalizeOnboardingConfig(
	config: VoiceConfig,
	options: { validated: boolean; source: "first-run" | "setup-command" },
): VoiceConfig {
	if (options.validated) {
		const timestamp = new Date().toISOString();
		return {
			...config,
			onboarding: {
				...config.onboarding,
				completed: true,
				schemaVersion: config.version,
				completedAt: timestamp,
				lastValidatedAt: timestamp,
				source: options.source,
				skippedAt: undefined,
			},
		};
	}

	return {
		...config,
		onboarding: {
			...config.onboarding,
			completed: false,
			schemaVersion: config.version,
			completedAt: undefined,
			lastValidatedAt: undefined,
			source: "repair",
			skippedAt: undefined,
		},
	};
}

export async function promptFirstRunOnboarding(ctx: VoiceUiContext): Promise<FirstRunDecision> {
	const choice = await ctx.ui.select("Set up pi-voice now?", [
		"Start voice setup",
		"Remind me later",
	]);

	return { action: choice === "Start voice setup" ? "start" : "later" };
}

/**
 * Pick a local model using fuzzy search — device-aware with fitness badges.
 * Shows recommended models first, with color-coded fitness indicators.
 */
export async function pickLocalModel(
	ctx: VoiceUiContext,
	currentModelId: string | undefined,
	language: string,
): Promise<LocalModelInfo | undefined> {
	const { Container, Input, Spacer, Text, fuzzyFilter, getEditorKeybindings } = await import("@mariozechner/pi-tui");

	const device = detectDevice();
	const allItems = LOCAL_MODELS.map(m => {
		const fitness = getModelFitness(m, device);
		const badge = fitnessLabel(fitness);
		return {
			...m,
			fitness,
			label: `${m.name} — ${m.size} ${badge} (${m.notes})`,
		};
	});

	// Sort: recommended → compatible → warning → incompatible, then by size (larger = more accurate)
	const fitnessOrder: Record<ModelFitness, number> = { recommended: 0, compatible: 1, warning: 2, incompatible: 3 };
	allItems.sort((a, b) => {
		const fitDiff = fitnessOrder[a.fitness] - fitnessOrder[b.fitness];
		if (fitDiff !== 0) return fitDiff;
		return b.sizeBytes - a.sizeBytes;
	});

	return ctx.ui.custom<LocalModelInfo | undefined>((tui, theme, _keybindings, done) => {
		const container = new Container();
		const searchInput = new Input();
		const listContainer = new Container();

		let filtered = allItems;
		let selectedIndex = 0;

		function updateList() {
			listContainer.clear();
			const maxVisible = 14;
			const start = Math.max(0, Math.min(selectedIndex - Math.floor(maxVisible / 2), filtered.length - maxVisible));
			const end = Math.min(start + maxVisible, filtered.length);

			for (let i = start; i < end; i++) {
				const item = filtered[i];
				if (!item) continue;
				const isSelected = i === selectedIndex;
				const isCurrent = item.id === currentModelId;
				const prefix = isSelected ? theme.fg("accent", "→ ") : "  ";
				const nameText = isSelected ? theme.fg("accent", item.name) : item.name;
				const sizeText = theme.fg("muted", ` — ${item.size}`);
				const badge = fitnessThemeBadge(item.fitness, theme);
				const notes = theme.fg("muted", ` (${item.notes})`);
				const check = isCurrent ? theme.fg("success", " ✓") : "";
				listContainer.addChild(new Text(`${prefix}${nameText}${sizeText} ${badge}${notes}${check}`, 0, 0));
			}

			if (filtered.length === 0) {
				listContainer.addChild(new Text(theme.fg("muted", "  No matching models"), 0, 0));
			} else if (start > 0 || end < filtered.length) {
				listContainer.addChild(new Text(theme.fg("muted", `  (${selectedIndex + 1}/${filtered.length})`), 0, 0));
			}

			tui.requestRender();
		}

		function filterList(query: string) {
			if (!query) {
				filtered = allItems;
			} else {
				filtered = fuzzyFilter(allItems, query, (item) => `${item.name} ${item.id} ${item.notes} ${item.langSupport}`);
			}
			selectedIndex = Math.min(selectedIndex, Math.max(0, filtered.length - 1));
			updateList();
		}

		// Build UI
		container.addChild(new Spacer(1));
		container.addChild(new Text(theme.fg("accent", `Choose local model (${formatDeviceSummary(device)})`), 1, 0));
		container.addChild(new Text(theme.fg("muted", "Type to search, ↑↓ to navigate, Enter to select, Esc to cancel"), 1, 0));
		container.addChild(new Spacer(1));
		container.addChild(searchInput);
		container.addChild(new Spacer(1));
		container.addChild(listContainer);
		container.addChild(new Spacer(1));

		updateList();

		const kb = getEditorKeybindings();
		(container as any).handleInput = (keyData: string) => {
			if (kb.matches(keyData, "selectUp")) {
				if (filtered.length === 0) return;
				selectedIndex = selectedIndex === 0 ? filtered.length - 1 : selectedIndex - 1;
				updateList();
			} else if (kb.matches(keyData, "selectDown")) {
				if (filtered.length === 0) return;
				selectedIndex = selectedIndex === filtered.length - 1 ? 0 : selectedIndex + 1;
				updateList();
			} else if (kb.matches(keyData, "selectConfirm") || keyData === "\n") {
				const item = filtered[selectedIndex];
				done(item ? LOCAL_MODELS.find(m => m.id === item.id) : undefined);
			} else if (kb.matches(keyData, "selectCancel")) {
				done(undefined);
			} else {
				searchInput.handleInput(keyData);
				filterList(searchInput.getValue());
			}
		};

		Object.defineProperty(container, "focused", {
			get: () => (searchInput as any).focused,
			set: (v: boolean) => { (searchInput as any).focused = v; },
		});

		return container;
	});
}

/** Fitness label for display */
function fitnessLabel(fitness: ModelFitness): string {
	switch (fitness) {
		case "recommended": return "[recommended]";
		case "compatible": return "[compatible]";
		case "warning": return "[may be slow]";
		case "incompatible": return "[too large]";
	}
}

/** Fitness badge with theme colors */
function fitnessThemeBadge(fitness: ModelFitness, theme: any): string {
	switch (fitness) {
		case "recommended": return theme.fg("success", "[recommended]");
		case "compatible": return theme.fg("accent", "[compatible]");
		case "warning": return theme.fg("warning", "[may be slow]");
		case "incompatible": return theme.fg("error", "[too large]");
	}
}

export async function runVoiceOnboarding(
	ctx: VoiceUiContext,
	currentConfig: VoiceConfig,
	options?: { isFirstRun?: boolean },
): Promise<OnboardingResult | undefined> {
	const isFirstRun = options?.isFirstRun ?? !currentConfig.onboarding.completed;

	// ─── Choose backend ──────────────────────────────────────
	const backendChoice = await ctx.ui.select(
		"Choose transcription backend:",
		[
			"Deepgram (cloud, real-time streaming, needs API key)",
			"Local model (offline, no API key, auto-download)",
		],
	);
	if (!backendChoice) return undefined;
	const selectedBackend: VoiceBackend = backendChoice.startsWith("Local") ? "local" : "deepgram";

	let localModel = currentConfig.localModel;
	let localEndpoint: string | undefined = currentConfig.localEndpoint;

	if (selectedBackend === "local") {
		// ─── Smart local backend setup ───────────────────────
		const device = detectDevice();
		const detectedLang = localeToLanguageCode(device.systemLocale);
		const language = currentConfig.language || detectedLang;

		// Auto-recommend the best model
		const recommended = autoRecommendModel(LOCAL_MODELS, device, language);
		const deviceSummary = formatDeviceSummary(device);

		if (recommended) {
			const fitness = getModelFitness(recommended, device);
			const setupChoice = await ctx.ui.select(
				`Detected: ${deviceSummary}`,
				[
					`Install ${recommended.name} (${recommended.size}) ${fitnessLabel(fitness)} — recommended`,
					"Choose a different model",
					"Advanced: use external server",
				],
			);
			if (!setupChoice) return undefined;

			if (setupChoice.startsWith("Install")) {
				// Accept recommendation — will auto-download on first use
				localModel = recommended.id;
				localEndpoint = undefined; // In-process, no server needed
				ctx.ui.notify(
					[
						`Selected: ${recommended.name} (${recommended.size})`,
						"Model will download automatically on first voice recording.",
						"No server setup needed — runs in-process.",
					].join("\n"),
					"info",
				);
			} else if (setupChoice.startsWith("Choose")) {
				// Full model list with fuzzy search
				const picked = await pickLocalModel(ctx, localModel, language);
				if (!picked) return undefined;
				localModel = picked.id;
				localEndpoint = undefined;
				ctx.ui.notify(`Selected: ${picked.name} (${picked.size})`, "info");
			} else {
				// Advanced: external server
				localEndpoint = await promptServerEndpoint(ctx);
				if (localEndpoint === undefined) return undefined;

				// Still pick a model for server
				const modelOptions = LOCAL_MODELS.map(m => `${m.name} — ${m.size} (${m.notes})`);
				const modelChoice = await ctx.ui.select("Choose model (for server):", modelOptions);
				if (!modelChoice) return undefined;
				const modelIndex = modelOptions.indexOf(modelChoice);
				localModel = LOCAL_MODELS[modelIndex]?.id || DEFAULT_LOCAL_MODEL;
			}
		} else {
			// No recommendation — show full picker
			const picked = await pickLocalModel(ctx, localModel, language);
			if (!picked) return undefined;
			localModel = picked.id;
			localEndpoint = undefined;
		}
	} else {
		// ─── Deepgram backend setup (unchanged logic) ────────
		const hasDeepgramKey = Boolean(process.env.DEEPGRAM_API_KEY || currentConfig.deepgramApiKey);

		if (!hasDeepgramKey) {
			const keyAction = await ctx.ui.select(
				"Deepgram API key not found. What would you like to do?",
				[
					"Paste API key now",
					"I'll set it up later (ask pi to help or export DEEPGRAM_API_KEY=...)",
				],
			);
			if (!keyAction) return undefined;

			if (keyAction.startsWith("Paste")) {
				ctx.ui.notify(
					[
						"Get your free Deepgram API key:",
						"  → https://dpgr.am/pi-voice",
						"  (Sign up → $200 free credits, no card needed)",
						"",
						"Paste your key below:",
					].join("\n"),
					"info",
				);
				const apiKey = await ctx.ui.input("DEEPGRAM_API_KEY");
				if (apiKey && apiKey.trim().length > 10) {
					const trimmedKey = apiKey.trim();
					const fs = await import("node:fs");
					const os = await import("node:os");
					const home = os.homedir();
					const envSecretsPath = `${home}/.env.secrets`;
					const zshrcPath = `${home}/.zshrc`;
					const exportLine = `export DEEPGRAM_API_KEY="${trimmedKey}"`;

					const targetFile = fs.existsSync(envSecretsPath) ? envSecretsPath : zshrcPath;
					const existing = fs.existsSync(targetFile) ? fs.readFileSync(targetFile, "utf-8") : "";

					if (existing.includes("DEEPGRAM_API_KEY")) {
						const updated = existing.replace(/^export DEEPGRAM_API_KEY=.*$/m, exportLine);
						fs.writeFileSync(targetFile, updated);
					} else {
						fs.appendFileSync(targetFile, `\n${exportLine}\n`);
					}

					process.env.DEEPGRAM_API_KEY = trimmedKey;

					ctx.ui.notify(
						`API key saved to ${targetFile}\nActive in this session. New terminals will pick it up automatically.`,
						"info",
					);
				} else if (apiKey !== undefined && apiKey !== null) {
					ctx.ui.notify(
						"Key looks too short — skipped. You can set it later:\n  export DEEPGRAM_API_KEY=\"your-key\"",
						"warning",
					);
				}
			} else {
				ctx.ui.notify(
					[
						"No problem! When you're ready:",
						"  1. Get a key → https://dpgr.am/pi-voice ($200 free credits)",
						"  2. Run: export DEEPGRAM_API_KEY=\"your-key\"",
						"  3. Or ask pi: \"help me set up my Deepgram API key\"",
					].join("\n"),
					"info",
				);
			}
		}
	}

	// ─── Choose language (first-run only) ────────────────────
	let langCode = currentConfig.language;
	if (isFirstRun) {
		if (selectedBackend === "local" && localModel) {
			const { languages, englishOnly } = getLanguagesForLocalModel(localModel);
			if (englishOnly) {
				// Single-language model — auto-set
				langCode = languages[0]?.code || "en";
				const langName = languages[0]?.name || "English";
				ctx.ui.notify(`Language set to ${langName} (only language supported by this model).`, "info");
			} else {
				// Auto-detect from system locale, let user confirm or change
				const device = detectDevice();
				const detectedLang = localeToLanguageCode(device.systemLocale);
				const detectedEntry = languages.find(l => l.code === detectedLang);
				if (detectedEntry) {
					langCode = detectedLang;
					ctx.ui.notify(`Language auto-detected: ${detectedEntry.name} (${detectedEntry.code}). Change with /voice-language.`, "info");
				} else {
					const picked = await pickLanguage(ctx, currentConfig.language, languages);
					if (!picked) return undefined;
					langCode = picked;
				}
			}
		} else {
			// Deepgram — show full language list
			const picked = await pickLanguage(ctx, currentConfig.language);
			if (!picked) return undefined;
			langCode = picked;
		}
	}

	// ─── Choose scope ────────────────────────────────────────
	const scopeChoice = await ctx.ui.select("Where should pi-voice settings be saved?", [
		"Global (all projects)",
		"Project only (this repo)",
	]);
	if (!scopeChoice) return undefined;
	const selectedScope: VoiceSettingsScope = scopeChoice.startsWith("Project") ? "project" : "global";

	const selectedModel = LOCAL_MODELS.find(m => m.id === localModel);
	const backendLabel = selectedBackend === "local"
		? `Local — ${selectedModel?.name || localModel}${localEndpoint ? ` at ${localEndpoint}` : " (in-process)"}`
		: "Deepgram Nova-3 (streaming)";

	const summaryLines = [
		`Backend: ${backendLabel}`,
		`Language: ${languageDisplayName(langCode)}${isFirstRun ? "" : " (change with /voice-language)"}`,
		`Scope: ${selectedScope}`,
		...(selectedBackend === "deepgram"
			? [`API key: ${process.env.DEEPGRAM_API_KEY ? "configured" : "not yet set"}`]
			: []),
	];

	const confirm = await ctx.ui.confirm("Confirm voice setup", summaryLines.join("\n"));
	if (!confirm) return undefined;

	return {
		selectedScope,
		summaryLines,
		config: {
			...currentConfig,
			language: langCode,
			scope: selectedScope,
			backend: selectedBackend,
			localModel: selectedBackend === "local" ? localModel : currentConfig.localModel,
			localEndpoint: selectedBackend === "local" ? localEndpoint : currentConfig.localEndpoint,
			onboarding: {
				...currentConfig.onboarding,
				completed: false,
				schemaVersion: currentConfig.version,
				source: "first-run",
			},
		},
	};
}

/** Prompt for external server URL. Returns undefined if cancelled. */
async function promptServerEndpoint(ctx: VoiceUiContext): Promise<string | undefined> {
	ctx.ui.notify(
		[
			"External server mode — you manage your own transcription server.",
			"",
			"Compatible servers: whisper.cpp, faster-whisper-server, transcribe-rs",
			"Must implement POST /v1/audio/transcriptions (OpenAI-compatible).",
		].join("\n"),
		"info",
	);

	const customEndpoint = await ctx.ui.input(`Server URL (Enter for ${DEFAULT_LOCAL_ENDPOINT})`);
	const endpoint = customEndpoint?.trim() || DEFAULT_LOCAL_ENDPOINT;

	const serverCheck = await checkLocalServer(endpoint);
	if (!serverCheck.ok) {
		ctx.ui.notify(
			`Server not reachable at ${endpoint}\n${serverCheck.error || ""}\nVoice will work once the server is running.`,
			"warning",
		);
	} else {
		ctx.ui.notify(`Server detected at ${endpoint}`, "info");
	}

	return endpoint;
}

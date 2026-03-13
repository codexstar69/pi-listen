import { getAgentDir } from "@mariozechner/pi-coding-agent";
import * as fs from "node:fs";
import * as path from "node:path";

export const SETTINGS_KEY = "voice";
export const VOICE_CONFIG_VERSION = 2;

export type VoiceSettingsScope = "global" | "project";
export type VoiceConfigSource = VoiceSettingsScope | "default";

export interface VoiceOnboardingState {
	completed: boolean;
	schemaVersion: number;
	completedAt?: string;
	lastValidatedAt?: string;
	source?: "first-run" | "setup-command" | "migration" | "repair";
	skippedAt?: string;
}

export interface VoiceConfig {
	version: number;
	enabled: boolean;
	language: string;
	scope: VoiceSettingsScope;
	onboarding: VoiceOnboardingState;
	/** Deepgram API key — stored in config so it's available even when env var isn't set */
	deepgramApiKey?: string;
}

export interface LoadedVoiceConfig {
	config: VoiceConfig;
	source: VoiceConfigSource;
	globalSettingsPath: string;
	projectSettingsPath: string;
}

export interface ConfigPathOptions {
	agentDir?: string;
}

export const DEFAULT_CONFIG: VoiceConfig = {
	version: VOICE_CONFIG_VERSION,
	enabled: true,
	language: "en",
	scope: "global",
	deepgramApiKey: undefined,
	onboarding: {
		completed: false,
		schemaVersion: VOICE_CONFIG_VERSION,
	},
};

export function readJsonFile(filePath: string): any {
	try {
		if (!fs.existsSync(filePath)) return {};
		return JSON.parse(fs.readFileSync(filePath, "utf8"));
	} catch {
		return {};
	}
}

export function getGlobalSettingsPath(options: ConfigPathOptions = {}): string {
	return path.join(options.agentDir ?? getAgentDir(), "settings.json");
}

export function getProjectSettingsPath(cwd: string): string {
	return path.join(cwd, ".pi", "settings.json");
}

function normalizeOnboarding(input: any, fallbackCompleted: boolean): VoiceOnboardingState {
	const completed = typeof input?.completed === "boolean" ? input.completed : fallbackCompleted;
	return {
		completed,
		schemaVersion: Number.isFinite(input?.schemaVersion) ? Number(input.schemaVersion) : VOICE_CONFIG_VERSION,
		completedAt: typeof input?.completedAt === "string" ? input.completedAt : undefined,
		lastValidatedAt: typeof input?.lastValidatedAt === "string" ? input.lastValidatedAt : undefined,
		source: typeof input?.source === "string" ? input.source : fallbackCompleted ? "migration" : undefined,
		skippedAt: typeof input?.skippedAt === "string" ? input.skippedAt : undefined,
	};
}

function migrateConfig(rawVoice: any, source: VoiceConfigSource): VoiceConfig {
	if (!rawVoice || typeof rawVoice !== "object") {
		return structuredClone(DEFAULT_CONFIG);
	}

	// Legacy configs may have backend+model — treat that as completed onboarding
	const hasMeaningfulLegacySetup =
		(typeof rawVoice.backend === "string" && typeof rawVoice.model === "string") ||
		rawVoice.onboarding?.completed === true;
	const fallbackCompleted = hasMeaningfulLegacySetup;

	return {
		version: VOICE_CONFIG_VERSION,
		enabled: typeof rawVoice.enabled === "boolean" ? rawVoice.enabled : DEFAULT_CONFIG.enabled,
		language: typeof rawVoice.language === "string" ? rawVoice.language : DEFAULT_CONFIG.language,
		scope: (rawVoice.scope as VoiceSettingsScope | undefined) ?? (source === "project" ? "project" : "global"),
		deepgramApiKey: typeof rawVoice.deepgramApiKey === "string" ? rawVoice.deepgramApiKey : undefined,
		onboarding: normalizeOnboarding(rawVoice.onboarding, fallbackCompleted),
	};
}

export function loadConfigWithSource(cwd: string, options: ConfigPathOptions = {}): LoadedVoiceConfig {
	const globalSettingsPath = getGlobalSettingsPath(options);
	const projectSettingsPath = getProjectSettingsPath(cwd);
	const globalVoice = readJsonFile(globalSettingsPath)[SETTINGS_KEY];
	const projectVoice = readJsonFile(projectSettingsPath)[SETTINGS_KEY];

	if (projectVoice && typeof projectVoice === "object") {
		return {
			config: migrateConfig(projectVoice, "project"),
			source: "project",
			globalSettingsPath,
			projectSettingsPath,
		};
	}

	if (globalVoice && typeof globalVoice === "object") {
		return {
			config: migrateConfig(globalVoice, "global"),
			source: "global",
			globalSettingsPath,
			projectSettingsPath,
		};
	}

	return {
		config: structuredClone(DEFAULT_CONFIG),
		source: "default",
		globalSettingsPath,
		projectSettingsPath,
	};
}

function serializeConfig(config: VoiceConfig, scope: VoiceSettingsScope): VoiceConfig {
	return {
		...config,
		scope,
		onboarding: {
			...config.onboarding,
			schemaVersion: VOICE_CONFIG_VERSION,
		},
	};
}

export function saveConfig(
	config: VoiceConfig,
	scope: VoiceSettingsScope,
	cwd: string,
	options: ConfigPathOptions = {},
): string {
	const settingsPath = scope === "project" ? getProjectSettingsPath(cwd) : getGlobalSettingsPath(options);
	const settings = readJsonFile(settingsPath);
	settings[SETTINGS_KEY] = serializeConfig(config, scope);
	fs.mkdirSync(path.dirname(settingsPath), { recursive: true });
	fs.writeFileSync(settingsPath, JSON.stringify(settings, null, 2) + "\n");
	return settingsPath;
}

export function needsOnboarding(config: VoiceConfig, source: VoiceConfigSource): boolean {
	const skippedAt = config.onboarding.skippedAt ? Date.parse(config.onboarding.skippedAt) : Number.NaN;
	const deferWindowMs = 1000 * 60 * 60 * 24;
	const recentlyDeferred = Number.isFinite(skippedAt) && Date.now() - skippedAt < deferWindowMs;
	if (recentlyDeferred) return false;
	if (source === "default") return true;
	return !config.onboarding.completed;
}

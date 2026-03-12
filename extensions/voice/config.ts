import { getAgentDir } from "@mariozechner/pi-coding-agent";
import * as crypto from "node:crypto";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

export const SETTINGS_KEY = "voice";
export const VOICE_CONFIG_VERSION = 2;

export type VoiceBackend = "faster-whisper" | "moonshine" | "whisper-cpp" | "deepgram" | "parakeet" | "auto";
export type VoiceMode = "auto" | "local" | "api";
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
	mode: VoiceMode;
	backend: VoiceBackend;
	model: string;
	scope: VoiceSettingsScope;
	btwEnabled: boolean;
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

export interface SocketPathOptions {
	scope: VoiceSettingsScope;
	cwd: string;
	backend: VoiceBackend;
	model: string;
}

export const DEFAULT_CONFIG: VoiceConfig = {
	version: VOICE_CONFIG_VERSION,
	enabled: true,
	language: "en",
	mode: "auto",
	backend: "auto",
	model: "small",
	scope: "global",
	btwEnabled: true,
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

function inferMode(backend: VoiceBackend): VoiceMode {
	if (backend === "deepgram") return "api";
	if (backend === "auto") return "auto";
	return "local";
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

	const backend = (rawVoice.backend ?? DEFAULT_CONFIG.backend) as VoiceBackend;
	const hasMeaningfulLegacySetup =
		(typeof rawVoice.backend === "string" && typeof rawVoice.model === "string") ||
		rawVoice.onboarding?.completed === true;
	const fallbackCompleted = hasMeaningfulLegacySetup;

	return {
		version: VOICE_CONFIG_VERSION,
		enabled: typeof rawVoice.enabled === "boolean" ? rawVoice.enabled : DEFAULT_CONFIG.enabled,
		language: typeof rawVoice.language === "string" ? rawVoice.language : DEFAULT_CONFIG.language,
		mode: (rawVoice.mode as VoiceMode | undefined) ?? inferMode(backend),
		backend,
		model: typeof rawVoice.model === "string" ? rawVoice.model : DEFAULT_CONFIG.model,
		scope: (rawVoice.scope as VoiceSettingsScope | undefined) ?? (source === "project" ? "project" : "global"),
		btwEnabled: typeof rawVoice.btwEnabled === "boolean" ? rawVoice.btwEnabled : DEFAULT_CONFIG.btwEnabled,
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

export function getSocketPath(options: SocketPathOptions): string {
	const fingerprint = crypto
		.createHash("sha1")
		.update(JSON.stringify({
			scope: options.scope,
			cwd: options.scope === "project" ? path.resolve(options.cwd) : "global",
			backend: options.backend,
			model: options.model,
		}))
		.digest("hex")
		.slice(0, 12);
	return path.join(os.tmpdir(), `pi-voice-${fingerprint}.sock`);
}

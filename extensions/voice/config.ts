import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

function getAgentDir(): string {
	return path.join(os.homedir(), ".pi", "agent");
}

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

export type VoiceBackend = "deepgram" | "local";

export interface VoiceConfig {
	version: number;
	enabled: boolean;
	language: string;
	scope: VoiceSettingsScope;
	onboarding: VoiceOnboardingState;
	/** Deepgram API key — stored in config so it's available even when env var isn't set */
	deepgramApiKey?: string;
	/** Transcription backend — "deepgram" (cloud streaming) or "local" (batch via local server) */
	backend?: VoiceBackend;
	/** Local model ID (e.g. "whisper-small", "whisper-turbo", "parakeet-v3") */
	localModel?: string;
	/** Local transcription server URL (default: http://localhost:8080) */
	localEndpoint?: string;
	/** Global-only shortcut used to toggle recording without hold-to-talk */
	toggleShortcut?: string;

	// ─── TTS (text-to-speech) ─────────────────────────────────────────
	// All TTS fields are opt-in (default: TTS disabled). New in v6.0.0.

	/** Master TTS toggle. When false, /voice-speak is a no-op. */
	ttsEnabled?: boolean;
	/** "local" (sherpa-onnx, offline) or "deepgram" (cloud REST). */
	ttsBackend?: "local" | "deepgram";
	/** Active local TTS model id (e.g. "kitten-nano-en-v0_2"). */
	ttsLocalModel?: string;
	/**
	 * Local backend speaker id (numeric `sid` per the model's voice
	 * catalog). Type-validated at config load — strings get rejected and
	 * fall back to the model's defaultSid.
	 */
	ttsLocalVoiceId?: number;
	/**
	 * Deepgram backend voice id (e.g. "aura-2-thalia-en"). Type-validated
	 * at config load — numbers get rejected and fall back to
	 * the default Aura-2 voice.
	 */
	ttsDeepgramVoiceId?: string;
	/** Speech rate multiplier; range 0.5–2.0. Default 1.0. */
	ttsSpeed?: number;
	/**
	 * If true, the agent's responses are spoken automatically after each
	 * turn. v7.1.1: defaults to true so users get audio responses out of
	 * the box once TTS is enabled — disable via /voice-settings or
	 * `voice.ttsAutoSpeak = false` in settings.json.
	 */
	ttsAutoSpeak?: boolean;
	/**
	 * v7.1.1: If true, an STT transcription is automatically submitted to
	 * the agent (turn triggered) instead of just being placed in the
	 * editor. The user-spoken text bypasses the manual `[enter]` step.
	 * Defaults to false — preserves the v7.0.x behavior so existing
	 * users aren't surprised.
	 */
	autoSubmitOnSpeak?: boolean;
	/**
	 * v7.1.3: hold-to-talk activation delay in milliseconds. Range
	 * [200, 3000]. Default 700ms — snappy without being trigger-happy.
	 * Override via `/voice-hold-delay <ms>` or settings.json.
	 */
	holdThresholdMs?: number;
	/**
	 * BCP-47 language tag for TTS (overrides `language`). Useful when
	 * STT and TTS should use different languages — e.g. user dictates in
	 * English but wants Spanish read-back.
	 */
	ttsLanguage?: string;
	/**
	 * v6.1 feature flag — opt-in WebSocket streaming for sub-200ms TTFB
	 * on the Deepgram backend. v6.0 ships REST only; this is documented
	 * but ignored.
	 */
	ttsDeepgramStreaming?: boolean;
	/**
	 * Set to true after `tts-onboarding.maybeShowTtsOnboarding()` has
	 * shown its first-run hint, so subsequent /voice-speak-toggle calls
	 * don't re-spam the same notification. New in v7.0.0.
	 */
	ttsOnboardingShown?: boolean;
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
	backend: undefined, // undefined = "deepgram" (default)
	localModel: undefined,
	localEndpoint: undefined,
	toggleShortcut: "ctrl+shift+v",
	// TTS defaults — all opt-in
	ttsEnabled: false,
	ttsBackend: "local",
	ttsLocalModel: "kitten-nano-en-v0_2",
	ttsLocalVoiceId: 0,
	ttsDeepgramVoiceId: "aura-2-thalia-en",
	ttsSpeed: 1.0,
	ttsAutoSpeak: true,
	autoSubmitOnSpeak: false,
	holdThresholdMs: 700,
	ttsLanguage: undefined,
	ttsDeepgramStreaming: false,
	ttsOnboardingShown: false,
	onboarding: {
		completed: false,
		schemaVersion: VOICE_CONFIG_VERSION,
	},
};

export function readJsonFile(filePath: string): Record<string, unknown> {
	try {
		if (!fs.existsSync(filePath)) return {};
		return JSON.parse(fs.readFileSync(filePath, "utf8"));
	} catch (err) {
		process.stderr.write(`[pi-voice] Warning: failed to read ${filePath}: ${err instanceof Error ? err.message : err}\n`);
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
		backend: rawVoice.backend === "local" ? "local" : undefined,
		localModel: typeof rawVoice.localModel === "string" ? rawVoice.localModel : undefined,
		localEndpoint: typeof rawVoice.localEndpoint === "string" ? rawVoice.localEndpoint : undefined,
		toggleShortcut: source !== "project" && typeof rawVoice.toggleShortcut === "string"
			? rawVoice.toggleShortcut
			: DEFAULT_CONFIG.toggleShortcut,
		// TTS fields — type-validated; mismatched persisted values fall
		// back to safe defaults so a hand-edited config can't poison the
		// engine. Notably: ttsLocalVoiceId rejects strings (would crash
		// sherpa's numeric sid arg) and ttsDeepgramVoiceId rejects numbers
		// (would 4xx at the Deepgram REST layer).
		ttsEnabled: typeof rawVoice.ttsEnabled === "boolean" ? rawVoice.ttsEnabled : DEFAULT_CONFIG.ttsEnabled,
		ttsBackend: rawVoice.ttsBackend === "deepgram" ? "deepgram" : DEFAULT_CONFIG.ttsBackend,
		ttsLocalModel: typeof rawVoice.ttsLocalModel === "string" && rawVoice.ttsLocalModel
			? rawVoice.ttsLocalModel
			: DEFAULT_CONFIG.ttsLocalModel,
		ttsLocalVoiceId: typeof rawVoice.ttsLocalVoiceId === "number" && Number.isFinite(rawVoice.ttsLocalVoiceId)
			? rawVoice.ttsLocalVoiceId
			: DEFAULT_CONFIG.ttsLocalVoiceId,
		ttsDeepgramVoiceId: typeof rawVoice.ttsDeepgramVoiceId === "string" && rawVoice.ttsDeepgramVoiceId
			? rawVoice.ttsDeepgramVoiceId
			: DEFAULT_CONFIG.ttsDeepgramVoiceId,
		ttsSpeed: typeof rawVoice.ttsSpeed === "number" && Number.isFinite(rawVoice.ttsSpeed)
			? Math.max(0.5, Math.min(2.0, rawVoice.ttsSpeed))
			: DEFAULT_CONFIG.ttsSpeed,
		ttsAutoSpeak: typeof rawVoice.ttsAutoSpeak === "boolean" ? rawVoice.ttsAutoSpeak : DEFAULT_CONFIG.ttsAutoSpeak,
		autoSubmitOnSpeak: typeof rawVoice.autoSubmitOnSpeak === "boolean"
			? rawVoice.autoSubmitOnSpeak
			: DEFAULT_CONFIG.autoSubmitOnSpeak,
		holdThresholdMs: typeof rawVoice.holdThresholdMs === "number"
			&& Number.isFinite(rawVoice.holdThresholdMs)
			&& rawVoice.holdThresholdMs >= 200
			&& rawVoice.holdThresholdMs <= 3000
			? rawVoice.holdThresholdMs
			: DEFAULT_CONFIG.holdThresholdMs,
		ttsLanguage: typeof rawVoice.ttsLanguage === "string" && rawVoice.ttsLanguage
			? rawVoice.ttsLanguage
			: undefined,
		ttsDeepgramStreaming: typeof rawVoice.ttsDeepgramStreaming === "boolean"
			? rawVoice.ttsDeepgramStreaming
			: DEFAULT_CONFIG.ttsDeepgramStreaming,
		ttsOnboardingShown: typeof rawVoice.ttsOnboardingShown === "boolean"
			? rawVoice.ttsOnboardingShown
			: false,
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

const VALID_MODIFIERS = new Set(["ctrl", "shift", "alt", "meta", "cmd", "super"]);
const SHORTCUT_PATTERN = /^[a-z0-9+]+$/;

/** Validate a shortcut string like "ctrl+shift+v". Returns true if structurally valid. */
export function isValidShortcut(shortcut: string): boolean {
	if (typeof shortcut !== "string" || shortcut.length === 0 || !SHORTCUT_PATTERN.test(shortcut)) return false;
	const parts = shortcut.split("+");
	if (parts.length < 1 || parts.length > 4) return false;
	const key = parts[parts.length - 1]!;
	if (key.length === 0) return false;
	const mods = parts.slice(0, -1);
	return mods.every((m) => VALID_MODIFIERS.has(m));
}

/**
 * Resolve the toggle shortcut from global config at startup.
 * Returns the validated shortcut or the default if invalid/missing.
 * Reads disk once — caller should cache the result.
 */
export function loadGlobalToggleShortcut(options: ConfigPathOptions = {}): string {
	const fallback = DEFAULT_CONFIG.toggleShortcut || "ctrl+shift+v";
	try {
		const globalSettingsPath = getGlobalSettingsPath(options);
		const globalVoice = readJsonFile(globalSettingsPath)[SETTINGS_KEY];
		if (globalVoice && typeof globalVoice === "object" && typeof (globalVoice as any).toggleShortcut === "string") {
			const candidate = (globalVoice as any).toggleShortcut;
			if (isValidShortcut(candidate)) return candidate;
			process.stderr.write(`[pi-voice] Warning: invalid toggleShortcut "${candidate}" in settings, using default "${fallback}"\n`);
		}
	} catch {
		// Fall through to default
	}
	return fallback;
}

/** Check if a URL points to a loopback address (localhost/127.0.0.1/::1). */
export function isLoopbackEndpoint(endpoint: string): boolean {
	try {
		const url = new URL(endpoint);
		const proto = url.protocol;
		if (proto !== "http:" && proto !== "https:") return false;
		const host = url.hostname;
		return host === "localhost" || host === "127.0.0.1" || host === "::1" || host === "[::1]";
	} catch {
		return false;
	}
}

export function getSessionStartPersistedConfig({
	config,
	envDeepgramApiKey,
}: {
	config: VoiceConfig;
	envDeepgramApiKey?: string;
}): VoiceConfig {
	if (!envDeepgramApiKey || config.deepgramApiKey) {
		return config;
	}

	return {
		...config,
		deepgramApiKey: undefined,
	};
}

function serializeConfig(config: VoiceConfig, scope: VoiceSettingsScope): VoiceConfig {
	return {
		...config,
		scope,
		// Never persist API keys into project-scoped config — prevents accidental repo commits
		deepgramApiKey: scope === "project" ? undefined : config.deepgramApiKey,
		// Only allow loopback endpoints in project config — prevents mic audio exfiltration
		localEndpoint: (scope === "project" && config.localEndpoint && !isLoopbackEndpoint(config.localEndpoint))
			? undefined
			: config.localEndpoint,
		// Shortcut registration is static at extension load time — project-scoped overrides cannot apply
		toggleShortcut: scope === "project" ? undefined : config.toggleShortcut,
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
	// Atomic write: temp file + rename prevents corruption from partial writes
	const tmpPath = `${settingsPath}.${process.pid}.tmp`;
	try {
		fs.writeFileSync(tmpPath, JSON.stringify(settings, null, 2) + "\n");
		fs.renameSync(tmpPath, settingsPath);
	} finally {
		try { if (fs.existsSync(tmpPath)) fs.unlinkSync(tmpPath); } catch {}
	}
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

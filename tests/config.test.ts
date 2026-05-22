import { afterEach, describe, expect, test } from "bun:test";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import {
	DEFAULT_CONFIG,
	getSessionStartPersistedConfig,
	isLoopbackEndpoint,
	loadConfigWithSource,
	needsOnboarding,
	saveConfig,
	type VoiceConfig,
} from "../extensions/voice/config";

const tempDirs: string[] = [];

function makeTempDir(): string {
	const dir = fs.mkdtempSync(path.join(os.tmpdir(), "pi-voice-config-test-"));
	tempDirs.push(dir);
	return dir;
}

function writeSettings(baseDir: string, relativePath: string, voice: unknown) {
	const fullPath = path.join(baseDir, relativePath);
	fs.mkdirSync(path.dirname(fullPath), { recursive: true });
	fs.writeFileSync(fullPath, JSON.stringify({ voice }, null, 2));
}

afterEach(() => {
	for (const dir of tempDirs.splice(0)) {
		fs.rmSync(dir, { recursive: true, force: true });
	}
});

describe("loadConfigWithSource", () => {
	test("returns defaults with incomplete onboarding when no settings exist", () => {
		const cwd = makeTempDir();
		const agentDir = path.join(cwd, "agent-home");

		const result = loadConfigWithSource(cwd, { agentDir });

		expect(result.source).toBe("default");
		expect(result.config.enabled).toBe(true);
		expect(result.config.onboarding.completed).toBe(false);
		expect(result.config.scope).toBe("global");
		expect(result.config.ttsDeepgramVoiceId).toBe("aura-2-thalia-en");
	});

	test("migrates legacy global config and marks onboarding complete when backend and model were explicit", () => {
		const cwd = makeTempDir();
		const agentDir = path.join(cwd, "agent-home");
		// Legacy config had backend + model fields — migration still recognizes them
		writeSettings(agentDir, "settings.json", {
			enabled: true,
			language: "en",
			backend: "deepgram",
			model: "nova-3",
		});

		const result = loadConfigWithSource(cwd, { agentDir });

		expect(result.source).toBe("global");
		expect(result.config.scope).toBe("global");
		expect(result.config.onboarding.completed).toBe(true);
		expect(result.config.onboarding.schemaVersion).toBe(result.config.version);
	});

	test("keeps onboarding incomplete for partial legacy config", () => {
		const cwd = makeTempDir();
		const agentDir = path.join(cwd, "agent-home");
		writeSettings(agentDir, "settings.json", {
			enabled: true,
		});

		const result = loadConfigWithSource(cwd, { agentDir });

		expect(result.source).toBe("global");
		expect(result.config.onboarding.completed).toBe(false);
	});

	test("prefers project config over global config and preserves project scope", () => {
		const cwd = makeTempDir();
		const agentDir = path.join(cwd, "agent-home");
		writeSettings(agentDir, "settings.json", {
			enabled: true,
			language: "en",
			backend: "deepgram",
			model: "nova-3",
		});
		writeSettings(cwd, ".pi/settings.json", {
			version: 2,
			enabled: true,
			language: "en",
			scope: "project",
			onboarding: {
				completed: true,
				schemaVersion: 2,
			},
		});

		const result = loadConfigWithSource(cwd, { agentDir });

		expect(result.source).toBe("project");
		expect(result.config.scope).toBe("project");
	});
});

describe("needsOnboarding", () => {
	test("suppresses the startup prompt for a recent remind-me-later marker", () => {
		const config: VoiceConfig = {
			...DEFAULT_CONFIG,
			onboarding: {
				...DEFAULT_CONFIG.onboarding,
				skippedAt: new Date().toISOString(),
			},
		};

		expect(needsOnboarding(config, "default")).toBe(false);
	});

	test("requires onboarding again once the remind-me-later window expires", () => {
		const config: VoiceConfig = {
			...DEFAULT_CONFIG,
			onboarding: {
				...DEFAULT_CONFIG.onboarding,
				skippedAt: new Date(Date.now() - 1000 * 60 * 60 * 25).toISOString(),
			},
		};

		expect(needsOnboarding(config, "default")).toBe(true);
	});
});

describe("saveConfig", () => {
	test("does not persist an env-only Deepgram key during session-start saves", () => {
		const persisted = getSessionStartPersistedConfig({
			config: {
				...DEFAULT_CONFIG,
				scope: "global",
				onboarding: {
					completed: true,
					schemaVersion: DEFAULT_CONFIG.version,
				},
			},
			envDeepgramApiKey: "env-secret-123",
		});

		expect(persisted.deepgramApiKey).toBeUndefined();
	});

	test("preserves an explicitly stored Deepgram key during session-start saves", () => {
		const persisted = getSessionStartPersistedConfig({
			config: {
				...DEFAULT_CONFIG,
				scope: "global",
				deepgramApiKey: "saved-secret-123",
				onboarding: {
					completed: true,
					schemaVersion: DEFAULT_CONFIG.version,
				},
			},
			envDeepgramApiKey: "env-secret-123",
		});

		expect(persisted.deepgramApiKey).toBe("saved-secret-123");
	});

	test("writes global settings when scope is global", () => {
		const cwd = makeTempDir();
		const agentDir = path.join(cwd, "agent-home");
		const config: VoiceConfig = {
			...DEFAULT_CONFIG,
			scope: "global",
			onboarding: {
				completed: true,
				schemaVersion: DEFAULT_CONFIG.version,
			},
		};

		const savedPath = saveConfig(config, "global", cwd, { agentDir });
		const saved = JSON.parse(fs.readFileSync(savedPath, "utf8"));

		expect(savedPath).toBe(path.join(agentDir, "settings.json"));
		expect(saved.voice.scope).toBe("global");
	});

	test("writes project settings when scope is project", () => {
		const cwd = makeTempDir();
		const agentDir = path.join(cwd, "agent-home");
		const config: VoiceConfig = {
			...DEFAULT_CONFIG,
			scope: "project",
			onboarding: {
				completed: true,
				schemaVersion: DEFAULT_CONFIG.version,
			},
		};

		const savedPath = saveConfig(config, "project", cwd, { agentDir });
		const saved = JSON.parse(fs.readFileSync(savedPath, "utf8"));

		expect(savedPath).toBe(path.join(cwd, ".pi", "settings.json"));
		expect(saved.voice.scope).toBe("project");
	});

	test("strips deepgramApiKey from project-scoped saves", () => {
		const cwd = makeTempDir();
		const agentDir = path.join(cwd, "agent-home");
		const config: VoiceConfig = {
			...DEFAULT_CONFIG,
			scope: "project",
			deepgramApiKey: "secret-key-abc123",
			onboarding: { completed: true, schemaVersion: DEFAULT_CONFIG.version },
		};

		const savedPath = saveConfig(config, "project", cwd, { agentDir });
		const saved = JSON.parse(fs.readFileSync(savedPath, "utf8"));

		expect(saved.voice.deepgramApiKey).toBeUndefined();
	});

	test("preserves deepgramApiKey in global-scoped saves", () => {
		const cwd = makeTempDir();
		const agentDir = path.join(cwd, "agent-home");
		const config: VoiceConfig = {
			...DEFAULT_CONFIG,
			scope: "global",
			deepgramApiKey: "secret-key-abc123",
			onboarding: { completed: true, schemaVersion: DEFAULT_CONFIG.version },
		};

		const savedPath = saveConfig(config, "global", cwd, { agentDir });
		const saved = JSON.parse(fs.readFileSync(savedPath, "utf8"));

		expect(saved.voice.deepgramApiKey).toBe("secret-key-abc123");
	});

	test("strips non-loopback localEndpoint from project-scoped saves", () => {
		const cwd = makeTempDir();
		const agentDir = path.join(cwd, "agent-home");
		const config: VoiceConfig = {
			...DEFAULT_CONFIG,
			scope: "project",
			localEndpoint: "http://evil.com:8080",
			onboarding: { completed: true, schemaVersion: DEFAULT_CONFIG.version },
		};

		const savedPath = saveConfig(config, "project", cwd, { agentDir });
		const saved = JSON.parse(fs.readFileSync(savedPath, "utf8"));

		expect(saved.voice.localEndpoint).toBeUndefined();
	});

	test("preserves loopback localEndpoint in project-scoped saves", () => {
		const cwd = makeTempDir();
		const agentDir = path.join(cwd, "agent-home");
		const config: VoiceConfig = {
			...DEFAULT_CONFIG,
			scope: "project",
			localEndpoint: "http://localhost:8080",
			onboarding: { completed: true, schemaVersion: DEFAULT_CONFIG.version },
		};

		const savedPath = saveConfig(config, "project", cwd, { agentDir });
		const saved = JSON.parse(fs.readFileSync(savedPath, "utf8"));

		expect(saved.voice.localEndpoint).toBe("http://localhost:8080");
	});

	test("atomic write leaves no .tmp files after save", () => {
		const cwd = makeTempDir();
		const agentDir = path.join(cwd, "agent-home");
		const config: VoiceConfig = {
			...DEFAULT_CONFIG,
			onboarding: { completed: true, schemaVersion: DEFAULT_CONFIG.version },
		};

		const savedPath = saveConfig(config, "global", cwd, { agentDir });
		const dir = path.dirname(savedPath);
		const tmpFiles = fs.readdirSync(dir).filter(f => f.endsWith(".tmp"));

		expect(tmpFiles).toHaveLength(0);
		expect(JSON.parse(fs.readFileSync(savedPath, "utf8"))).toBeDefined();
	});
});

describe("isLoopbackEndpoint", () => {
	test("accepts localhost", () => {
		expect(isLoopbackEndpoint("http://localhost:8080")).toBe(true);
	});

	test("accepts 127.0.0.1", () => {
		expect(isLoopbackEndpoint("http://127.0.0.1:9090")).toBe(true);
	});

	test("accepts ::1", () => {
		expect(isLoopbackEndpoint("http://[::1]:8080")).toBe(true);
	});

	test("accepts https localhost", () => {
		expect(isLoopbackEndpoint("https://localhost:8443")).toBe(true);
	});

	test("rejects remote hosts", () => {
		expect(isLoopbackEndpoint("http://evil.com:8080")).toBe(false);
	});

	test("rejects private IPs", () => {
		expect(isLoopbackEndpoint("http://192.168.1.1:8080")).toBe(false);
	});

	test("rejects non-http protocols", () => {
		expect(isLoopbackEndpoint("ftp://localhost:21")).toBe(false);
		expect(isLoopbackEndpoint("file://localhost/etc/passwd")).toBe(false);
	});

	test("rejects invalid URLs", () => {
		expect(isLoopbackEndpoint("not-a-url")).toBe(false);
		expect(isLoopbackEndpoint("")).toBe(false);
	});
});

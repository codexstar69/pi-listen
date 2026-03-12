/**
 * pi-voice — Voice input + BTW side conversations for Pi CLI.
 *
 * Features:
 *   1. Hold-spacebar to talk (Kitty protocol key release detection)
 *      Fallback: Ctrl+Shift+V toggle for non-Kitty terminals
 *   2. BTW side conversations (/btw <msg>, /btw:new, /btw:clear, /btw:inject, /btw:summarize)
 *   3. Voice → BTW glue: Ctrl+Shift+B = hold to record → auto-send as /btw
 *
 * Records audio via SoX, transcribes via persistent daemon (daemon.py) or fallback subprocess.
 * STT backends: faster-whisper, moonshine, whisper.cpp, deepgram, parakeet.
 *
 * Config in ~/.pi/agent/settings.json or <project>/.pi/settings.json:
 * {
 *   "voice": {
 *     "enabled": true,
 *     "language": "en",
 *     "backend": "faster-whisper",
 *     "model": "small"
 *   }
 * }
 */

import type {
	ExtensionAPI,
	ExtensionContext,
	ExtensionCommandContext,
	MessageRenderOptions,
} from "@mariozechner/pi-coding-agent";
import type { CustomMessage } from "@mariozechner/pi-coding-agent/dist/core/messages.js";
import { streamSimple } from "@mariozechner/pi-ai";
import type { AssistantMessageEvent } from "@mariozechner/pi-ai";
import { isKeyRelease, isKeyRepeat, matchesKey } from "@mariozechner/pi-tui";
import type { Component } from "@mariozechner/pi-tui";
import type { Theme } from "@mariozechner/pi-coding-agent/dist/modes/interactive/theme/theme.js";
import { spawn, spawnSync, type ChildProcess } from "node:child_process";
import * as fs from "node:fs";
import * as net from "node:net";
import * as os from "node:os";
import * as path from "node:path";
import {
	DEFAULT_CONFIG,
	getSocketPath,
	loadConfigWithSource,
	needsOnboarding,
	saveConfig,
	type VoiceConfig,
	type VoiceSettingsScope,
} from "./voice/config";
import { getModelReadiness, recommendVoiceSetup, scanEnvironment } from "./voice/diagnostics";
import { finalizeOnboardingConfig, promptFirstRunOnboarding, runVoiceOnboarding } from "./voice/onboarding";
import { buildProvisioningPlan } from "./voice/install";

// ─── Types ───────────────────────────────────────────────────────────────────

type VoiceState = "idle" | "recording" | "transcribing";

interface BtwExchange {
	question: string;
	thinking?: string;
	answer: string;
	model?: string;
	timestamp: number;
}

// ─── Constants ───────────────────────────────────────────────────────────────

const SAMPLE_RATE = 16000;
const EXT_DIR = path.dirname(new URL(import.meta.url).pathname);
const PROJECT_ROOT = path.join(EXT_DIR, "..");
const DAEMON_SCRIPT = path.join(PROJECT_ROOT, "daemon.py");
const TRANSCRIBE_SCRIPT = path.join(PROJECT_ROOT, "transcribe.py");

function commandExists(cmd: string): boolean {
	return spawnSync("which", [cmd], { stdio: "pipe", timeout: 3000 }).status === 0;
}

// ─── Daemon Communication ────────────────────────────────────────────────────

let activeSocketPath = getSocketPath({
	scope: DEFAULT_CONFIG.scope,
	cwd: process.cwd(),
	backend: DEFAULT_CONFIG.backend,
	model: DEFAULT_CONFIG.model,
});

function updateSocketPath(config: VoiceConfig, cwd: string) {
	activeSocketPath = getSocketPath({
		scope: config.scope,
		cwd,
		backend: config.backend,
		model: config.model,
	});
}

function daemonSend(
	cmd: Record<string, unknown>,
	timeout = 30000,
	socketPath = activeSocketPath,
): Promise<Record<string, unknown>> {
	return new Promise((resolve) => {
		const sock = net.createConnection(socketPath);
		let buf = "";
		const timer = setTimeout(() => {
			sock.destroy();
			resolve({ error: "Daemon timeout" });
		}, timeout);

		sock.on("connect", () => {
			sock.write(JSON.stringify(cmd) + "\n");
		});
		sock.on("data", (chunk) => {
			buf += chunk.toString();
			if (buf.includes("\n")) {
				clearTimeout(timer);
				sock.destroy();
				try {
					resolve(JSON.parse(buf.trim()));
				} catch {
					resolve({ error: "Invalid daemon response" });
				}
			}
		});
		sock.on("error", () => {
			clearTimeout(timer);
			resolve({ error: "Daemon not running" });
		});
	});
}

async function isDaemonRunning(socketPath = activeSocketPath): Promise<boolean> {
	const resp = await daemonSend({ cmd: "ping" }, 2000, socketPath);
	return resp.status === "ok";
}

async function ensureDaemon(config: VoiceConfig): Promise<boolean> {
	if (await isDaemonRunning(activeSocketPath)) {
		const status = await daemonSend({ cmd: "status" }, 3000, activeSocketPath);
		// When backend is 'auto', accept any loaded backend — the daemon already
		// resolved 'auto' to a concrete backend, so we don't need to reload.
		if (config.backend === "auto" || (status.backend === config.backend && status.model === config.model)) return true;
		const reloaded = await daemonSend({
			cmd: "load",
			backend: config.backend,
			model: config.model,
		}, 15000, activeSocketPath);
		return !reloaded.error;
	}

	return new Promise((resolve) => {
		const args = [DAEMON_SCRIPT, "start", "--socket", activeSocketPath];
		if (config.backend !== "auto") {
			args.push("--backend", config.backend);
		}
		if (config.model) {
			args.push("--model", config.model);
		}

		const proc = spawn("python3", args, {
			stdio: ["pipe", "pipe", "pipe"],
			detached: true,
		});

		let started = false;
		proc.stdout?.on("data", (d) => {
			const line = d.toString().trim();
			try {
				const msg = JSON.parse(line);
				if (msg.status === "started") {
					started = true;
					proc.unref();
					resolve(true);
				}
			} catch {}
		});

		proc.on("error", () => resolve(false));

		// Timeout: if daemon doesn't start in 10s, kill orphan and fall back
		setTimeout(() => {
			if (!started) {
				try { proc.kill(); } catch {}
				resolve(false);
			}
		}, 10000);
	});
}

// ─── Audio Recording ─────────────────────────────────────────────────────────

let recProcess: ChildProcess | null = null;

function startRecordingToFile(outPath: string): boolean {
	if (recProcess) {
		recProcess.kill("SIGTERM");
		recProcess = null;
	}

	if (!commandExists("rec")) return false;

	recProcess = spawn("rec", [
		"-q", "-r", String(SAMPLE_RATE), "-c", "1", "-b", "16", outPath,
	], { stdio: ["pipe", "pipe", "pipe"] });

	recProcess.stderr?.on("data", () => {});
	recProcess.on("error", () => { recProcess = null; });
	return true;
}

function stopRecording(): Promise<void> {
	return new Promise((resolve) => {
		if (!recProcess) { resolve(); return; }
		recProcess.on("close", () => { recProcess = null; resolve(); });
		recProcess.kill("SIGTERM");
		setTimeout(() => {
			if (recProcess) { recProcess.kill("SIGKILL"); recProcess = null; }
			resolve();
		}, 2000);
	});
}

// ─── Transcription (daemon or fallback) ──────────────────────────────────────

async function transcribeAudio(
	audioPath: string,
	config: VoiceConfig,
): Promise<{ text: string; duration: number; error?: string }> {
	// Try daemon first
	if (await isDaemonRunning()) {
		const resp = await daemonSend({
			cmd: "transcribe",
			audio: audioPath,
			backend: config.backend,
			model: config.model,
			language: config.language,
			vad: true,
		});
		if (!resp.error) {
			return resp as { text: string; duration: number };
		}
	}

	// Fallback: direct subprocess
	return new Promise((resolve) => {
		const args = [TRANSCRIBE_SCRIPT, "--language", config.language, audioPath];
		if (config.backend !== "auto") args.splice(1, 0, "--backend", config.backend);
		if (config.model) args.splice(1, 0, "--model", config.model);

		const proc = spawn("python3", args, { stdio: ["pipe", "pipe", "pipe"] });
		let stdout = "";
		let stderr = "";
		proc.stdout?.on("data", (d: Buffer) => { stdout += d.toString(); });
		proc.stderr?.on("data", (d: Buffer) => { stderr += d.toString(); });
		proc.on("close", (code) => {
			try { resolve(JSON.parse(stdout.trim())); }
			catch { resolve({ text: "", duration: 0, error: stderr || `Exit ${code}` }); }
		});
		proc.on("error", (err) => resolve({ text: "", duration: 0, error: err.message }));
	});
}

// ─── Extension ───────────────────────────────────────────────────────────────

export default function (pi: ExtensionAPI) {
	let config = DEFAULT_CONFIG;
	let configSource: VoiceSettingsScope | "default" = "default";
	let currentCwd = process.cwd();
	let voiceState: VoiceState = "idle";
	let ctx: ExtensionContext | null = null;
	let tempFile: string | null = null;
	let recordingStart = 0;
	let statusTimer: ReturnType<typeof setInterval> | null = null;
	let terminalInputUnsub: (() => void) | null = null;
	let isHolding = false;

	// ─── BTW State ───────────────────────────────────────────────────────────

	let btwThread: BtwExchange[] = [];
	let btwWidgetVisible = false;

	// ─── Voice UI ────────────────────────────────────────────────────────────

	function updateVoiceStatus() {
		if (!ctx?.hasUI) return;
		switch (voiceState) {
			case "idle": {
				if (!config.enabled) {
					ctx.ui.setStatus("voice", undefined);
					break;
				}
				const modeTag = !config.onboarding.completed
					? "SETUP"
					: config.mode === "api"
						? "API"
						: config.mode === "local"
							? "LOCAL"
							: "AUTO";
				ctx.ui.setStatus("voice", `MIC ${modeTag}`);
				break;
			}
			case "recording": {
				const secs = Math.round((Date.now() - recordingStart) / 1000);
				ctx.ui.setStatus("voice", `REC ${secs}s`);
				break;
			}
			case "transcribing":
				ctx.ui.setStatus("voice", "STT...");
				break;
		}
	}

	function setVoiceState(newState: VoiceState) {
		voiceState = newState;
		updateVoiceStatus();
	}

	function voiceCleanup() {
		if (statusTimer) { clearInterval(statusTimer); statusTimer = null; }
		if (recProcess) { recProcess.kill("SIGTERM"); recProcess = null; }
		if (tempFile) { try { fs.unlinkSync(tempFile); } catch {} tempFile = null; }
		isHolding = false;
		setVoiceState("idle");
	}

	async function finalizeAndSaveSetup(
		uiCtx: ExtensionContext | ExtensionCommandContext,
		nextConfig: VoiceConfig,
		selectedScope: VoiceSettingsScope,
		summaryLines: string[],
		source: "first-run" | "setup-command",
	) {
		updateSocketPath(nextConfig, currentCwd);
		const diagnostics = scanEnvironment(TRANSCRIBE_SCRIPT);
		const provisioningPlan = buildProvisioningPlan(nextConfig, diagnostics);
		let validated = provisioningPlan.ready;
		if (validated && nextConfig.enabled) {
			validated = await ensureDaemon(nextConfig);
		}

		config = finalizeOnboardingConfig(nextConfig, { validated, source });
		configSource = selectedScope;
		updateSocketPath(config, currentCwd);
		const savedPath = saveConfig(config, selectedScope, currentCwd);
		const statusHeader = validated ? "Voice setup complete." : "Voice setup saved, but action is still required.";
		uiCtx.ui.notify([
			statusHeader,
			...summaryLines,
			"",
			`Saved to ${savedPath}`,
		].join("\n"), validated ? "info" : "warning");
	}

	// ─── Voice: Start / Stop / Transcribe ────────────────────────────────────

	async function startVoiceRecording(target: "editor" | "btw" = "editor"): Promise<boolean> {
		if (voiceState !== "idle" || !ctx) return false;

		tempFile = path.join(os.tmpdir(), `pi-voice-${Date.now()}.wav`);
		if (!startRecordingToFile(tempFile)) {
			ctx.ui.notify("Voice requires SoX. Install: brew install sox", "error");
			return false;
		}

		recordingStart = Date.now();
		setVoiceState("recording");
		statusTimer = setInterval(() => {
			if (voiceState === "recording") updateVoiceStatus();
		}, 1000);

		if (ctx.hasUI) {
			ctx.ui.setWidget("voice-recording", [
				target === "btw"
					? " BTW Recording... release to send"
					: " Recording... release to transcribe",
			], { placement: "aboveEditor" });
		}
		return true;
	}

	async function stopVoiceRecording(target: "editor" | "btw" = "editor") {
		if (voiceState !== "recording" || !ctx) return;
		if (statusTimer) { clearInterval(statusTimer); statusTimer = null; }

		const elapsed = ((Date.now() - recordingStart) / 1000).toFixed(1);
		const audioFile = tempFile; // capture before cleanup can null it
		setVoiceState("transcribing");
		ctx.ui.setWidget("voice-recording", undefined);

		await stopRecording();

		if (!audioFile || !fs.existsSync(audioFile)) {
			ctx.ui.notify("No audio recorded.", "warning");
			setVoiceState("idle");
			return;
		}

		const stats = fs.statSync(audioFile);
		if (stats.size < 1000) {
			ctx.ui.notify("Recording too short.", "warning");
			try { fs.unlinkSync(audioFile); } catch {}
			tempFile = null;
			setVoiceState("idle");
			return;
		}

		// Ensure daemon is up before transcribing — await so the warm path
		// is available for this request instead of falling through to the
		// cold subprocess fallback.
		await ensureDaemon(config).catch(() => {});

		const result = await transcribeAudio(audioFile, config);
		try { fs.unlinkSync(audioFile); } catch {}
		if (tempFile === audioFile) tempFile = null;

		if (result.error) {
			ctx.ui.notify(`STT error: ${result.error}`, "error");
			setVoiceState("idle");
			return;
		}

		const transcript = (result.text || "").trim();
		if (!transcript) {
			ctx.ui.notify("No speech detected.", "warning");
			setVoiceState("idle");
			return;
		}

		if (target === "btw") {
			await handleBtw(transcript);
		} else {
			// Inject into editor
			if (ctx.hasUI) {
				const existing = ctx.ui.getEditorText();
				ctx.ui.setEditorText(existing ? existing + " " + transcript : transcript);
				ctx.ui.notify(
					`STT (${elapsed}s): ${transcript.slice(0, 80)}${transcript.length > 80 ? "..." : ""}`,
					"info",
				);
			}
		}

		setVoiceState("idle");
	}

	// ─── Hold-to-talk via Kitty protocol ─────────────────────────────────────

	function setupHoldToTalk() {
		if (!ctx?.hasUI) return;

		// Remove previous listener
		if (terminalInputUnsub) { terminalInputUnsub(); terminalInputUnsub = null; }

		terminalInputUnsub = ctx.ui.onTerminalInput((data: string) => {
			if (!config.enabled) return undefined;

			// Hold SPACE → talk → release → transcribe to editor
			if (matchesKey(data, "space")) {
				// Only activate when editor is empty (avoid conflicting with typing)
				const editorText = ctx?.hasUI ? ctx.ui.getEditorText() : "";
				if (editorText && editorText.trim().length > 0) return undefined;

				if (isKeyRelease(data)) {
					if (isHolding) {
						isHolding = false;
						stopVoiceRecording("editor");
						return { consume: true };
					}
					return undefined;
				}

				if (isKeyRepeat(data)) {
					if (isHolding) return { consume: true };
					return undefined;
				}

				// Key press — start recording
				if (voiceState === "idle" && !isHolding) {
					isHolding = true;
					startVoiceRecording("editor").then((ok) => {
						if (!ok) isHolding = false;
					});
					return { consume: true };
				}

				if (isHolding) return { consume: true };
				return undefined;
			}

			// Hold Ctrl+Shift+B → talk → release → auto-btw
			if (matchesKey(data, "ctrl+shift+b")) {
				if (isKeyRelease(data)) {
					if (isHolding) {
						isHolding = false;
						stopVoiceRecording("btw");
						return { consume: true };
					}
					return undefined;
				}

				if (isKeyRepeat(data)) {
					if (isHolding) return { consume: true };
					return undefined;
				}

				if (voiceState === "idle" && !isHolding) {
					isHolding = true;
					startVoiceRecording("btw").then((ok) => {
						if (!ok) isHolding = false;
					});
					return { consume: true };
				}

				if (isHolding) return { consume: true };
				return undefined;
			}

			// Any other key while holding = cancel
			if (isHolding && voiceState === "recording") {
				// Don't cancel on modifier-only events
				return undefined;
			}

			return undefined;
		});
	}

	// ─── BTW: Side Conversations ─────────────────────────────────────────────

	function buildBtwContext(): string {
		// Build context from main session + btw thread
		const systemPrompt = ctx?.getSystemPrompt() ?? "";
		let btwContext = "You are a helpful side-channel assistant. ";
		btwContext += "The user is having a parallel conversation while their main Pi agent works. ";
		btwContext += "Be concise and direct. This is a quick side question, not the main task.\n\n";

		if (systemPrompt) {
			btwContext += `Main session context (for reference):\n${systemPrompt.slice(0, 2000)}\n\n`;
		}

		if (btwThread.length > 0) {
			btwContext += "Prior btw thread:\n";
			for (const ex of btwThread) {
				btwContext += `User: ${ex.question}\nAssistant: ${ex.answer}\n\n`;
			}
		}

		return btwContext;
	}

	function updateBtwWidget() {
		if (!ctx?.hasUI) return;

		if (!btwWidgetVisible || btwThread.length === 0) {
			ctx.ui.setWidget("btw", undefined);
			return;
		}

		const last = btwThread[btwThread.length - 1];
		const lines: string[] = [
			` BTW (${btwThread.length} exchange${btwThread.length > 1 ? "s" : ""})`,
			"",
		];

		// Show last exchange
		lines.push(`  Q: ${last.question.slice(0, 100)}${last.question.length > 100 ? "..." : ""}`);
		const answerLines = last.answer.split("\n");
		for (const line of answerLines.slice(0, 8)) {
			lines.push(`  ${line}`);
		}
		if (answerLines.length > 8) lines.push("  ...");

		lines.push("");
		lines.push("  /btw:clear to dismiss | /btw:inject to send to agent");

		ctx.ui.setWidget("btw", lines, { placement: "aboveEditor" });
	}

	async function handleBtw(message: string) {
		if (!ctx) return;

		btwWidgetVisible = true;

		// Show thinking state
		ctx.ui.setWidget("btw", [
			" BTW",
			"",
			`  Q: ${message.slice(0, 100)}${message.length > 100 ? "..." : ""}`,
			"",
			"  Thinking...",
		], { placement: "aboveEditor" });

		// Build context for LLM
		const btwContext = buildBtwContext();

		// Use the model registry to get current model
		const model = ctx.model;
		if (!model) {
			const exchange: BtwExchange = {
				question: message,
				answer: "(No model available — is the session active?)",
				timestamp: Date.now(),
			};
			btwThread.push(exchange);
			pi.appendEntry("btw", exchange);
			updateBtwWidget();
			return;
		}

		try {
			// Stream the response
			let answer = "";
			const eventStream = streamSimple(model, {
				systemPrompt: btwContext,
				messages: [{ role: "user" as const, content: message, timestamp: Date.now() }],
			});

			for await (const event of eventStream) {
				if (event.type === "text_delta") {
					answer += event.delta;
				} else if (event.type === "error") {
					answer = "(Error during BTW streaming)";
					break;
				} else if (event.type === "done") {
					break;
				}

				// Update widget with streaming response
				const displayLines: string[] = [
					` BTW`,
					"",
					`  Q: ${message.slice(0, 80)}`,
					"",
				];
				for (const line of answer.split("\n").slice(0, 10)) {
					displayLines.push(`  ${line}`);
				}
				ctx!.ui.setWidget("btw", displayLines, { placement: "aboveEditor" });
			}

			const exchange: BtwExchange = {
				question: message,
				answer: answer.trim() || "(empty response)",
				model: model.name,
				timestamp: Date.now(),
			};

			btwThread.push(exchange);
			pi.appendEntry("btw", exchange);
			updateBtwWidget();
		} catch (err: any) {
			// Fallback: send as a follow-up message to the main agent
			const exchange: BtwExchange = {
				question: message,
				answer: `(BTW streaming failed: ${err.message}. Falling back to sendUserMessage.)`,
				timestamp: Date.now(),
			};
			btwThread.push(exchange);
			pi.appendEntry("btw", exchange);
			updateBtwWidget();

			// Use sendUserMessage as alternative
			pi.sendUserMessage(
				`[BTW question]: ${message}`,
				{ deliverAs: "followUp" },
			);
		}
	}

	// ─── Shortcuts ───────────────────────────────────────────────────────────

	// Ctrl+Shift+V = toggle voice (fallback for non-Kitty terminals)
	pi.registerShortcut("ctrl+shift+v", {
		description: "Toggle voice recording (start/stop)",
		handler: async (handlerCtx) => {
			ctx = handlerCtx;
			if (!config.enabled) {
				handlerCtx.ui.notify("Voice disabled. Use /voice on", "warning");
				return;
			}
			if (voiceState === "idle") {
				await startVoiceRecording("editor");
			} else if (voiceState === "recording") {
				await stopVoiceRecording("editor");
			}
		},
	});

	// ─── Lifecycle ───────────────────────────────────────────────────────────

	pi.on("session_start", async (_event, startCtx) => {
		ctx = startCtx;
		currentCwd = startCtx.cwd;
		const loaded = loadConfigWithSource(startCtx.cwd);
		config = loaded.config;
		configSource = loaded.source;
		updateSocketPath(config, currentCwd);
		updateVoiceStatus();
		setupHoldToTalk();

		if (needsOnboarding(config, configSource) && startCtx.hasUI) {
			const decision = await promptFirstRunOnboarding(startCtx);
			if (decision.action === "start") {
				const diagnostics = scanEnvironment(TRANSCRIBE_SCRIPT);
				const result = await runVoiceOnboarding(startCtx, config, diagnostics);
				if (result) {
					await finalizeAndSaveSetup(startCtx, result.config, result.selectedScope, result.summaryLines, "first-run");
				} else {
					config = {
						...config,
						onboarding: {
							...config.onboarding,
							skippedAt: new Date().toISOString(),
						},
					};
					saveConfig(config, config.scope, currentCwd);
					startCtx.ui.notify("Voice setup skipped. Run /voice setup anytime.", "warning");
				}
			} else {
				config = {
					...config,
					onboarding: {
						...config.onboarding,
						skippedAt: new Date().toISOString(),
					},
				};
				saveConfig(config, config.scope, currentCwd);
				startCtx.ui.notify("Voice setup skipped. Run /voice setup anytime.", "warning");
			}
			return;
		}

		// Start daemon in background only after the user has completed onboarding.
		if (config.enabled && config.onboarding.completed && !needsOnboarding(config, configSource)) {
			ensureDaemon(config).catch(() => {});
		}
	});

	pi.on("session_shutdown", async () => {
		voiceCleanup();
		if (terminalInputUnsub) { terminalInputUnsub(); terminalInputUnsub = null; }
	});

	pi.on("session_switch", async (_event, switchCtx) => {
		ctx = switchCtx;
		currentCwd = switchCtx.cwd;
		const loaded = loadConfigWithSource(switchCtx.cwd);
		config = loaded.config;
		configSource = loaded.source;
		updateSocketPath(config, currentCwd);
		btwThread = [];
		btwWidgetVisible = false;
		if (ctx.hasUI) ctx.ui.setWidget("btw", undefined);
		setupHoldToTalk();
		updateVoiceStatus();
	});

	// ─── BTW Message Renderer ────────────────────────────────────────────────

	pi.registerMessageRenderer("btw", (message: CustomMessage<BtwExchange>, _options: MessageRenderOptions, _theme: Theme): Component | undefined => {
		const ex = message.details;
		if (!ex) return undefined;
		return {
			invalidate() {},
			render(width: number): string[] {
				const maxW = Math.max(width - 4, 20);
				const q = ex.question.length > maxW ? ex.question.slice(0, maxW) + "…" : ex.question;
				const a = ex.answer.length > 200 ? ex.answer.slice(0, 200) + "…" : ex.answer;
				return [
					`BTW Q: ${q}`,
					`BTW A: ${a}`,
				];
			},
		};
	});

	// ─── /voice command ──────────────────────────────────────────────────────

	pi.registerCommand("voice", {
		description: "Voice input: /voice [on|off|test|info|setup|reconfigure|doctor|backends|daemon]",
		handler: async (args, cmdCtx) => {
			ctx = cmdCtx;
			const sub = (args || "").trim().toLowerCase();

			if (sub === "on") {
				config.enabled = true;
				updateVoiceStatus();
				setupHoldToTalk();
				ensureDaemon(config).catch(() => {});
				cmdCtx.ui.notify("Voice enabled. Hold SPACE (empty editor) to record.", "info");
				return;
			}

			if (sub === "off") {
				config.enabled = false;
				voiceCleanup();
				if (terminalInputUnsub) { terminalInputUnsub(); terminalInputUnsub = null; }
				ctx.ui.setStatus("voice", undefined);
				cmdCtx.ui.notify("Voice disabled.", "info");
				return;
			}

			if (sub === "test") {
				cmdCtx.ui.notify("Testing voice setup...", "info");
				const diagnostics = scanEnvironment(TRANSCRIBE_SCRIPT);
				const daemonUp = await isDaemonRunning();
				const provisioningPlan = buildProvisioningPlan(config, diagnostics);
				const selectedBackend = diagnostics.backends.find((backend) => backend.name === config.backend);
				const modelReadiness = getModelReadiness(selectedBackend, config.model);

				const lines = [
					"Voice test:",
					`  mode: ${config.mode}`,
					`  backend: ${config.backend}`,
					`  model: ${config.model}`,
					`  model status: ${modelReadiness}`,
					`  language: ${config.language}`,
					`  onboarding: ${config.onboarding.completed ? "complete" : "incomplete"}`,
					`  python3: ${diagnostics.hasPython ? "OK" : "missing"}`,
					`  sox/rec: ${diagnostics.hasSox ? "OK" : "missing"}`,
					`  daemon: ${daemonUp ? "running" : "not running"}`,
				];

				if (diagnostics.hasSox) {
					const testFile = path.join(os.tmpdir(), "pi-voice-test.wav");
					const testProc = spawn("rec", ["-q", "-r", "16000", "-c", "1", "-b", "16", "-d", "1", testFile], { stdio: "pipe" });
					await new Promise<void>((resolve) => {
						testProc.on("close", () => resolve());
						setTimeout(() => { testProc.kill(); resolve(); }, 2000);
					});
					if (fs.existsSync(testFile)) {
						lines.push(`  mic sample: OK (${fs.statSync(testFile).size} bytes)`);
						try { fs.unlinkSync(testFile); } catch {}
					} else {
						lines.push("  mic sample: missing audio");
					}
				}

				lines.push("", "Suggested commands:");
				lines.push(...(provisioningPlan.commands.length > 0 ? provisioningPlan.commands.map((command) => `  - ${command}`) : ["  - none"]));
				if (provisioningPlan.manualSteps.length > 0) {
					lines.push("", "Manual steps:");
					lines.push(...provisioningPlan.manualSteps.map((step) => `  - ${step}`));
				}

				cmdCtx.ui.notify(lines.join("\n"), provisioningPlan.ready ? "info" : "warning");
				return;
			}

			if (sub === "info") {
				const daemonUp = await isDaemonRunning();
				let daemonInfo = "";
				if (daemonUp) {
					const status = await daemonSend({ cmd: "status" });
					daemonInfo = `\n  daemon:   running (pid ${status.pid}, ${status.requests} requests, ${status.uptime}s uptime)`;
				}
				const diagnostics = scanEnvironment(TRANSCRIBE_SCRIPT);
				const selectedBackend = diagnostics.backends.find((backend) => backend.name === config.backend);
				const modelReadiness = getModelReadiness(selectedBackend, config.model);

				cmdCtx.ui.notify([
					`Voice config:`,
					`  enabled:  ${config.enabled}`,
					`  mode:     ${config.mode}`,
					`  scope:    ${config.scope}`,
					`  backend:  ${config.backend}`,
					`  model:    ${config.model}`,
					`  model status: ${modelReadiness}`,
					`  language: ${config.language}`,
					`  state:    ${voiceState}`,
					`  setup:    ${config.onboarding.completed ? `complete (${config.onboarding.source ?? "unknown"})` : "incomplete"}`,
					`  socket:   ${activeSocketPath}`,
					`  daemon:   ${daemonUp ? "running" : "stopped"}${daemonInfo}`,
					`  hold-key: SPACE (editor empty) or Ctrl+Shift+V (toggle)`,
					`  btw-key:  Ctrl+Shift+B (hold to record → auto-btw)`,
				].join("\n"), "info");
				return;
			}

			if (sub === "daemon" || sub === "daemon start") {
				cmdCtx.ui.notify("Starting STT daemon...", "info");
				const ok = await ensureDaemon(config);
				cmdCtx.ui.notify(ok ? "Daemon started." : "Failed to start daemon.", ok ? "info" : "error");
				return;
			}

			if (sub === "daemon stop") {
				const resp = await daemonSend({ cmd: "shutdown" });
				cmdCtx.ui.notify(resp.error ? "Daemon not running." : "Daemon stopped.", "info");
				return;
			}

			if (sub === "daemon status") {
				const resp = await daemonSend({ cmd: "status" });
				if (resp.error) {
					cmdCtx.ui.notify("Daemon not running.", "info");
				} else {
					cmdCtx.ui.notify([
						`Daemon status:`,
						`  pid: ${resp.pid}`,
						`  uptime: ${resp.uptime}s`,
						`  requests: ${resp.requests}`,
						`  backend: ${resp.backend}`,
						`  model: ${resp.model}`,
						`  model loaded: ${resp.model_loaded}`,
					].join("\n"), "info");
				}
				return;
			}

			if (sub === "setup" || sub === "reconfigure") {
				const diagnostics = scanEnvironment(TRANSCRIBE_SCRIPT);
				const result = await runVoiceOnboarding(cmdCtx, config, diagnostics);
				if (!result) {
					cmdCtx.ui.notify("Voice setup cancelled.", "warning");
					return;
				}

				await finalizeAndSaveSetup(cmdCtx, result.config, result.selectedScope, result.summaryLines, "setup-command");
				return;
			}

			if (sub === "backends") {
				const backendsJson = spawnSync("python3", [TRANSCRIBE_SCRIPT, "--list-backends"], {
					stdio: ["pipe", "pipe", "pipe"], timeout: 10000,
				});
				try {
					const backends = JSON.parse(backendsJson.stdout?.toString() || "[]");
					const lines = backends.flatMap((b: any) => {
						const installedModels = Array.isArray(b.installed_models) ? b.installed_models : [];
						const readiness = b.type === "cloud"
							? (b.available ? "api ready" : `needs setup: ${b.install}`)
							: installedModels.length > 0
								? `installed: ${installedModels.join(", ")}`
								: b.available
									? "no confirmed installed models"
									: `install: ${b.install}`;
						const summary = `${b.available ? "+" : "-"} ${String(b.name).padEnd(16)} ${String(b.type).padEnd(6)} ${readiness}`;
						const detection = b.install_detection ? `    detection: ${b.install_detection}` : undefined;
						return detection ? [summary, detection] : [summary];
					});
					cmdCtx.ui.notify("Backends:\n" + lines.join("\n"), "info");
				} catch {
					cmdCtx.ui.notify("Failed to scan backends.", "error");
				}
				return;
			}

			if (sub === "doctor") {
				const diagnostics = scanEnvironment(TRANSCRIBE_SCRIPT);
				const recommendation = recommendVoiceSetup(diagnostics, "balanced");
				const currentPlan = buildProvisioningPlan(config, diagnostics);
				const currentBackend = diagnostics.backends.find((backend) => backend.name === config.backend);
				const currentModelReadiness = getModelReadiness(currentBackend, config.model);
				const recommendedConfig: VoiceConfig = {
					...config,
					mode: recommendation.mode,
					backend: recommendation.backend,
					model: recommendation.model,
				};
				const recommendedPlan = buildProvisioningPlan(recommendedConfig, diagnostics);
				const recommendedBackend = diagnostics.backends.find((backend) => backend.name === recommendation.backend);
				const recommendedModelReadiness = getModelReadiness(recommendedBackend, recommendation.model);
				const availableBackends = diagnostics.backends
					.filter((backend) => backend.available)
					.map((backend) => `${backend.name} (${backend.type})`);
				const lines = [
					"Voice doctor:",
					`  python3: ${diagnostics.hasPython ? "OK" : "missing"}`,
					`  sox/rec: ${diagnostics.hasSox ? "OK" : "missing"}`,
					`  brew: ${diagnostics.hasHomebrew ? "OK" : "missing"}`,
					`  deepgram key: ${diagnostics.hasDeepgramKey ? "configured" : "missing"}`,
					`  available backends: ${availableBackends.length > 0 ? availableBackends.join(", ") : "none"}`,
					"",
					`Current config: ${config.mode}/${config.backend}/${config.model}`,
					`Current model status: ${currentModelReadiness}`,
					"Repair current setup:",
					...(currentPlan.commands.length > 0 ? currentPlan.commands.map((command) => `  - ${command}`) : ["  - no install commands required"]),
					...(currentPlan.manualSteps.length > 0 ? currentPlan.manualSteps.map((step) => `  - ${step}`) : []),
					"",
					`Recommended alternative: ${recommendation.mode}/${recommendation.backend}/${recommendation.model}`,
					`Recommended model status: ${recommendedModelReadiness}`,
					`Why: ${recommendation.reason}`,
					...(recommendation.fixableIssues.length > 0 ? ["Fixable issues:", ...recommendation.fixableIssues.map((issue) => `  - ${issue}`)] : ["Fixable issues:", "  - none"]),
					"Suggested commands for the recommendation:",
					...(recommendedPlan.commands.length > 0 ? recommendedPlan.commands.map((command) => `  - ${command}`) : ["  - none"]),
					...(recommendedPlan.manualSteps.length > 0 ? recommendedPlan.manualSteps.map((step) => `  - ${step}`) : []),
				];
				cmdCtx.ui.notify(lines.join("\n"), !currentPlan.ready || recommendation.fixableIssues.length > 0 ? "warning" : "info");
				return;
			}

			// Default: toggle
			config.enabled = !config.enabled;
			if (!config.enabled) voiceCleanup();
			else setupHoldToTalk();
			updateVoiceStatus();
			cmdCtx.ui.notify(`Voice ${config.enabled ? "enabled" : "disabled"}.`, "info");
		},
	});

	// ─── /btw commands ───────────────────────────────────────────────────────

	pi.registerCommand("btw", {
		description: "Side conversation: /btw <message> | /btw:new | /btw:clear | /btw:inject | /btw:summarize",
		handler: async (args, cmdCtx) => {
			ctx = cmdCtx;
			const message = (args || "").trim();

			if (!message) {
				cmdCtx.ui.notify(
					"Usage: /btw <message>\n" +
					"  /btw:new [msg]  — new thread\n" +
					"  /btw:clear      — dismiss widget\n" +
					"  /btw:inject     — push thread to agent\n" +
					"  /btw:summarize  — summarize then inject",
					"info",
				);
				return;
			}

			await handleBtw(message);
		},
	});

	pi.registerCommand("btw:new", {
		description: "Start a fresh BTW thread",
		handler: async (args, cmdCtx) => {
			ctx = cmdCtx;
			btwThread = [];
			pi.appendEntry("btw-reset", { timestamp: Date.now() });
			btwWidgetVisible = false;
			cmdCtx.ui.setWidget("btw", undefined);
			cmdCtx.ui.notify("BTW thread cleared.", "info");

			const message = (args || "").trim();
			if (message) {
				await handleBtw(message);
			}
		},
	});

	pi.registerCommand("btw:clear", {
		description: "Dismiss BTW widget and clear thread",
		handler: async (_args, cmdCtx) => {
			ctx = cmdCtx;
			btwThread = [];
			btwWidgetVisible = false;
			cmdCtx.ui.setWidget("btw", undefined);
			pi.appendEntry("btw-reset", { timestamp: Date.now() });
			cmdCtx.ui.notify("BTW cleared.", "info");
		},
	});

	pi.registerCommand("btw:inject", {
		description: "Push BTW thread into main agent context",
		handler: async (args, cmdCtx) => {
			ctx = cmdCtx;

			if (btwThread.length === 0) {
				cmdCtx.ui.notify("No BTW thread to inject.", "warning");
				return;
			}

			const instructions = (args || "").trim();
			let content = "Here is a side conversation I had (via /btw). Please take it into account:\n\n";

			for (const ex of btwThread) {
				content += `Me: ${ex.question}\nAssistant: ${ex.answer}\n\n`;
			}

			if (instructions) {
				content += `\nAdditional instructions: ${instructions}`;
			}

			pi.sendUserMessage(content, { deliverAs: "followUp" });

			// Clear after injection
			btwThread = [];
			btwWidgetVisible = false;
			cmdCtx.ui.setWidget("btw", undefined);
			pi.appendEntry("btw-reset", { timestamp: Date.now(), reason: "injected" });
			cmdCtx.ui.notify("BTW thread injected into main context.", "info");
		},
	});

	pi.registerCommand("btw:summarize", {
		description: "Summarize BTW thread then inject into main agent",
		handler: async (args, cmdCtx) => {
			ctx = cmdCtx;

			if (btwThread.length === 0) {
				cmdCtx.ui.notify("No BTW thread to summarize.", "warning");
				return;
			}

			const instructions = (args || "").trim();
			let threadText = "";
			for (const ex of btwThread) {
				threadText += `Q: ${ex.question}\nA: ${ex.answer}\n\n`;
			}

			// Ask the model to summarize
			const model = ctx.model;
			if (!model) {
				cmdCtx.ui.notify("No model available for summarization.", "error");
				return;
			}

			cmdCtx.ui.notify("Summarizing BTW thread...", "info");

			try {
				let summary = "";
				const summaryPrompt = instructions
					? `Summarize this conversation thread concisely. Focus on: ${instructions}\n\n${threadText}`
					: `Summarize this conversation thread concisely, capturing key decisions and information:\n\n${threadText}`;

				const eventStream = streamSimple(model, {
					systemPrompt: "You are a concise summarizer. Output only the summary, no preamble.",
					messages: [{ role: "user" as const, content: summaryPrompt, timestamp: Date.now() }],
				});

				for await (const event of eventStream) {
					if (event.type === "text_delta") summary += event.delta;
					else if (event.type === "error") break;
					else if (event.type === "done") break;
				}

				summary = summary.trim();
				if (!summary) {
					cmdCtx.ui.notify("Empty summary — injecting full thread instead.", "warning");
					summary = threadText;
				}

				pi.sendUserMessage(
					`Here's a summary of a side conversation I had (via /btw):\n\n${summary}`,
					{ deliverAs: "followUp" },
				);

				btwThread = [];
				btwWidgetVisible = false;
				cmdCtx.ui.setWidget("btw", undefined);
				pi.appendEntry("btw-reset", { timestamp: Date.now(), reason: "summarized" });
				cmdCtx.ui.notify("BTW summary injected.", "info");
			} catch (err: any) {
				cmdCtx.ui.notify(`Summarization failed: ${err.message}`, "error");
			}
		},
	});
}

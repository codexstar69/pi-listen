/**
 * pi-voice — Enterprise-grade voice STT for Pi CLI.
 *
 * Architecture (modeled after Claude Code's voice pipeline):
 *
 *   STATE MACHINE
 *   ─────────────
 *   idle → warmup → recording → finalizing → idle
 *              ↑         │
 *              └─────────┘  (rapid re-press recovery)
 *
 *   warmup:     User holds SPACE for ≥ HOLD_THRESHOLD_MS (2000ms).
 *               A "keep holding…" hint with countdown is shown. If released before
 *               the threshold, a normal space character is typed (or "hold longer" hint shown).
 *
 *   recording:  SoX captures PCM → Deepgram WebSocket streaming.
 *               Live interim + final transcripts update the widget.
 *               Release SPACE (or press again in toggle mode) → stop.
 *
 *   finalizing: CloseStream sent to Deepgram. Waiting for final
 *               transcript. Safety timeout auto-completes.
 *
 *   HOLD-TO-TALK DETECTION
 *   ──────────────────────
 *   Two paths depending on terminal capabilities:
 *
 *   A) Kitty protocol (Ghostty on Linux, Kitty, WezTerm):
 *      True key-down/repeat/release events available.
 *      First SPACE press → enter warmup immediately (show countdown).
 *      Released < 300ms → tap → type a space.
 *      Released 300ms–2s → show "hold longer" hint.
 *      Held ≥ 2s → activate recording.
 *      True release event stops recording.
 *
 *   B) Non-Kitty (macOS Terminal, Ghostty on macOS):
 *      No key-release event. Holding sends rapid press events (~30-90ms apart).
 *      First SPACE press → record time, start release-detect timer (500ms).
 *      No more presses within 500ms → TAP → type a space.
 *      Rapid presses detected → user is HOLDING.
 *      After REPEAT_CONFIRM_COUNT (3) rapid presses → enter warmup.
 *      After HOLD_THRESHOLD_MS (2000ms) from first press → activate recording.
 *      Gap > RELEASE_DETECT_MS (500ms) after RECORDING_GRACE_MS (1000ms) → stop.
 *
 *   ENTERPRISE FALLBACKS
 *   ────────────────────
 *   • Session corruption guard: new recording request during
 *     finalizing automatically cancels the stale session first.
 *   • Transient failure retry: on WebSocket error during rapid
 *     push-to-talk re-press, auto-retry once after 300ms.
 *   • Stale transcript cleanup: any prior transcript is cleared
 *     before new recording begins.
 *   • Silence vs. no-speech: distinguishes "mic captured silence"
 *     from "no speech detected" with distinct user messages.
 *
 * Activation:
 *   - Hold SPACE (≥500ms) → release to finalize
 *   - Ctrl+Shift+V → toggle start/stop (always works)

 *
 * Config in ~/.pi/agent/settings.json under "voice": { ... }
 */

import type {
	ExtensionAPI,
	ExtensionContext,
	ExtensionCommandContext,
} from "@mariozechner/pi-coding-agent";
import { isKeyRelease, isKeyRepeat, matchesKey } from "@mariozechner/pi-tui";
import type { Component } from "@mariozechner/pi-tui";
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

/**
 * Voice state machine — strict transitions only:
 *   idle → warmup → recording → finalizing → idle
 *   warmup → idle  (released before threshold)
 *   recording → idle  (on error)
 *   finalizing → idle  (on completion or timeout)
 */
type VoiceState = "idle" | "warmup" | "recording" | "finalizing";

// ─── Constants ───────────────────────────────────────────────────────────────

const SAMPLE_RATE = 16000;        // Target sample rate for Deepgram
const CHANNELS = 1;
const ENCODING = "linear16";
const DEEPGRAM_WS_URL = "wss://api.deepgram.com/v1/listen";
const KEEPALIVE_INTERVAL_MS = 8000;
const FINALIZE_SAFETY_TIMEOUT_MS = 2500;  // Max wait for remaining transcripts after CloseStream
const FINALIZE_NO_DATA_TIMEOUT_MS = 600;  // If no data for 600ms after CloseStream, assume done
const MAX_RECORDING_SECS = 120;

// Hold-to-talk timing
const HOLD_THRESHOLD_MS = 800;    // Must hold for 0.8s before voice activates
const RELEASE_DETECT_MS = 500;    // Gap in key-repeat that means "released" (non-Kitty)
                                   // macOS default InitialKeyRepeat is ~375ms, so 500ms
                                   // ensures the first repeat arrives before we decide "tap"
const REPEAT_CONFIRM_COUNT = 3;   // Need this many rapid repeat presses to confirm "holding"
const REPEAT_CONFIRM_MS = 700;    // Max gap between presses to count as rapid repeat
                                  // macOS initial key-repeat delay is ~417-583ms depending on settings
                                   // Must be > macOS InitialKeyRepeat (~375ms)
const RECORDING_GRACE_MS = 1200;  // After recording starts, ignore release for this long
                                   // The holdActivationTimer + async startVoiceRecording
                                   // creates a gap where no key events arrive, so this must
                                   // be generous to avoid false stops
const RELEASE_DETECT_RECORDING_MS = 3000; // During active recording, gap before we consider
                                           // the key released. Must be very generous because:
                                           // 1. macOS key repeat is ~90ms but event loop may
                                           //    be busy with Deepgram WebSocket I/O
                                           // 2. False stops are far worse than slow stop
                                           // 3. User expects recording to continue while holding
const RETRY_DELAY_MS = 300;       // Auto-retry on transient failure during rapid re-press
const MAX_RETRY_ATTEMPTS = 1;     // Max retries per activation attempt
const CORRUPTION_GUARD_MS = 200;  // Min gap between stop and restart

// Debug logging — set PI_VOICE_DEBUG=1 to enable
const VOICE_DEBUG = !!process.env.PI_VOICE_DEBUG;
const VOICE_LOG_FILE = path.join(os.tmpdir(), "pi-voice-debug.log");

// ─── Audio level tracking (module scope so streaming can access) ──────
let audioLevel = 0;
let audioLevelSmoothed = 0;

function updateAudioLevel(chunk: Buffer) {
	// Compute RMS of 16-bit PCM samples
	let sum = 0;
	const samples = Math.floor(chunk.length / 2);
	if (samples === 0) return;
	for (let i = 0; i < chunk.length - 1; i += 2) {
		const sample = chunk.readInt16LE(i);
		sum += sample * sample;
	}
	const rms = Math.sqrt(sum / samples);
	// Normalize to 0-1 range (16-bit max is 32768)
	audioLevel = Math.min(1, rms / 8000);
	// Smooth with exponential decay
	audioLevelSmoothed = audioLevelSmoothed * 0.6 + audioLevel * 0.4;
}

function voiceDebug(...args: unknown[]) {
	// Always write to log file for post-mortem debugging
	const ts = new Date().toISOString().split("T")[1];
	const line = `[voice ${ts}] ${args.map(a => typeof a === "object" ? JSON.stringify(a) : String(a)).join(" ")}\n`;
	try { fs.appendFileSync(VOICE_LOG_FILE, line); } catch {}
	if (VOICE_DEBUG) {
		process.stderr.write(line);
	}
}

const EXT_DIR = path.dirname(new URL(import.meta.url).pathname);
const PROJECT_ROOT = path.join(EXT_DIR, "..");
const DAEMON_SCRIPT = path.join(PROJECT_ROOT, "daemon.py");
const TRANSCRIBE_SCRIPT = path.join(PROJECT_ROOT, "transcribe.py");

// ── Guard: only use python scripts if they actually exist ──
const HAS_TRANSCRIBE_SCRIPT = fs.existsSync(TRANSCRIBE_SCRIPT);
const HAS_DAEMON_SCRIPT = fs.existsSync(DAEMON_SCRIPT);

function commandExists(cmd: string): boolean {
	const which = process.platform === "win32" ? "where" : "which";
	return spawnSync(which, [cmd], { stdio: "pipe", timeout: 3000 }).status === 0;
}

// ─── Daemon Communication (kept for non-deepgram local backends) ─────────────

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
	if (!HAS_DAEMON_SCRIPT) return false;
	if (await isDaemonRunning(activeSocketPath)) {
		const status = await daemonSend({ cmd: "status" }, 3000, activeSocketPath);
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

		setTimeout(() => {
			if (!started) {
				try { proc.kill(); } catch {}
				resolve(false);
			}
		}, 10000);
	});
}

// ─── Legacy file-based transcription (for non-deepgram backends) ─────────────

let legacyRecProcess: ChildProcess | null = null;

function startLegacyRecordingToFile(outPath: string): boolean {
	if (legacyRecProcess) {
		try { legacyRecProcess.kill("SIGTERM"); } catch {}
		legacyRecProcess = null;
	}
	if (!commandExists("rec")) return false;
	legacyRecProcess = spawn("rec", [
		"-q", "-c", "1", "-b", "16", "-r", String(SAMPLE_RATE), outPath,
		"rate", String(SAMPLE_RATE),
	], { stdio: ["pipe", "pipe", "pipe"] });
	legacyRecProcess.stderr?.on("data", () => {});
	legacyRecProcess.on("error", () => { legacyRecProcess = null; });
	return true;
}

function stopLegacyRecording(): Promise<void> {
	return new Promise((resolve) => {
		if (!legacyRecProcess) { resolve(); return; }
		legacyRecProcess.on("close", () => { legacyRecProcess = null; resolve(); });
		try { legacyRecProcess.kill("SIGTERM"); } catch {}
		setTimeout(() => {
			if (legacyRecProcess) {
				try { legacyRecProcess.kill("SIGKILL"); } catch {}
				legacyRecProcess = null;
			}
			resolve();
		}, 2000);
	});
}

async function transcribeAudioFile(
	audioPath: string,
	config: VoiceConfig,
): Promise<{ text: string; duration: number; error?: string }> {
	if (!HAS_TRANSCRIBE_SCRIPT) {
		return { text: "", duration: 0, error: "transcribe.py not found — use Deepgram streaming instead" };
	}
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

// ─── Deepgram WebSocket Streaming ────────────────────────────────────────────

interface StreamingSession {
	ws: WebSocket;
	recProcess: ChildProcess;
	interimText: string;
	finalizedParts: string[];
	keepAliveTimer: ReturnType<typeof setInterval> | null;
	closed: boolean;
	hadAudioData: boolean;       // Track if we received any audio data
	hadSpeech: boolean;          // Track if Deepgram detected any speech
	onTranscript: (interim: string, finals: string[]) => void;
	onDone: (fullText: string, meta: { hadAudio: boolean; hadSpeech: boolean }) => void;
	onError: (err: string) => void;
}

function resolveDeepgramApiKey(config: VoiceConfig): string | null {
	return process.env.DEEPGRAM_API_KEY || config.deepgramApiKey || null;
}

function isDeepgramStreaming(config: VoiceConfig): boolean {
	const key = resolveDeepgramApiKey(config);
	if (!key) return false;
	return config.backend === "deepgram" || (config.backend === "auto" && !!key);
}

function buildDeepgramWsUrl(config: VoiceConfig): string {
	const params = new URLSearchParams({
		encoding: ENCODING,
		sample_rate: String(SAMPLE_RATE),
		channels: String(CHANNELS),
		endpointing: "300",
		utterance_end_ms: "1000",
		language: config.language || "en",
		model: config.model || "nova-3",
		smart_format: "true",
		interim_results: "true",
	});
	return `${DEEPGRAM_WS_URL}?${params.toString()}`;
}

function startStreamingSession(
	config: VoiceConfig,
	callbacks: {
		onTranscript: (interim: string, finals: string[]) => void;
		onDone: (fullText: string, meta: { hadAudio: boolean; hadSpeech: boolean }) => void;
		onError: (err: string) => void;
	},
): StreamingSession | null {
	const apiKey = resolveDeepgramApiKey(config);
	voiceDebug("startStreamingSession", { hasApiKey: !!apiKey, backend: config.backend });
	if (!apiKey) {
		voiceDebug("startStreamingSession → no API key, calling onError");
		callbacks.onError("DEEPGRAM_API_KEY not set");
		return null;
	}

	if (!commandExists("rec")) {
		voiceDebug("startStreamingSession → no SoX, calling onError");
		callbacks.onError("Voice requires SoX. Install: brew install sox");
		return null;
	}

	// On macOS, SoX's rec ignores -r for input (CoreAudio captures at native rate,
	// usually 48kHz). We must use the `rate` effect to downsample to 16kHz.
	// Without this, Deepgram receives 48kHz audio labeled as 16kHz → garbled/silence.
	// Use smaller buffer (-b 4096) for lower latency audio capture.
	const recProc = spawn("rec", [
		"-q",
		"--buffer", "4096",           // 4096 bytes prevents CoreAudio buffer overruns
		"-c", String(CHANNELS),
		"-b", "16",
		"-e", "signed-integer",
		"-t", "raw",
		"-",          // output to stdout
		"rate", String(SAMPLE_RATE),  // SoX effect: resample to 16kHz
	], { stdio: ["pipe", "pipe", "pipe"] });

	recProc.stderr?.on("data", (d: Buffer) => {
		const msg = d.toString().trim();
		// Suppress CoreAudio buffer overrun spam — only log once
		if (msg.includes("buffer overrun")) return;
		voiceDebug("SoX stderr:", msg);
	});

	const wsUrl = buildDeepgramWsUrl(config);
	const ws = new WebSocket(wsUrl, {
		headers: {
			"Authorization": `Token ${apiKey}`,
		},
	} as any);

	const session: StreamingSession = {
		ws,
		recProcess: recProc,
		interimText: "",
		finalizedParts: [],
		keepAliveTimer: null,
		closed: false,
		hadAudioData: false,
		hadSpeech: false,
		onTranscript: callbacks.onTranscript,
		onDone: callbacks.onDone,
		onError: callbacks.onError,
	};

	// Handle HTTP error responses before WebSocket upgrade (e.g., 400 Bad Request, 401 Unauthorized)
	// Only available with Node.js `ws` package — skip if using browser-style WebSocket
	if (typeof (ws as any).on === "function") {
		(ws as any).on("unexpected-response", (_req: any, res: any) => {
			let body = "";
			res.on("data", (d: Buffer) => { body += d.toString(); });
			res.on("end", () => {
				voiceDebug("WebSocket unexpected-response", { status: res.statusCode, body });
				if (!session.closed) {
					session.onError(`Deepgram HTTP ${res.statusCode}: ${body.slice(0, 200)}`);
					session.closed = true;
					try { recProc.kill("SIGTERM"); } catch {}
				}
			});
		});
	}

	ws.onopen = () => {
		voiceDebug("WebSocket onopen → streaming audio");
		try { ws.send(JSON.stringify({ type: "KeepAlive" })); } catch {}

		session.keepAliveTimer = setInterval(() => {
			if (ws.readyState === WebSocket.OPEN) {
				try { ws.send(JSON.stringify({ type: "KeepAlive" })); } catch {}
			}
		}, KEEPALIVE_INTERVAL_MS);

		recProc.stdout?.on("data", (chunk: Buffer) => {
			if (ws.readyState === WebSocket.OPEN) {
				session.hadAudioData = true;
				try { ws.send(chunk); } catch {}
				// Feed audio data to level meter for reactive waveform
				updateAudioLevel(chunk);
			}
		});
	};

	ws.onmessage = (event: MessageEvent) => {
		try {
			const msg = typeof event.data === "string" ? JSON.parse(event.data) : null;
			if (!msg) return;

			if (msg.type === "Results") {
				const alt = msg.channel?.alternatives?.[0];
				const transcript = alt?.transcript || "";

				if (transcript.trim()) {
					session.hadSpeech = true;
				}

				if (msg.is_final) {
					if (transcript.trim()) {
						session.finalizedParts.push(transcript.trim());
					}
					session.interimText = "";
				} else {
					session.interimText = transcript;
				}

				session.onTranscript(session.interimText, session.finalizedParts);
			} else if (msg.type === "Error" || msg.type === "error") {
				session.onError(msg.message || msg.description || "Deepgram error");
			}
		} catch {}
	};

	ws.onerror = (ev) => {
		const errMsg = (ev as any)?.message || (ev as any)?.error?.message || "unknown";
		voiceDebug("WebSocket onerror", { readyState: ws.readyState, error: errMsg });
		if (!session.closed) {
			session.onError(`WebSocket error: ${errMsg}`);
		}
	};

	ws.onclose = (ev) => {
		voiceDebug("WebSocket onclose", { code: (ev as any)?.code, reason: (ev as any)?.reason, closed: session.closed });
		if (!session.closed) {
			finalizeSession(session);
		}
	};

	recProc.on("error", (err) => {
		voiceDebug("SoX process error:", err.message);
		session.onError(`SoX error: ${err.message}`);
	});

	recProc.on("close", (code, signal) => {
		voiceDebug("SoX process closed", { code, signal, wsClosed: session.closed, wsState: ws.readyState });
		if (ws.readyState === WebSocket.OPEN) {
			try { ws.send(JSON.stringify({ type: "CloseStream" })); } catch {}
		}
	});

	return session;
}

function stopStreamingSession(session: StreamingSession): void {
	if (session.closed) return;

	try { session.recProcess.kill("SIGTERM"); } catch {}

	if (session.ws.readyState === WebSocket.OPEN) {
		try { session.ws.send(JSON.stringify({ type: "CloseStream" })); } catch {}
	}

	// Safety timeout
	setTimeout(() => {
		if (!session.closed) {
			finalizeSession(session);
		}
	}, FINALIZE_SAFETY_TIMEOUT_MS);

	// Quick finalize if no new data
	let lastDataTime = Date.now();
	const origOnMessage = session.ws.onmessage;
	session.ws.onmessage = (event: MessageEvent) => {
		lastDataTime = Date.now();
		if (origOnMessage) origOnMessage.call(session.ws, event);
	};

	const noDataCheck = setInterval(() => {
		if (session.closed) {
			clearInterval(noDataCheck);
			return;
		}
		if (Date.now() - lastDataTime > FINALIZE_NO_DATA_TIMEOUT_MS) {
			clearInterval(noDataCheck);
			if (!session.closed) {
				finalizeSession(session);
			}
		}
	}, 500);
}

function finalizeSession(session: StreamingSession): void {
	if (session.closed) return;
	session.closed = true;
	voiceDebug("finalizeSession", { hadAudio: session.hadAudioData, hadSpeech: session.hadSpeech, parts: session.finalizedParts.length });

	if (session.keepAliveTimer) {
		clearInterval(session.keepAliveTimer);
		session.keepAliveTimer = null;
	}

	try { session.ws.close(); } catch {}
	try { session.recProcess.kill("SIGKILL"); } catch {}

	const fullText = session.finalizedParts.join(" ").trim();
	session.onDone(fullText, {
		hadAudio: session.hadAudioData,
		hadSpeech: session.hadSpeech,
	});
}

// ─── Abort helper — nuke everything synchronously ────────────────────────────

function abortSession(session: StreamingSession | null): void {
	if (!session || session.closed) return;
	session.closed = true;
	if (session.keepAliveTimer) {
		clearInterval(session.keepAliveTimer);
		session.keepAliveTimer = null;
	}
	try { session.ws.close(); } catch {}
	try { session.recProcess.kill("SIGKILL"); } catch {}
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

	// Streaming session state
	let activeSession: StreamingSession | null = null;

	let retryAttempts = 0;
	let lastStopTime = 0;    // For corruption guard
	let recordingStartedAt = 0; // When recording actually started (for grace period)
	let editorTextBeforeVoice = ""; // Snapshot of editor text before recording started

	// Hold-to-talk state
	let kittyReleaseDetected = false;
	let spaceDownTime: number | null = null;
	let holdActivationTimer: ReturnType<typeof setTimeout> | null = null;
	let spaceConsumed = false;        // True once threshold passed and recording started
	let releaseDetectTimer: ReturnType<typeof setTimeout> | null = null;
	let warmupWidgetTimer: ReturnType<typeof setInterval> | null = null;
	let spacePressCount = 0;          // Count of rapid space presses (for non-Kitty hold detection)
	let lastSpacePressTime = 0;       // Timestamp of last space press event
	let holdConfirmed = false;        // True once we've confirmed user is holding (not tapping)
	let errorCooldownUntil = 0;       // After an error, block re-activation until this timestamp

	// ─── Recording History ───────────────────────────────────────────────────

	interface RecordingHistoryEntry {
		text: string;
		timestamp: number;
		duration: number;
		mode: "hold" | "toggle" | "dictate";
	}

	const recordingHistory: RecordingHistoryEntry[] = [];
	const MAX_HISTORY = 50;

	function addToHistory(text: string, duration: number, mode: "hold" | "toggle" | "dictate" = "hold") {
		recordingHistory.unshift({ text, timestamp: Date.now(), duration, mode });
		if (recordingHistory.length > MAX_HISTORY) recordingHistory.pop();
	}

	// ─── Continuous Dictation Mode ───────────────────────────────────────────

	let dictationMode = false;
	let dictationText = "";

	// ─── Voice Command Detection ─────────────────────────────────────────────

	const VOICE_COMMAND_PREFIXES = ["hey pi", "pi ", "hey pie", "run ", "execute ", "commit ", "search for ", "open ", "go to "];
	const VOICE_COMMAND_MAP: Record<string, (args: string) => string> = {
		"run tests": () => "bun run test",
		"run test": () => "bun run test",
		"run the tests": () => "bun run test",
		"run all tests": () => "bun run test",
		"run typecheck": () => "bun run typecheck",
		"type check": () => "bun run typecheck",
		"run lint": () => "bun run lint",
		"lint this": () => "bun run lint",
		"commit this": () => "git add -A && git commit",
		"commit": () => "git add -A && git commit",
		"git status": () => "git status",
		"git diff": () => "git diff",
		"undo": () => "__UNDO__",
		"undo that": () => "__UNDO__",
		"clear": () => "__CLEAR__",
		"clear all": () => "__CLEAR__",
		"select all": () => "__SELECT_ALL__",
		"new line": () => "__NEWLINE__",
		"submit": () => "__SUBMIT__",
		"send": () => "__SUBMIT__",
		"send it": () => "__SUBMIT__",
	};

	function detectVoiceCommand(text: string): { isCommand: boolean; action?: string; args?: string } {
		const lower = text.toLowerCase().trim();

		// Direct command matches
		for (const [trigger, handler] of Object.entries(VOICE_COMMAND_MAP)) {
			if (lower === trigger || lower.replace(/[.,!?]/g, "") === trigger) {
				return { isCommand: true, action: handler("") };
			}
		}

		// Prefix-based commands: "hey pi, run the tests"
		for (const prefix of VOICE_COMMAND_PREFIXES) {
			if (lower.startsWith(prefix)) {
				const rest = lower.slice(prefix.length).trim().replace(/[.,!?]/g, "");
				for (const [trigger, handler] of Object.entries(VOICE_COMMAND_MAP)) {
					if (rest === trigger || rest.startsWith(trigger)) {
						return { isCommand: true, action: handler(rest.slice(trigger.length).trim()) };
					}
				}
				// "hey pi, search for X" → search
				if (rest.startsWith("search for ") || rest.startsWith("search ")) {
					const query = rest.replace(/^search (for )?/, "").trim();
					return { isCommand: true, action: `__SEARCH__${query}` };
				}
				// Generic: "hey pi, <anything>" → send as user message
				if (prefix === "hey pi" || prefix === "hey pie" || prefix === "pi ") {
					return { isCommand: true, action: `__MESSAGE__${rest}` };
				}
			}
		}

		// Voice shortcuts embedded in dictation
		if (lower === "new line" || lower === "newline") return { isCommand: true, action: "__NEWLINE__" };
		if (lower === "submit" || lower === "send it" || lower === "send") return { isCommand: true, action: "__SUBMIT__" };

		return { isCommand: false };
	}

	function executeVoiceCommand(action: string): boolean {
		if (!ctx?.hasUI) return false;

		if (action === "__UNDO__") {
			// Remove last word from editor
			const text = ctx.ui.getEditorText() || "";
			const words = text.trim().split(/\s+/);
			words.pop();
			ctx.ui.setEditorText(words.join(" ") + (words.length ? " " : ""));
			ctx.ui.notify("↩ Undo last word", "info");
			return true;
		}

		if (action === "__CLEAR__") {
			ctx.ui.setEditorText("");
			ctx.ui.notify("Cleared editor", "info");
			return true;
		}

		if (action === "__SELECT_ALL__") {
			// Can't truly select-all in the editor, but useful feedback
			ctx.ui.notify("(Select all not available in terminal editor)", "info");
			return true;
		}

		if (action === "__NEWLINE__") {
			const text = ctx.ui.getEditorText() || "";
			ctx.ui.setEditorText(text + "\n");
			return true;
		}

		if (action === "__SUBMIT__") {
			// Simulate Enter — send the editor text as a message
			const text = (ctx.ui.getEditorText() || "").trim();
			if (text) {
				pi.sendUserMessage(text);
				ctx.ui.setEditorText("");
				ctx.ui.notify("✓ Submitted via voice", "info");
			}
			return true;
		}

		if (action.startsWith("__SEARCH__")) {
			const query = action.slice("__SEARCH__".length);
			ctx.ui.setEditorText(`search for ${query}`);
			ctx.ui.notify(`🔍 Search: ${query}`, "info");
			return true;
		}

		if (action.startsWith("__MESSAGE__")) {
			const msg = action.slice("__MESSAGE__".length);
			pi.sendUserMessage(msg);
			ctx.ui.setEditorText("");
			ctx.ui.notify("✓ Sent to agent via voice", "info");
			return true;
		}

		// Shell command — put in editor for user to review
		ctx.ui.setEditorText(action);
		ctx.ui.notify(`🎤 Voice command: ${action}`, "info");
		return true;
	}

	// ─── Voice Shortcut Processing ───────────────────────────────────────────
	// Processes text for inline shortcuts like "new line", "period", etc.

	function processVoiceShortcuts(text: string): string {
		return text
			.replace(/\bnew line\b/gi, "\n")
			.replace(/\bnewline\b/gi, "\n")
			.replace(/\bperiod\b/gi, ".")
			.replace(/\bcomma\b/gi, ",")
			.replace(/\bquestion mark\b/gi, "?")
			.replace(/\bexclamation mark\b/gi, "!")
			.replace(/\bcolon\b/gi, ":")
			.replace(/\bsemicolon\b/gi, ";")
			.replace(/\bopen parenthesis\b/gi, "(")
			.replace(/\bclose parenthesis\b/gi, ")")
			.replace(/\bopen bracket\b/gi, "[")
			.replace(/\bclose bracket\b/gi, "]")
			.replace(/\bopen brace\b/gi, "{")
			.replace(/\bclose brace\b/gi, "}")
			.replace(/\bbackslash\b/gi, "\\")
			.replace(/\bforward slash\b/gi, "/");
	}

	// ─── Sound Feedback ──────────────────────────────────────────────────────

	function playSound(type: "start" | "stop" | "error") {
		// Use macOS system sounds — lightweight, no dependencies
		try {
			const soundMap: Record<string, string> = {
				start: "/System/Library/Sounds/Tink.aiff",
				stop: "/System/Library/Sounds/Pop.aiff",
				error: "/System/Library/Sounds/Basso.aiff",
			};
			const soundFile = soundMap[type];
			if (soundFile && fs.existsSync(soundFile)) {
				spawn("afplay", [soundFile], { stdio: "ignore", detached: true }).unref();
			}
		} catch {}
	}

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
					: isDeepgramStreaming(config)
						? "STREAM"
						: config.mode === "api"
							? "API"
							: config.mode === "local"
								? "LOCAL"
								: "AUTO";
				ctx.ui.setStatus("voice", `MIC ${modeTag}`);
				break;
			}
			case "warmup":
				ctx.ui.setStatus("voice", "MIC HOLD...");
				break;
			case "recording": {
				const secs = Math.round((Date.now() - recordingStart) / 1000);
				// Live audio level meter in status bar
				const meterLen = 4;
				const meterFilled = Math.round(audioLevelSmoothed * meterLen);
				const meter = "█".repeat(meterFilled) + "░".repeat(meterLen - meterFilled);
				ctx.ui.setStatus("voice", `REC ${secs}s ${meter}`);
				break;
			}
			case "finalizing":
				// Don't show "STT..." — live transcript handles it
				ctx.ui.setStatus("voice", "");
				break;
		}
	}

	function setVoiceState(newState: VoiceState) {
		const prev = voiceState;
		voiceState = newState;
		if (prev !== newState) {
			voiceDebug(`STATE: ${prev} → ${newState}`, { stack: new Error().stack?.split("\n").slice(1, 4).map(s => s.trim()).join(" | ") });
		}
		updateVoiceStatus();
	}

	// ─── Cleanup helpers ─────────────────────────────────────────────────────

	function clearHoldTimer() {
		if (holdActivationTimer) {
			clearTimeout(holdActivationTimer);
			holdActivationTimer = null;
		}
	}

	function clearReleaseTimer() {
		if (releaseDetectTimer) {
			clearTimeout(releaseDetectTimer);
			releaseDetectTimer = null;
		}
	}

	function clearWarmupWidget() {
		if (warmupWidgetTimer) {
			clearInterval(warmupWidgetTimer);
			warmupWidgetTimer = null;
		}
	}

	function clearRecordingAnimTimer() {
		const timer = (showRecordingWidget as any)?._animTimer;
		if (timer) {
			clearInterval(timer);
			(showRecordingWidget as any)._animTimer = null;
		}
		const finTimer = (showFinalizingWidget as any)?._timer;
		if (finTimer) {
			clearInterval(finTimer);
			(showFinalizingWidget as any)._timer = null;
		}
	}

	function hideWidget() {
		if (ctx?.hasUI) ctx.ui.setWidget("voice-recording", undefined);
	}

	function voiceCleanup() {
		if (statusTimer) { clearInterval(statusTimer); statusTimer = null; }
		clearHoldTimer();
		clearReleaseTimer();
		clearWarmupWidget();
		clearRecordingAnimTimer();
		// Reset audio levels
		audioLevel = 0;
		audioLevelSmoothed = 0;
		if (activeSession) {
			abortSession(activeSession);
			activeSession = null;
		}
		if (legacyRecProcess) {
			try { legacyRecProcess.kill("SIGTERM"); } catch {}
			legacyRecProcess = null;
		}
		if (tempFile) { try { fs.unlinkSync(tempFile); } catch {} tempFile = null; }
		spaceConsumed = false;
		spaceDownTime = null;
		spacePressCount = 0;
		lastSpacePressTime = 0;
		holdConfirmed = false;
		retryAttempts = 0;
		errorCooldownUntil = 0;
		editorTextBeforeVoice = "";
		dictationMode = false;
		dictationText = "";
		hideWidget();
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
		if (validated && nextConfig.enabled && !isDeepgramStreaming(nextConfig)) {
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

	// ─── Warmup Widget ──────────────────────────────────────────────────────
	//
	// Premium minimal cat: calm, clean, and subtle.

	function stripAnsi(text: string): string {
		return text.replace(/\[[^m]*m/g, "");
	}

	// ═══════════════════════════════════════════════════════════════════════
	// ─── Face: (◕▽◕) — eyes react to voice ─────────────────────────────

	let _nextBlinkAt = Date.now() + 3000 + Math.random() * 2000;
	let _blinkUntil = 0;
	let _doubleBlink = false;

	function getFace(audioLevel: number): string {
		const now = Date.now();

		// ── Natural blink ──
		if (now >= _nextBlinkAt && now > _blinkUntil) {
			_blinkUntil = now + 130;
			if (Math.random() < 0.18) {
				_doubleBlink = true;
				_nextBlinkAt = now + 280;
			} else {
				_doubleBlink = false;
				_nextBlinkAt = now + 2500 + Math.random() * 3500;
			}
		}
		if (_doubleBlink && now >= _nextBlinkAt && now > _blinkUntil) {
			_blinkUntil = now + 100;
			_doubleBlink = false;
			_nextBlinkAt = now + 2500 + Math.random() * 3500;
		}
		const blinking = now < _blinkUntil;

		// ── Eyes dilate with audio level ──
		let eyes: string;
		if (blinking)               eyes = "–▽–";
		else if (audioLevel > 0.42) eyes = "◉▽◉";
		else if (audioLevel > 0.15) eyes = "●▽●";
		else                        eyes = "◕▽◕";

		return `(${eyes})`;
	}


	// ─── Braille Waveform ───────────────────────────────────────────────
	const _noiseTable: number[] = [];
	for (let i = 0; i < 256; i++) _noiseTable.push(Math.random());
	function smoothNoise(x: number): number {
		const xi = Math.floor(x) & 255;
		const xf = x - Math.floor(x);
		const t = xf * xf * (3 - 2 * xf);
		return _noiseTable[xi] * (1 - t) + _noiseTable[(xi + 1) & 255] * t;
	}
	function fbmNoise(x: number, octaves: number): number {
		let val = 0, amp = 0.5, freq = 1;
		for (let i = 0; i < octaves; i++) {
			val += smoothNoise(x * freq) * amp;
			amp *= 0.5;
			freq *= 2.1;
		}
		return val;
	}

	function buildPremiumWave(_frame: number, width: number, level: number): string {
		const t = Date.now() / 1000;
		const bars = "⠁⠃⠇⡇⣇⣧⣷⣿";
		const len = Math.max(8, Math.min(width, 16));
		let out = "";
		const energy = Math.pow(level, 0.7);
		const speed = 1.8 + energy * 4.5;
		const octaves = energy > 0.3 ? 4 : 3;
		const baseAmp = 0.10 + energy * 0.90;
		for (let i = 0; i < len; i++) {
			const pos = i / len;
			const n1 = fbmNoise(pos * 3.0 + t * speed * 0.4, octaves);
			const n2 = fbmNoise(pos * 5.5 + t * speed * 0.9 + 100, 2) * 0.3;
			const center = 1.0 - Math.abs(pos - 0.5) * 1.4;
			const breath = Math.sin(t * 1.2) * 0.1 * center;
			const raw = (n1 + n2 + breath) * baseAmp;
			const value = Math.max(0, Math.min(1, raw));
			const idx = Math.min(bars.length - 1, Math.round(value * (bars.length - 1)));
			out += bars[idx];
		}
		return out;
	}

	// ─── Record Dot ─────────────────────────────────────────────────────
	function getRecordDot(): string {
		const phase = (Math.sin(Date.now() / 600) + 1) / 2;
		if (phase > 0.65) return "●";
		if (phase > 0.35) return "◉";
		return "○";
	}

	// ─── Warmup Widget ──────────────────────────────────────────────────
	function showWarmupWidget() {
		if (!ctx?.hasUI) return;

		const startTime = Date.now();

		const renderWarmup = () => {
			if (!ctx?.hasUI) return;
			const elapsed = Date.now() - startTime;
			const progress = Math.min(elapsed / HOLD_THRESHOLD_MS, 1);

			ctx.ui.setWidget("voice-recording", (tui, theme) => {
				return {
					invalidate() {},
					render(width: number): string[] {
						const maxW = Math.max(0, Math.min(width - 4, 72));
						const face = getFace(0);
						const meterLen = Math.max(4, Math.min(10, maxW - 30));
						const filled = Math.round(progress * meterLen);
						const meter = theme.fg("accent", "⣿".repeat(filled)) + theme.fg("muted", "⠁".repeat(meterLen - filled));
						const hint = theme.fg("dim", progress < 1 ? "hold…" : "ready!");
						const out: string[] = [];
						out.push(` ${theme.fg("accent", face)}  ${meter} ${hint}`);
						return out;
					},
				};
			}, { placement: "aboveEditor" });
		};

		renderWarmup();
		warmupWidgetTimer = setInterval(renderWarmup, 90);
	}

	// ─── Recording Widget ───────────────────────────────────────────────
	function showRecordingWidget() {
		if (!ctx?.hasUI) return;

		(showRecordingWidget as any)._frame = 0;
		(showRecordingWidget as any)._liveText = "";

		const animTimer = setInterval(() => {
			(showRecordingWidget as any)._frame = ((showRecordingWidget as any)._frame || 0) + 1;
			showRecordingWidgetFrame((showRecordingWidget as any)._frame);
		}, 120);

		(showRecordingWidget as any)._animTimer = animTimer;
		showRecordingWidgetFrame(0);
	}

	function showRecordingWidgetFrame(frame: number) {
		if (!ctx?.hasUI) return;

		ctx.ui.setWidget("voice-recording", (tui, theme) => {
			return {
				invalidate() {},
				render(width: number): string[] {
					const maxW = Math.max(0, Math.min(width - 4, 72));
					const elapsed = Math.round((Date.now() - recordingStart) / 1000);
					const mins = Math.floor(elapsed / 60);
					const secs = elapsed % 60;
					const timeStr = mins > 0 ? `${mins}:${String(secs).padStart(2, "0")}` : `${secs}s`;
					const wave = buildPremiumWave(frame, Math.max(6, Math.min(12, maxW - 30)), audioLevelSmoothed);
					const hint = theme.fg("dim", "⌴ release");
					const face = getFace(audioLevelSmoothed);

					return [` ${theme.fg("error", getRecordDot())} ${theme.fg("accent", face)}  ${theme.fg("accent", wave)} ${theme.fg("muted", timeStr)} ${hint}`];
				},
			};
		}, { placement: "aboveEditor" });
	}


	// ─── Live Transcript ────────────────────────────────────────────────────
	// Instead of showing transcript in a widget, put it directly in the editor
	// input area so users see it where they type.

	function updateLiveTranscriptWidget(interim: string, finals: string[]) {
		if (!ctx?.hasUI) return;

		// DON'T stop the waveform animation — keep it running!
		// We still want the ● REC waveform + timer to show.
		// Just update the editor text with the live transcript.

		const finalized = finals.join(" ");
		const displayText = finalized + (interim ? (finalized ? " " : "") + interim : "");

		// Show live text directly in the editor input (prepend any existing text)
		if (displayText.trim()) {
			const prefix = editorTextBeforeVoice ? editorTextBeforeVoice + " " : "";
			ctx.ui.setEditorText(prefix + displayText);
		}

		// Update the waveform widget to also show a transcript preview line
		// but keep the full animation running (don't replace it)
		(showRecordingWidget as any)._liveText = displayText;
	}

	// ─── Finalizing Widget ──────────────────────────────────────────────────

	function showFinalizingWidget() {
		if (!ctx?.hasUI) return;

		let finalizingAnimTimer: ReturnType<typeof setInterval> | null = null;

		const render = () => {
			if (!ctx?.hasUI) return;
			ctx.ui.setWidget("voice-recording", (tui, theme) => {
				return {
					invalidate() {},
					render(width: number): string[] {
						const maxW = Math.max(0, Math.min(width - 4, 56));
						const dots = ".".repeat(1 + (Math.floor(Date.now() / 400) % 3));
						const content = ` ${theme.fg("accent", ">")} ${theme.fg("dim", `Processing${dots}`)}`;
						const stripped = content.replace(/\x1b\[[^m]*m/g, "").length;
						const pad = Math.max(0, maxW - stripped);

						const border = theme.fg("border", "─".repeat(maxW));
						return [border, content + " ".repeat(pad), border];
					},
				};
			}, { placement: "aboveEditor" });
		};

		render();
		finalizingAnimTimer = setInterval(render, 400);
		(showFinalizingWidget as any)._timer = finalizingAnimTimer;
	}

	// ─── Voice: Start / Stop (Streaming or Legacy) ───────────────────────────

	async function startVoiceRecording(): Promise<boolean> {
		voiceDebug("startVoiceRecording called", { voiceState, isDeepgram: isDeepgramStreaming(config), hasUI: !!ctx?.hasUI });
		if (!ctx?.hasUI) return false;

		// ── SESSION CORRUPTION GUARD ──
		// If we're still finalizing from a previous recording, abort it first.
		// This prevents the "slow connection overlaps new recording" bug.
		if (voiceState === "finalizing" || voiceState === "recording") {
			abortSession(activeSession);
			activeSession = null;
			clearRecordingAnimTimer();
			clearWarmupWidget();
			hideWidget();
			setVoiceState("idle");
			// Brief pause to let resources release
			await new Promise((r) => setTimeout(r, CORRUPTION_GUARD_MS));
		}

		// ── STALE TRANSCRIPT CLEANUP ──
		// Clear any prior transcript from the widget
		hideWidget();

		recordingStart = Date.now();
		retryAttempts = 0;

		// Snapshot editor text before voice overwrites it with live transcript
		editorTextBeforeVoice = ctx?.hasUI ? (ctx.ui.getEditorText() || "") : "";

		if (isDeepgramStreaming(config)) {
			return startStreamingRecording();
		} else {
			return startLegacyRecording();
		}
	}

	async function startStreamingRecording(): Promise<boolean> {
		voiceDebug("startStreamingRecording called", { configBackend: config.backend, hasKey: !!resolveDeepgramApiKey(config) });
		setVoiceState("recording");

		const session = startStreamingSession(config, {
			onTranscript: (interim, finals) => {
				// Live transcript update — this is the key UX feature
				updateLiveTranscriptWidget(interim, finals);
				updateVoiceStatus();
			},
			onDone: (fullText, meta) => {
				voiceDebug("onDone callback", { fullText: fullText.slice(0, 100), meta, voiceState, spaceConsumed });
				activeSession = null;
				clearRecordingAnimTimer();
				lastStopTime = Date.now();

				if (!fullText.trim()) {
					// ── DISTINGUISH SILENCE VS NO SPEECH ──
					hideWidget();
					playSound("error");
					// Full state reset on empty result
					spaceConsumed = false;
					spaceDownTime = null;
					spacePressCount = 0;
					holdConfirmed = false;
					clearHoldTimer();
					clearReleaseTimer();
					errorCooldownUntil = Date.now() + 3000; // Block re-activation for 3s
					if (!meta.hadAudio) {
						ctx?.ui.notify("Microphone captured no audio. Check mic permissions.", "error");
					} else if (!meta.hadSpeech) {
						ctx?.ui.notify("Microphone captured silence — no speech detected.", "warning");
					} else {
						ctx?.ui.notify("No speech detected.", "warning");
					}
					setVoiceState("idle");
					return;
				}

				// Live transcript already showed text in editor — just finalize quietly.
				// No check mark widget, no "Processing" animation, no STT notification needed.
				hideWidget();

				if (ctx?.hasUI) {
					// Check for voice commands first
					const cmd = detectVoiceCommand(fullText);
					if (cmd.isCommand && cmd.action) {
						executeVoiceCommand(cmd.action);
						playSound("stop");
						addToHistory(fullText, (Date.now() - recordingStart) / 1000);
						setVoiceState("idle");
						return;
					}

					// Process voice shortcuts (new line, period, etc.)
					const processedText = processVoiceShortcuts(fullText);

					// The editor already has the live transcript via updateLiveTranscriptWidget.
					// Only set final text if the editor still has content.
					// If user already hit Enter (editor cleared), don't re-insert.
					const currentEditorText = ctx.ui.getEditorText?.() ?? "";
					if (currentEditorText.trim()) {
						const prefix = editorTextBeforeVoice ? editorTextBeforeVoice + " " : "";
						ctx.ui.setEditorText(prefix + processedText);
					}
					const elapsed = ((Date.now() - recordingStart) / 1000).toFixed(1);
					addToHistory(fullText, parseFloat(elapsed));
				}
				playSound("stop");
				// Full state reset on successful completion
				spaceConsumed = false;
				spaceDownTime = null;
				spacePressCount = 0;
				holdConfirmed = false;
				clearHoldTimer();
				clearReleaseTimer();
				setVoiceState("idle");
			},
			onError: (err) => {
				activeSession = null;
				clearRecordingAnimTimer();
				hideWidget();

				// ── STOP THE LOOP ──
				// On error, fully reset ALL hold state AND set a cooldown
				// so incoming key-repeat events can't re-trigger activation.
				spaceConsumed = false;
				spaceDownTime = null;
				spacePressCount = 0;
				holdConfirmed = false;
				clearHoldTimer();
				clearReleaseTimer();
				clearWarmupWidget();
				retryAttempts = 0;
				errorCooldownUntil = Date.now() + 5000; // Block re-activation for 5 seconds

				ctx?.ui.notify(`Voice error: ${err}`, "error");
				playSound("error");
				setVoiceState("idle");
			},
		});

		if (!session) {
			// startStreamingSession returned null — reset ALL state
			spaceConsumed = false;
			spaceDownTime = null;
			spacePressCount = 0;
			holdConfirmed = false;
			clearHoldTimer();
			clearReleaseTimer();
			setVoiceState("idle");
			return false;
		}

		activeSession = session;

		// Status timer for elapsed time
		statusTimer = setInterval(() => {
			if (voiceState === "recording") {
				updateVoiceStatus();
				const elapsed = (Date.now() - recordingStart) / 1000;
				if (elapsed >= MAX_RECORDING_SECS) {
					stopVoiceRecording();
				}
			}
		}, 1000);

		showRecordingWidget();
		playSound("start");
		return true;
	}

	async function startLegacyRecording(): Promise<boolean> {
		if (!ctx) return false;
		tempFile = path.join(os.tmpdir(), `pi-voice-${Date.now()}.wav`);
		if (!startLegacyRecordingToFile(tempFile)) {
			ctx.ui.notify("Voice requires SoX. Install: brew install sox", "error");
			return false;
		}

		setVoiceState("recording");
		statusTimer = setInterval(() => {
			if (voiceState === "recording") {
				updateVoiceStatus();
				const elapsed = (Date.now() - recordingStart) / 1000;
				if (elapsed >= MAX_RECORDING_SECS) {
					stopVoiceRecording();
				}
			}
		}, 1000);

		showRecordingWidget();
		playSound("start");
		return true;
	}

	async function stopVoiceRecording() {
		voiceDebug("stopVoiceRecording called", { voiceState, hasActiveSession: !!activeSession });
		if (voiceState !== "recording" || !ctx) return;
		if (statusTimer) { clearInterval(statusTimer); statusTimer = null; }

		if (activeSession) {
			setVoiceState("finalizing");
			clearRecordingAnimTimer();
			// Don't show "Processing..." widget — live transcript is already in the editor.
			// Just stop the recording widget and finalize silently.
			hideWidget();
			stopStreamingSession(activeSession);
			return;
		}

		// Legacy path
		const elapsed = ((Date.now() - recordingStart) / 1000).toFixed(1);
		const audioFile = tempFile;
		setVoiceState("finalizing");
		clearRecordingAnimTimer();
		showFinalizingWidget();

		await stopLegacyRecording();

		if (!audioFile || !fs.existsSync(audioFile)) {
			ctx.ui.notify("No audio recorded.", "warning");
			hideWidget();
			setVoiceState("idle");
			return;
		}

		const stats = fs.statSync(audioFile);
		if (stats.size < 1000) {
			ctx.ui.notify("Recording too short — mic captured silence.", "warning");
			try { fs.unlinkSync(audioFile); } catch {}
			tempFile = null;
			hideWidget();
			setVoiceState("idle");
			return;
		}

		await ensureDaemon(config).catch(() => {});

		const result = await transcribeAudioFile(audioFile, config);
		try { fs.unlinkSync(audioFile); } catch {}
		if (tempFile === audioFile) tempFile = null;

		hideWidget();

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

		if (ctx.hasUI) {
			const prefix = editorTextBeforeVoice ? editorTextBeforeVoice + " " : "";
			ctx.ui.setEditorText(prefix + transcript);
			ctx.ui.notify(
				`STT (${elapsed}s): ${transcript.slice(0, 80)}${transcript.length > 80 ? "…" : ""}`,
				"info",
			);
		}

		setVoiceState("idle");
	}

	// ─── Hold-to-Talk State Machine ─────────────────────────────────────────
	//
	// SPACE key handling with STRICT hold-duration detection.
	//
	// TWO TERMINAL MODES:
	//
	// A) KITTY PROTOCOL (Ghostty on Linux, Kitty, WezTerm, etc.):
	//    True key-down/repeat/release events. On first SPACE press,
	//    immediately enter warmup (show countdown). If released before
	//    HOLD_THRESHOLD_MS → cancel warmup, type a space. If held past
	//    threshold → start recording. True release event stops recording.
	//    No timer-based release detection needed.
	//
	// B) NON-KITTY (macOS Terminal, Ghostty on macOS, etc.):
	//    No key-release event. Holding sends rapid press events (~30-90ms apart).
	//    A single tap sends exactly ONE press.
	//    Algorithm:
	//      1. First SPACE press → record time, start release-detect timer.
	//      2. No more presses within RELEASE_DETECT_MS (500ms) → TAP → type space.
	//      3. Rapid presses arrive → user is HOLDING. After REPEAT_CONFIRM_COUNT
	//         rapid presses → enter warmup, show countdown.
	//      4. After HOLD_THRESHOLD_MS (2000ms) from first press → start recording.
	//      5. Recording continues while key-repeat events arrive.
	//         Gap > RELEASE_DETECT_MS after RECORDING_GRACE_MS → stop.
	//
	// The RECORDING_GRACE_MS prevents the state transition at recording start
	// from being mistaken for a key release (brief gap in events).

	function onSpaceReleaseDetected() {
		releaseDetectTimer = null;
		voiceDebug("onSpaceReleaseDetected", { voiceState, holdConfirmed, spaceConsumed, spaceDownTime, spacePressCount, timeSinceRecStart: spaceConsumed ? Date.now() - recordingStartedAt : null });

		// If we never confirmed this was a hold (< REPEAT_CONFIRM_COUNT rapid presses),
		// then it was a TAP → space already passed through naturally (not consumed)
		if (!holdConfirmed && voiceState === "idle") {
			clearHoldTimer();
			clearWarmupWidget();
			hideWidget();
			spaceDownTime = null;
			spaceConsumed = false;
			spacePressCount = 0;
			holdConfirmed = false;
			// No need to type a space — the first press was NOT consumed,
			// so it already reached the focused UI component naturally.
			return;
		}

		// Released during warmup — cancel (user held but not long enough)
		if (voiceState === "warmup") {
			clearHoldTimer();
			clearWarmupWidget();
			hideWidget();
			setVoiceState("idle");
			spaceDownTime = null;
			spaceConsumed = false;
			spacePressCount = 0;
			holdConfirmed = false;
			// Don't type a space — user clearly intended to trigger voice but let go too early
			ctx?.ui.notify("Hold SPACE for 2 seconds to activate voice.", "info");
			return;
		}

		// Released during recording — but ONLY if grace period has passed.
		// The grace period prevents the recording-start transition from being
		// mistaken for a key release.
		if (spaceConsumed && voiceState === "recording") {
			const timeSinceRecordingStart = Date.now() - recordingStartedAt;
			voiceDebug("release detected during recording", { timeSinceRecordingStart, RECORDING_GRACE_MS });
			if (timeSinceRecordingStart < RECORDING_GRACE_MS) {
				// Too soon after recording started — this is likely a false release
				// caused by the state transition. Re-arm the release detector.
				voiceDebug("  → too soon, re-arming (grace period)");
				resetReleaseDetect();
				return;
			}
			voiceDebug("  → stopping recording");
			spaceConsumed = false;
			spaceDownTime = null;
			spacePressCount = 0;
			holdConfirmed = false;
			stopVoiceRecording();
		}
	}

	function resetReleaseDetect() {
		clearReleaseTimer();
		if (voiceState === "warmup" || voiceState === "recording" || spaceDownTime || spaceConsumed || holdConfirmed) {
			// Use longer timeout during active recording — key repeats can be
			// irregular when the system is under load (Deepgram streaming, etc.)
			const timeout = (voiceState === "recording" || spaceConsumed)
				? RELEASE_DETECT_RECORDING_MS
				: RELEASE_DETECT_MS;
			voiceDebug("resetReleaseDetect", { timeout, voiceState, spaceConsumed });
			releaseDetectTimer = setTimeout(onSpaceReleaseDetected, timeout);
		}
	}

	function setupHoldToTalk() {
		if (!ctx?.hasUI) return;

		if (terminalInputUnsub) { terminalInputUnsub(); terminalInputUnsub = null; }

		terminalInputUnsub = ctx.ui.onTerminalInput((data: string) => {
			if (!config.enabled) return undefined;

			// ── SPACE handling ──
			if (matchesKey(data, "space")) {
				// ── ERROR COOLDOWN: block all voice activation for 5s after an error ──
				if (errorCooldownUntil > Date.now()) {
					// During cooldown, let space through as a normal character
					return undefined;
				}

				voiceDebug("SPACE event", {
					isRelease: isKeyRelease(data),
					isRepeat: isKeyRepeat(data),
					voiceState,
					kittyReleaseDetected,
					holdConfirmed,
					spaceConsumed,
					spacePressCount,
					spaceDownTime: spaceDownTime ? Date.now() - spaceDownTime : null,
					dataHex: Buffer.from(data).toString("hex"),
				});

				// ── Kitty key-release (true release event) ──
				if (isKeyRelease(data)) {
					kittyReleaseDetected = true;
					clearReleaseTimer();

					// Released during warmup → cancel
					// If released very quickly (< 300ms), it was a tap → type a space
					// If released after 300ms+, user was trying voice → show hint
					if (voiceState === "warmup") {
						const holdDuration = spaceDownTime ? Date.now() - spaceDownTime : 0;
						clearHoldTimer();
						clearWarmupWidget();
						hideWidget();
						setVoiceState("idle");
						spaceDownTime = null;
						spaceConsumed = false;
						spacePressCount = 0;
						holdConfirmed = false;
						if (holdDuration < 300) {
							// Quick tap — just type a space
							if (ctx?.hasUI) ctx.ui.setEditorText((ctx.ui.getEditorText() || "") + " ");
						} else {
							// Held long enough to see warmup but let go → show hint
							ctx?.ui.notify("Hold SPACE for 2 seconds to activate voice.", "info");
						}
						return { consume: true };
					}

					// Tap: released before warmup even started (shouldn't happen in
					// Kitty path since we enter warmup on first press, but handle anyway)
					if (spaceDownTime && !holdConfirmed && voiceState === "idle") {
						clearHoldTimer();
						spaceDownTime = null;
						spacePressCount = 0;
						holdConfirmed = false;
						if (ctx?.hasUI) ctx.ui.setEditorText((ctx.ui.getEditorText() || "") + " ");
						return { consume: true };
					}

					// Released during recording → stop
					if (spaceConsumed && voiceState === "recording") {
						spaceConsumed = false;
						spaceDownTime = null;
						spacePressCount = 0;
						holdConfirmed = false;
						stopVoiceRecording();
						return { consume: true };
					}

					spaceDownTime = null;
					spaceConsumed = false;
					spacePressCount = 0;
					holdConfirmed = false;
					return undefined;
				}

				// ── Kitty key-repeat ──
				if (isKeyRepeat(data)) {
					// Already in recording/finalizing — just consume
					if (voiceState === "recording" || voiceState === "finalizing" || spaceConsumed) {
						return { consume: true };
					}
					// Already in warmup — consume (hold timer is running)
					if (voiceState === "warmup") {
						return { consume: true };
					}

					// During initial hold detection: if we took PATH B on first
					// press (because kittyReleaseDetected was false), we need to
					// count these repeats to confirm the hold. Update state so
					// onSpaceReleaseDetected won't fire a false tap.
					if (spaceDownTime && !holdConfirmed) {
						kittyReleaseDetected = true; // Now we know it's Kitty
						clearReleaseTimer(); // Cancel non-Kitty release timer

						const now = Date.now();
						spacePressCount++;
						lastSpacePressTime = now;

						// Enough repeats to confirm hold — enter warmup
						if (spacePressCount >= REPEAT_CONFIRM_COUNT) {
							holdConfirmed = true;
							setVoiceState("warmup");
							showWarmupWidget();

							const alreadyElapsed = now - (spaceDownTime || now);
							const remaining = Math.max(0, HOLD_THRESHOLD_MS - alreadyElapsed);

							holdActivationTimer = setTimeout(() => {
								holdActivationTimer = null;
								if (voiceState === "warmup") {
									clearWarmupWidget();
									spaceConsumed = true;
									recordingStartedAt = Date.now();
									// Kitty repeat handler: clear release timer during
									// async recording startup to prevent false stop
									clearReleaseTimer();
									voiceDebug("holdActivationTimer fired → starting recording (Kitty repeat path)");
									startVoiceRecording().then((ok) => {
										if (!ok) {
											spaceConsumed = false;
											spaceDownTime = null;
											spacePressCount = 0;
											holdConfirmed = false;
											setVoiceState("idle");
										}
									}).catch((err) => {
										voiceDebug('startVoiceRecording THREW', { error: String(err) });
										spaceConsumed = false;
										spaceDownTime = null;
										spacePressCount = 0;
										holdConfirmed = false;
										errorCooldownUntil = Date.now() + 5000;
										setVoiceState('idle');
									});
								} else {
									spaceDownTime = null;
									spaceConsumed = false;
									spacePressCount = 0;
									holdConfirmed = false;
								}
							}, remaining);
						}
						return { consume: true };
					}

					return { consume: true };
				}

				// === Key PRESS (not repeat, not release) ===
				//
				// TWO TERMINAL MODES:
				//
				// A) Kitty protocol (kittyReleaseDetected = true):
				//    Press fires ONCE on key-down. Repeats come as isKeyRepeat().
				//    Release comes as isKeyRelease(). NO timer-based release detection
				//    needed — the true release event handles everything.
				//    On first press: enter warmup immediately and start hold timer.
				//    (No need to wait for repeats to confirm hold.)
				//
				// B) Non-Kitty (macOS Terminal, etc.):
				//    Holding a key sends rapid "press" events (~30-90ms apart).
				//    A single tap sends exactly ONE press. There is NO release event.
				//    We detect "tap vs hold" by counting rapid presses, and detect
				//    "release" when no press arrives within RELEASE_DETECT_MS.

				// If finalizing → ignore
				if (voiceState === "finalizing") {
					return { consume: true };
				}

				// If already recording → just consume
				if (voiceState === "recording") {
					if (!kittyReleaseDetected) {
						voiceDebug("SPACE during recording → re-arm release detect");
						resetReleaseDetect();
					}
					return { consume: true };
				}

				// If already in warmup → consume
				if (voiceState === "warmup") {
					if (!kittyReleaseDetected) {
						voiceDebug("SPACE during warmup → re-arm release detect");
						resetReleaseDetect();
					}
					return { consume: true };
				}

				// If we've already consumed space for this hold → consume
				// This handles the gap between holdActivationTimer firing and
				// voiceState transitioning to "recording" (async gap)
				if (spaceConsumed) {
					if (!kittyReleaseDetected) {
						voiceDebug("SPACE while spaceConsumed (async gap) → re-arm release detect");
						resetReleaseDetect();
					}
					return { consume: true };
				}

				// ──────────────────────────────────────────────────────────
				// PATH A: Kitty protocol — true key events available
				// ──────────────────────────────────────────────────────────
				if (kittyReleaseDetected) {
					// First press → immediately enter warmup (release event
					// will cancel if it was a tap)
					if (voiceState === "idle") {
						spaceDownTime = Date.now();
						spaceConsumed = false;
						spacePressCount = 1;
						lastSpacePressTime = Date.now();
						holdConfirmed = true; // Kitty: trust the press, release cancels

						setVoiceState("warmup");
						showWarmupWidget();

						holdActivationTimer = setTimeout(() => {
							holdActivationTimer = null;
							if (voiceState === "warmup") {
								clearWarmupWidget();
								spaceConsumed = true;
								recordingStartedAt = Date.now();
								voiceDebug("holdActivationTimer fired → starting recording (Kitty path)");
								startVoiceRecording().then((ok) => {
									if (!ok) {
										spaceConsumed = false;
										spaceDownTime = null;
										spacePressCount = 0;
										holdConfirmed = false;
										setVoiceState("idle");
									}
								}).catch((err) => {
									voiceDebug('startVoiceRecording THREW', { error: String(err) });
									spaceConsumed = false;
									spaceDownTime = null;
									spacePressCount = 0;
									holdConfirmed = false;
									errorCooldownUntil = Date.now() + 5000;
									setVoiceState('idle');
								});
							} else {
								spaceDownTime = null;
								spaceConsumed = false;
								spacePressCount = 0;
								holdConfirmed = false;
							}
						}, HOLD_THRESHOLD_MS);

						return { consume: true };
					}
					return { consume: true };
				}

				// ──────────────────────────────────────────────────────────
				// PATH B: Non-Kitty — gap-based hold/release detection
				// ──────────────────────────────────────────────────────────
				// Holding a key sends rapid press events.
				// We count presses and measure gaps to detect holds vs taps.
				if (spaceDownTime) {
					const now = Date.now();
					const gap = now - lastSpacePressTime;

					if (gap < REPEAT_CONFIRM_MS) {
						// Rapid press = user is holding
						spacePressCount++;
						lastSpacePressTime = now;

						if (spacePressCount >= REPEAT_CONFIRM_COUNT && !holdConfirmed) {
							holdConfirmed = true;
							setVoiceState("warmup");
							showWarmupWidget();

							const alreadyElapsed = now - spaceDownTime;
							const remaining = Math.max(0, HOLD_THRESHOLD_MS - alreadyElapsed);

							holdActivationTimer = setTimeout(() => {
								holdActivationTimer = null;
								if (voiceState === "warmup") {
									clearWarmupWidget();
									spaceConsumed = true;
									recordingStartedAt = Date.now();
									// CRITICAL: Clear release timer and DO NOT re-arm.
									// The next key-repeat press event will re-arm it.
									// Without this, the async startVoiceRecording creates
									// a gap where the release timer fires falsely.
									clearReleaseTimer();
									voiceDebug("holdActivationTimer fired → starting recording (non-Kitty)");
									startVoiceRecording().then((ok) => {
										if (!ok) {
											spaceConsumed = false;
											spaceDownTime = null;
											spacePressCount = 0;
											holdConfirmed = false;
											setVoiceState("idle");
										}
										// Do NOT re-arm release detect here!
										// The next SPACE key-repeat event will do it.
										// Re-arming here causes false stops because
										// the timer fires during the async gap.
									}).catch((err) => {
										voiceDebug('startVoiceRecording THREW', { error: String(err) });
										spaceConsumed = false;
										spaceDownTime = null;
										spacePressCount = 0;
										holdConfirmed = false;
										errorCooldownUntil = Date.now() + 5000;
										setVoiceState('idle');
									});
								} else {
									spaceDownTime = null;
									spaceConsumed = false;
									spacePressCount = 0;
									holdConfirmed = false;
								}
							}, remaining);
						}

						resetReleaseDetect();
						return { consume: true };
					} else {
						// Gap too large → previous hold abandoned, new tap
						clearHoldTimer();
						clearReleaseTimer();
						clearWarmupWidget();
						hideWidget();
						const wasInWarmup = (voiceState as VoiceState) === "warmup";
						if (wasInWarmup) setVoiceState("idle");
						// Only type a space if we weren't already in warmup
						// (if we were in warmup, user was trying to activate voice, not type)
						// Note: first space already passed through naturally (not consumed)
						// so we don't need to manually type it here
						spaceDownTime = null;
						spacePressCount = 0;
						holdConfirmed = false;
						spaceConsumed = false;
						// Fall through to treat this as a new first press
					}
				}

				// IDLE — first SPACE press (non-Kitty path)
				// Do NOT consume — let it pass through to whatever UI is focused
				// (editor, search box, picker, etc.). Only start consuming after
				// we confirm it's a hold via REPEAT_CONFIRM_COUNT rapid presses.
				if (voiceState === "idle") {
					spaceDownTime = Date.now();
					spaceConsumed = false;
					spacePressCount = 1;
					lastSpacePressTime = Date.now();
					holdConfirmed = false;

					resetReleaseDetect();

					// Don't consume — let the space reach the focused UI component
					return undefined;
				}

				if (spaceConsumed) return { consume: true };
				return undefined;
			}

			// ── Any other key pressed → cancel potential hold ──
			if (spaceDownTime && !holdConfirmed && voiceState === "idle") {
				clearHoldTimer();
				clearReleaseTimer();
				spaceDownTime = null;
				spacePressCount = 0;
				holdConfirmed = false;
				spaceConsumed = false;
				// No need to insert a space manually — the first space press was
				// already allowed to pass through to the focused UI component.
				return undefined;
			}

			if (voiceState === "warmup" && holdConfirmed && !spaceConsumed) {
				clearHoldTimer();
				clearReleaseTimer();
				clearWarmupWidget();
				hideWidget();
				setVoiceState("idle");
				spaceDownTime = null;
				spacePressCount = 0;
				holdConfirmed = false;
				spaceConsumed = false;
				return undefined;
			}

			// ── Escape key — clear voice transcript ──
			if (matchesKey(data, "escape")) {
				// During recording: cancel recording and clear transcript
				if (voiceState === "recording" || voiceState === "warmup" || voiceState === "finalizing") {
					voiceDebug("Escape pressed → canceling voice");
					if (activeSession) {
						abortSession(activeSession);
						activeSession = null;
					}
					clearRecordingAnimTimer();
					clearWarmupWidget();
					clearHoldTimer();
					clearReleaseTimer();
					hideWidget();
					if (statusTimer) { clearInterval(statusTimer); statusTimer = null; }
					// Restore editor text to what it was before recording
					if (ctx?.hasUI) ctx.ui.setEditorText(editorTextBeforeVoice);
					spaceConsumed = false;
					spaceDownTime = null;
					spacePressCount = 0;
					holdConfirmed = false;
					playSound("error");
					setVoiceState("idle");
					return { consume: true };
				}

				// In idle: if editor has text from a recent voice transcription, clear it
				if (voiceState === "idle" && lastStopTime > 0 && (Date.now() - lastStopTime) < 30000) {
					if (ctx?.hasUI) {
						const currentText = ctx.ui.getEditorText() || "";
						if (currentText.trim()) {
							ctx.ui.setEditorText("");
							return { consume: true };
						}
					}
				}


			}

			return undefined;
		});
	}


	// ─── Shortcuts ───────────────────────────────────────────────────────────

	pi.registerShortcut("ctrl+shift+v", {
		description: "Toggle voice recording (start/stop)",
		handler: async (handlerCtx) => {
			ctx = handlerCtx;
			if (!config.enabled) {
				handlerCtx.ui.notify("Voice disabled. Use /voice on", "warning");
				return;
			}
			if (dictationMode) {
				// Ctrl+Shift+V stops dictation mode
				dictationMode = false;
				if (voiceState === "recording") {
					await stopVoiceRecording();
				}
				handlerCtx.ui.notify("Dictation mode stopped.", "info");
				return;
			}
			if (voiceState === "idle") {
				spaceConsumed = true;
				const ok = await startVoiceRecording();
				if (!ok) {
					spaceConsumed = false;
				}
			} else if (voiceState === "recording") {
				spaceConsumed = false;
				spaceDownTime = null;
				clearHoldTimer();
				await stopVoiceRecording();
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

		// Auto-capture DEEPGRAM_API_KEY from env into config
		if (process.env.DEEPGRAM_API_KEY && !config.deepgramApiKey) {
			config.deepgramApiKey = process.env.DEEPGRAM_API_KEY;
			if (configSource !== "default") {
				saveConfig(config, config.scope, currentCwd);
			}
		}

		// Try to load DEEPGRAM_API_KEY from shell if not available
		if (!resolveDeepgramApiKey(config) && config.backend === "deepgram") {
			try {
				const result = spawnSync("zsh", ["-ic", "echo $DEEPGRAM_API_KEY"], {
					stdio: ["pipe", "pipe", "pipe"],
					timeout: 3000,
					env: { ...process.env, HOME: os.homedir() },
				});
				const shellKey = result.stdout?.toString().trim();
				if (shellKey && shellKey.length > 5) {
					config.deepgramApiKey = shellKey;
					process.env.DEEPGRAM_API_KEY = shellKey;
					if (configSource !== "default") {
						saveConfig(config, config.scope, currentCwd);
					}
				}
			} catch {}
		}

		if (config.enabled && config.onboarding.completed) {
			updateVoiceStatus();
			setupHoldToTalk();
			if (!isDeepgramStreaming(config)) {
				ensureDaemon(config).catch(() => {});
			}
		}
	});

	pi.on("session_shutdown", async () => {
		voiceCleanup();
		if (terminalInputUnsub) { terminalInputUnsub(); terminalInputUnsub = null; }
	});

	pi.on("session_switch", async (_event, switchCtx) => {
		// Clean up any active recording before switching
		voiceCleanup();
		ctx = switchCtx;
		currentCwd = switchCtx.cwd;
		const loaded = loadConfigWithSource(switchCtx.cwd);
		config = loaded.config;
		configSource = loaded.source;
		updateSocketPath(config, currentCwd);
		setupHoldToTalk();
		updateVoiceStatus();
	});

	// ─── /voice command ──────────────────────────────────────────────────────

	pi.registerCommand("voice", {
		description: "Voice: /voice [on|off|stop|dictate|history|test|info|setup|doctor|backends|daemon|cat on|cat off]",
		handler: async (args, cmdCtx) => {
			ctx = cmdCtx;
			const sub = (args || "").trim().toLowerCase();

			if (sub === "cat on" || sub === "style cat") {
				config.catAnimation = true;
				saveConfig(config, config.scope, currentCwd);
				cmdCtx.ui.notify("Voice cat animation enabled.", "info");
				return;
			}

			if (sub === "cat off" || sub === "style minimal") {
				config.catAnimation = false;
				saveConfig(config, config.scope, currentCwd);
				cmdCtx.ui.notify("Voice minimal widget enabled (cat animation off).", "info");
				return;
			}

			if (sub === "on") {
				config.enabled = true;
				updateVoiceStatus();
				setupHoldToTalk();
				if (!isDeepgramStreaming(config)) {
					ensureDaemon(config).catch(() => {});
				}
				const mode = isDeepgramStreaming(config) ? "Deepgram streaming" : config.backend;
				cmdCtx.ui.notify([
					`Voice enabled (${mode}).`,
					"",
					"  Hold SPACE (2s) → release to transcribe",
					"  Ctrl+Shift+V → toggle recording on/off",
					"  Quick SPACE tap → types a space (no voice)",
					"  /voice dictate  → continuous mode (no hold)",
					"  /voice history  → recent transcriptions",
					"",
					"  Voice commands: 'hey pi, run tests' → auto-executes",
					"  Shortcuts: 'new line', 'period', 'submit'",
					"",
					"  Live transcription shown while speaking",
				].join("\n"), "info");
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

			if (sub === "stop") {
				if (dictationMode) {
					dictationMode = false;
					if (voiceState === "recording") {
						await stopVoiceRecording();
					}
					cmdCtx.ui.notify("Dictation mode stopped.", "info");
				} else if (voiceState === "recording") {
					await stopVoiceRecording();
					cmdCtx.ui.notify("Recording stopped and transcribed.", "info");
				} else if (voiceState === "warmup") {
					clearHoldTimer();
					clearWarmupWidget();
					hideWidget();
					setVoiceState("idle");
					cmdCtx.ui.notify("Warmup cancelled.", "info");
				} else {
					cmdCtx.ui.notify("No recording in progress.", "info");
				}
				return;
			}

			// /voice dictate — continuous dictation mode
			if (sub === "dictate") {
				if (!config.enabled) {
					cmdCtx.ui.notify("Voice disabled. Use /voice on", "warning");
					return;
				}
				if (dictationMode) {
					cmdCtx.ui.notify("Already in dictation mode. /voice stop to end.", "info");
					return;
				}
				dictationMode = true;
				dictationText = ctx?.hasUI ? (ctx.ui.getEditorText() || "") : "";
				editorTextBeforeVoice = dictationText;
				const ok = await startVoiceRecording();
				if (ok) {
					cmdCtx.ui.notify([
						"🎤 Continuous dictation mode active.",
						"",
						"  Speak freely — no need to hold SPACE.",
						"  /voice stop → finalize and stop",
						"  Ctrl+Shift+V → also stops dictation",
					].join("\n"), "info");
				} else {
					dictationMode = false;
					cmdCtx.ui.notify("Failed to start dictation.", "error");
				}
				return;
			}

			// /voice history — show recent transcriptions
			if (sub === "history") {
				if (recordingHistory.length === 0) {
					cmdCtx.ui.notify("No recording history yet.", "info");
					return;
				}
				const lines = ["📜 Recent transcriptions:", ""];
				const show = recordingHistory.slice(0, 20);
				for (const entry of show) {
					const time = new Date(entry.timestamp).toLocaleTimeString();
					const dur = entry.duration.toFixed(1);
					const preview = entry.text.slice(0, 60) + (entry.text.length > 60 ? "…" : "");
					lines.push(`  ${time} (${dur}s): ${preview}`);
				}
				if (recordingHistory.length > 20) {
					lines.push(`  … and ${recordingHistory.length - 20} more`);
				}
				cmdCtx.ui.notify(lines.join("\n"), "info");
				return;
			}

			if (sub === "test") {
				cmdCtx.ui.notify("Testing voice setup…", "info");
				const diagnostics = scanEnvironment(TRANSCRIBE_SCRIPT);
				const dgKey = resolveDeepgramApiKey(config);
				const streaming = isDeepgramStreaming(config);
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
					`  streaming: ${streaming ? "YES (Deepgram WS)" : "NO (batch)"}`,
					`  DEEPGRAM_API_KEY: ${dgKey ? "set (" + dgKey.slice(0, 8) + "…)" : "NOT SET"}`,
					`  onboarding: ${config.onboarding.completed ? "complete" : "incomplete"}`,
					`  python3: ${diagnostics.hasPython ? "OK" : "missing"}`,
					`  sox/rec: ${diagnostics.hasSox ? "OK" : "missing"}`,
					`  daemon: ${daemonUp ? "running" : "not running"}`,
					`  state: ${voiceState}`,
					`  hold threshold: ${HOLD_THRESHOLD_MS}ms`,
					`  release detect: ${RELEASE_DETECT_MS}ms`,
					`  kitty protocol: ${kittyReleaseDetected ? "detected" : "not detected"}`,
				];

				if (diagnostics.hasSox) {
					const testFile = path.join(os.tmpdir(), "pi-voice-test.wav");
					const testProc = spawn("rec", ["-q", "-r", "16000", "-c", "1", "-b", "16", "-d", "1", testFile], { stdio: "pipe" });
					testProc.on("error", () => {}); // Prevent unhandled error crash
					await new Promise<void>((resolve) => {
						testProc.on("close", () => resolve());
						setTimeout(() => { try { testProc.kill(); } catch {} resolve(); }, 2000);
					});
					if (fs.existsSync(testFile)) {
						const size = fs.statSync(testFile).size;
						lines.push(`  mic sample: OK (${size} bytes)`);
						try { fs.unlinkSync(testFile); } catch {}
					} else {
						lines.push("  mic sample: missing audio");
					}
				}

				if (!dgKey && config.backend === "deepgram") {
					lines.push("");
					lines.push("⚠️  DEEPGRAM_API_KEY not set! Add to ~/.zshrc or ~/.env.secrets");
					lines.push("   export DEEPGRAM_API_KEY=your_key_here");
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
				const streaming = isDeepgramStreaming(config);

				cmdCtx.ui.notify([
					`Voice config:`,
					`  enabled:    ${config.enabled}`,
					`  mode:       ${config.mode}`,
					`  scope:      ${config.scope}`,
					`  backend:    ${config.backend}`,
					`  model:      ${config.model}`,
					`  model stat: ${modelReadiness}`,
					`  language:   ${config.language}`,
					`  streaming:  ${streaming ? "YES (Deepgram WebSocket)" : "NO (batch)"}`,
					`  state:      ${voiceState}`,
					`  setup:      ${config.onboarding.completed ? `complete (${config.onboarding.source ?? "unknown"})` : "incomplete"}`,
					`  socket:     ${activeSocketPath}`,
					`  daemon:     ${daemonUp ? "running" : "stopped"}${daemonInfo}`,
					`  hold-key:   SPACE (hold ≥${HOLD_THRESHOLD_MS}ms) or Ctrl+Shift+V (toggle)`,
					`  kitty:      ${kittyReleaseDetected ? "yes" : "no"}`,
				].join("\n"), "info");
				return;
			}

			if (sub === "daemon" || sub === "daemon start") {
				cmdCtx.ui.notify("Starting STT daemon…", "info");
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
				if (!HAS_TRANSCRIBE_SCRIPT) {
					cmdCtx.ui.notify("Using Deepgram streaming (no local backends needed).", "info");
					return;
				}
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

	// ─── Dedicated setup command ─────────────────────────────────────────────

	pi.registerCommand("voice-setup", {
		description: "Configure voice input — select backend, model, and language",
		handler: async (_args, cmdCtx) => {
			ctx = cmdCtx;
			const diagnostics = scanEnvironment(TRANSCRIBE_SCRIPT);
			const result = await runVoiceOnboarding(cmdCtx, config, diagnostics);
			if (!result) {
				cmdCtx.ui.notify("Voice setup cancelled.", "warning");
				return;
			}
			await finalizeAndSaveSetup(cmdCtx, result.config, result.selectedScope, result.summaryLines, "setup-command");
		},
	});

}

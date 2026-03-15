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
 *   warmup:     User holds SPACE for ≥ HOLD_THRESHOLD_MS (1200ms).
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
 *      Held ≥ 1.2s → activate recording.
 *      True release event stops recording.
 *
 *   B) Non-Kitty (macOS Terminal, Ghostty on macOS):
 *      No key-release event. Holding sends rapid press events (~30-90ms apart).
 *      First SPACE press → record time, start release-detect timer (500ms).
 *      No more presses within 500ms → TAP → type a space.
 *      Rapid presses detected → user is HOLDING.
 *      After REPEAT_CONFIRM_COUNT (6) rapid presses → enter warmup.
 *      After HOLD_THRESHOLD_MS (1200ms) from first press → activate recording.
 *      Gap > RELEASE_DETECT_MS (500ms) after RECORDING_GRACE_MS (800ms) → stop.
 *
 *   ENTERPRISE FALLBACKS
 *   ────────────────────
 *   • Session corruption guard: new recording request during
 *     finalizing automatically cancels the stale session first.
 *   • Stale transcript cleanup: any prior transcript is cleared
 *     before new recording begins.
 *   • Silence vs. no-speech: distinguishes "mic captured silence"
 *     from "no speech detected" with distinct user messages.
 *
 * Activation:
 *   - Hold SPACE (≥1200ms) → release to finalize
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

import { spawn, spawnSync, type ChildProcess } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import {
	DEFAULT_CONFIG,
	loadConfigWithSource,
	saveConfig,
	type VoiceConfig,
	type VoiceSettingsScope,
} from "./voice/config";
import { finalizeOnboardingConfig, runVoiceOnboarding, pickLanguage, languageDisplayName, modelForLanguage } from "./voice/onboarding";
import { buildDeepgramWsUrl, resolveDeepgramApiKey, SAMPLE_RATE, CHANNELS } from "./voice/deepgram";
import {
	startLocalSession, stopLocalSession, abortLocalSession,
	checkLocalServer, LOCAL_MODELS, DEFAULT_LOCAL_ENDPOINT,
	getLanguagesForLocalModel, isLanguageSupportedByModel, localLanguageDisplayName,
	type LocalSession,
} from "./voice/local";


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

const KEEPALIVE_INTERVAL_MS = 8000;
const MAX_RECORDING_SECS = 120;

// Hold-to-talk timing — Apple-style deliberate hold detection
// The goal: typing normally should NEVER accidentally trigger voice.
// Only a clearly intentional long press activates it.
const HOLD_THRESHOLD_MS = 1200;   // Must hold for 1.2s before voice activates
                                   // (Apple Caps Lock uses ~1s — we use slightly more to be safe)
const RELEASE_DETECT_MS = 500;    // Gap in key-repeat that means "released" (non-Kitty)
                                   // macOS default InitialKeyRepeat is ~375ms, so 500ms
                                   // ensures the first repeat arrives before we decide "tap"
const REPEAT_CONFIRM_COUNT = 6;   // Need this many rapid repeat presses to confirm "holding"
                                   // At ~30ms repeat rate, 6 presses ≈ 180ms of continuous holding
                                   // This filters out brief pauses while typing
const REPEAT_CONFIRM_MS = 700;    // Max gap between presses to count as rapid repeat
                                  // macOS initial key-repeat delay is ~417-583ms depending on settings
                                   // Must be > macOS InitialKeyRepeat (~375ms)
const RECORDING_GRACE_MS = 800;   // After recording starts, ignore release for this long
                                   // Covers async gap from holdActivationTimer → startVoiceRecording
const RELEASE_DETECT_RECORDING_MS = 250; // During active recording, gap before we consider
                                          // the key released (non-Kitty only). macOS Terminal
                                          // key repeat fires every ~30-50ms. 250ms gap = released.
const TYPING_COOLDOWN_MS = 400;   // If ANY non-space key was pressed within this window,
                                   // ignore space holds (user is typing, not activating voice)
const TAIL_RECORDING_MS = 1500;   // Keep recording for 1.5s after space release to catch
                                   // trailing words. If user re-presses space within this
                                   // window, cancel the delayed stop and keep recording.
const CORRUPTION_GUARD_MS = 200;  // Min gap between stop and restart

// Debug logging — set PI_VOICE_DEBUG=1 to enable
const VOICE_DEBUG = !!process.env.PI_VOICE_DEBUG;
const VOICE_LOG_FILE = path.join(os.tmpdir(), "pi-voice-debug.log");

// ─── Audio level tracking (module scope so streaming can access) ──────
let audioLevel = 0;
let audioLevelSmoothed = 0;

function updateAudioLevel(chunk: Buffer) {
	const len = chunk.length;
	if (len < 2) return;
	const samples = len >> 1;
	let sum = 0;
	// Use Int16Array view when alignment permits (2-byte aligned), else fall back
	if ((chunk.byteOffset & 1) === 0) {
		const view = new Int16Array(chunk.buffer, chunk.byteOffset, samples);
		for (let i = 0; i < view.length; i++) {
			sum += view[i] * view[i];
		}
	} else {
		for (let i = 0; i < len - 1; i += 2) {
			const s = chunk.readInt16LE(i);
			sum += s * s;
		}
	}
	const rms = Math.sqrt(sum / samples);
	// Lower ceiling (2500) so normal speech hits 0.5-0.9 instead of 0.1-0.3
	// Power curve (^0.6) boosts quiet sounds for more visible reactivity
	audioLevel = Math.min(1, Math.pow(Math.min(rms / 2500, 1), 0.6));
	// Faster attack (0.35 old), slower decay — snappy peaks, smooth falloff
	audioLevelSmoothed = audioLevel > audioLevelSmoothed
		? audioLevelSmoothed * 0.35 + audioLevel * 0.65
		: audioLevelSmoothed * 0.75 + audioLevel * 0.25;
	// Shared state for other extensions (e.g. pi-pompom mouth animation).
	// Using a namespaced globalThis object instead of pi.events because
	// audio levels update at ~60Hz — event emission would be wasteful.
	const shared = ((globalThis as any).__piListen ??= {});
	shared.audioLevel = audioLevelSmoothed;
}

function voiceDebug(...args: unknown[]) {
	if (!VOICE_DEBUG) return;
	const ts = new Date().toISOString().split("T")[1];
	const line = `[voice ${ts}] ${args.map(a => typeof a === "object" ? JSON.stringify(a) : String(a)).join(" ")}\n`;
	try { fs.appendFileSync(VOICE_LOG_FILE, line); } catch {}
	process.stderr.write(line);
}

// Cache command existence checks — avoid sync spawnSync on every recording start
const _cmdExistsCache = new Map<string, boolean>();
function commandExists(cmd: string): boolean {
	const cached = _cmdExistsCache.get(cmd);
	if (cached !== undefined) return cached;
	const which = process.platform === "win32" ? "where" : "which";
	const result = spawnSync(which, [cmd], { stdio: "pipe", timeout: 3000 }).status === 0;
	_cmdExistsCache.set(cmd, result);
	return result;
}

interface AudioCaptureTool { name: string; cmd: string; args: string[]; }

// Try available audio capture tools in order of preference
let _cachedAudioTool: AudioCaptureTool | null | undefined;
function detectAudioCaptureTool(): AudioCaptureTool | null {
	if (_cachedAudioTool !== undefined) return _cachedAudioTool;

	// 1. SoX rec — purpose-built for recording, best quality
	if (commandExists("rec")) {
		_cachedAudioTool = {
			name: "sox",
			cmd: "rec",
			args: [
				"-q",
				"--buffer", "4096",
				"-c", String(CHANNELS),
				"-b", "16",
				"-e", "signed-integer",
				"-t", "raw",
				"-",
				"rate", String(SAMPLE_RATE),
			],
		};
		return _cachedAudioTool;
	}

	// 2. ffmpeg — widely installed, captures from default mic
	if (commandExists("ffmpeg")) {
		const isLinux = process.platform === "linux";
		const isMac = process.platform === "darwin";
		// Input device varies by platform
		const inputArgs = isMac
			? ["-f", "avfoundation", "-i", ":default"]
			: isLinux
				? ["-f", "pulse", "-i", "default"]
				: ["-f", "dshow", "-i", "audio=default"];
		_cachedAudioTool = {
			name: "ffmpeg",
			cmd: "ffmpeg",
			args: [
				...inputArgs,
				"-ac", String(CHANNELS),
				"-ar", String(SAMPLE_RATE),
				"-sample_fmt", "s16",
				"-f", "s16le",
				"-loglevel", "error",
				"pipe:1",
			],
		};
		return _cachedAudioTool;
	}

	// 3. arecord — built into Linux ALSA, zero install
	if (process.platform === "linux" && commandExists("arecord")) {
		_cachedAudioTool = {
			name: "arecord",
			cmd: "arecord",
			args: [
				"-q",
				"-f", "S16_LE",
				"-r", String(SAMPLE_RATE),
				"-c", String(CHANNELS),
				"-t", "raw",
			],
		};
		return _cachedAudioTool;
	}

	_cachedAudioTool = null;
	return null;
}

// ─── Deepgram WebSocket Streaming ────────────────────────────────────────────

interface StreamingSession {
	backend: "deepgram";
	ws: WebSocket;
	recProcess: ChildProcess;
	interimText: string;
	finalizedParts: string[];
	keepAliveTimer: ReturnType<typeof setInterval> | null;
	staleSessionTimer: ReturnType<typeof setTimeout> | null;
	closed: boolean;
	hadAudioData: boolean;       // Track if we received any audio data
	hadSpeech: boolean;          // Track if Deepgram detected any speech
	receivedMessage: boolean;    // Track if we got ANY message from Deepgram
	onTranscript: (interim: string, finals: string[]) => void;
	onDone: (fullText: string, meta: { hadAudio: boolean; hadSpeech: boolean }) => void;
	onError: (err: string) => void;
}

/** Union of session types — Deepgram streaming or local batch */
type VoiceSession = StreamingSession | LocalSession;

function startStreamingSession(
	config: VoiceConfig,
	callbacks: {
		onTranscript: (interim: string, finals: string[]) => void;
		onDone: (fullText: string, meta: { hadAudio: boolean; hadSpeech: boolean }) => void;
		onError: (err: string) => void;
	},
): StreamingSession | null {
	const apiKey = resolveDeepgramApiKey(config);
	voiceDebug("startStreamingSession", { hasApiKey: !!apiKey });
	if (!apiKey) {
		voiceDebug("startStreamingSession → no API key, calling onError");
		callbacks.onError("DEEPGRAM_API_KEY not set");
		return null;
	}

	// ── Audio capture: try rec (SoX) → ffmpeg → arecord (Linux ALSA) ──
	const audioTool = detectAudioCaptureTool();
	if (!audioTool) {
		voiceDebug("startStreamingSession → no audio capture tool found");
		callbacks.onError("No audio capture tool found. Install one of: sox, ffmpeg, or arecord (Linux)");
		return null;
	}
	voiceDebug("Using audio capture tool:", audioTool.name);

	const recProc = spawn(audioTool.cmd, audioTool.args, { stdio: ["pipe", "pipe", "pipe"] });

	recProc.stderr?.on("data", (d: Buffer) => {
		const msg = d.toString().trim();
		// Suppress noisy but harmless messages
		if (msg.includes("buffer overrun") || msg.includes("Discarding") || msg.includes("Last message repeated")) return;
		voiceDebug(`${audioTool.name} stderr:`, msg);
	});

	const wsUrl = buildDeepgramWsUrl(config);
	const ws = new WebSocket(wsUrl, {
		headers: {
			"Authorization": `Token ${apiKey}`,
		},
	} as any);

	// Connection timeout — abort if Deepgram doesn't respond within 10s
	const wsConnectTimeout = setTimeout(() => {
		if (ws.readyState !== WebSocket.OPEN) {
			voiceDebug("WebSocket connection timeout (10s)");
			try { ws.close(); } catch {}
			try { recProc.kill("SIGTERM"); } catch {}
			callbacks.onError("Deepgram connection timed out (10s). Check your network.");
		}
	}, 10_000);

	const session: StreamingSession = {
		backend: "deepgram",
		ws,
		recProcess: recProc,
		interimText: "",
		finalizedParts: [],
		keepAliveTimer: null,
		staleSessionTimer: null,
		closed: false,
		hadAudioData: false,
		hadSpeech: false,
		receivedMessage: false,
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
		clearTimeout(wsConnectTimeout);
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
				// Start stale-session watchdog on first audio chunk
				if (!session.staleSessionTimer && !session.receivedMessage) {
					session.staleSessionTimer = setTimeout(() => {
						if (!session.closed && !session.receivedMessage) {
							voiceDebug("Stale session: no Deepgram response after 15s of audio");
							session.onError("No response from Deepgram (15s). Check your API key and network.");
						}
					}, 15_000);
				}
			}
		});
	};

	ws.onmessage = (event: MessageEvent) => {
		try {
			const msg = typeof event.data === "string" ? JSON.parse(event.data) : null;
			if (!msg) return;

			// Cancel stale-session watchdog on first response
			if (!session.receivedMessage) {
				session.receivedMessage = true;
				if (session.staleSessionTimer) {
					clearTimeout(session.staleSessionTimer);
					session.staleSessionTimer = null;
				}
			}

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
		} catch (err) {
			voiceDebug("onmessage parse error", { error: String(err) });
		}
	};

	ws.onerror = (ev) => {
		clearTimeout(wsConnectTimeout);
		const errMsg = (ev as any)?.message || (ev as any)?.error?.message || "unknown";
		voiceDebug("WebSocket onerror", { readyState: ws.readyState, error: errMsg });
		if (!session.closed) {
			session.onError(`WebSocket error: ${errMsg}`);
		}
	};

	ws.onclose = (ev) => {
		clearTimeout(wsConnectTimeout);
		const code = (ev as any)?.code;
		const reason = (ev as any)?.reason;
		voiceDebug("WebSocket onclose", { code, reason, closed: session.closed });
		if (!session.closed) {
			// Unexpected close — distinguish normal completion from network drops
			if (code === 1000 || code === 1001 || session.finalizedParts.length > 0) {
				// Normal close or we have usable transcript data — finalize
				finalizeSession(session);
			} else {
				// Abnormal close with no transcript — treat as error
				session.onError(`Connection lost (code ${code ?? "unknown"}${reason ? `: ${reason}` : ""})`);
			}
		}
	};

	recProc.on("error", (err) => {
		voiceDebug("SoX process error:", err.message);
		if (!session.closed) {
			session.onError(`SoX error: ${err.message}`);
		}
	});

	recProc.on("close", (code, signal) => {
		voiceDebug("SoX process closed", { code, signal, closed: session.closed, wsState: ws.readyState });
		// Only send CloseStream if the session isn't already being torn down
		// (stopStreamingSession sends its own CloseStream before killing SoX)
		if (!session.closed && ws.readyState === WebSocket.OPEN) {
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

	// Finalize immediately — the live stream already captured all finals.
	// Include any trailing interim text that Deepgram hasn't promoted yet.
	if (session.interimText.trim()) {
		session.finalizedParts.push(session.interimText.trim());
		session.interimText = "";
	}
	finalizeSession(session);
}

function finalizeSession(session: StreamingSession): void {
	if (session.closed) return;
	session.closed = true;
	voiceDebug("finalizeSession", { hadAudio: session.hadAudioData, hadSpeech: session.hadSpeech, parts: session.finalizedParts.length });

	if (session.staleSessionTimer) {
		clearTimeout(session.staleSessionTimer);
		session.staleSessionTimer = null;
	}
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

function abortSession(session: VoiceSession | null): void {
	if (!session || session.closed) return;
	if (session.backend === "local") {
		abortLocalSession(session);
		return;
	}
	session.closed = true;
	if (session.staleSessionTimer) {
		clearTimeout(session.staleSessionTimer);
		session.staleSessionTimer = null;
	}
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
	let recordingStart = 0;
	let statusTimer: ReturnType<typeof setInterval> | null = null;
	let terminalInputUnsub: (() => void) | null = null;

	// Streaming session state
	let activeSession: VoiceSession | null = null;
	let preRecordingSession: StreamingSession | null = null;  // Started during warmup, promoted on confirm (Deepgram only)

	let lastStopTime = 0;    // For Escape-to-clear-editor within 30s of recording
	let lastEscapeTime = 0;  // For double-escape to clear editor
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
	let lastNonSpaceKeyTime = 0;      // Timestamp of last non-space keypress (typing cooldown)
	let tailRecordingTimer: ReturnType<typeof setTimeout> | null = null; // Delayed stop after release

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

	// ─── Sound Feedback ──────────────────────────────────────────────────────

	// Pre-resolve sound paths once at load time (not per-play)
	const _soundPaths: Record<string, string | null> = {};
	for (const [type, file] of Object.entries({
		start: "/System/Library/Sounds/Tink.aiff",
		stop: "/System/Library/Sounds/Pop.aiff",
		error: "/System/Library/Sounds/Basso.aiff",
	})) {
		_soundPaths[type] = fs.existsSync(file) ? file : null;
	}

	function playSound(type: "start" | "stop" | "error") {
		const file = _soundPaths[type];
		if (!file) return;
		try {
			const proc = spawn("afplay", [file], { stdio: "ignore", detached: true });
			proc.unref();
			proc.on("error", () => {}); // Prevent unhandled error crash
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
				const modeTag = !config.onboarding.completed ? "SETUP" : config.backend === "local" ? "LOCAL" : "STREAM";
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
				if (config.backend === "local") {
					ctx.ui.setStatus("voice", "STT...");
				} else {
					// Don't show "STT..." — live transcript handles it
					ctx.ui.setStatus("voice", "");
				}
				break;
		}
	}

	function setVoiceState(newState: VoiceState) {
		const prev = voiceState;
		voiceState = newState;
		if (prev !== newState) {
			voiceDebug(`STATE: ${prev} → ${newState}`);
		}
		const shared = ((globalThis as any).__piListen ??= {});
		shared.recording = newState === "recording";
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
		if (_recWidgetAnimTimer) {
			clearInterval(_recWidgetAnimTimer);
			_recWidgetAnimTimer = null;
		}
	}

	function hideWidget() {
		if (ctx?.hasUI) ctx.ui.setWidget("voice-recording", undefined);
	}

	/** Reset all hold-to-talk state to idle. Call after any recording stop/error/cancel. */
	function resetHoldState(opts?: { cooldown?: number }) {
		spaceConsumed = false;
		spaceDownTime = null;
		spacePressCount = 0;
		holdConfirmed = false;
		clearHoldTimer();
		clearReleaseTimer();
		abortPreRecording();
		if (opts?.cooldown) errorCooldownUntil = Date.now() + opts.cooldown;
	}

	function voiceCleanup() {
		if (statusTimer) { clearInterval(statusTimer); statusTimer = null; }
		cancelDelayedStop();
		clearWarmupWidget();
		clearRecordingAnimTimer();
		// Reset audio levels
		audioLevel = 0;
		audioLevelSmoothed = 0;
		if (activeSession) {
			abortSession(activeSession);
			activeSession = null;
		}

		resetHoldState(); // includes clearHoldTimer + clearReleaseTimer
		_startingRecording = false;
		lastSpacePressTime = 0;
		lastNonSpaceKeyTime = 0;
		errorCooldownUntil = 0;
		editorTextBeforeVoice = "";
		dictationMode = false;
		recordingStart = 0;
		recordingStartedAt = 0;
		lastStopTime = 0;
		if (terminalInputUnsub) { terminalInputUnsub(); terminalInputUnsub = null; }
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
		const hasKey = !!resolveDeepgramApiKey(nextConfig);
		config = finalizeOnboardingConfig(nextConfig, { validated: hasKey, source });
		configSource = selectedScope;
		const savedPath = saveConfig(config, selectedScope, currentCwd);
		const statusHeader = hasKey ? "Voice setup complete." : "Voice setup saved, but DEEPGRAM_API_KEY is still required.";
		uiCtx.ui.notify([
			statusHeader,
			...summaryLines,
			"",
			`Saved to ${savedPath}`,
		].join("\n"), hasKey ? "info" : "warning");
	}

	// ─── Warmup Widget ──────────────────────────────────────────────────────
	// ─── Minimal Voice Indicators ──────────────────────────────────────

	function getRecordDot(): string {
		const phase = (Math.sin(Date.now() / 600) + 1) / 2;
		if (phase > 0.65) return "●";
		if (phase > 0.35) return "◉";
		return "○";
	}

	function buildMiniWave(level: number): string {
		const bars = "▁▂▃▄▅▆▇█";
		const len = 12;
		let out = "";
		const t = Date.now() / 1000;
		const energy = Math.pow(level, 0.7); // Boost low levels
		for (let i = 0; i < len; i++) {
			const pos = i / len;
			// Multi-frequency sine for organic movement
			const wave1 = Math.sin(t * 4.5 + i * 0.9) * 0.35;
			const wave2 = Math.sin(t * 7.2 + i * 1.4 + 2.0) * 0.15;
			// Center emphasis — bars in the middle are taller
			const center = 1.0 - Math.abs(pos - 0.5) * 1.2;
			const base = 0.15 + energy * 0.85; // Always show some movement
			const value = Math.max(0, Math.min(1, (wave1 + wave2 + 0.5) * base * center));
			const idx = Math.min(bars.length - 1, Math.round(value * (bars.length - 1)));
			out += bars[idx];
		}
		return out;
	}

	// ─── Warmup Widget ──────────────────────────────────────────────────
	function showWarmupWidget() {
		if (!ctx?.hasUI) return;

		const startTime = Date.now();

		const renderWarmup = () => {
			if (!ctx?.hasUI) return;
			const elapsed = Date.now() - startTime;
			const progress = Math.min(elapsed / HOLD_THRESHOLD_MS, 1);

			ctx.ui.setWidget("voice-recording", (_tui, theme) => {
				return {
					invalidate() {},
					render(width: number): string[] {
						const meterLen = Math.max(4, Math.min(12, Math.floor(width * 0.15)));
						const filled = Math.round(progress * meterLen);
						const meter = "█".repeat(filled) + "░".repeat(meterLen - filled);
						const hint = progress < 1 ? "hold…" : "ready!";
						return [` ${theme.fg("accent", "🎤")} ${theme.fg("accent", meter)} ${theme.fg("dim", hint)}`];
					},
				};
			}, { placement: "belowEditor" });
		};

		renderWarmup();
		warmupWidgetTimer = setInterval(renderWarmup, 90);
	}

	// ─── Recording Widget ───────────────────────────────────────────────
	let _recWidgetAnimTimer: ReturnType<typeof setInterval> | null = null;

	function showRecordingWidget() {
		if (!ctx?.hasUI) return;

		// Stop warmup animation if still running — seamless takeover,
		// no gap between warmup and recording widgets (same widget ID).
		clearWarmupWidget();

		_recWidgetAnimTimer = setInterval(() => {
			showRecordingWidgetFrame();
		}, 150);

		showRecordingWidgetFrame();
	}

	function showRecordingWidgetFrame() {
		if (!ctx?.hasUI) return;

		// Minimal recording indicator below editor
		ctx.ui.setWidget("voice-recording", (_tui, theme) => {
			return {
				invalidate() {},
				render(_width: number): string[] {
					const elapsed = Math.round((Date.now() - recordingStart) / 1000);
					const mins = Math.floor(elapsed / 60);
					const secs = elapsed % 60;
					const timeStr = mins > 0 ? `${mins}:${String(secs).padStart(2, "0")}` : `${secs}s`;
					const wave = buildMiniWave(audioLevelSmoothed);
					const dot = theme.fg("error", getRecordDot());
					return [` ${dot} ${theme.fg("accent", wave)} ${theme.fg("muted", timeStr)} ${theme.fg("dim", "⌴ release")}`];
				},
			};
		}, { placement: "belowEditor" });
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


	}

	// ─── Voice: Start / Stop ─────────────────────────────────────────────────

	let _startingRecording = false; // Re-entrancy guard for startVoiceRecording

	async function startVoiceRecording(): Promise<boolean> {
		voiceDebug("startVoiceRecording called", { voiceState, hasUI: !!ctx?.hasUI, starting: _startingRecording });
		if (!ctx?.hasUI) return false;
		if (_startingRecording) return false; // Prevent overlapping starts during corruption guard sleep
		_startingRecording = true;

		try {
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
		// Don't hideWidget() here — the warmup widget is still showing and
		// showRecordingWidget() will seamlessly replace it using the same
		// widget ID. Hiding it first causes a visible gap (jitter).

		recordingStart = Date.now();

		// Snapshot editor text before voice overwrites it with live transcript
		editorTextBeforeVoice = ctx?.hasUI ? (ctx.ui.getEditorText() || "") : "";

		return startStreamingRecording();
		} finally {
			_startingRecording = false;
		}
	}

	// ── Pre-recording: start capturing audio during warmup so we don't miss words ──
	function startPreRecording() {
		if (preRecordingSession) return; // Already started
		if (config.backend === "local") return;      // No pre-recording for local batch mode
		if (!resolveDeepgramApiKey(config)) return; // No key — skip silently
		if (!detectAudioCaptureTool()) return;       // No audio tool — skip silently

		voiceDebug("startPreRecording → capturing audio during warmup");

		const session = startStreamingSession(config, {
			onTranscript: (interim, finals) => {
				// During warmup, silently accumulate transcript
				// (don't update UI — user hasn't committed to voice yet)
				voiceDebug("preRecording transcript", { interim: interim.slice(0, 50), finals: finals.length });
			},
			onDone: (fullText, meta) => {
				// Pre-recording ended (user released during warmup) — discard
				voiceDebug("preRecording onDone (discarded)", { fullText: fullText.slice(0, 50) });
				if (preRecordingSession === session) preRecordingSession = null;
			},
			onError: (err: string) => {
				voiceDebug("preRecording onError (ignored)", { err });
				if (preRecordingSession === session) preRecordingSession = null;
			},
		});

		if (session) {
			preRecordingSession = session;
		}
	}

	function abortPreRecording() {
		if (preRecordingSession) {
			voiceDebug("abortPreRecording → discarding warmup audio");
			abortSession(preRecordingSession);
			preRecordingSession = null;
		}
	}

	async function startStreamingRecording(): Promise<boolean> {
		voiceDebug("startStreamingRecording called", { hasKey: !!resolveDeepgramApiKey(config), hasPreRecording: !!preRecordingSession });
		setVoiceState("recording");

		// ── Callbacks for the active recording session ──
		const recordingCallbacks = {
			onTranscript: (interim: string, finals: string[]) => {
				// Live transcript update — this is the key UX feature
				updateLiveTranscriptWidget(interim, finals);
				updateVoiceStatus();
			},
			onDone: (fullText: string, meta: { hadAudio: boolean; hadSpeech: boolean }) => {
				voiceDebug("onDone callback", { fullText: fullText.slice(0, 100), meta, voiceState, spaceConsumed });
				activeSession = null;
				clearRecordingAnimTimer();
				if (statusTimer) { clearInterval(statusTimer); statusTimer = null; }
				lastStopTime = Date.now();

				if (!fullText.trim()) {
					// ── DISTINGUISH SILENCE VS NO SPEECH ──
					hideWidget();
					playSound("error");
					// Full state reset on empty result
					resetHoldState({ cooldown: 3000 });
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
					// The editor already has the live transcript via updateLiveTranscriptWidget.
					// Only set final text if the editor still has content.
					// If user already hit Enter (editor cleared), don't re-insert.
					const currentEditorText = ctx.ui.getEditorText?.() ?? "";
					if (currentEditorText.trim()) {
						const prefix = editorTextBeforeVoice ? editorTextBeforeVoice + " " : "";
						ctx.ui.setEditorText(prefix + fullText);
					}
					const elapsed = ((Date.now() - recordingStart) / 1000).toFixed(1);
					addToHistory(fullText, parseFloat(elapsed));
				}
				playSound("stop");
				// Full state reset on successful completion
				resetHoldState();
				setVoiceState("idle");
			},
			onError: (err: string) => {
				activeSession = null;
				clearRecordingAnimTimer();
				if (statusTimer) { clearInterval(statusTimer); statusTimer = null; }
				hideWidget();

				// ── STOP THE LOOP ──
				// On error, fully reset ALL hold state AND set a cooldown
				// so incoming key-repeat events can't re-trigger activation.
				resetHoldState({ cooldown: 5000 });
				clearWarmupWidget();

				ctx?.ui.notify(`Voice error: ${err}`, "error");
				playSound("error");
				setVoiceState("idle");
			},
		};

		// ── Promote pre-recording, start local, or start streaming ──
		let session: VoiceSession | null;

		if (config.backend === "local") {
			// Local backend: buffer audio, transcribe on stop
			const audioTool = detectAudioCaptureTool();
			if (!audioTool) {
				recordingCallbacks.onError("No audio capture tool found. Install one of: sox, ffmpeg, or arecord (Linux)");
				resetHoldState();
				setVoiceState("idle");
				return false;
			}
			const recProc = spawn(audioTool.cmd, audioTool.args, { stdio: ["pipe", "pipe", "pipe"] });
			recProc.stderr?.on("data", (d: Buffer) => {
				const msg = d.toString().trim();
				if (msg.includes("buffer overrun") || msg.includes("Discarding") || msg.includes("Last message repeated")) return;
				voiceDebug(`${audioTool.name} stderr:`, msg);
			});
			session = startLocalSession(recProc, recordingCallbacks);

			// Feed audio level meter for waveform animation
			recProc.stdout?.on("data", (chunk: Buffer) => {
				updateAudioLevel(chunk);
			});
		} else if (preRecordingSession) {
			// Promote: swap callbacks so pre-recorded audio feeds into real UI
			voiceDebug("Promoting pre-recording session to active");
			session = preRecordingSession;
			preRecordingSession = null;
			session.onTranscript = recordingCallbacks.onTranscript;
			session.onDone = recordingCallbacks.onDone;
			session.onError = recordingCallbacks.onError;
			// Flush any transcript already accumulated during warmup
			if (session.finalizedParts.length > 0 || session.interimText) {
				updateLiveTranscriptWidget(session.interimText, session.finalizedParts);
			}
		} else {
			session = startStreamingSession(config, recordingCallbacks);
		}

		if (!session) {
			// startStreamingSession returned null — reset ALL state
			resetHoldState();
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

	// ── Tail recording: keep capturing for 1.5s after space release ──
	function scheduleDelayedStop() {
		cancelDelayedStop(); // Clear any existing timer
		voiceDebug("scheduleDelayedStop → will stop in", TAIL_RECORDING_MS, "ms");
		tailRecordingTimer = setTimeout(() => {
			tailRecordingTimer = null;
			voiceDebug("tailRecordingTimer fired → stopping recording");
			stopVoiceRecording();
		}, TAIL_RECORDING_MS);
	}

	function cancelDelayedStop() {
		if (tailRecordingTimer) {
			clearTimeout(tailRecordingTimer);
			tailRecordingTimer = null;
			voiceDebug("cancelDelayedStop → tail recording timer cleared");
		}
	}

	async function stopVoiceRecording() {
		cancelDelayedStop(); // Safety: clear any pending delayed stop
		voiceDebug("stopVoiceRecording called", { voiceState, hasActiveSession: !!activeSession });
		if (voiceState !== "recording" || !ctx) return;
		if (statusTimer) { clearInterval(statusTimer); statusTimer = null; }

		if (activeSession) {
			setVoiceState("finalizing");
			clearRecordingAnimTimer();
			hideWidget();
			if (activeSession.backend === "local") {
				// Local: show "Transcribing..." while model processes audio
				ctx?.ui.notify("Transcribing with local model...", "info");
				await stopLocalSession(activeSession, config);
			} else {
				stopStreamingSession(activeSession);
			}
		} else {
			// No active session — shouldn't happen, but recover gracefully
			voiceDebug("stopVoiceRecording: no active session, resetting to idle");
			hideWidget();
			setVoiceState("idle");
		}
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
	//      4. After HOLD_THRESHOLD_MS (1200ms) from first press → start recording.
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
			resetHoldState();
			clearWarmupWidget();
			hideWidget();
			// No need to type a space — the first press was NOT consumed,
			// so it already reached the focused UI component naturally.
			return;
		}

		// Released during warmup — cancel (user held but not long enough)
		if (voiceState === "warmup") {
			resetHoldState();
			abortPreRecording();
			clearWarmupWidget();
			hideWidget();
			setVoiceState("idle");
			spaceDownTime = null;
			spaceConsumed = false;
			spacePressCount = 0;
			holdConfirmed = false;
			// Don't type a space — user clearly intended to trigger voice but let go too early
			ctx?.ui.notify("Hold SPACE longer to activate voice.", "info");
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
			voiceDebug("  → scheduling delayed stop (tail recording)");
			resetHoldState();
			scheduleDelayedStop();
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

			// ── Track non-space keypresses for typing cooldown ──
			// If user was just typing (non-space key within TYPING_COOLDOWN_MS),
			// don't let space holds activate voice — they're just typing.
			if (!matchesKey(data, "space") && !isKeyRelease(data) && !isKeyRepeat(data)) {
				// Regular keypress that isn't space — user is typing
				if (data.length > 0 && data.charCodeAt(0) >= 32) {
					lastNonSpaceKeyTime = Date.now();
				}
			}

			// ── SPACE handling ──
			if (matchesKey(data, "space")) {
				// ── ERROR COOLDOWN: block all voice activation for 5s after an error ──
				if (errorCooldownUntil > Date.now()) {
					// During cooldown, let space through as a normal character
					return undefined;
				}

				// ── TYPING COOLDOWN: if user was just typing, let space through ──
				// Apple-style: if a non-space key was pressed recently, this space
				// is part of typing (e.g., "hello world"), not a voice activation.
				// Only applies to NEW activations — don't interrupt active recording.
				if (voiceState === "idle" && !spaceConsumed &&
					lastNonSpaceKeyTime > 0 && (Date.now() - lastNonSpaceKeyTime) < TYPING_COOLDOWN_MS) {
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
						resetHoldState();
						abortPreRecording();
						clearWarmupWidget();
						hideWidget();
						setVoiceState("idle");
						if (holdDuration < 300) {
							// Quick tap — just type a space
							if (ctx?.hasUI) ctx.ui.setEditorText((ctx.ui.getEditorText() || "") + " ");
						} else {
							// Held long enough to see warmup but let go → show hint
							ctx?.ui.notify("Hold SPACE longer to activate voice.", "info");
						}
						return { consume: true };
					}

					// Tap: released before warmup even started (shouldn't happen in
					// Kitty path since we enter warmup on first press, but handle anyway)
					if (spaceDownTime && !holdConfirmed && voiceState === "idle") {
						resetHoldState();
						if (ctx?.hasUI) ctx.ui.setEditorText((ctx.ui.getEditorText() || "") + " ");
						return { consume: true };
					}

					// Released during recording → schedule delayed stop (tail recording)
					if (spaceConsumed && voiceState === "recording") {
						resetHoldState();
						scheduleDelayedStop();
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
						// NOTE: Do NOT set kittyReleaseDetected here!
						// Ghostty on macOS sends repeat events but NO release events.
						// Only a true isKeyRelease() should flip the Kitty flag.

						const now = Date.now();
						spacePressCount++;
						lastSpacePressTime = now;

						// Enough repeats to confirm hold — enter warmup
						if (spacePressCount >= REPEAT_CONFIRM_COUNT) {
							holdConfirmed = true;
							setVoiceState("warmup");
							showWarmupWidget();
							startPreRecording();

							const alreadyElapsed = now - (spaceDownTime || now);
							const remaining = Math.max(0, HOLD_THRESHOLD_MS - alreadyElapsed);

							holdActivationTimer = setTimeout(() => {
								holdActivationTimer = null;
								if (voiceState === "warmup") {
									// Don't clearWarmupWidget() here — showRecordingWidget()
									// seamlessly replaces it using the same widget ID.
									spaceConsumed = true;
									recordingStartedAt = Date.now();
									// Clear release timer during async recording startup
									// to prevent false stop. Next repeat re-arms it.
									clearReleaseTimer();
									voiceDebug("holdActivationTimer fired → starting recording (Kitty repeat path)");
									startVoiceRecording().then((ok) => {
										if (!ok) {
											resetHoldState();
											setVoiceState("idle");
										}
									}).catch((err) => {
										voiceDebug("startVoiceRecording THREW", { error: String(err) });
										resetHoldState({ cooldown: 5000 });
										setVoiceState("idle");
									});
								} else {
									spaceDownTime = null;
									spaceConsumed = false;
									spacePressCount = 0;
									holdConfirmed = false;
								}
							}, remaining);
						}

						// Re-arm gap-based release detection — this is how
						// Ghostty-on-macOS (repeats but no release) detects
						// the key being released.
						resetReleaseDetect();
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

				// If already recording → cancel any pending delayed stop and keep going
				if (voiceState === "recording") {
					cancelDelayedStop(); // User re-pressed — they want to keep recording
					spaceConsumed = true; // Re-arm hold state for the continued recording
					spaceDownTime = Date.now();
					holdConfirmed = true;
					if (!kittyReleaseDetected) {
						voiceDebug("SPACE during recording → cancel delayed stop, re-arm release detect");
						resetReleaseDetect();
					} else {
						voiceDebug("SPACE during recording → cancel delayed stop (Kitty)");
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
						startPreRecording();

						holdActivationTimer = setTimeout(() => {
							holdActivationTimer = null;
							if (voiceState === "warmup") {
								// Don't clearWarmupWidget() here — showRecordingWidget()
								// seamlessly replaces it using the same widget ID.
								spaceConsumed = true;
								recordingStartedAt = Date.now();
								voiceDebug("holdActivationTimer fired → starting recording (Kitty path)");
								startVoiceRecording().then((ok) => {
									if (!ok) {
										resetHoldState();
										setVoiceState("idle");
									}
								}).catch((err) => {
									voiceDebug("startVoiceRecording THREW", { error: String(err) });
									resetHoldState({ cooldown: 5000 });
									setVoiceState("idle");
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
							startPreRecording();

							const alreadyElapsed = now - spaceDownTime;
							const remaining = Math.max(0, HOLD_THRESHOLD_MS - alreadyElapsed);

							holdActivationTimer = setTimeout(() => {
								holdActivationTimer = null;
								if (voiceState === "warmup") {
									// Don't clearWarmupWidget() here — showRecordingWidget()
									// seamlessly replaces it using the same widget ID.
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
											resetHoldState();
											setVoiceState("idle");
										}
										// Do NOT re-arm release detect here!
										// The next SPACE key-repeat event will do it.
										// Re-arming here causes false stops because
										// the timer fires during the async gap.
									}).catch((err) => {
										voiceDebug("startVoiceRecording THREW", { error: String(err) });
										resetHoldState({ cooldown: 5000 });
										setVoiceState("idle");
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
						const wasInWarmup = (voiceState as VoiceState) === "warmup";
						resetHoldState();
						abortPreRecording();
						clearWarmupWidget();
						hideWidget();
						if (wasInWarmup) setVoiceState("idle");
						// Only type a space if we weren't already in warmup
						// (if we were in warmup, user was trying to activate voice, not type)
						// Note: first space already passed through naturally (not consumed)
						// so we don't need to manually type it here
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
				resetHoldState();
				// No need to insert a space manually — the first space press was
				// already allowed to pass through to the focused UI component.
				return undefined;
			}

			if (voiceState === "warmup" && holdConfirmed && !spaceConsumed) {
				clearWarmupWidget();
				hideWidget();
				resetHoldState();
				setVoiceState("idle");
				return undefined;
			}

			// ── Escape key — cancel voice / double-escape clears editor ──
			// Skip release/repeat events — only act on actual presses
			if (matchesKey(data, "escape") && !isKeyRelease(data) && !isKeyRepeat(data)) {
				// During recording: cancel recording and clear transcript
				if (voiceState === "recording" || voiceState === "warmup" || voiceState === "finalizing") {
					voiceDebug("Escape pressed → canceling voice");
					abortPreRecording();
					if (activeSession) {
						abortSession(activeSession);
						activeSession = null;
					}
					clearRecordingAnimTimer();
					clearWarmupWidget();
					hideWidget();
					if (statusTimer) { clearInterval(statusTimer); statusTimer = null; }
					// Restore editor text to what it was before recording
					if (ctx?.hasUI) ctx.ui.setEditorText(editorTextBeforeVoice);
					resetHoldState();
					playSound("error");
					setVoiceState("idle");
					lastEscapeTime = Date.now();
					return { consume: true };
				}

				// In idle: double-escape (two presses within 500ms) clears editor
				if (voiceState === "idle") {
					const now = Date.now();
					if (lastEscapeTime > 0 && (now - lastEscapeTime) < 500) {
						if (ctx?.hasUI) {
							const currentText = ctx.ui.getEditorText() || "";
							if (currentText.trim()) {
								ctx.ui.setEditorText("");
								lastEscapeTime = 0;
								return { consume: true };
							}
						}
					}
					lastEscapeTime = now;
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
				resetHoldState();
				await stopVoiceRecording();
			} else if (voiceState === "warmup") {
				// Cancel warmup
				abortPreRecording();
				clearWarmupWidget();
				hideWidget();
				resetHoldState();
				setVoiceState("idle");
			}
			// voiceState === "finalizing" → ignore (wait for transcript)
		},
	});

	// ─── Lifecycle ───────────────────────────────────────────────────────────

	pi.on("session_start", async (_event, startCtx) => {
		ctx = startCtx;
		currentCwd = startCtx.cwd;
		const loaded = loadConfigWithSource(startCtx.cwd);
		config = loaded.config;
		configSource = loaded.source;

		// Auto-capture DEEPGRAM_API_KEY from env into config
		if (process.env.DEEPGRAM_API_KEY && !config.deepgramApiKey) {
			config.deepgramApiKey = process.env.DEEPGRAM_API_KEY;
			if (configSource !== "default") {
				saveConfig(config, config.scope, currentCwd);
			}
		}

		if (config.enabled && config.onboarding.completed) {
			updateVoiceStatus();
			setupHoldToTalk();
		} else if (!config.onboarding.completed) {
			// First-time hint — show once, non-intrusive
			const hasKey = !!resolveDeepgramApiKey(config);
			if (startCtx.hasUI) {
				const audioTool = detectAudioCaptureTool();
				if (hasKey) {
					// Key exists but onboarding not completed — just activate
					config.onboarding.completed = true;
					config.onboarding.completedAt = new Date().toISOString();
					config.onboarding.source = "migration";
					saveConfig(config, config.scope === "project" ? "project" : "global", currentCwd);
					updateVoiceStatus();
					setupHoldToTalk();
					const lines = [
						"pi-voice ready!",
						"",
						"  Hold SPACE to record → release to transcribe",
						"  Ctrl+Shift+V to toggle recording",
						"  Escape × 2 to clear the editor",
						`  Audio: ${audioTool ? `${audioTool.name}` : "NONE — install sox or ffmpeg"}`,
						"  /voice test to verify setup",
					];
					startCtx.ui.notify(lines.join("\n"), audioTool ? "info" : "warning");
				} else {
					const lines = [
						"pi-voice installed — voice input for Pi",
						"",
						"  Hold SPACE to record, release to transcribe.",
						"",
						"  Setup:",
						"  1. Get a Deepgram API key → https://dpgr.am/pi-voice ($200 free credit)",
						`  2. ${audioTool ? `Audio capture: ${audioTool.name} ✓` : "Install sox or ffmpeg (audio capture)"}`,
						"  3. Run /voice-setup to configure",
					];
					startCtx.ui.notify(lines.join("\n"), "info");
				}
			}
		}
	});

	pi.on("session_shutdown", async () => {
		voiceCleanup();
		ctx = null;
	});

	pi.on("session_switch", async (_event, switchCtx) => {
		// Clean up any active recording before switching
		voiceCleanup();
		ctx = switchCtx;
		currentCwd = switchCtx.cwd;
		const loaded = loadConfigWithSource(switchCtx.cwd);
		config = loaded.config;
		configSource = loaded.source;
		if (config.enabled && config.onboarding.completed) {
			setupHoldToTalk();
		}
		updateVoiceStatus();
	});

	// ─── /voice command ──────────────────────────────────────────────────────

	pi.registerCommand("voice", {
		description: "Voice: /voice [on|off|stop|dictate|history|test|info|setup]",
		handler: async (args, cmdCtx) => {
			ctx = cmdCtx;
			const sub = (args || "").trim().toLowerCase();

			if (sub === "on") {
				config.enabled = true;
				updateVoiceStatus();
				setupHoldToTalk();
				const backendInfo = config.backend === "local"
					? `Voice enabled (local model: ${config.localModel || "whisper-small"}).`
					: "Voice enabled (Deepgram streaming).";
				cmdCtx.ui.notify([
					backendInfo,
					"",
					"  Hold SPACE → release to transcribe",
					"  Ctrl+Shift+V → toggle recording on/off",
					"  Quick SPACE tap → types a space (no voice)",
					"  Escape × 2 → clear editor",
					"",
					"  /voice-language → change language (56+ supported)",
					"  /voice dictate  → continuous mode (no hold)",
					"  /voice test     → verify setup",
					"",
					"  Say 'undo', 'clear', 'new line', 'period' during dictation",
				].join("\n"), "info");
				return;
			}

			if (sub === "off") {
				config.enabled = false;
				voiceCleanup();
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
					abortPreRecording();
					clearWarmupWidget();
					hideWidget();
					resetHoldState();
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
				editorTextBeforeVoice = ctx?.hasUI ? (ctx.ui.getEditorText() || "") : "";
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
				const isLocal = config.backend === "local";
				const dgKey = resolveDeepgramApiKey(config);
				const tool = detectAudioCaptureTool();

				const lines = [
					"Voice diagnostics:",
					"",
					`  Backend: ${isLocal ? "local" : "deepgram"}`,
					"",
					"  Audio capture:",
					`    tool:              ${tool ? `${tool.name} (${tool.cmd})` : "NONE FOUND"}`,
				];
				if (!tool) {
					lines.push("    available:         sox ✗  ffmpeg ✗  arecord ✗");
					lines.push("    install one:       brew install sox (or ffmpeg)");
				}

				if (isLocal) {
					lines.push(`    local model:       ${config.localModel || "whisper-small"}`);
					lines.push(`    local endpoint:    ${config.localEndpoint || DEFAULT_LOCAL_ENDPOINT}`);
				} else {
					lines.push(`    DEEPGRAM_API_KEY:  ${dgKey ? "set (" + dgKey.slice(0, 8) + "…)" : "NOT SET"}`);
				}
				lines.push("");
				lines.push("  Config:");
				lines.push(`    language:          ${config.language}`);
				lines.push(`    onboarding:        ${config.onboarding.completed ? "complete" : "incomplete"}`);
				lines.push(`    hold threshold:    ${HOLD_THRESHOLD_MS}ms`);
				lines.push(`    kitty protocol:    ${kittyReleaseDetected ? "detected" : "not detected"}`);
				lines.push(`    state:             ${voiceState}`);

				// Mic capture test using detected tool
				if (tool) {
					const testFile = path.join(os.tmpdir(), "pi-voice-test.wav");
					let testProc;
					if (tool.name === "sox") {
						testProc = spawn("rec", ["-q", "-r", "16000", "-c", "1", "-b", "16", "-d", "1", testFile], { stdio: "pipe" });
					} else if (tool.name === "ffmpeg") {
						const isMac = process.platform === "darwin";
						const isLinux = process.platform === "linux";
						const inputArgs = isMac ? ["-f", "avfoundation", "-i", ":default"] : isLinux ? ["-f", "pulse", "-i", "default"] : ["-f", "dshow", "-i", "audio=default"];
						testProc = spawn("ffmpeg", [...inputArgs, "-t", "1", "-ar", "16000", "-ac", "1", "-y", "-loglevel", "error", testFile], { stdio: "pipe" });
					} else {
						testProc = spawn("arecord", ["-q", "-f", "S16_LE", "-r", "16000", "-c", "1", "-d", "1", testFile], { stdio: "pipe" });
					}
					testProc.on("error", () => {});
					await new Promise<void>((resolve) => {
						let resolved = false;
						const done = () => { if (!resolved) { resolved = true; resolve(); } };
						testProc.on("close", done);
						setTimeout(() => { try { testProc.kill(); } catch {} done(); }, 3000);
					});
					if (fs.existsSync(testFile)) {
						const size = fs.statSync(testFile).size;
						lines.push(`    mic capture:       OK (${size} bytes via ${tool.name})`);
						try { fs.unlinkSync(testFile); } catch {}
					} else {
						lines.push(`    mic capture:       FAILED — ${tool.name} ran but no audio captured`);
					}
				} else {
					lines.push("    mic capture:       skipped (no audio tool)");
				}

				if (isLocal) {
					// Local server connectivity check
					const serverCheck = await checkLocalServer(config.localEndpoint);
					if (serverCheck.ok) {
						lines.push("    local server:      OK (reachable)");
					} else {
						lines.push(`    local server:      NOT REACHABLE — ${serverCheck.error || "connection refused"}`);
					}
				} else if (dgKey) {
					// Deepgram API key validation
					try {
						const res = await fetch("https://api.deepgram.com/v1/projects", {
							method: "GET",
							headers: { "Authorization": `Token ${dgKey}` },
							signal: AbortSignal.timeout(5000),
						});
						if (res.ok) {
							lines.push("    Deepgram API:      OK (key validated)");
						} else if (res.status === 401 || res.status === 403) {
							lines.push("    Deepgram API:      INVALID KEY — check your API key");
						} else {
							lines.push(`    Deepgram API:      ERROR (HTTP ${res.status})`);
						}
					} catch (err) {
						const msg = err instanceof Error ? err.message : String(err);
						lines.push(`    Deepgram API:      UNREACHABLE — ${msg}`);
					}
				}

				// Summary
				lines.push("");
				let ready: boolean;
				if (isLocal) {
					const serverOk = (await checkLocalServer(config.localEndpoint)).ok;
					ready = !!tool && serverOk;
					if (!tool) {
						lines.push("  Setup needed — install any one of:");
						lines.push("    brew install sox       # macOS (recommended)");
						lines.push("    apt install sox        # Linux");
					} else if (!serverOk) {
						lines.push("  Setup needed — start a local transcription server:");
						lines.push("    whisper.cpp: ./build/bin/whisper-server -m models/ggml-small.bin --port 8080");
						lines.push("    Or any OpenAI-compatible transcription server");
					} else {
						lines.push("  All checks passed — voice is ready!");
						lines.push("  Hold SPACE to record, or use Ctrl+Shift+V to toggle.");
					}
				} else {
					ready = !!dgKey && !!tool;
					if (!dgKey) {
						lines.push("  Setup needed:");
						lines.push("    1. Get a free key → https://dpgr.am/pi-voice ($200 free credit)");
						lines.push("    2. export DEEPGRAM_API_KEY=\"your-key\" (add to ~/.zshrc)");
						lines.push("    3. Or run /voice-setup to paste it interactively");
					} else if (!tool) {
						lines.push("  Setup needed — install any one of:");
						lines.push("    brew install sox       # macOS (recommended)");
						lines.push("    brew install ffmpeg    # macOS (alternative)");
						lines.push("    apt install sox        # Linux");
						lines.push("    apt install ffmpeg     # Linux (alternative)");
						lines.push("    choco install sox      # Windows");
					} else {
						lines.push("  All checks passed — voice is ready!");
						lines.push("  Hold SPACE to record, or use Ctrl+Shift+V to toggle.");
					}
				}

				cmdCtx.ui.notify(lines.join("\n"), ready ? "info" : "warning");
				return;
			}

			// /voice language — fuzzy search picker
			if (sub === "language" || sub === "lang") {
				if (config.backend === "local") {
					const localModelId = config.localModel || "whisper-small";
					const { languages, englishOnly } = getLanguagesForLocalModel(localModelId);
					if (englishOnly) {
						const modelName = LOCAL_MODELS.find(m => m.id === localModelId)?.name || localModelId;
						const langName = languages[0]?.name || "English";
						cmdCtx.ui.notify(`${modelName} only supports ${langName}.`, "warning");
						return;
					}
					const langCode = await pickLanguage(cmdCtx, config.language, languages);
					if (langCode) {
						config.language = langCode;
						saveConfig(config, config.scope === "project" ? "project" : "global", currentCwd);
						cmdCtx.ui.notify(`Language: ${localLanguageDisplayName(langCode)}\nModel: ${localModelId}`, "info");
					}
				} else {
					const langCode = await pickLanguage(cmdCtx, config.language);
					if (langCode) {
						config.language = langCode;
						saveConfig(config, config.scope === "project" ? "project" : "global", currentCwd);
						const model = modelForLanguage(langCode);
						cmdCtx.ui.notify(`Language: ${languageDisplayName(langCode)}\nModel: ${model}`, "info");
					}
				}
				return;
			}
			// /voice language <code> — direct set without picker
			if (sub.startsWith("language ") || sub.startsWith("lang ")) {
				const langCode = sub.replace(/^lang(uage)?\s+/, "").trim();
				if (config.backend === "local") {
					const localModelId = config.localModel || "whisper-small";
					if (!isLanguageSupportedByModel(localModelId, langCode)) {
						const modelName = LOCAL_MODELS.find(m => m.id === localModelId)?.name || localModelId;
						cmdCtx.ui.notify(`"${langCode}" is not supported by ${modelName}.\nUse /voice language to see supported languages.`, "warning");
						return;
					}
				}
				config.language = langCode;
				saveConfig(config, config.scope === "project" ? "project" : "global", currentCwd);
				if (config.backend === "local") {
					cmdCtx.ui.notify(`Language: ${localLanguageDisplayName(langCode)}\nModel: ${config.localModel || "whisper-small"}`, "info");
				} else {
					const model = modelForLanguage(langCode);
					cmdCtx.ui.notify(`Language: ${languageDisplayName(langCode)}\nModel: ${model}`, "info");
				}
				return;
			}

			if (sub === "info") {
				const dgKey = resolveDeepgramApiKey(config);
				const isLocal = config.backend === "local";
				const infoLines = [
					`Voice config:`,
					`  backend:    ${isLocal ? "local" : "deepgram"}`,
					`  enabled:    ${config.enabled}`,
					`  scope:      ${config.scope}`,
					`  language:   ${config.language}`,
				];
				if (isLocal) {
					infoLines.push(`  model:      ${config.localModel || "whisper-small"}`);
					infoLines.push(`  endpoint:   ${config.localEndpoint || DEFAULT_LOCAL_ENDPOINT}`);
					infoLines.push(`  mode:       batch (transcribe after recording)`);
				} else {
					infoLines.push(`  streaming:  YES (Deepgram WebSocket)`);
					infoLines.push(`  api key:    ${dgKey ? "set (" + dgKey.slice(0, 8) + "…)" : "NOT SET"}`);
				}
				infoLines.push(
					`  state:      ${voiceState}`,
					`  setup:      ${config.onboarding.completed ? `complete (${config.onboarding.source ?? "unknown"})` : "incomplete"}`,
					`  hold-key:   SPACE (hold ≥${HOLD_THRESHOLD_MS}ms) or Ctrl+Shift+V (toggle)`,
					`  kitty:      ${kittyReleaseDetected ? "yes" : "no"}`,
				);
				cmdCtx.ui.notify(infoLines.join("\n"), "info");
				return;
			}

			if (sub === "setup" || sub === "reconfigure") {
				const result = await runVoiceOnboarding(cmdCtx, config, { isFirstRun: !config.onboarding.completed });
				if (!result) {
					cmdCtx.ui.notify("Voice setup cancelled.", "warning");
					return;
				}
				await finalizeAndSaveSetup(cmdCtx, result.config, result.selectedScope, result.summaryLines, "setup-command");
				return;
			}


			// Default: toggle
			config.enabled = !config.enabled;
			if (!config.enabled) { voiceCleanup(); }
			else { setupHoldToTalk(); }
			updateVoiceStatus();
			cmdCtx.ui.notify(`Voice ${config.enabled ? "enabled" : "disabled"}.`, "info");
		},
	});

	// ─── Dedicated setup command ─────────────────────────────────────────────

	pi.registerCommand("voice-setup", {
		description: "Configure voice input — Deepgram API key and settings",
		handler: async (_args, cmdCtx) => {
			ctx = cmdCtx;
			const result = await runVoiceOnboarding(cmdCtx, config, { isFirstRun: !config.onboarding.completed });
			if (!result) {
				cmdCtx.ui.notify("Voice setup cancelled.", "warning");
				return;
			}
			await finalizeAndSaveSetup(cmdCtx, result.config, result.selectedScope, result.summaryLines, "setup-command");
		},
	});

	// ─── /voice-language — dedicated language picker ────────────────────────

	pi.registerCommand("voice-language", {
		description: "Change voice transcription language",
		handler: async (args, cmdCtx) => {
			ctx = cmdCtx;
			const isLocal = config.backend === "local";
			const localModelId = config.localModel || "whisper-small";

			// Direct set: /voice-language hi
			const direct = (args || "").trim().toLowerCase();
			if (direct) {
				if (isLocal) {
					if (!isLanguageSupportedByModel(localModelId, direct)) {
						const modelName = LOCAL_MODELS.find(m => m.id === localModelId)?.name || localModelId;
						cmdCtx.ui.notify(`"${direct}" is not supported by ${modelName}.\nUse /voice-language to see supported languages.`, "warning");
						return;
					}
				}
				config.language = direct;
				saveConfig(config, config.scope === "project" ? "project" : "global", currentCwd);
				if (isLocal) {
					cmdCtx.ui.notify(`Language: ${localLanguageDisplayName(direct)}\nModel: ${localModelId}`, "info");
				} else {
					const model = modelForLanguage(direct);
					cmdCtx.ui.notify(`Language: ${languageDisplayName(direct)}\nModel: ${model}`, "info");
				}
				return;
			}

			// Fuzzy picker — show model-appropriate languages
			if (isLocal) {
				const { languages, englishOnly } = getLanguagesForLocalModel(localModelId);
				if (englishOnly) {
					const modelName = LOCAL_MODELS.find(m => m.id === localModelId)?.name || localModelId;
					const langName = languages[0]?.name || "English";
					cmdCtx.ui.notify(`${modelName} only supports ${langName}.`, "warning");
					return;
				}
				const langCode = await pickLanguage(cmdCtx, config.language, languages);
				if (langCode) {
					config.language = langCode;
					saveConfig(config, config.scope === "project" ? "project" : "global", currentCwd);
					cmdCtx.ui.notify(`Language: ${localLanguageDisplayName(langCode)}\nModel: ${localModelId}`, "info");
				}
			} else {
				const langCode = await pickLanguage(cmdCtx, config.language);
				if (langCode) {
					config.language = langCode;
					saveConfig(config, config.scope === "project" ? "project" : "global", currentCwd);
					const model = modelForLanguage(langCode);
					cmdCtx.ui.notify(`Language: ${languageDisplayName(langCode)}\nModel: ${model}`, "info");
				}
			}
		},
	});

	// ─── /voice-settings — show current config ─────────────────────────────

	pi.registerCommand("voice-settings", {
		description: "Show voice configuration and settings",
		handler: async (_args, cmdCtx) => {
			ctx = cmdCtx;
			const dgKey = resolveDeepgramApiKey(config);
			const isLocal = config.backend === "local";
			const model = isLocal ? (config.localModel || "whisper-small") : modelForLanguage(config.language);
			const backendLabel = isLocal ? "local" : "deepgram";
			const lines = [
				"Voice settings:",
				"",
				`  backend:     ${backendLabel}`,
				`  enabled:     ${config.enabled}`,
				`  language:    ${languageDisplayName(config.language)}`,
				`  model:       ${model}`,
				`  scope:       ${config.scope}`,
			];
			if (isLocal) {
				lines.push(`  endpoint:    ${config.localEndpoint || DEFAULT_LOCAL_ENDPOINT}`);
			} else {
				lines.push(`  api key:     ${dgKey ? "set (" + dgKey.slice(0, 8) + "…)" : "NOT SET"}`);
			}
			lines.push(
				`  onboarding:  ${config.onboarding.completed ? "complete" : "incomplete"}`,
				`  state:       ${voiceState}`,
				`  kitty:       ${kittyReleaseDetected ? "yes" : "no"}`,
				"",
				"Commands:",
				"  /voice-setup      Reconfigure (backend, API key, language, scope)",
				"  /voice-language   Change language (56+ supported)",
				"  /voice test       Run diagnostics",
				"  /voice on|off     Enable/disable",
			);
			cmdCtx.ui.notify(lines.join("\n"), "info");
		},
	});

}

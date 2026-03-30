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
 *   warmup:     User holds SPACE for ≥ HOLD_THRESHOLD_MS (500ms).
 *               On kitty-protocol terminals, warmup UI waits for a brief intent delay
 *               so normal space taps don't flash the mic indicator. Releasing before
 *               warmup starts types a space; releasing during warmup cancels silently.
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
 *      First SPACE press → start short intent delay (~200ms).
 *      Still holding after intent delay → enter warmup (show countdown).
 *      Released before warmup starts → type a space.
 *      Released during warmup → cancel without inserting a space.
 *      Held ≥ 500ms from initial press → activate recording.
 *      True release event stops recording.
 *
 *   B) Non-Kitty (macOS Terminal, Ghostty on macOS):
 *      No key-release event. Holding sends rapid press events (~30-90ms apart).
 *      First SPACE press → record time, start release-detect timer (500ms).
 *      No more presses within 500ms → TAP → type a space.
 *      Rapid presses detected → user is HOLDING.
 *      After REPEAT_CONFIRM_COUNT (6) rapid presses → enter warmup.
 *      After HOLD_THRESHOLD_MS (500ms) from first press → activate recording.
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
 *   - Hold SPACE (≥500ms) → release to finalize
 *   - Configurable shortcut → toggle start/stop (always works)

 *
 * Config in ~/.pi/agent/settings.json under "voice": { ... }
 */

import type {
	ExtensionAPI,
	ExtensionContext,
	ExtensionCommandContext,
} from "@mariozechner/pi-coding-agent";
import { isKeyRelease, isKeyRepeat, isKittyProtocolActive, matchesKey } from "@mariozechner/pi-tui";

import { spawn, spawnSync, type ChildProcess } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import {
	DEFAULT_CONFIG,
	getSessionStartPersistedConfig,
	loadConfigWithSource,
	loadGlobalConfig,
	saveConfig,
	type VoiceConfig,
	type VoiceSettingsScope,
} from "./voice/config";
import { formatVoiceStatus } from "./voice/status-labels";
import {
	getKittyHoldTiming,
	shouldInsertSpaceOnKittyReleaseBeforeRecording,
	shouldStartRecordingFromWarmup,
	stripInsertedSpaceOnWarmupStart,
} from "./voice/hold-to-talk";
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
const STREAM_FINALIZE_TIMEOUT_MS = 2500;

// Hold-to-talk timing — Apple-style deliberate hold detection
// The goal: typing normally should NEVER accidentally trigger voice.
// Only a clearly intentional long press activates it.
const HOLD_THRESHOLD_MS = 500;    // Must hold for 0.5s before voice activates
                                   // Fast enough for short dictation, still deliberate vs normal typing.
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
const KITTY_INTENT_DELAY_MS = 200; // Brief buffer before kitty terminals show warmup/consume voice UI.
                                    // Quick taps stay feeling like normal spaces.
const TAIL_RECORDING_MS = 1500;   // Keep recording for 1.5s after space release to catch
                                   // trailing words. If user re-presses space within this
                                   // window, cancel the delayed stop and keep recording.
const CORRUPTION_GUARD_MS = 200;  // Min gap between stop and restart
const WARMUP_RELEASE_GRACE_MS = 120; // Linux terminals may emit repeated release events
                                     // while a key is still physically held

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

/** Detect the first Windows DirectShow audio input device name via ffmpeg.
 * DirectShow has no "default" alias — must enumerate and pick the first audio device.
 */
function detectWindowsAudioDevice(): string | null {
	try {
		const result = spawnSync("ffmpeg", ["-f", "dshow", "-list_devices", "true", "-i", "dummy"], {
			timeout: 3000, encoding: "utf-8", stdio: ["pipe", "pipe", "pipe"],
		});
		// ffmpeg outputs device list to stderr
		const output = result.stderr || "";
		// Match lines like: [dshow @ ...] "Microphone (Realtek HD Audio)" (audio)
		const match = output.match(/"([^"]+)"\s*\(audio\)/);
		return match?.[1] || null;
	} catch {
		return null;
	}
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
		const isWin = process.platform === "win32";
		// Input device varies by platform
		let inputArgs: string[];
		if (isMac) {
			inputArgs = ["-f", "avfoundation", "-i", ":default"];
		} else if (isLinux) {
			inputArgs = ["-f", "pulse", "-i", "default"];
		} else if (isWin) {
			// DirectShow has no "default" alias — enumerate devices and pick the first audio device
			const dshowDevice = detectWindowsAudioDevice();
			inputArgs = dshowDevice
				? ["-f", "dshow", "-i", `audio=${dshowDevice}`]
				: ["-f", "dshow", "-i", "audio=Microphone"]; // last-resort guess
		} else {
			inputArgs = ["-f", "pulse", "-i", "default"]; // fallback for other platforms
		}
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
	finalizeTimer: ReturnType<typeof setTimeout> | null;
	closed: boolean;
	stopRequested: boolean;
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
		finalizeTimer: null,
		closed: false,
		stopRequested: false,
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
					failStreamingSession(session, `Deepgram HTTP ${res.statusCode}: ${body.slice(0, 200)}`);
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
							failStreamingSession(session, "No response from Deepgram (15s). Check your API key and network.");
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
				failStreamingSession(session, msg.message || msg.description || "Deepgram error");
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
			failStreamingSession(session, `WebSocket error: ${errMsg}`);
		}
	};

	ws.onclose = (ev) => {
		clearTimeout(wsConnectTimeout);
		const code = (ev as any)?.code;
		const reason = (ev as any)?.reason;
		voiceDebug("WebSocket onclose", { code, reason, closed: session.closed });
		if (!session.closed) {
			// Unexpected close — distinguish normal completion from network drops
			if (session.stopRequested || code === 1000 || code === 1001 || session.finalizedParts.length > 0) {
				if (session.interimText.trim()) {
					session.finalizedParts.push(session.interimText.trim());
					session.interimText = "";
				}
				// Normal close or we have usable transcript data — finalize
				finalizeSession(session);
			} else {
				// Abnormal close with no transcript — treat as error
				failStreamingSession(session, `Connection lost (code ${code ?? "unknown"}${reason ? `: ${reason}` : ""})`);
			}
		}
	};

	recProc.on("error", (err) => {
		voiceDebug("SoX process error:", err.message);
		if (!session.closed) {
			failStreamingSession(session, `SoX error: ${err.message}`);
		}
	});

	recProc.on("close", (code, signal) => {
		voiceDebug("SoX process closed", { code, signal, closed: session.closed, wsState: ws.readyState });
		// Only send CloseStream if the session isn't already being torn down
		// (stopStreamingSession sends its own CloseStream before killing SoX)
		if (!session.closed && !session.stopRequested && ws.readyState === WebSocket.OPEN) {
			try { ws.send(JSON.stringify({ type: "CloseStream" })); } catch {}
		}
	});

	return session;
}

function stopStreamingSession(session: StreamingSession): void {
	if (session.closed) return;
	session.stopRequested = true;

	try { session.recProcess.kill("SIGTERM"); } catch {}

	if (session.ws.readyState === WebSocket.OPEN) {
		try { session.ws.send(JSON.stringify({ type: "CloseStream" })); } catch {}
	}

	if (!session.finalizeTimer) {
		session.finalizeTimer = setTimeout(() => {
			session.finalizeTimer = null;
			if (session.closed) return;
			if (session.interimText.trim()) {
				session.finalizedParts.push(session.interimText.trim());
				session.interimText = "";
			}
			finalizeSession(session);
		}, STREAM_FINALIZE_TIMEOUT_MS);
	}
}

function finalizeSession(session: StreamingSession): void {
	if (session.closed) return;
	session.closed = true;
	voiceDebug("finalizeSession", { hadAudio: session.hadAudioData, hadSpeech: session.hadSpeech, parts: session.finalizedParts.length });

	if (session.staleSessionTimer) {
		clearTimeout(session.staleSessionTimer);
		session.staleSessionTimer = null;
	}
	if (session.finalizeTimer) {
		clearTimeout(session.finalizeTimer);
		session.finalizeTimer = null;
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

function failStreamingSession(session: StreamingSession, err: string): void {
	if (session.closed) return;
	session.closed = true;
	session.stopRequested = true;

	if (session.staleSessionTimer) {
		clearTimeout(session.staleSessionTimer);
		session.staleSessionTimer = null;
	}
	if (session.finalizeTimer) {
		clearTimeout(session.finalizeTimer);
		session.finalizeTimer = null;
	}
	if (session.keepAliveTimer) {
		clearInterval(session.keepAliveTimer);
		session.keepAliveTimer = null;
	}

	try { session.ws.close(); } catch {}
	try { session.recProcess.kill("SIGKILL"); } catch {}
	session.onError(err);
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
	let kittyIntentTimer: ReturnType<typeof setTimeout> | null = null;
	let kittyWarmupReleasePending = false;
	let editorTextBeforeSpacePress = "";
	let spaceConsumed = false;        // True once threshold passed and recording started
	let releaseDetectTimer: ReturnType<typeof setTimeout> | null = null;
	let warmupReleaseTimer: ReturnType<typeof setTimeout> | null = null;
	let warmupWidgetTimer: ReturnType<typeof setInterval> | null = null;
	let spacePressCount = 0;          // Count of rapid space presses (for non-Kitty hold detection)
	let lastSpacePressTime = 0;       // Timestamp of last space press event
	let holdConfirmed = false;        // True once we've confirmed user is holding (not tapping)
	let errorCooldownUntil = 0;       // After an error, block re-activation until this timestamp
	let lastNonSpaceKeyTime = 0;      // Timestamp of last non-space keypress (typing cooldown)
	let tailRecordingTimer: ReturnType<typeof setTimeout> | null = null; // Delayed stop after release

	function getToggleShortcut(): string {
		return loadGlobalConfig().toggleShortcut || DEFAULT_CONFIG.toggleShortcut || "ctrl+shift+v";
	}

	function getShortcutLabel(shortcut: string): string {
		return shortcut
			.split("+")
			.map((part) => part.length <= 1 ? part.toUpperCase() : part[0]!.toUpperCase() + part.slice(1))
			.join("+");
	}

	function shouldUseKittyHoldPath(): boolean {
		return kittyReleaseDetected || (process.platform === "linux" && isKittyProtocolActive());
	}

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
		ctx.ui.setStatus("voice", formatVoiceStatus({
			state: voiceState,
			enabled: config.enabled,
			onboardingCompleted: config.onboarding.completed,
			backend: config.backend,
			style: config.statusLabelStyle,
			nowMs: Date.now(),
			recordingStartMs: recordingStart,
			audioLevelSmoothed,
		}));
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

	function clearKittyIntentTimer() {
		if (kittyIntentTimer) {
			clearTimeout(kittyIntentTimer);
			kittyIntentTimer = null;
		}
	}

	function clearReleaseTimer() {
		if (releaseDetectTimer) {
			clearTimeout(releaseDetectTimer);
			releaseDetectTimer = null;
		}
	}

	function clearWarmupReleaseTimer() {
		if (warmupReleaseTimer) {
			clearTimeout(warmupReleaseTimer);
			warmupReleaseTimer = null;
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
		kittyWarmupReleasePending = false;
		editorTextBeforeSpacePress = "";
		clearHoldTimer();
		clearKittyIntentTimer();
		clearReleaseTimer();
		clearWarmupReleaseTimer();
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
		const isLocal = nextConfig.backend === "local";
		const hasKey = !!resolveDeepgramApiKey(nextConfig);
		// Local backend is always valid (sherpa handles everything). Deepgram needs API key.
		const validated = isLocal || hasKey;
		config = finalizeOnboardingConfig(nextConfig, { validated, source });
		configSource = selectedScope;
		const savedPath = saveConfig(config, selectedScope, currentCwd);
		const statusHeader = validated
			? "Voice setup complete."
			: "Voice setup saved, but DEEPGRAM_API_KEY is still required.";
		uiCtx.ui.notify([
			statusHeader,
			...summaryLines,
			"",
			`Saved to ${savedPath}`,
		].join("\n"), validated ? "info" : "warning");
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
			const holdStart = spaceDownTime ?? startTime;
			const elapsed = Date.now() - holdStart;
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

				hideWidget();

				if (ctx?.hasUI) {
					const prefix = editorTextBeforeVoice ? editorTextBeforeVoice + " " : "";
					const isLocal = config.backend === "local";

					if (isLocal) {
						// Local backend (batch mode): no interim transcripts were sent to the editor,
						// so we must always insert the final text. This is the ONLY place it arrives.
						ctx.ui.setEditorText(prefix + fullText);
					} else {
						// Streaming backend: interim transcripts already updated the editor live.
						// Only set final text if the editor still has content (user didn't hit Enter).
						const currentEditorText = ctx.ui.getEditorText?.() ?? "";
						if (currentEditorText.trim()) {
							ctx.ui.setEditorText(prefix + fullText);
						}
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
				// Local: show which model is transcribing + estimated time
				const modelName = LOCAL_MODELS.find(m => m.id === (config.localModel || "whisper-small"))?.name || config.localModel || "local model";
				ctx?.ui.notify(`Transcribing with ${modelName}…`, "info");
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
	//    start a short intent delay before entering warmup. If released
	//    before warmup starts, type a space. If released during warmup,
	//    cancel. If held past threshold → start recording. True release
	//    event stops recording.
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
	//      4. After HOLD_THRESHOLD_MS (500ms) from first press → start recording.
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

	function scheduleKittyActivation() {
		if (!spaceDownTime) return;
		kittyWarmupReleasePending = false;
		clearHoldTimer();
		const { activationDelayMs } = getKittyHoldTiming({
			heldMs: Date.now() - spaceDownTime,
			intentDelayMs: KITTY_INTENT_DELAY_MS,
			holdThresholdMs: HOLD_THRESHOLD_MS,
		});

		holdActivationTimer = setTimeout(() => {
			holdActivationTimer = null;
			if (shouldStartRecordingFromWarmup({ state: voiceState, releasePending: kittyWarmupReleasePending })) {
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
		}, activationDelayMs);
	}

	function scheduleKittyWarmup() {
		clearKittyIntentTimer();
		kittyIntentTimer = setTimeout(() => {
			kittyIntentTimer = null;
			if (!spaceDownTime || voiceState !== "idle" || spaceConsumed || holdConfirmed) return;

			holdConfirmed = true;
			kittyWarmupReleasePending = false;
			if (ctx?.hasUI) {
				const currentText = ctx.ui.getEditorText() || "";
				const nextText = stripInsertedSpaceOnWarmupStart({
					textBeforePress: editorTextBeforeSpacePress,
					currentText,
				});
				if (nextText !== currentText) ctx.ui.setEditorText(nextText);
			}
			setVoiceState("warmup");
			showWarmupWidget();
			startPreRecording();
			scheduleKittyActivation();
		}, KITTY_INTENT_DELAY_MS);
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
					kittyPath: shouldUseKittyHoldPath(),
					holdConfirmed,
					spaceConsumed,
					spacePressCount,
					spaceDownTime: spaceDownTime ? Date.now() - spaceDownTime : null,
					dataHex: Buffer.from(data).toString("hex"),
				});

				if (!isKeyRelease(data)) {
					clearWarmupReleaseTimer();
				}

				// ── Kitty key-release (true release event) ──
				if (isKeyRelease(data)) {
					kittyReleaseDetected = true;
					clearKittyIntentTimer();
					clearReleaseTimer();

					if (voiceState === "warmup") {
						kittyWarmupReleasePending = true;
						clearHoldTimer();
						clearWarmupReleaseTimer();
						warmupReleaseTimer = setTimeout(() => {
							warmupReleaseTimer = null;
							if (voiceState !== "warmup" || !kittyWarmupReleasePending) return;
							const shouldInsertSpace = shouldInsertSpaceOnKittyReleaseBeforeRecording(voiceState, spaceConsumed);
							resetHoldState();
							abortPreRecording();
							clearWarmupWidget();
							hideWidget();
							setVoiceState("idle");
							if (shouldInsertSpace && ctx?.hasUI) {
								ctx.ui.setEditorText((ctx.ui.getEditorText() || "") + " ");
							}
						}, WARMUP_RELEASE_GRACE_MS);
						return { consume: true };
					}

					if (spaceDownTime && shouldInsertSpaceOnKittyReleaseBeforeRecording(voiceState, spaceConsumed)) {
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
						if (kittyWarmupReleasePending) {
							scheduleKittyActivation();
						}
						return { consume: true };
					}
					if (shouldUseKittyHoldPath()) {
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
				// A) Kitty protocol (true release events available):
				//    Press fires ONCE on key-down. Repeats come as isKeyRepeat().
				//    Release comes as isKeyRelease(). NO timer-based release detection
				//    needed — the true release event handles everything.
				//    On first press: start a short intent delay. If the key is still
				//    held, enter warmup and continue toward activation.
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
					if (!shouldUseKittyHoldPath()) {
						voiceDebug("SPACE during recording → cancel delayed stop, re-arm release detect");
						resetReleaseDetect();
					} else {
						voiceDebug("SPACE during recording → cancel delayed stop (Kitty)");
					}
					return { consume: true };
				}

				// If already in warmup → consume
				if (voiceState === "warmup") {
					if (shouldUseKittyHoldPath() && kittyWarmupReleasePending) {
						scheduleKittyActivation();
					}
					if (!shouldUseKittyHoldPath()) {
						voiceDebug("SPACE during warmup → re-arm release detect");
						resetReleaseDetect();
					}
					return { consume: true };
				}

				// If we've already consumed space for this hold → consume
				// This handles the gap between holdActivationTimer firing and
				// voiceState transitioning to "recording" (async gap)
				if (spaceConsumed) {
					if (!shouldUseKittyHoldPath()) {
						voiceDebug("SPACE while spaceConsumed (async gap) → re-arm release detect");
						resetReleaseDetect();
					}
					return { consume: true };
				}

				// ──────────────────────────────────────────────────────────
				// PATH A: Kitty protocol — true key events available
				// ──────────────────────────────────────────────────────────
				if (shouldUseKittyHoldPath()) {
					if (voiceState === "idle") {
						const now = Date.now();
						spaceDownTime = now;
						spaceConsumed = false;
						spacePressCount = 1;
						lastSpacePressTime = now;
						holdConfirmed = false;
						kittyWarmupReleasePending = false;
						editorTextBeforeSpacePress = ctx?.hasUI ? (ctx.ui.getEditorText() || "") : "";

						const { warmupDelayMs } = getKittyHoldTiming({
							heldMs: 0,
							intentDelayMs: KITTY_INTENT_DELAY_MS,
							holdThresholdMs: HOLD_THRESHOLD_MS,
						});
						voiceDebug("kitty intent delay", { warmupDelayMs, holdThresholdMs: HOLD_THRESHOLD_MS });
						scheduleKittyWarmup();
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

	pi.registerShortcut(loadGlobalConfig().toggleShortcut || DEFAULT_CONFIG.toggleShortcut || "ctrl+shift+v", {
		description: "Toggle voice recording (start/stop)",
		handler: async (handlerCtx) => {
			ctx = handlerCtx;
			if (!config.enabled) {
				handlerCtx.ui.notify("Voice disabled. Use /voice on", "warning");
				return;
			}
			if (dictationMode) {
				// The configured toggle shortcut stops dictation mode
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

		if (config.enabled && config.onboarding.completed) {
			updateVoiceStatus();
			setupHoldToTalk();
		} else if (!config.onboarding.completed) {
			// First-time hint — show once, non-intrusive
			const hasKey = !!resolveDeepgramApiKey(config);
			const hasLocalModel = config.backend === "local" && !!config.localModel;
			if (startCtx.hasUI) {
				const audioTool = detectAudioCaptureTool();
				if (hasKey || hasLocalModel) {
					// Backend configured (Deepgram key or local model) — auto-activate
					config.onboarding.completed = true;
					config.onboarding.completedAt = new Date().toISOString();
					config.onboarding.source = "migration";
					const configToSave = getSessionStartPersistedConfig({
						config,
						envDeepgramApiKey: process.env.DEEPGRAM_API_KEY,
					});
					saveConfig(configToSave, config.scope === "project" ? "project" : "global", currentCwd);
					updateVoiceStatus();
					setupHoldToTalk();
					const backendLabel = hasLocalModel
						? `Local model: ${LOCAL_MODELS.find(m => m.id === config.localModel)?.name || config.localModel} (offline, batch mode)`
						: "Deepgram Nova-3 (cloud, live streaming)";
					const lines = [
						"pi-listen ready!",
						"",
						"  Hold SPACE to record → release to transcribe",
						`  ${getShortcutLabel(getToggleShortcut())} to toggle recording`,
						`  Backend: ${backendLabel}`,
						`  Audio: ${audioTool ? `${audioTool.name}` : "NONE — install sox or ffmpeg"}`,
						"",
						"  /voice-settings to change backend, model, or language",
					];
					startCtx.ui.notify(lines.join("\n"), audioTool ? "info" : "warning");
				} else {
					const lines = [
						"pi-listen installed — voice input for Pi",
						"",
						"  Two backends available:",
						"  • Deepgram — cloud, live streaming, $200 free credit (6–12 months of use)",
						"  • Local models — fully offline, no API key, auto-downloads on first use",
						"",
						`  Audio capture: ${audioTool ? `${audioTool.name} ✓` : "not found — install sox or ffmpeg"}`,
						"",
						"  Run /voice-settings to choose your backend and get started.",
					];
					startCtx.ui.notify(lines.join("\n"), "info");
				}
			}
		}
	});

	pi.on("session_shutdown", async () => {
		voiceCleanup();
		// Clean up sherpa recognizer cache
		try {
			const { clearRecognizerCache } = await import("./voice/sherpa-engine");
			clearRecognizerCache();
		} catch {}
		ctx = null;
	});

	pi.on("session_switch", async (_event, switchCtx) => {
		// Clean up any active recording before switching
		voiceCleanup();
		// Clear cached recognizer — new project may use different model/language
		try {
			const { clearRecognizerCache } = await import("./voice/sherpa-engine");
			clearRecognizerCache();
		} catch {}
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
					`  ${getShortcutLabel(getToggleShortcut())} → toggle recording on/off`,
					"  Quick SPACE tap → types a space (no voice)",
					"  Escape × 2 → clear editor",
					"",
					"  /voice-settings → open settings panel",
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
						`  ${getShortcutLabel(getToggleShortcut())} → also stops dictation`,
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
				lines.push(`    toggle shortcut:   ${getToggleShortcut()}`);
				lines.push(`    kitty protocol:    ${shouldUseKittyHoldPath() ? "detected" : "not detected"}`);
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
						let testInputArgs: string[];
						if (isMac) testInputArgs = ["-f", "avfoundation", "-i", ":default"];
						else if (isLinux) testInputArgs = ["-f", "pulse", "-i", "default"];
						else {
							const dshowDev = detectWindowsAudioDevice();
							testInputArgs = dshowDev ? ["-f", "dshow", "-i", `audio=${dshowDev}`] : ["-f", "dshow", "-i", "audio=Microphone"];
						}
						const inputArgs = testInputArgs;
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

				if (isLocal && config.localEndpoint) {
					// External local server connectivity check
					const serverCheck = await checkLocalServer(config.localEndpoint);
					if (serverCheck.ok) {
						lines.push("    local server:      OK (reachable)");
					} else {
						lines.push(`    local server:      NOT REACHABLE — ${serverCheck.error || "connection refused"}`);
					}
				} else if (isLocal) {
					// In-process sherpa-onnx mode — check module availability
					try {
						const { initSherpa, isSherpaAvailable } = await import("./voice/sherpa-engine");
						if (!isSherpaAvailable()) await initSherpa();
						const { isSherpaAvailable: checkAgain, getSherpaError } = await import("./voice/sherpa-engine");
						if (checkAgain()) {
							lines.push("    sherpa-onnx:       OK (in-process mode)");
						} else {
							lines.push(`    sherpa-onnx:       NOT AVAILABLE — ${getSherpaError() || "unknown"}`);
						}
					} catch (e: any) {
						lines.push(`    sherpa-onnx:       NOT AVAILABLE — ${e?.message || e}`);
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
				if (isLocal && config.localEndpoint) {
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
						lines.push(`  Hold SPACE to record, or use ${getShortcutLabel(getToggleShortcut())} to toggle.`);
					}
				} else if (isLocal) {
					// In-process sherpa-onnx mode — no server needed
					ready = !!tool;
					if (!tool) {
						lines.push("  Setup needed — install any one of:");
						lines.push("    brew install sox       # macOS (recommended)");
						lines.push("    apt install sox        # Linux");
					} else {
						lines.push("  All checks passed — voice is ready (in-process sherpa-onnx)!");
						lines.push(`  Hold SPACE to record, or use ${getShortcutLabel(getToggleShortcut())} to toggle.`);
					}
				} else {
					ready = !!dgKey && !!tool;
					if (!dgKey) {
						lines.push("  Setup needed:");
						lines.push("    1. Get a free key → https://dpgr.am/pi-voice ($200 free credit)");
						lines.push("    2. export DEEPGRAM_API_KEY=\"your-key\" (add to ~/.zshrc)");
						lines.push("    3. Or run /voice-settings to configure");
					} else if (!tool) {
						lines.push("  Setup needed — install any one of:");
						lines.push("    brew install sox       # macOS (recommended)");
						lines.push("    brew install ffmpeg    # macOS (alternative)");
						lines.push("    apt install sox        # Linux");
						lines.push("    apt install ffmpeg     # Linux (alternative)");
						lines.push("    choco install sox      # Windows");
					} else {
						lines.push("  All checks passed — voice is ready!");
						lines.push(`  Hold SPACE to record, or use ${getShortcutLabel(getToggleShortcut())} to toggle.`);
					}
				}

				cmdCtx.ui.notify(lines.join("\n"), ready ? "info" : "warning");
				return;
			}

			// /voice language, /voice setup, /voice info → open settings panel
			if (sub === "language" || sub === "lang" || sub.startsWith("language ") || sub.startsWith("lang ")) {
				await openSettingsPanel(cmdCtx);
				return;
			}

			if (sub === "info" || sub === "setup" || sub === "reconfigure" || sub === "settings" || sub === "config") {
				await openSettingsPanel(cmdCtx);
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

	// ─── /voice-setup → redirects to settings panel ─────────────────────────

	pi.registerCommand("voice-setup", {
		description: "Open pi-listen settings panel",
		handler: async (_args, cmdCtx) => openSettingsPanel(cmdCtx),
	});

	// ─── /voice-language → redirects to settings panel ───────────────────────

	pi.registerCommand("voice-language", {
		description: "Open pi-listen settings to change language",
		handler: async (_args, cmdCtx) => openSettingsPanel(cmdCtx),
	});

	// ─── Settings panel (shared handler) ────────────────────────────────────

	async function openSettingsPanel(cmdCtx: ExtensionCommandContext, initialTab?: number) {
		ctx = cmdCtx;

		const { detectDevice, getModelFitness, formatDeviceSummary } = await import("./voice/device");
		const { getDownloadedModels, deleteModel, ensureModelDownloaded } = await import("./voice/model-download");
		const { isSherpaAvailable, clearRecognizerCache } = await import("./voice/sherpa-engine");
		const { VoiceSettingsPanel } = await import("./voice/settings-panel");
		type PanelAction = import("./voice/settings-panel").PanelAction;
		const { LANGUAGES } = await import("./voice/onboarding");
		const { resolveDeepgramApiKey } = await import("./voice/deepgram");

		const device = detectDevice();
		const panel = new VoiceSettingsPanel({
			config,
			device,
			cwd: currentCwd,
			getModelFitness,
			getDownloadedModels,
			deleteModel,
			isSherpaAvailable,
			formatDeviceSummary,
			saveConfig: (cfg, scope, cwd) => saveConfig(cfg, scope, cwd),
			clearRecognizerCache: () => { try { clearRecognizerCache(); } catch {} },
			resolveApiKey: () => resolveDeepgramApiKey(config) ?? undefined,
			deepgramLanguages: LANGUAGES.map(l => ({ name: l.name, code: l.code, popular: l.popular })),
		}, initialTab);

		const result = await cmdCtx.ui.custom<PanelAction>(
			(_tui, _theme, _kb, done) => {
				panel.onClose = (action) => done(action);
				return panel;
			},
			{
				overlay: true,
				overlayOptions: {
					width: "70%",
					minWidth: 44,
					maxHeight: "80%",
					anchor: "center",
				},
			},
		);

		// Post-close: handle download action with full pre-checks + progress
		if (result?.type === "download" && result.modelId) {
			const model = LOCAL_MODELS.find(m => m.id === result.modelId);
			if (model) {
				const {
					checkDownloadPrereqs, createProgressTracker, verifyDownload, formatBytes,
				} = await import("./voice/model-download");
				const { initSherpa, isSherpaAvailable, getSherpaError } = await import("./voice/sherpa-engine");

				// ── Step 1: Check sherpa-onnx dependency ──
				if (!isSherpaAvailable()) {
					cmdCtx.ui.notify("Initializing sherpa-onnx runtime…", "info");
					const ok = await initSherpa();
					if (!ok) {
						cmdCtx.ui.notify(
							[
								"sherpa-onnx is required for local models but failed to initialize.",
								`Error: ${getSherpaError() || "unknown"}`,
								"",
								"To fix:",
								"  1. Ensure sherpa-onnx-node is installed: bun add sherpa-onnx-node",
								"  2. Check platform compatibility (macOS/Linux x64/arm64)",
								"  3. Or switch to Deepgram (cloud) backend in /voice-settings",
							].join("\n"),
							"error",
						);
						return;
					}
				}

				// ── Step 2: Pre-download checks (disk, network, permissions) ──
				cmdCtx.ui.notify(`Checking prerequisites for ${model.name} (${model.size})…`, "info");
				const preCheck = await checkDownloadPrereqs(model.sherpaModel.downloadUrls, model.sizeBytes);
				if (!preCheck.ok) {
					cmdCtx.ui.notify(
						[
							`Cannot download ${model.name}:`,
							"",
							...preCheck.issues.map(i => `  • ${i}`),
							"",
							"Resolve the above and try again via /voice-models.",
						].join("\n"),
						"error",
					);
					return;
				}

				// ── Step 3: Download with real-time progress ──
				const tracker = createProgressTracker(model.name);
				cmdCtx.ui.notify(`Starting download: ${model.name} (${model.size})…`, "info");

				try {
					await ensureModelDownloaded(
						model.id,
						model.sherpaModel.downloadUrls,
						model.sizeBytes,
						(raw) => {
							const rich = tracker(raw);
							if (rich) cmdCtx.ui.notify(rich.line, "info");
						},
					);
				} catch (err: any) {
					const msg = err?.message || String(err);
					const lines = [`Download failed: ${model.name}`];
					if (msg.includes("timed out") || msg.includes("Timeout")) {
						lines.push("The download timed out. Check your internet speed and try again.");
					} else if (msg.includes("ENOSPC") || msg.includes("no space")) {
						lines.push("Disk is full. Free up space and try again.");
					} else if (msg.includes("HTTP 4") || msg.includes("HTTP 5")) {
						lines.push(`Server error: ${msg}`);
						lines.push("The model server may be temporarily down. Try again in a few minutes.");
					} else {
						lines.push(`Error: ${msg}`);
					}
					lines.push("", "Partial downloads are auto-resumed on next attempt.");
					cmdCtx.ui.notify(lines.join("\n"), "error");
					return;
				}

				// ── Step 4: Post-download verification ──
				const verification = verifyDownload(model.id, model.sherpaModel.downloadUrls, model.sizeBytes);
				if (!verification.ok) {
					cmdCtx.ui.notify(
						[
							`${model.name} downloaded but verification failed:`,
							"",
							...verification.issues.map(i => `  • ${i}`),
							"",
							"Try: /voice-models → Downloaded tab → delete and re-download.",
						].join("\n"),
						"warning",
					);
					return;
				}

				cmdCtx.ui.notify(
					`${model.name} downloaded and verified (${model.size}). Ready to use.`,
					"info",
				);
			}
		}

		// Sync voice state after panel changes
		if (config.enabled) { setupHoldToTalk(); }
		else { voiceCleanup(); }
		updateVoiceStatus();
	}

	// ─── /voice-settings — unified pi-listen settings panel ─────────────

	pi.registerCommand("voice-settings", {
		description: "Open pi-listen settings — backend, models, language, device",
		handler: async (_args, cmdCtx) => openSettingsPanel(cmdCtx),
	});

	// ─── /voice-models — opens settings panel on Models tab ─────────────

	pi.registerCommand("voice-models", {
		description: "Manage local voice models (opens settings panel)",
		handler: async (_args, cmdCtx) => openSettingsPanel(cmdCtx, 1),
	});

}

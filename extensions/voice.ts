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
 *   - Configurable shortcut (default Ctrl+Shift+V) → toggle start/stop (always works)

 *
 * Config in ~/.pi/agent/settings.json under "voice": { ... }
 */

import type {
	ExtensionAPI,
	ExtensionContext,
	ExtensionCommandContext,
} from "@mariozechner/pi-coding-agent";
import { isKeyRelease, isKeyRepeat, matchesKey, Key, type KeyId } from "@mariozechner/pi-tui";

import { spawn, spawnSync, type ChildProcess } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import {
	DEFAULT_CONFIG,
	getSessionStartPersistedConfig,
	loadConfigWithSource,
	loadGlobalToggleShortcut,
	saveConfig,
	type VoiceConfig,
	type VoiceSettingsScope,
} from "./voice/config";
import { finalizeOnboardingConfig, runVoiceOnboarding, pickLanguage, languageDisplayName, modelForLanguage } from "./voice/onboarding";
import { makeWidgetRegistry, type WidgetRegistry } from "./voice/ui-widget-base";
import { makeRenderTicker, type RenderTicker } from "./voice/ui-render-ticker";
import { TtsInstallProgressWidget } from "./voice/tts-install-progress";
import { TtsPlaybackIndicator } from "./voice/tts-playback-indicator";
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
// v7.1.3 — hold threshold is configurable via `voice.holdThresholdMs`.
// Default lowered from 1200ms → 700ms for snappier hold-to-talk activation.
// Apple Caps Lock uses ~1s; modern push-to-talk apps tend to use 500-800ms.
// 700ms strikes a balance: too short risks accidental activation while typing
// (a single tap on the spacebar at the end of a word is ~80ms), too long
// feels laggy. Users can dial higher via /voice-hold-delay or settings.json.
const HOLD_THRESHOLD_DEFAULT_MS = 700;
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
				const audioFrame = chunk.buffer.slice(
					chunk.byteOffset,
					chunk.byteOffset + chunk.byteLength,
				) as ArrayBuffer;
				try { ws.send(audioFrame); } catch {}
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
	// Replace callbacks with no-ops BEFORE the backend-specific abort path so any
	// in-flight async work (sherpa transcription, late ws messages, recProcess close)
	// that resolves after abort cannot reach the recording state machine — including
	// across session replacement, where the surviving callbacks would otherwise
	// write into the new session's editor / fire notifications on the new ctx.
	session.onTranscript = () => {};
	session.onDone = () => {};
	session.onError = () => {};
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

	// ─── v7.1 Settings UI: widget registry + render ticker (per session) ──
	// Created lazily on first use (session_start) and torn down in
	// voiceCleanup. Owns all DisposableWidget instances for the session.
	let widgetRegistry: WidgetRegistry | null = null;
	let renderTicker: RenderTicker | null = null;
	function getOrInitVoiceUi(): { registry: WidgetRegistry; ticker: RenderTicker } {
		if (!widgetRegistry) widgetRegistry = makeWidgetRegistry();
		if (!renderTicker) renderTicker = makeRenderTicker();
		return { registry: widgetRegistry, ticker: renderTicker };
	}
	/** Currently-mounted install widgets keyed by modelId — for [esc] routing. */
	const activeInstallWidgets = new Map<string, TtsInstallProgressWidget>();
	/** Currently-mounted playback indicator (one at a time) for [esc] routing. */
	let activePlaybackIndicator: TtsPlaybackIndicator | null = null;

	/**
	 * v7.1.3 — read the active hold-threshold (ms) from config with bounds.
	 * Outside [200, 3000] falls back to the default (700ms). Bounds chosen
	 * so a typo can't make the experience completely broken: 200ms is
	 * effectively single-tap, 3000ms is "RSI-friendly slow press."
	 */
	function getHoldThresholdMs(): number {
		const v = (config as any).holdThresholdMs;
		if (typeof v === "number" && Number.isFinite(v) && v >= 200 && v <= 3000) return v;
		return HOLD_THRESHOLD_DEFAULT_MS;
	}

	// ─── Toggle Shortcut (resolved once at startup, used everywhere) ──────
	const resolvedToggleShortcut = loadGlobalToggleShortcut();
	const toggleShortcutLabel = resolvedToggleShortcut
		.split("+")
		.map((p) => p.length <= 1 ? p.toUpperCase() : p[0]!.toUpperCase() + p.slice(1))
		.join("+");

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
		// v7.1: cancel in-flight installs FIRST so their AbortControllers
		// fire before we drop UI state. Without this, a session_shutdown
		// during a download would leave the network/disk work running
		// after the widget slot is cleared (Codex v6 finding #3).
		for (const w of Array.from(activeInstallWidgets.values())) {
			try { w.cancel(); } catch (err) { voiceDebug("install widget cancel threw during voiceCleanup", String(err)); }
		}
		activeInstallWidgets.clear();
		// Stop any active playback the same way.
		try { activePlaybackIndicator?.stop(); } catch (err) { voiceDebug("playback stop threw during voiceCleanup", String(err)); }
		activePlaybackIndicator = null;
		// Drain registry — each widget self-clears its slot via dispose().
		try { widgetRegistry?.disposeAll(); } catch (err) { voiceDebug("widgetRegistry.disposeAll threw", String(err)); }
		try { renderTicker?.dispose(); } catch (err) { voiceDebug("renderTicker.dispose threw", String(err)); }
		widgetRegistry = null;
		renderTicker = null;

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
		// v7.2 — soft pulse between full and dim. Two-state instead of
		// three-state (cleaner, more like a breathing LED indicator
		// than the v7.0 multi-glyph approach).
		const phase = (Math.sin(Date.now() / 700) + 1) / 2;
		return phase > 0.5 ? "●" : "○";
	}

	function buildMiniWave(level: number): string {
		// Legacy block-bar wave — retained for compatibility with any
		// caller that still uses it. New code should use buildAuroraWave.
		const bars = "▁▂▃▄▅▆▇█";
		const len = 12;
		let out = "";
		const t = Date.now() / 1000;
		const energy = Math.pow(level, 0.7);
		for (let i = 0; i < len; i++) {
			const pos = i / len;
			const wave1 = Math.sin(t * 4.5 + i * 0.9) * 0.35;
			const wave2 = Math.sin(t * 7.2 + i * 1.4 + 2.0) * 0.15;
			const center = 1.0 - Math.abs(pos - 0.5) * 1.2;
			const base = 0.15 + energy * 0.85;
			const value = Math.max(0, Math.min(1, (wave1 + wave2 + 0.5) * base * center));
			const idx = Math.min(bars.length - 1, Math.round(value * (bars.length - 1)));
			out += bars[idx];
		}
		return out;
	}

	/**
	 * v7.2 world-class — Liquid Braille audio waveform with truecolor
	 * Aurora gradient. Per Gemini design recommendation:
	 *   - 8 effective vertical levels per CELL (4 dots × 2 columns) via
	 *     braille — vs 8 levels per cell with block-bars.
	 *   - 2 audio samples per cell width — 2× density of block-bar wave.
	 *   - Per-cell color picks an aurora stop based on local peak;
	 *     loud peaks "burn" into hot peach/red, soft tails stay cool
	 *     lavender. RGB interpolated at runtime.
	 * Output is `cells` cells wide rendered as a single string with
	 * inline ANSI 24-bit escapes; ends with a reset. Caller-side
	 * width math should use `cells` not the byte-length of the result.
	 */
	function buildAuroraWave(level: number, cells = 16): string {
		// Generate 2*cells audio "samples" via multi-frequency sine
		// + the live RMS energy. Same organic motion as the legacy
		// wave but at higher density.
		const samples = 2 * cells;
		const t = Date.now() / 1000;
		const energy = Math.pow(level, 0.7);
		const arr: number[] = [];
		for (let i = 0; i < samples; i++) {
			const pos = i / samples;
			const wave1 = Math.sin(t * 4.5 + i * 0.45) * 0.35;
			const wave2 = Math.sin(t * 7.2 + i * 0.7 + 2.0) * 0.15;
			const center = 1.0 - Math.abs(pos - 0.5) * 1.0;
			const base = 0.10 + energy * 0.90;
			const value = Math.max(0, Math.min(1, (wave1 + wave2 + 0.5) * base * center));
			arr.push(value);
		}
		// Lazy-import to keep voice.ts hot path fast on cold start.
		const { liquidBraille, auroraColor } = require("./voice/ui-aura") as typeof import("./voice/ui-aura");
		return liquidBraille(arr, auroraColor);
	}

	// ─── Warmup Widget ──────────────────────────────────────────────────
	function showWarmupWidget() {
		if (!ctx?.hasUI) return;

		const startTime = Date.now();

		const renderWarmup = () => {
			if (!ctx?.hasUI) return;
			const elapsed = Date.now() - startTime;
			const progress = Math.min(elapsed / getHoldThresholdMs(), 1);

			ctx.ui.setWidget("voice-recording", (_tui, theme) => {
				return {
					invalidate() {},
					render(width: number): string[] {
						// v7.2 world-class — Same Floating Island chrome
						// as the active recording widget, with the
						// progress bar inside. Establishes visual
						// continuity between warmup → recording (same
						// island, content shifts, no jump).
						const { island, auroraColor, titleBreathe } = require("./voice/ui-aura") as typeof import("./voice/ui-aura");
						const dim = (s: string) => theme.fg("dim", s);
						const muted = (s: string) => theme.fg("muted", s);
						const accent = (s: string) => theme.fg("accent", s);
						const islandW = Math.max(36, Math.min(46, width - 2));

						// Aurora gradient progress: ▰ filled / ▱ empty
						// with truecolor across the filled portion.
						const innerW = islandW - 2;
						const fixedW = 3 /* " ○ " */ + 1 /* trail */;
						const meterCells = Math.max(12, innerW - fixedW);
						const filled = Math.round(progress * meterCells);
						let bar = "";
						for (let i = 0; i < meterCells; i++) {
							if (i < filled) {
								// Color stop based on position along filled portion.
								const t = filled === 0 ? 0 : i / Math.max(1, meterCells - 1);
								bar += auroraColor(t) + "▰";
							} else {
								bar += dim("▱");
							}
						}
						bar += "\x1b[0m";

						const dot = progress < 1 ? muted("○") : accent("●");
						const content = ` ${dot} ${bar} `;
						// Breathing title — same aurora-cycle as recording widget
						// so the warmup → recording transition feels seamless.
						const titleStyled = titleBreathe(Date.now()) + "\x1b[1mVoice Mode\x1b[0m";
						const footer = progress < 1
							? dim("hold to record")
							: accent("ready");
						return island({ width: islandW, title: titleStyled, content, footer, dim });
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
				render(width: number): string[] {
					// v7.2 world-class — Floating Island + Liquid Braille +
					// Aurora gradient + breathing title + activity chip
					// + 300 ms fade-in transition from warmup.
					const { island, titleBreathe, activityTag } = require("./voice/ui-aura") as typeof import("./voice/ui-aura");
					const now = Date.now();
					const elapsed = (now - recordingStart) / 1000;
					const mins = Math.floor(elapsed / 60);
					const secs = elapsed % 60;
					const timeStr = mins > 0
						? `${mins}:${String(Math.floor(secs)).padStart(2, "0")}`
						: `${secs.toFixed(1)}s`;
					const dim = (s: string) => theme.fg("dim", s);
					const muted = (s: string) => theme.fg("muted", s);
					const accent = (s: string) => theme.fg("accent", s);

					// Activity chip — one-glance "is sound coming in?"
					const chip = activityTag(audioLevelSmoothed, dim);
					const chipPlain = chip.replace(/\x1b\[[\d;]*[A-Za-z]/g, "");

					// Compact 36-48 cols. Reserved cells:
					//   " ● "(3) + wave + " · TIME "(timer+4) + "  CHIP "(chip+3) + " "(1)
					const islandW = Math.max(36, Math.min(48, width - 2));
					const innerW = islandW - 2;
					const fixedW = 3 + (4 + timeStr.length) + (3 + chipPlain.length) + 1;
					const waveCells = Math.max(8, innerW - fixedW);

					// Fade-in over first 300 ms — wave amplitude
					// scales 0→1 so the recording widget grows out
					// of warmup rather than snapping in.
					const sinceStart = Math.max(0, now - recordingStart);
					const fade = Math.min(1, sinceStart / 300);

					const dot = theme.fg("error", getRecordDot());
					const wave = buildAuroraWave(audioLevelSmoothed * fade, waveCells);

					// Breathing title — slow aurora-color cycle.
					const titleStyled = titleBreathe(now) + "\x1b[1mVoice Input\x1b[0m";

					const content = ` ${dot} ${wave} ${dim("·")} ${muted(timeStr)}  ${chip} `;
					const footerStyled = `${dim("release")} ${accent("↑")}`;
					return island({ width: islandW, title: titleStyled, content, footer: footerStyled, dim });
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

		abortActiveSpeak();

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
		abortActiveSpeak();
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
					const finalText = prefix + fullText;

					if (isLocal) {
						// Local backend (batch mode): no interim transcripts were sent to the editor,
						// so we must always insert the final text. This is the ONLY place it arrives.
						ctx.ui.setEditorText(finalText);
					} else {
						// Streaming backend: interim transcripts already updated the editor live.
						// Only set final text if the editor still has content (user didn't hit Enter).
						const currentEditorText = ctx.ui.getEditorText?.() ?? "";
						if (currentEditorText.trim()) {
							ctx.ui.setEditorText(finalText);
						}
					}

					// v7.1.1 — auto-submit on STT (config.autoSubmitOnSpeak).
					// When enabled, the transcribed text is sent to the
					// agent immediately instead of sitting in the editor
					// waiting for [enter]. Defaults OFF; user toggles via
					// /voice-autosubmit or settings panel.
					if (config.autoSubmitOnSpeak === true && finalText.trim().length > 0) {
						// v7.2.3 — if the agent is currently mid-turn
						// (especially mid-retry), DON'T auto-submit.
						// followUp queueing during a retry pile-up
						// makes the agent look like it's "looping" —
						// each queued message gets processed only
						// after the broken turn finishes. Better UX:
						// keep transcribed text in the editor, notify
						// the user, and let them press [enter]
						// manually when ready.
						if (agentBusy) {
							voiceDebug("autoSubmitOnSpeak: agent busy — leaving text in editor");
							try {
								ctx.ui.notify(
									"Agent is busy — voice text held in editor. Press [↵] to send when ready.",
									"info",
								);
							} catch { /* notify may fail silently */ }
							// Skip dispatch but DON'T early-return —
							// the rest of the onDone handler still
							// needs to run state cleanup
							// (resetHoldState / setVoiceState("idle")).
						} else {
						// `sendUserMessage` lives on the `pi` ExtensionAPI
						// surface, NOT on `ctx`. Always triggers a turn.
						// `deliverAs: "followUp"` queues mid-stream
						// messages instead of throwing "Agent is already
						// processing".
						//
						// v7.2.2 (godspeed architect finding) — only
						// clear the editor AFTER send confirms. If
						// send rejects synchronously OR returns a
						// rejected Promise, the user's dictated text
						// stays visible so they can re-send.
						const send = (pi as any).sendUserMessage as ((text: string, opts?: any) => unknown) | undefined;
						if (typeof send === "function") {
							voiceDebug("autoSubmitOnSpeak: dispatching", { len: finalText.length });
							// godspeed architect finding — only clear if the
							// editor STILL contains exactly the dispatched
							// transcript. If the user typed more chars or
							// edited it while send was pending, leave their
							// edits alone.
							const dispatchedText = finalText;
							const clearAfterSuccess = () => {
								try {
									const cur = ctx?.ui.getEditorText?.() ?? "";
									if (cur === dispatchedText) {
										ctx?.ui.setEditorText("");
									}
								} catch { /* ui may be gone */ }
								editorTextBeforeVoice = "";
							};
							try {
								const r = send(finalText, { deliverAs: "followUp" });
								if (r && typeof (r as Promise<unknown>).then === "function") {
									(r as Promise<unknown>).then(clearAfterSuccess).catch((err) => {
										voiceDebug("autoSubmitOnSpeak: sendUserMessage rejected", String(err));
										// Editor stays populated — user can re-press [enter].
									});
								} else {
									// Sync return (or undefined) — assume success.
									clearAfterSuccess();
								}
							} catch (err) {
								voiceDebug("autoSubmitOnSpeak: sendUserMessage threw sync", String(err));
								// Editor stays populated.
							}
						} else {
							voiceDebug("autoSubmitOnSpeak: pi.sendUserMessage not available on this Pi version");
							ctx.ui.notify(
								"Auto-submit ON but unavailable on this Pi version (need pi.sendUserMessage). " +
								"Press [enter] to send, or update Pi.",
								"warning",
							);
						}
						} // end else (agent not busy)
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

			// v7.1 §4 — escape priority routing for v7.1 widgets. When
			// no overlay (panel/help/picker) is in front (those run
			// inside ctx.ui.custom() and consume their own input), the
			// fallthrough order is: install widget → playback indicator
			// → editor. Most recent install wins precedence.
			if (matchesKey(data, Key.escape)) {
				if (activeInstallWidgets.size > 0) {
					// Most recent install — Maps preserve insertion order.
					const ids = Array.from(activeInstallWidgets.keys());
					const lastId = ids[ids.length - 1]!;
					const w = activeInstallWidgets.get(lastId);
					if (w) { w.cancel(); return { consume: true }; }
				}
				if (activePlaybackIndicator) {
					activePlaybackIndicator.stop();
					return { consume: true };
				}
			}

			// v7.1 §11 — F1 always opens help, regardless of context.
			// `?` is intentionally NOT bound here because the user is
			// in the editor and ? is a literal character. Help is
			// reachable by F1 or /voice-help instead.
			if (matchesKey(data, Key.f1) && ctx?.hasUI) {
				openHelpOverlay(ctx as unknown as ExtensionCommandContext).catch(() => {});
				return { consume: true };
			}

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
							const remaining = Math.max(0, getHoldThresholdMs() - alreadyElapsed);

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
						}, getHoldThresholdMs());

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
							const remaining = Math.max(0, getHoldThresholdMs() - alreadyElapsed);

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

	// resolvedToggleShortcut is a runtime config string, but pi.registerShortcut
	// wants the literal-union KeyId type. The string was validated by
	// isValidShortcut() in loadGlobalToggleShortcut() (modifier+key shape with
	// a known modifier set), so asserting `as KeyId` is honest — `as any` was
	// strictly worse because it hid the intent and would have masked a real
	// signature change. If pi-tui later promotes KeyId to `string`, this
	// assertion just becomes a no-op.
	pi.registerShortcut(resolvedToggleShortcut as KeyId, {
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

	pi.on("session_start", async (event, startCtx) => {
		// `event.reason` was added in pi-mono 0.65.0 ("startup" | "reload" | "new" |
		// "resume" | "fork"). Older Pi versions don't include it. Narrow defensively
		// so the same code typechecks against both 0.57-era and 0.65+ types.
		const reason = (event as { reason?: string } | undefined)?.reason ?? "startup";
		const isStartup = reason === "startup";

		// Non-startup transitions: pi-mono >= 0.68.0 fires session_shutdown first
		// (and awaits it), so voiceCleanup() has already run. This is just belt-
		// and-suspenders for older Pi versions that may skip session_shutdown on
		// session replacement. The try/catch matches session_shutdown — a throw
		// inside voiceCleanup (e.g. a child process kill EPERM under load) must
		// not abort handler execution and leave ctx unassigned.
		if (!isStartup) {
			try { voiceCleanup(); } catch (err) {
				voiceDebug("voiceCleanup threw during session_start", { error: String(err) });
			}
		}

		ctx = startCtx;
		currentCwd = startCtx.cwd;
		const loaded = loadConfigWithSource(startCtx.cwd);
		config = loaded.config;
		configSource = loaded.source;

		// v7.1.3 — version banner emitted to debug log on every session
		// start. Lets users / support verify which extension build is
		// actually loaded after a `pi install .` (path-installed
		// extensions cache modules across `pi install` reinstalls — only
		// a fresh `pi` process picks up source changes).
		voiceDebug("pi-listen v7.1.3 loaded", { reason });

		// Migration / setup runs on EVERY session_start, regardless of reason.
		// Only the first-run notification is gated on isStartup.
		if (config.onboarding.completed) {
			// Always refresh the status bar — when voice is disabled,
			// updateVoiceStatus() clears the entry so users don't see stale
			// "MIC STREAM" text from a prior session. Hold-to-talk wiring
			// only runs when enabled.
			updateVoiceStatus();
			if (config.enabled) {
				setupHoldToTalk();
			}
			return;
		}

		// Onboarding not complete. Bail before the migration / hint UI work if the
		// session has no UI surface — non-interactive sessions can't display
		// notifications anyway, and migration wiring (setupHoldToTalk) requires UI.
		if (!startCtx.hasUI) return;

		// Try migration if a backend is already configured.
		const hasKey = !!resolveDeepgramApiKey(config);
		const hasLocalModel = config.backend === "local" && !!config.localModel;
		const audioTool = detectAudioCaptureTool();
		if (hasKey || hasLocalModel) {
			// Backend configured (Deepgram key or local model) — auto-activate.
			// Migration runs every transition; the welcome notification is gated
			// on isStartup so /new and /resume don't re-spam the hint.
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
			if (!isStartup) return;
			const backendLabel = hasLocalModel
				? `Local model: ${LOCAL_MODELS.find(m => m.id === config.localModel)?.name || config.localModel} (offline, batch mode)`
				: "Deepgram Nova-3 (cloud, live streaming)";
			const lines = [
				"pi-listen ready!",
				"",
				"  Hold SPACE to record → release to transcribe",
				`  ${toggleShortcutLabel} to toggle recording`,
				`  Backend: ${backendLabel}`,
				`  Audio: ${audioTool ? `${audioTool.name}` : "NONE — install sox or ffmpeg"}`,
				"",
				"  /voice-settings to change backend, model, or language",
			];
			startCtx.ui.notify(lines.join("\n"), audioTool ? "info" : "warning");
			return;
		}

		// No backend configured — show install hint only on actual startup.
		if (!isStartup) return;
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
	});

	pi.on("session_shutdown", async (event) => {
		// Synchronous teardown FIRST. Pi-mono >= 0.65.0 awaits the handler promise
		// before firing the replacement session_start, so the late-ctx-null race
		// the audit flagged is not present on current Pi. Keeping the order
		// (cleanup → null → await import) is still cheap insurance for older Pi
		// versions whose replacement path may not await. The try/catch is so a
		// throw inside voiceCleanup (e.g. a child process kill EPERM under load)
		// can't leak ctx or skip the recognizer cache clear.
		try { voiceCleanup(); } catch (err) {
			voiceDebug("voiceCleanup threw during shutdown", { error: String(err) });
		}
		ctx = null;

		// Clear the sherpa recognizer cache ONLY on terminal quit. On older Pi
		// versions (< 0.65.0) shutdown handlers are not awaited before the
		// replacement session_start, so an `await import()` here can race with
		// the new session re-initializing the recognizer — and our late
		// clearRecognizerCache() would wipe the recognizer the new session just
		// created. Keeping the cache across non-quit transitions is also faster:
		// /reload, /new, /fork, /resume typically reuse the same model+language,
		// so the recognizer is still hot. Per-session language/model changes are
		// already invalidated in voice/settings-panel.ts when the user picks a
		// different model. Reason is undefined on pre-0.65 Pi (where shutdown
		// only ever fired on quit), so we treat undefined the same as "quit".
		const reason = (event as { reason?: string } | undefined)?.reason;
		if (reason === "quit" || reason === undefined) {
			try {
				const { clearRecognizerCache } = await import("./voice/sherpa-engine");
				clearRecognizerCache();
			} catch {}
		}
	});

	// Note: pi-mono < 0.65.0 fired a discrete "session_switch" event for
	// /new, /resume, /fork. That event was removed in 0.65.0 in favor of the
	// session_shutdown → session_start (with reason) flow handled above.
	// We don't register a shim here because package.json:peerDependencies
	// requires "@mariozechner/pi-coding-agent": ">=0.65.0", so a host without
	// the new flow can't install this extension in the first place.

	// ─── Auto-speak (TTS after assistant turn ends) ─────────────────────
	//
	// When ttsAutoSpeak is true AND ttsEnabled is true, subscribe to
	// `turn_end` and pipe the assistant's response through speak() with
	// the same text-filter and length-cap logic the manual command uses.
	//
	// The handler is always registered; it short-circuits when the flags
	// are off. This way toggling `ttsAutoSpeak` via /voice-speak-toggle
	// doesn't require a session restart.
	//
	// Rate limit: track the last auto-speak timestamp and skip if a new
	// turn ends within ~3 seconds. Prevents the agent's rapid-fire
	// short responses from queueing up unread audio.
	let lastAutoSpeakAt = 0;
	const AUTO_SPEAK_RATE_LIMIT_MS = 3000;

	// v7.2.3 — track agent busy state so autoSubmit can skip dispatch
	// when the agent is mid-turn (and especially mid-retry). Otherwise
	// holding space during a "Retrying (n/3) in Xs..." cycle queues
	// followUp messages that pile onto the failing turn — feels like
	// the voice extension is "looping" because the agent never recovers.
	let agentBusy = false;
	pi.on("agent_start", async () => { agentBusy = true; });
	pi.on("agent_end", async () => { agentBusy = false; });

	// v7.2.1 — streaming auto-speak. Instead of waiting for `turn_end`
	// (which only fires AFTER the full response is generated), subscribe
	// to `message_update` and `message_end`:
	//   - message_update fires as the LLM streams tokens. We extract
	//     the accumulated text, find new sentence boundaries since
	//     the last speak, and queue those sentences for synthesis.
	//   - message_end flushes any remaining buffer (one final speak
	//     for trailing text without a sentence terminator).
	// Result: TTS starts speaking the FIRST sentence within ~1 second
	// of the agent producing it, instead of after the entire response
	// completes. Sentence chunking + pipelined synth (already in
	// speak.ts) keeps audio flowing continuously.
	//
	// Per-message tracking: each assistant message has its own
	// `spokenLen` cursor so concurrent messages (compaction, sub-turns)
	// don't cross-talk. Map keyed by message id.

	interface MessageStreamState {
		spokenLen: number;          // chars of accumulated text already queued for speech
		pending: Promise<void>;     // chain of in-flight speak() calls — serialize per message
	}
	const messageStreams = new Map<string, MessageStreamState>();

	const SENTENCE_TERMINATORS = /[.!?](?=\s|$)|[\n]{1,}/g;

	function extractAccumulatedText(message: any): string {
		if (!message || !Array.isArray(message.content)) return "";
		return (message.content as any[])
			.filter(c => c?.type === "text" && typeof c.text === "string")
			.map(c => c.text as string)
			.join("");
	}

	/** Detect last sentence boundary at or before `maxIdx` in `text`.
	 *  Returns the index AFTER the terminator (so [0..idx] is a complete
	 *  block). Returns -1 if none found. */
	function lastSentenceEnd(text: string, fromIdx: number): number {
		let lastEnd = -1;
		const re = /[.!?](?=\s|$)|\n+/g;
		let m: RegExpExecArray | null;
		while ((m = re.exec(text)) !== null) {
			const end = m.index + m[0].length;
			if (end > fromIdx) lastEnd = end;
		}
		return lastEnd;
	}

	// Eager-speak threshold — if the buffered new text grows past this
	// many chars without a sentence terminator (e.g. very long
	// sentence mid-stream), speak what we have at the next clause
	// boundary (`,` `;` ` — `) or flush as-is. Keeps latency low for
	// long-winded responses.
	const EAGER_SPEAK_CHARS = 80;
	const EAGER_CLAUSE_RE = /[,;:—](?=\s)/g;

	function lastClauseEnd(text: string): number {
		let last = -1;
		EAGER_CLAUSE_RE.lastIndex = 0;
		let m: RegExpExecArray | null;
		while ((m = EAGER_CLAUSE_RE.exec(text)) !== null) {
			last = m.index + m[0].length;
		}
		return last;
	}

	async function maybeSpeakNew(messageId: string, fullText: string, isFinal: boolean): Promise<void> {
		if (!config.ttsEnabled || !config.ttsAutoSpeak) return;
		// Don't speak while STT is hot — feedback loop hazard.
		if (voiceState === "warmup" || voiceState === "recording" || voiceState === "finalizing") return;
		if (!fullText) return;

		let state = messageStreams.get(messageId);
		if (!state) {
			state = { spokenLen: 0, pending: Promise.resolve() };
			messageStreams.set(messageId, state);
		}

		const newText = fullText.slice(state.spokenLen);
		if (!newText) return;

		let speakUpTo: number;
		if (isFinal) {
			// Flush everything remaining.
			speakUpTo = newText.length;
		} else {
			// Prefer sentence boundary; fall back to clause boundary
			// if we've buffered > EAGER_SPEAK_CHARS without one.
			const boundary = lastSentenceEnd(newText, 0);
			if (boundary > 0) {
				speakUpTo = boundary;
			} else if (newText.length >= EAGER_SPEAK_CHARS) {
				const clause = lastClauseEnd(newText);
				if (clause <= 0) return; // not even a clause yet — wait
				speakUpTo = clause;
			} else {
				return; // wait for more text
			}
		}

		const chunk = newText.slice(0, speakUpTo).trim();
		state.spokenLen += speakUpTo;
		if (!chunk) return;
		voiceDebug("autoSpeak.streaming", { id: messageId, chars: chunk.length, isFinal });

		// Strip code blocks / links / emojis / abbreviations to spoken form.
		const { prepareForSpeech } = await import("./voice/tts-text-filter");
		const prepared = prepareForSpeech(chunk, {
			maxChars: 2000,
			stripCodeBlocks: true,
			collapseLinks: true,
		});
		if (prepared.skipped || !prepared.text.trim()) return;

		// Serialize per-message: chain onto pending so chunks play in
		// order, never overlapping for the same message.
		const text = prepared.text;
		state.pending = state.pending.then(async () => {
			try {
				if (ctx) await runSpeak(ctx, text);
			} catch {
				// Auto-speak failures are non-blocking.
			}
		});
	}

	// Diagnostic: stream log shows when message_update / message_end
	// actually fire (some Pi versions don't emit message_update
	// per-token). Lets us verify the streaming path is alive.
	const streamDiag = (s: string) => {
		try {
			const fs2 = require("node:fs") as typeof import("node:fs");
			fs2.appendFileSync("/tmp/pi-listen-stream.log", `[${new Date().toISOString()}] ${s}\n`);
		} catch { /* best-effort */ }
	};
	let mu_count = 0;
	pi.on("message_update", async (event) => {
		const msg = (event as any)?.message;
		if (!msg || msg.role !== "assistant") return;
		const id = (msg.id as string) || "current";
		const fullText = extractAccumulatedText(msg);
		mu_count++;
		if (mu_count <= 5 || mu_count % 10 === 0) {
			streamDiag(`message_update #${mu_count} id=${id} chars=${fullText.length}`);
		}
		await maybeSpeakNew(id, fullText, false);
	});

	pi.on("message_end", async (event) => {
		const msg = (event as any)?.message;
		if (!msg || msg.role !== "assistant") return;
		const id = (msg.id as string) || "current";
		const fullText = extractAccumulatedText(msg);
		streamDiag(`message_end id=${id} chars=${fullText.length} updates_seen=${mu_count}`);
		mu_count = 0;
		await maybeSpeakNew(id, fullText, true);
		// Drop the stream state once flushed — prevents unbounded growth.
		const state = messageStreams.get(id);
		if (state) {
			state.pending.finally(() => {
				if (messageStreams.get(id) === state) messageStreams.delete(id);
			});
		}
	});

	// Legacy turn_end fallback — only relevant on Pi versions that
	// don't fire message_update yet. Tracked via lastAutoSpeakAt so
	// it doesn't double-speak when message_update already covered the
	// content.
	pi.on("turn_end", async (event, _evtCtx) => {
		if (!config.ttsEnabled || !config.ttsAutoSpeak) return;
		const message = (event as any)?.message;
		if (!message || message.role !== "assistant") return;
		const id = (message.id as string) || "current";
		// If message_update already drained this message's stream
		// state, the streaming path handled it — skip.
		const state = messageStreams.get(id);
		if (state) return;
		if (voiceState === "warmup" || voiceState === "recording" || voiceState === "finalizing") return;
		const now = Date.now();
		if (now - lastAutoSpeakAt < AUTO_SPEAK_RATE_LIMIT_MS) return;
		const text = extractAccumulatedText(message).trim();
		if (!text) return;
		const { prepareForSpeech } = await import("./voice/tts-text-filter");
		const prepared = prepareForSpeech(text, { maxChars: 2000, stripCodeBlocks: true, collapseLinks: true });
		if (prepared.skipped) return;
		lastAutoSpeakAt = now;
		try { if (ctx) await runSpeak(ctx, prepared.text); } catch { /* non-blocking */ }
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
					`  ${toggleShortcutLabel} → toggle recording on/off`,
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
						`  ${toggleShortcutLabel} → also stops dictation`,
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
				lines.push(`    hold threshold:    ${getHoldThresholdMs()}ms`);
				lines.push(`    toggle shortcut:   ${resolvedToggleShortcut}`);
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
						lines.push(`  Hold SPACE to record, or use ${toggleShortcutLabel} to toggle.`);
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
						lines.push(`  Hold SPACE to record, or use ${toggleShortcutLabel} to toggle.`);
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
						lines.push(`  Hold SPACE to record, or use ${toggleShortcutLabel} to toggle.`);
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

	// ─── /voice-help → v7.1 §11 keyboard / command reference ────────────────

	pi.registerCommand("voice-help", {
		description: "Show pi-listen keyboard + command reference",
		handler: async (_args, cmdCtx) => openHelpOverlay(cmdCtx),
	});

	async function openHelpOverlay(cmdCtx: ExtensionCommandContext): Promise<void> {
		if (!cmdCtx.hasUI) {
			cmdCtx.ui.notify(
				"pi-listen: hold space=record · /voice-speak <text> · /voice-settings · /voice-help",
				"info",
			);
			return;
		}
		const { HelpOverlay } = await import("./voice/ui-help-overlay");
		await cmdCtx.ui.custom<void>(
			(_tui, theme, _kb, done) => new HelpOverlay({ theme }, done),
			{
				overlay: true,
				overlayOptions: { width: "70%", minWidth: 60, maxHeight: "80%", anchor: "center" },
			},
		);
	}

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
		// Construct the panel inside the custom() callback so the host theme
		// is in scope. Without this the panel falls back to raw ANSI which
		// clashes with non-default themes (Catppuccin Mocha etc.).
		const panelDeps = {
			config,
			device,
			cwd: currentCwd,
			getModelFitness,
			getDownloadedModels,
			deleteModel,
			isSherpaAvailable,
			formatDeviceSummary,
			saveConfig: (cfg: VoiceConfig, scope: VoiceSettingsScope, cwd: string) => saveConfig(cfg, scope, cwd),
			clearRecognizerCache: () => { try { clearRecognizerCache(); } catch {} },
			resolveApiKey: () => resolveDeepgramApiKey(config) ?? undefined,
			deepgramLanguages: LANGUAGES.map(l => ({ name: l.name, code: l.code, popular: l.popular })),
		};

		let panel!: InstanceType<typeof VoiceSettingsPanel>;
		const result = await cmdCtx.ui.custom<PanelAction>(
			(_tui, theme, _kb, done) => {
				panel = new VoiceSettingsPanel({ ...panelDeps, theme }, initialTab);
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

		// Post-close: handle the speak-test action by re-using the
		// /voice-speak-test command path. We do this AFTER the panel has
		// closed so the test sample plays without the picker overlay
		// interfering with the audio cue (Pi's terminal renderer paints
		// the panel as an overlay; closing it first gives a clean
		// playback experience).
		if (result?.type === "speak-test") {
			await runSpeak(cmdCtx, "The quick brown fox jumps over the lazy dog.", { forceEnabled: true });
			return;
		}

		// Post-close: handle the TTS install action triggered by selecting
		// a not-yet-installed model in the Speak tab Model picker.
		// v7.1: surfaces progress through the new sticky install widget
		// (`tts-install-progress.ts`) keyed by modelId so concurrent
		// installs of different models coexist without slot collision.
		if (result?.type === "tts-install" && result.modelId) {
			const { ensureTtsModelInstalled, getTtsModel } = await import("./voice/tts-local-models");
			const model = getTtsModel(result.modelId);
			// runInstallWithWidget rethrows on failure (Codex v6 #2) so
			// callers like runSpeak can short-circuit. This call site is
			// terminal — no further work — so swallow the rethrow after
			// notifications were already emitted by the helper itself.
			try {
				await runInstallWithWidget(cmdCtx, model.id, model.name, model.sizeBytes ?? 0, ensureTtsModelInstalled);
			} catch { /* notify already emitted in runInstallWithWidget */ }
			return;
		}

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

	// ─── TTS commands (v6.0.0+) ─────────────────────────────────────────
	//
	// The active speech AbortController lives at extension scope so
	// `/voice-speak-stop` can cancel whatever is currently playing.
	// Re-entrant `/voice-speak` calls abort the prior one before starting
	// (no overlapping audio); `null` means nothing is in-flight.
	let activeSpeak: AbortController | null = null;

	function abortActiveSpeak(): boolean {
		if (!activeSpeak) return false;
		try { activeSpeak.abort(); } catch {}
		activeSpeak = null;
		return true;
	}

	/**
	 * v7.1: run an install with the new sticky `TtsInstallProgressWidget`.
	 * Replaces the v7.0.x notify-spam loop. Mounts a per-model-id slot
	 * (`installWidgetKey(modelId)`) so two concurrent installs for
	 * different models coexist without clobbering. The widget owns its
	 * own AbortController (currently abort-only via cancel()); the
	 * existing in-flight Map in `ensureTtsModelInstalled` serializes
	 * same-id calls.
	 */
	async function runInstallWithWidget(
		cmdCtx: ExtensionCommandContext | ExtensionContext,
		modelId: string,
		modelName: string,
		totalBytesEstimate: number,
		ensureTtsModelInstalled: (
			id: string,
			opts: { signal?: AbortSignal; onProgress?: (info: any) => void },
		) => Promise<unknown>,
		// godspeed architect finding: accept caller signal so
		// /voice-speak-stop or TTS-disable propagates into the install.
		// Caller's signal cascades: aborting it triggers our own
		// AbortController and tears down the widget cleanly.
		callerSignal?: AbortSignal,
	): Promise<void> {
		if (!cmdCtx.hasUI) {
			// Headless / scripted mode — fall back to a single notify so
			// users running pi without a TUI still get progress feedback.
			cmdCtx.ui.notify(`Installing ${modelName}…`, "info");
			try {
				await ensureTtsModelInstalled(modelId, {});
				cmdCtx.ui.notify(`${modelName} ready.`, "info");
			} catch (err: any) {
				// Codex v6.5: rethrow so callers like runSpeak short-
				// circuit instead of proceeding with a model that
				// isn't installed. Match the TUI branch's
				// `__alreadyNotified` contract so the outer catch
				// doesn't emit a duplicate notify.
				if (err?.name === "AbortError") {
					cmdCtx.ui.notify(`Install cancelled: ${modelName}`, "warning");
				} else {
					cmdCtx.ui.notify(`Install failed: ${err?.message ?? err}`, "error");
				}
				if (err && typeof err === "object") {
					try { (err as any).__alreadyNotified = true; } catch { /* frozen errors */ }
				}
				throw err;
			}
			return;
		}

		const { registry, ticker } = getOrInitVoiceUi();
		const controller = new AbortController();
		// Cascade caller signal → controller. If runSpeak's activeSpeak
		// signal aborts (user runs /voice-speak-stop or disables TTS),
		// the install cancels too. Listener removed in finally.
		let callerAbortListener: (() => void) | null = null;
		if (callerSignal) {
			if (callerSignal.aborted) {
				try { controller.abort(); } catch {}
			} else {
				callerAbortListener = () => { try { controller.abort(); } catch {} };
				callerSignal.addEventListener("abort", callerAbortListener);
			}
		}
		const widget = new TtsInstallProgressWidget({
			ui: cmdCtx.ui,
			modelId,
			modelName,
			totalBytesEstimate,
			registry,
			ticker,
			controller,
		});
		activeInstallWidgets.set(modelId, widget);
		try {
			await ensureTtsModelInstalled(modelId, {
				signal: controller.signal,
				onProgress: (info) => widget.onProgress(info),
			});
			// Widget self-disposes on phase=done; if ensure resolved
			// without firing done (very unlikely), make sure cleanup
			// runs anyway.
			widget.dispose();
			cmdCtx.ui.notify(`${modelName} ready.`, "info");
		} catch (err: any) {
			widget.dispose();
			// Codex v6 finding #2: rethrow so the caller (e.g. runSpeak)
			// can short-circuit instead of proceeding into speak() with
			// a model that isn't installed. We notify once here with the
			// install context; the rethrown error will be caught by the
			// caller's outer try/catch but tagged with `__alreadyNotified`
			// so the caller can skip its generic notify.
			if (err?.name === "AbortError") {
				cmdCtx.ui.notify(`Install cancelled: ${modelName}`, "warning");
			} else {
				cmdCtx.ui.notify(`Install failed: ${err?.message ?? err}`, "error");
			}
			if (err && typeof err === "object") {
				try { (err as any).__alreadyNotified = true; } catch { /* frozen errors */ }
			}
			throw err;
		} finally {
			// Codex v6 finding #4: owner-checked delete so an older
			// finally cannot evict a newer same-id widget. (The
			// in-flight Map in ensureTtsModelInstalled already serializes
			// same-id installs, but the side-table here doesn't piggy-
			// back on that guarantee — defensive owner check.)
			if (activeInstallWidgets.get(modelId) === widget) {
				activeInstallWidgets.delete(modelId);
			}
			// Always remove the caller-signal listener.
			if (callerAbortListener && callerSignal) {
				try { callerSignal.removeEventListener("abort", callerAbortListener); } catch {}
			}
		}
	}

	async function runSpeak(cmdCtx: ExtensionCommandContext | ExtensionContext, text: string, opts: { forceEnabled?: boolean } = {}): Promise<void> {
		// `forceEnabled` lets /voice-speak-test bypass the gate without
		// mutating shared config. The previous mutate-snapshot-restore
		// pattern raced against /voice-speak-toggle and could clobber the
		// user's explicit toggle.
		if (!config.ttsEnabled && !opts.forceEnabled) {
			cmdCtx.ui.notify("TTS is disabled. Enable in /voice-settings.", "warning");
			return;
		}
		if (voiceState === "recording" || voiceState === "finalizing") {
			// Speaking while the mic is hot would feedback into STT.
			cmdCtx.ui.notify("Cannot speak while recording. Stop recording first.", "warning");
			return;
		}

		// Cancel any in-flight speech so the new request takes the floor.
		abortActiveSpeak();
		const controller = new AbortController();
		activeSpeak = controller;

		try {
			const { speak } = await import("./voice/speak");
			const { getInstalledTtsModelDir, ensureTtsModelInstalled, getTtsModel } = await import("./voice/tts-local-models");

			// On the local backend, fetch the model on-demand if missing.
			// Deepgram backend skips this branch entirely. v7.1: surface
			// progress through the sticky install widget instead of the
			// v7.0.x notify-spam loop.
			if ((config.ttsBackend ?? "local") === "local") {
				const modelId = config.ttsLocalModel || "kitten-nano-en-v0_2";
				try {
					getInstalledTtsModelDir(modelId);
				} catch {
					const model = getTtsModel(modelId);
					await runInstallWithWidget(cmdCtx, modelId, model.name, model.sizeBytes ?? 0, ensureTtsModelInstalled, controller.signal);
				}
			}

			// v7.1: mount the honest playback indicator (§6 of plan).
			// Spinner + state word with no fake amplitude meter. Until
			// `speak()` exposes a phase callback (v7.2), the indicator
			// stays on "playing" for the whole synth+play cycle —
			// honest because audio IS in flight throughout. Disposed
			// in finally regardless of success/abort/error. Tracked
			// on `activePlaybackIndicator` so the [esc] router can
			// stop playback when no install widget owns escape.
			let indicator: TtsPlaybackIndicator | null = null;
			if (cmdCtx.hasUI) {
				const { registry, ticker } = getOrInitVoiceUi();
				indicator = new TtsPlaybackIndicator({
					ui: cmdCtx.ui,
					registry,
					ticker,
					onStop: () => abortActiveSpeak(),
				});
				indicator.setState("playing");
				activePlaybackIndicator = indicator;
			}
			try {
				await speak({
					text,
					config,
					signal: controller.signal,
					resolveModelDir: (id) => getInstalledTtsModelDir(id),
				});
			} finally {
				indicator?.setState("idle"); // self-disposes
				// Owner-checked clear, mirroring runInstallWithWidget's
				// Codex v6 #4 fix.
				if (activePlaybackIndicator === indicator) activePlaybackIndicator = null;
			}
		} catch (err: any) {
			if (err?.name === "AbortError") return;
			// If the install widget already notified the user with a
			// scoped message ("Install failed: <model>"), don't emit a
			// duplicate generic "Speak failed:" notify. The install
			// widget's notify is more informative for that path.
			if (!err?.__alreadyNotified) {
				cmdCtx.ui.notify(`Speak failed: ${err?.message ?? err}`, "error");
			}
		} finally {
			if (activeSpeak === controller) activeSpeak = null;
		}
	}

	pi.registerCommand("voice-speak", {
		description: "Speak the given text (text-to-speech)",
		handler: async (args, cmdCtx) => {
			ctx = cmdCtx;
			const text = (args || "").trim();
			if (!text) {
				cmdCtx.ui.notify("Usage: /voice-speak <text>", "warning");
				return;
			}
			await runSpeak(cmdCtx, text);
		},
	});

	// v7.1.3 — toggle Deepgram WebSocket streaming TTS (cloud backend).
	// When ON, /voice-speak uses wss://api.deepgram.com/v1/speak so audio
	// frames stream into the local player as they arrive (sub-200ms
	// TTFA in good network conditions). When OFF, the REST `/v1/speak`
	// path returns a complete WAV.
	pi.registerCommand("voice-stream", {
		description: "Toggle Deepgram WebSocket streaming TTS (cloud)",
		handler: async (args, cmdCtx) => {
			ctx = cmdCtx;
			const trimmed = (args || "").trim().toLowerCase();
			let next: boolean;
			if (trimmed === "on") next = true;
			else if (trimmed === "off") next = false;
			else next = !(config.ttsDeepgramStreaming === true);
			config.ttsDeepgramStreaming = next;
			saveConfig(config, config.scope === "project" ? "project" : "global", currentCwd);
			cmdCtx.ui.notify(`Deepgram WebSocket streaming TTS: ${next ? "ON" : "OFF"}`, "info");
		},
	});

	// v7.1.3 — tune the hold-to-talk activation delay.
	pi.registerCommand("voice-hold-delay", {
		description: "Set hold-to-talk delay in ms (200-3000, default 700)",
		handler: async (args, cmdCtx) => {
			ctx = cmdCtx;
			const trimmed = (args || "").trim();
			if (!trimmed) {
				cmdCtx.ui.notify(`Hold delay: ${getHoldThresholdMs()}ms (default 700)`, "info");
				return;
			}
			const ms = parseInt(trimmed, 10);
			if (!Number.isFinite(ms) || ms < 200 || ms > 3000) {
				cmdCtx.ui.notify(`Invalid value: ${trimmed}. Must be 200-3000 ms.`, "warning");
				return;
			}
			(config as any).holdThresholdMs = ms;
			saveConfig(config, config.scope === "project" ? "project" : "global", currentCwd);
			cmdCtx.ui.notify(`Hold delay set to ${ms}ms.`, "info");
		},
	});

	// v7.1.1 — toggle auto-submit on STT (sends transcribed text
	// directly to the agent instead of just placing it in the editor).
	pi.registerCommand("voice-autosubmit", {
		description: "Toggle auto-submit on STT — sends spoken text to the agent immediately",
		handler: async (args, cmdCtx) => {
			ctx = cmdCtx;
			const trimmed = (args || "").trim().toLowerCase();
			let next: boolean;
			if (trimmed === "on") next = true;
			else if (trimmed === "off") next = false;
			else next = !(config.autoSubmitOnSpeak === true);
			config.autoSubmitOnSpeak = next;
			saveConfig(config, config.scope === "project" ? "project" : "global", currentCwd);
			cmdCtx.ui.notify(`Auto-submit on speak: ${next ? "ON" : "OFF"}`, "info");
		},
	});

	pi.registerCommand("voice-speak-stop", {
		description: "Stop in-flight TTS playback",
		handler: async (_args, cmdCtx) => {
			ctx = cmdCtx;
			if (abortActiveSpeak()) {
				cmdCtx.ui.notify("Speech stopped.", "info");
			} else {
				cmdCtx.ui.notify("No active speech.", "info");
			}
		},
	});

	pi.registerCommand("voice-speak-toggle", {
		description: "Toggle TTS on/off (master switch)",
		handler: async (_args, cmdCtx) => {
			ctx = cmdCtx;
			const nowEnabling = !config.ttsEnabled;
			config.ttsEnabled = nowEnabling;
			saveConfig(config, config.scope === "project" ? "project" : "global", currentCwd);
			if (!nowEnabling) abortActiveSpeak();
			cmdCtx.ui.notify(`TTS ${nowEnabling ? "enabled" : "disabled"}.`, "info");
			// First-time enable → v7.1 §9 rich onboarding overlay with
			// three explicit actions (try / pick model / skip).
			// Subsequent toggles are silent.
			if (nowEnabling && !(config as any).ttsOnboardingShown && cmdCtx.hasUI) {
				try {
					const { detectDevice } = await import("./voice/device");
					const { TtsOnboardingOverlay } = await import("./voice/tts-onboarding-overlay");
					const device = detectDevice();
					// §9: persist `ttsOnboardingShown = true` BEFORE any
					// async work so a failed install/cancel never re-prompts.
					(config as any).ttsOnboardingShown = true;
					saveConfig(config, config.scope === "project" ? "project" : "global", currentCwd);

					const result = await cmdCtx.ui.custom<import("./voice/tts-onboarding-overlay").OnboardingResult>(
						(_tui, theme, _kb, done) => {
							return new TtsOnboardingOverlay({ systemLocale: device.systemLocale, theme }, done);
						},
						{
							overlay: true,
							overlayOptions: { width: "70%", minWidth: 60, maxHeight: "60%", anchor: "center" },
						},
					);
					if (result?.kind === "test") {
						await runSpeak(cmdCtx, "The quick brown fox jumps over the lazy dog.", { forceEnabled: true });
					} else if (result?.kind === "pickModel") {
						// Open the settings panel directly on the Speak tab.
						// Tab order is general/models/downloaded/speak/device → idx 3.
						await openSettingsPanel(cmdCtx, 3);
					}
				} catch (err) {
					voiceDebug("onboarding overlay threw", String(err));
				}
			} else if (nowEnabling && !cmdCtx.hasUI) {
				// Headless mode — fall back to the v7.0 notify-based hint.
				try {
					const { detectDevice } = await import("./voice/device");
					const { maybeShowTtsOnboarding } = await import("./voice/tts-onboarding");
					maybeShowTtsOnboarding({
						ctx: cmdCtx,
						config,
						device: detectDevice(),
						cwd: currentCwd,
						saveConfig: (cfg, scope, cwd) => saveConfig(cfg, scope, cwd),
					});
				} catch { /* onboarding hint is best-effort */ }
			}
		},
	});

	pi.registerCommand("voice-speak-test", {
		description: "Synthesize a sample sentence in the current voice",
		handler: async (_args, cmdCtx) => {
			ctx = cmdCtx;
			// Pass forceEnabled so the test runs even when ttsEnabled is
			// false — without mutating shared config (a concurrent
			// /voice-speak-toggle would otherwise be clobbered when the
			// test's finally restored its snapshot).
			await runSpeak(cmdCtx, "The quick brown fox jumps over the lazy dog.", { forceEnabled: true });
		},
	});

	pi.registerCommand("voice-speak-info", {
		description: "Show TTS configuration: backend, model, voice, install state",
		handler: async (_args, cmdCtx) => {
			ctx = cmdCtx;
			const { getTtsModel, isTtsModelInstalled, TTS_LOCAL_MODELS } =
				await import("./voice/tts-local-models");
			const { DEEPGRAM_TTS_VOICES, DEFAULT_DEEPGRAM_TTS_VOICE } = await import("./voice/tts-deepgram");
			const { resolveDeepgramApiKey } = await import("./voice/deepgram");

			const isLocal = (config.ttsBackend ?? "local") === "local";
			const lines: string[] = ["TTS configuration:", ""];
			lines.push(`  Enabled:      ${config.ttsEnabled ? "yes" : "no"}`);
			lines.push(`  Backend:      ${isLocal ? "local (sherpa-onnx)" : "deepgram (cloud REST)"}`);
			lines.push(`  Language:     ${config.ttsLanguage ?? config.language ?? "en"}`);
			lines.push(`  Speed:        ${(config.ttsSpeed ?? 1.0).toFixed(2)}x`);
			lines.push(`  Auto-speak:   ${config.ttsAutoSpeak ? "yes" : "no"}`);
			lines.push("");

			if (isLocal) {
				const modelId = config.ttsLocalModel ?? "kitten-nano-en-v0_2";
				let model;
				try { model = getTtsModel(modelId); } catch { model = undefined; }
				const installed = isTtsModelInstalled(modelId);
				lines.push("  Local backend:");
				lines.push(`    Model:      ${modelId}${model ? ` (${model.name}, ${model.size})` : " — unknown id"}`);
				lines.push(`    Installed:  ${installed ? "yes" : "NO — first speak will download"}`);
				if (model) {
					const sid = typeof config.ttsLocalVoiceId === "number" ? config.ttsLocalVoiceId : model.defaultSid;
					const voice = model.voices.find(v => v.sid === sid);
					lines.push(`    Voice sid:  ${sid}${voice ? ` (${voice.name})` : ""}`);
					lines.push(`    Languages:  ${model.languages.join(", ")}`);
					lines.push(`    Sample rate: ${model.sampleRate} Hz`);
				}
				lines.push("");
				lines.push(`  Catalog:      ${TTS_LOCAL_MODELS.length} models available — /voice-speak-models to browse`);
			} else {
				const voiceId = config.ttsDeepgramVoiceId ?? DEFAULT_DEEPGRAM_TTS_VOICE;
				const voice = DEEPGRAM_TTS_VOICES.find(v => v.id === voiceId);
				const apiKey = resolveDeepgramApiKey(config);
				lines.push("  Deepgram backend:");
				lines.push(`    Voice:      ${voiceId}${voice ? ` (${voice.name})` : ""}`);
				lines.push(`    API key:    ${apiKey ? `set (${apiKey.slice(0, 8)}…)` : "NOT SET — set DEEPGRAM_API_KEY"}`);
				lines.push(`    Catalog:    ${DEEPGRAM_TTS_VOICES.length} Aura voices surfaced`);
			}

			lines.push("");
			lines.push("  Commands: /voice-speak <text>  ·  /voice-speak-stop  ·  /voice-speak-toggle");
			lines.push("            /voice-speak-test    ·  /voice-speak-models  ·  /voice-settings");
			cmdCtx.ui.notify(lines.join("\n"), "info");
		},
	});

	pi.registerCommand("voice-speak-models", {
		description: "Browse and install TTS models (opens settings panel on Speak tab)",
		// initialTab=3 → Speak tab (index 0=General, 1=Models, 2=Downloaded, 3=Speak, 4=Device)
		handler: async (_args, cmdCtx) => openSettingsPanel(cmdCtx, 3),
	});

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

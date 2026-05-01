/**
 * TTS audio playback — write WAV to a temp file, spawn a platform player,
 * abort cleanly on signal. v6.0 ships file-based playback for simplicity;
 * stdin-streaming for sub-200ms TTFB is a v6.1 optimization.
 *
 * Both backends produce a complete WAV blob:
 *   - Local engine returns Float32Array PCM → encoded to WAV here
 *   - Deepgram REST returns WAV bytes directly (container=wav)
 *
 * Concurrency contract: each play() call owns its own temp file. Two
 * concurrent calls write to distinct UUID-named files and spawn distinct
 * player processes. The caller is responsible for serializing if it
 * doesn't want overlapping audio (the speak orchestrator does this).
 *
 * Security model:
 *   - Player invoked via `child_process.spawn(cmd, [args])` — argument
 *     array, no shell, no string interpolation. Cannot be hijacked by a
 *     malicious TMPDIR with shell metacharacters.
 *   - Windows uses an env-var indirection ($env:PI_SPEAK_PATH) so paths
 *     containing single quotes (e.g. C:\Users\O'Neil\...) cannot inject
 *     into the PowerShell command string.
 *   - Temp filenames are randomUUID — no user input in the name.
 *   - Files are written 0600 and asserted to live under os.tmpdir().
 *   - Cleanup uses a single-ownership token: the playback Promise's
 *     `finally` block is the ONLY code path that unlinks. Abort kills
 *     the player but leaves cleanup to that finally.
 */

import { spawn, spawnSync, type ChildProcess } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { randomUUID } from "node:crypto";

// ─── Types ────────────────────────────────────────────────────────────────────

/**
 * Audio source for playback. Either:
 *   - { wav: Uint8Array }                     — pre-encoded WAV bytes
 *   - { samples: Float32Array; sampleRate }   — raw float PCM, encoded here
 */
export type PlaybackSource =
	| { wav: Uint8Array }
	| { samples: Float32Array; sampleRate: number };

export interface PlayOpts {
	source: PlaybackSource;
	signal?: AbortSignal;
	/**
	 * Override the player command for testing. Production callers leave
	 * this unset so we pick by `process.platform`.
	 */
	playerOverride?: PlayerSpec;
}

// ─── Public API ───────────────────────────────────────────────────────────────

/**
 * Play `source` to the user's default audio output and resolve when
 * playback finishes. Aborts cleanly via `opts.signal`.
 *
 * Resolves with `void` on successful completion. Rejects with:
 *   - DOMException("AbortError") if signal fires
 *   - Error("No audio player found...") if platform player can't be spawned
 *   - Error("Audio player exited with code N") on non-zero exit
 */
export async function play(opts: PlayOpts): Promise<void> {
	const { source, signal } = opts;

	if (signal?.aborted) {
		throw makeAbortError();
	}

	const wav = "wav" in source
		? source.wav
		: encodeWav(source.samples, source.sampleRate);

	const tmpFile = createTempWavPath();
	let cleanupDone = false;
	const cleanup = () => {
		if (cleanupDone) return;
		cleanupDone = true;
		try { fs.unlinkSync(tmpFile); } catch { /* may already be gone */ }
	};

	// Single-ownership unlink: the `finally` below is the ONLY code path
	// that removes the temp file. Abort kills the player via Node's
	// native `signal` option on spawn(); the player exit triggers the
	// same finally. No double-delete possible.

	try {
		// Write the WAV with 0600 perms so other users on a multi-user box
		// cannot read TTS output (transcripts can be sensitive even though
		// they're agent-generated).
		fs.writeFileSync(tmpFile, wav, { mode: 0o600 });

		// Re-check abort after the sync write — if user hit Escape during
		// the write, no point spawning the player.
		if (signal?.aborted) throw makeAbortError();

		const player = opts.playerOverride ?? choosePlayer();
		const env = player.env ? { ...process.env, ...player.env(tmpFile) } : process.env;

		const proc: ChildProcess = spawn(player.cmd, player.args(tmpFile), {
			stdio: ["ignore", "ignore", "pipe"], // capture stderr for error messages
			env,
			// Node's native abort plumbing — when `signal` aborts, Node
			// kills the child process atomically. Single source of kills,
			// no race window between "abort fires" and "we look up proc to
			// kill" (which a hand-rolled addEventListener would have).
			...(signal ? { signal } : {}),
		});

		await new Promise<void>((resolve, reject) => {
			// Node can emit BOTH "error" (with AbortError) and "close" for
			// the same termination — the order is racy. `settled` ensures
			// exactly one settlement reaches the await.
			let settled = false;
			const settle = (action: () => void) => {
				if (settled) return;
				settled = true;
				action();
			};

			let stderr = "";
			const STDERR_CAP = 2048;
			proc.stderr?.on("data", (d: Buffer) => {
				// Cap BEFORE appending so a single multi-MB chunk can't
				// blow past the budget. Truncate the chunk to the
				// remaining headroom; once full, drop further chunks.
				if (stderr.length >= STDERR_CAP) return;
				const headroom = STDERR_CAP - stderr.length;
				const text = d.toString();
				stderr += text.length > headroom ? text.slice(0, headroom) : text;
			});
			proc.on("error", (err: NodeJS.ErrnoException) => {
				settle(() => {
					// Node fires "error" with AbortError when the native
					// signal aborts; also when spawn() itself fails (ENOENT
					// etc.). Distinguish by err.name.
					if (err.name === "AbortError" || signal?.aborted) {
						reject(makeAbortError());
					} else {
						reject(new Error(`Audio player ${player.cmd} failed to start: ${err.message}`));
					}
				});
			});
			proc.on("close", (code, sig) => {
				settle(() => {
					// Order matters: a clean exit (code === 0) ALWAYS wins,
					// even if the abort signal fired in the microtask gap
					// between the player finishing and the close handler
					// running. The reverse — surfacing AbortError on a
					// successfully-played audio — would be wrong UX.
					if (code === 0) {
						resolve();
					} else if (signal?.aborted) {
						reject(makeAbortError());
					} else if (sig) {
						reject(new Error(`Audio player ${player.cmd} terminated by ${sig}`));
					} else {
						const tail = stderr.trim().slice(-200);
						reject(new Error(
							`Audio player ${player.cmd} exited with code ${code}` +
							(tail ? ` (${tail})` : ""),
						));
					}
				});
			});
		});
	} finally {
		cleanup();
	}
}

// ─── v7.1.3 streaming playback ─────────────────────────────────────────────────

/**
 * Streaming playback sink. Synthesis writes PCM as it produces samples;
 * the sink pipes them to a long-lived audio process (`sox` / `paplay`)
 * so audio starts playing the moment the first chunk arrives. Drops
 * file-write/open/start latency vs the file-based `play()` path.
 *
 * Supported writes: `Int16Array` mono samples at the configured
 * `sampleRate`. Float32 inputs must be int16-converted by the caller —
 * keeps this layer narrow and avoids per-write float→int conversions
 * if the source already has int16 (e.g. Deepgram WS TTS).
 *
 * Lifecycle: `end()` flushes any buffered writes, then sends EOF to the
 * player and resolves `done()` when playback finishes. `cancel()`
 * kills the player immediately. Both are idempotent.
 *
 * Backpressure: `writePcm` returns a Promise that resolves once the
 * data is accepted by Node's writable stream (immediately if the
 * write returns true, or on `'drain'` if it returns false). Callers
 * MUST await it to avoid unbounded memory growth and the "many
 * once('drain')" race where multiple listeners fire on a single
 * drain event before all writes have actually been buffered.
 */
export interface PlaybackStream {
	writePcm(int16: Int16Array): Promise<void>;
	end(): Promise<void>;
	cancel(): void;
	done(): Promise<void>;
}

export interface OpenPlaybackStreamOpts {
	sampleRate: number;
	signal?: AbortSignal;
}

/**
 * Open a streaming playback sink. Returns `null` when no streaming-capable
 * player is found on PATH — the caller should fall back to the file-based
 * `play()` path. Player priority: `sox` (preferred — works on macOS via
 * homebrew, Linux via apt/yum, ships with most pi-listen STT installs) →
 * `paplay` (Linux PulseAudio) → null.
 *
 * Windows is intentionally unsupported here — PowerShell SoundPlayer
 * can't accept piped PCM. Windows users get the file-based fallback.
 */
export function openPlaybackStream(opts: OpenPlaybackStreamOpts): PlaybackStream | null {
	const { sampleRate, signal } = opts;
	if (signal?.aborted) return null;
	if (process.platform === "win32") return null;

	const player = pickStreamingPlayer(sampleRate);
	if (!player) return null;

	let cancelled = false;
	let ended = false;

	const proc: ChildProcess = spawn(player.cmd, player.args, {
		stdio: ["pipe", "ignore", "pipe"],
		...(signal ? { signal } : {}),
	});

	// v7.1.3 diagnostic: append every byte-count + lifecycle event to a
	// stable log path so production truncation issues can be diagnosed
	// without re-running with PI_VOICE_DEBUG. Best-effort — silently
	// drops if /tmp isn't writable.
	const diagLog = (s: string) => {
		try {
			const fs2 = require("node:fs") as typeof import("node:fs");
			fs2.appendFileSync("/tmp/pi-listen-stream.log", `[${new Date().toISOString()}] ${s}\n`);
		} catch { /* best-effort */ }
	};
	diagLog(`opened ${player.cmd} sampleRate=${sampleRate} pid=${proc.pid}`);
	let totalBytesAccepted = 0;
	let totalWrites = 0;

	let stderr = "";
	const STDERR_CAP = 2048;
	proc.stderr?.on("data", (d: Buffer) => {
		if (stderr.length >= STDERR_CAP) return;
		const headroom = STDERR_CAP - stderr.length;
		const text = d.toString();
		stderr += text.length > headroom ? text.slice(0, headroom) : text;
	});

	// Defensive: attach an internal handler to the donePromise so a
	// cancel-then-not-await flow (caller throws before reaching
	// `await stream.done()`) doesn't surface as an UnhandledPromiseRejection.
	// The caller-facing `done()` returns the same promise; awaiters still
	// see the rejection. (godspeed runtime/gemini-3.1-pro finding.)
	const donePromise = attachSafetyCatch(new Promise<void>((resolve, reject) => {
		let settled = false;
		const settle = (action: () => void) => {
			if (settled) return;
			settled = true;
			action();
		};
		proc.on("error", (err: NodeJS.ErrnoException) => {
			settle(() => {
				if (err.name === "AbortError" || signal?.aborted || cancelled) {
					reject(makeAbortError());
				} else {
					reject(new Error(`Streaming player ${player.cmd} failed: ${err.message}`));
				}
			});
		});
		proc.on("close", (code, sig) => {
			diagLog(`proc.close code=${code} sig=${sig} cancelled=${cancelled} aborted=${signal?.aborted}`);
			settle(() => {
				if (cancelled || signal?.aborted) {
					reject(makeAbortError());
				} else if (code === 0) {
					resolve();
				} else if (sig) {
					reject(new Error(`Streaming player ${player.cmd} terminated by ${sig}`));
				} else {
					const tail = stderr.trim().slice(-200);
					reject(new Error(`Streaming player ${player.cmd} exited with code ${code}${tail ? ` (${tail})` : ""}`));
				}
			});
		});
		// EPIPE if player exits before we finish writing — common when the
		// user aborts; settled by 'close' above.
		proc.stdin?.on("error", () => {});
	}));

	// Serialize writes via a chained promise. Each writePcm awaits the
	// previous write's drain (if backpressured) before issuing the next
	// stdin.write(). This is the only correct backpressure pattern for
	// `once('drain')` — multiple listeners on the same event all fire
	// simultaneously on the first drain, so Promise.all on independent
	// drain promises resolves prematurely.
	let writeTail: Promise<void> = Promise.resolve();

	// v7.1.3 — chunked write to avoid sox-with-large-stdin-burst bug.
	// When we pour a multi-MB Float32→Int16 PCM blob into sox stdin in
	// one go, sox/CoreAudio appears to underrun mid-playback (consuming
	// only the first ~64KB OS pipe buffer, exiting cleanly with code 0
	// at ~1.5s in regardless of how much we queued). Splitting into
	// CHUNK_BYTES-sized writes keeps sox fed in real time without
	// overflowing — one chunk of audio per ~250ms of playback at
	// 24kHz mono int16.
	const CHUNK_BYTES = 12_000;     // ~250ms @ 24kHz, ~270ms @ 22kHz

	const writeOne = async (view: Uint8Array): Promise<void> => {
		if (!proc.stdin || proc.stdin.destroyed) {
			diagLog(`writeOne: stdin destroyed, dropping ${view.byteLength} bytes`);
			return;
		}
		totalWrites++;
		totalBytesAccepted += view.byteLength;
		diagLog(`writeOne[${totalWrites}]: ${view.byteLength} bytes (total ${totalBytesAccepted})`);

		// Slice-and-write loop with backpressure awareness.
		for (let off = 0; off < view.byteLength; off += CHUNK_BYTES) {
			if (!proc.stdin || proc.stdin.destroyed) {
				diagLog(`writeOne: stdin destroyed mid-chunk at offset ${off}`);
				return;
			}
			const slice = view.subarray(off, Math.min(off + CHUNK_BYTES, view.byteLength));
			const ok = proc.stdin.write(slice);
			if (!ok) {
				await new Promise<void>((res) => {
					const stdin = proc.stdin!;
					const cleanup = () => {
						stdin.off("drain", onDrain);
						stdin.off("close", onClose);
						stdin.off("error", onError);
					};
					const onDrain = () => { cleanup(); res(); };
					const onClose = () => { cleanup(); res(); };
					const onError = () => { cleanup(); res(); };
					stdin.once("drain", onDrain);
					stdin.once("close", onClose);
					stdin.once("error", onError);
				});
			}
		}
	};

	return {
		writePcm(int16: Int16Array): Promise<void> {
			if (cancelled || ended) return Promise.resolve();
			const view = new Uint8Array(int16.buffer, int16.byteOffset, int16.byteLength);
			// Chain: caller can either await this OR fire-and-forget; if
			// they fire-and-forget, end() awaits the same tail.
			writeTail = writeTail.then(() => writeOne(view)).catch(() => { /* EPIPE ok */ });
			return writeTail;
		},
		async end(): Promise<void> {
			if (ended) return;
			ended = true;
			diagLog(`end() called — awaiting ${totalWrites} writes (${totalBytesAccepted} bytes)`);
			// Drain all pending writes before signaling EOF.
			try { await writeTail; } catch { /* swallowed */ }
			// Sox closes the audio device immediately on EOF, dropping
			// any audio still in the OS hardware buffer (~1-2s on macOS
			// CoreAudio). Append a tail of silence so the trailing real
			// audio is fully flushed before sox tears down the device.
			// 1.0 sec at sampleRate samples = sampleRate * 2 bytes.
			const SILENCE_TAIL_SECS = 1;
			const silence = new Int16Array(sampleRate * SILENCE_TAIL_SECS);
			try {
				await writeOne(new Uint8Array(silence.buffer, silence.byteOffset, silence.byteLength));
			} catch { /* EPIPE ok */ }
			diagLog(`end() — silence tail written, calling stdin.end()`);
			try { proc.stdin?.end(); } catch { /* already closed */ }
		},
		cancel(): void {
			if (cancelled) return;
			cancelled = true;
			diagLog(`cancel() called after ${totalWrites} writes (${totalBytesAccepted} bytes)`);
			try { proc.stdin?.destroy(); } catch {}
			try { proc.kill("SIGTERM"); } catch {}
		},
		done(): Promise<void> {
			return donePromise;
		},
	};
}

/** Internal: prevent unhandled rejection when the caller never awaits done(). */
function attachSafetyCatch<T>(p: Promise<T>): Promise<T> {
	p.catch(() => { /* swallow — real awaiters see the rejection */ });
	return p;
}

interface StreamingPlayerSpec { cmd: string; args: string[]; }

function pickStreamingPlayer(sampleRate: number): StreamingPlayerSpec | null {
	// v7.1.3 — ffplay is the most-reliable streaming PCM consumer on
	// macOS: it's designed for real-time piped audio and doesn't suffer
	// the sox-with-CoreAudio underrun where sox exits cleanly after
	// playing only the first ~1.5s of a multi-MB stdin write. Prefer
	// ffplay when present; fall back to paplay (Linux) or sox.
	if (binaryAvailable("ffplay")) {
		return {
			cmd: "ffplay",
			args: [
				"-nodisp",            // no video window
				"-autoexit",          // exit when input EOFs
				"-loglevel", "quiet",
				"-f", "s16le",
				"-ar", String(sampleRate),
				"-ch_layout", "mono", // ffmpeg 8+ uses ch_layout instead of -ac
				"-i", "pipe:0",
			],
		};
	}
	// paplay (Linux PulseAudio / PipeWire-pulse): pipe PCM via stdin.
	if (process.platform === "linux" && binaryAvailable("paplay")) {
		return {
			cmd: "paplay",
			args: [
				"--raw",
				`--rate=${sampleRate}`,
				"--format=s16le",
				"--channels=1",
				"--client-name=pi-listen",
			],
		};
	}
	// sox last-resort: cross-platform but has the macOS CoreAudio
	// underrun issue noted above. Used when ffplay/paplay missing.
	if (binaryAvailable("sox")) {
		return {
			cmd: "sox",
			args: [
				"-t", "raw",
				"-r", String(sampleRate),
				"-e", "signed-integer",
				"-b", "16",
				"-c", "1",
				"-q",
				"-",
				"-d",
			],
		};
	}
	return null;
}

const _binaryCache = new Map<string, boolean>();
function binaryAvailable(cmd: string): boolean {
	const cached = _binaryCache.get(cmd);
	if (cached !== undefined) return cached;
	try {
		const r = spawnSync(cmd, ["--version"], { stdio: "ignore" });
		const ok = r.status === 0 || r.status === 1; // some tools return 1 for --version
		_binaryCache.set(cmd, ok);
		return ok;
	} catch {
		_binaryCache.set(cmd, false);
		return false;
	}
}

/** Helper — convert Float32 [-1, 1] PCM to Int16 with NaN guard + clamp. */
export function float32ToInt16(samples: Float32Array): Int16Array {
	const out = new Int16Array(samples.length);
	for (let i = 0; i < samples.length; i++) {
		const raw = samples[i]!;
		const finite = Number.isFinite(raw) ? raw : 0;
		const s = Math.max(-1, Math.min(1, finite));
		out[i] = s < 0 ? Math.round(s * 0x8000) : Math.round(s * 0x7fff);
	}
	return out;
}

// ─── Player selection ─────────────────────────────────────────────────────────

interface PlayerSpec {
	cmd: string;
	args: (path: string) => string[];
	/**
	 * Optional environment variables. The Windows player uses this to pass
	 * the path via $env:PI_SPEAK_PATH instead of substituting it into the
	 * PowerShell command string — defeats injection via paths containing `'`.
	 */
	env?: (path: string) => NodeJS.ProcessEnv;
}

/**
 * Choose a platform-appropriate player. Throws with an actionable message
 * if no player is recognized — the message guides the user to install
 * something compatible.
 *
 * Linux prefers paplay (PulseAudio / PipeWire compat) but falls back to
 * aplay (raw ALSA). The fallback is decided at spawn time, not here, so
 * we pick paplay first and let the caller observe spawn failure to retry
 * with aplay. See the inline comment on linuxPlayer below.
 */
function choosePlayer(): PlayerSpec {
	switch (process.platform) {
		case "darwin":
			return {
				cmd: "afplay",
				args: (p) => [p],
			};
		case "linux":
			return linuxPlayer();
		case "win32":
			// PowerShell SoundPlayer reads from $env:PI_SPEAK_PATH so the
			// path is never interpolated into the command string. Defends
			// against any path that contains single quotes or other
			// PowerShell metacharacters.
			return {
				cmd: "powershell",
				args: () => [
					"-NoProfile",
					"-Command",
					"$p = $env:PI_SPEAK_PATH; (New-Object Media.SoundPlayer $p).PlaySync()",
				],
				env: (p) => ({ PI_SPEAK_PATH: p }),
			};
		default:
			throw new Error(
				`No audio player configured for platform: ${process.platform}. ` +
				`Supported: darwin, linux, win32.`,
			);
	}
}

/**
 * Linux player selection. We default to paplay (PulseAudio / PipeWire
 * compat shim) because almost all modern desktop distros run pulse or
 * pipewire-pulse. If paplay isn't installed, callers will get
 * `spawn paplay ENOENT`; the caller (speak orchestrator) can detect that
 * and surface "install paplay or aplay" — we don't probe here because
 * `which paplay` would add an extra spawn per playback.
 *
 * Users who only have aplay can override via the (future) settings-panel
 * "audio player" option. v6.0 does not surface that knob; v6.1 adds it
 * if field reports show it's needed.
 */
function linuxPlayer(): PlayerSpec {
	return {
		cmd: "paplay",
		args: (p) => [p],
	};
}

// ─── Temp file ────────────────────────────────────────────────────────────────

function createTempWavPath(): string {
	const tmpdir = os.tmpdir();
	const file = path.join(tmpdir, `pi-speak-${randomUUID()}.wav`);
	// Defense in depth: assert the file lives under tmpdir, in case
	// path.join somehow ate a `..` (it shouldn't, but the assertion is
	// nearly free and pins down the invariant).
	const rel = path.relative(tmpdir, file);
	if (rel.startsWith("..") || path.isAbsolute(rel)) {
		throw new Error(`Refusing to write outside tmpdir: ${file}`);
	}
	return file;
}

// ─── WAV encoding ─────────────────────────────────────────────────────────────

/**
 * Encode Float32 PCM samples in [-1, 1] as a mono 16-bit signed-LE WAV.
 * Standard 44-byte RIFF header followed by sample data.
 *
 * No external deps so this works in the smoke test sandbox where
 * sherpa.writeWave isn't loaded. Float-to-int16 clamps to [-32768, 32767]
 * to handle out-of-range values from the engine without wrap-around
 * artifacts.
 */
export function encodeWav(samples: Float32Array, sampleRate: number): Uint8Array {
	if (!Number.isFinite(sampleRate) || sampleRate <= 0) {
		throw new Error(`Invalid sample rate: ${sampleRate}`);
	}
	const numSamples = samples.length;
	// WAV header chunk-size fields are uint32 — `36 + dataLen` must fit.
	// 4 GiB total is the spec maximum; we cap at ~2 GiB of PCM data
	// (1,073,741,800 bytes) which is roughly 6 hours at 24 kHz mono.
	// Anything longer is almost certainly a programmer error in chunking
	// upstream — surface it loudly rather than emitting a corrupt header.
	const MAX_DATA_BYTES = 0xFFFFFFFF - 36;
	if (numSamples > MAX_DATA_BYTES / 2) {
		throw new Error(
			`encodeWav: ${numSamples} samples exceeds WAV uint32 limit. ` +
			`Chunk the input upstream (e.g. via Intl.Segmenter sentence chunking).`,
		);
	}
	const byteRate = sampleRate * 2; // mono * 16-bit (channels * bytesPerSample)
	const dataLen = numSamples * 2;
	const buf = new ArrayBuffer(44 + dataLen);
	const view = new DataView(buf);

	// "RIFF" chunk descriptor
	writeAscii(view, 0, "RIFF");
	view.setUint32(4, 36 + dataLen, true); // chunk size = file size - 8
	writeAscii(view, 8, "WAVE");

	// "fmt " sub-chunk
	writeAscii(view, 12, "fmt ");
	view.setUint32(16, 16, true);          // PCM fmt chunk size
	view.setUint16(20, 1, true);           // format = 1 (PCM)
	view.setUint16(22, 1, true);           // channels = 1 (mono)
	view.setUint32(24, sampleRate, true);
	view.setUint32(28, byteRate, true);
	view.setUint16(32, 2, true);           // block align = channels * bytes-per-sample
	view.setUint16(34, 16, true);          // bits per sample

	// "data" sub-chunk
	writeAscii(view, 36, "data");
	view.setUint32(40, dataLen, true);

	// PCM samples — replace non-finite values with 0 (silence) instead of
	// letting Math.max/min coerce NaN to -1. A NaN sample slipping through
	// would otherwise produce a single-sample DC offset spike on output.
	// 0 is the correct silent-sample value for signed PCM.
	let offset = 44;
	for (let i = 0; i < numSamples; i++) {
		const raw = samples[i]!;
		const finite = Number.isFinite(raw) ? raw : 0;
		const s = Math.max(-1, Math.min(1, finite));
		const i16 = s < 0 ? Math.round(s * 0x8000) : Math.round(s * 0x7fff);
		view.setInt16(offset, i16, true);
		offset += 2;
	}

	return new Uint8Array(buf);
}

function writeAscii(view: DataView, offset: number, str: string): void {
	for (let i = 0; i < str.length; i++) {
		view.setUint8(offset + i, str.charCodeAt(i));
	}
}

// ─── Errors ───────────────────────────────────────────────────────────────────

function makeAbortError(): Error {
	if (typeof DOMException === "function") {
		return new DOMException("Audio playback aborted", "AbortError");
	}
	const e = new Error("Audio playback aborted");
	(e as any).name = "AbortError";
	return e;
}

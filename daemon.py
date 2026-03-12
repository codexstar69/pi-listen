#!/usr/bin/env python3
"""
pi-voice STT daemon — keeps model warm in memory for zero cold-start transcription.

Protocol: Unix domain socket, newline-delimited JSON.

Commands (send JSON, receive JSON response):
  {"cmd": "transcribe", "audio": "/path/to/file.wav"}
  {"cmd": "transcribe", "audio": "/path/to/file.wav", "backend": "faster-whisper", "model": "small", "language": "en"}
  {"cmd": "status"}
  {"cmd": "load", "backend": "faster-whisper", "model": "small"}
  {"cmd": "shutdown"}
  {"cmd": "ping"}

The daemon auto-exits after 5 minutes of inactivity.
"""
import sys
import os
import json
import time
import socket
import signal
import argparse
import threading
import tempfile
import traceback

# Add our own directory to path so we can import transcribe module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transcribe import BACKENDS, detect_backend


# ─── Config ──────────────────────────────────────────────────────────────────

DEFAULT_SOCKET = os.path.join(tempfile.gettempdir(), "pi-voice-daemon.sock")
IDLE_TIMEOUT = 300  # 5 minutes
MAX_MSG_SIZE = 1024 * 1024  # 1MB


# ─── Warm Model Cache ────────────────────────────────────────────────────────

class ModelCache:
    """Keeps a single STT model loaded in memory."""

    def __init__(self):
        self.backend_name: str | None = None
        self.model_name: str | None = None
        self._model = None
        self._lock = threading.Lock()

    def load(self, backend: str, model: str) -> dict:
        with self._lock:
            if self.backend_name == backend and self.model_name == model and self._model is not None:
                return {"status": "already_loaded", "backend": backend, "model": model}

            # Unload previous
            self._model = None
            self.backend_name = backend
            self.model_name = model

            start = time.time()
            try:
                if backend == "faster-whisper":
                    from faster_whisper import WhisperModel
                    self._model = WhisperModel(model, device="cpu", compute_type="int8")
                elif backend == "parakeet":
                    import nemo.collections.asr as nemo_asr
                    self._model = nemo_asr.models.ASRModel.from_pretrained(model)
                else:
                    # moonshine, whisper-cpp, deepgram don't have persistent model objects
                    # We still mark them as "loaded" so the daemon knows what to use
                    self._model = "external"

                elapsed = round(time.time() - start, 2)
                return {"status": "loaded", "backend": backend, "model": model, "load_time": elapsed}
            except Exception as e:
                self._model = None
                return {"status": "error", "error": str(e)}

    def transcribe(self, audio_path: str, language: str = "en") -> dict:
        with self._lock:
            if not self.backend_name:
                return {"error": "No model loaded. Send 'load' first."}

            backend = self.backend_name
            model = self.model_name
            start = time.time()

            try:
                if backend == "faster-whisper" and self._model and self._model != "external":
                    segments, info = self._model.transcribe(
                        audio_path,
                        language=language if language != "auto" else None,
                        beam_size=1,
                        vad_filter=True,
                    )
                    text = " ".join(seg.text.strip() for seg in segments)
                    return {
                        "text": text,
                        "duration": round(time.time() - start, 2),
                        "backend": backend,
                        "model": model,
                        "language": getattr(info, 'language', language),
                    }

                elif backend == "parakeet" and self._model and self._model != "external":
                    result = self._model.transcribe([audio_path])[0]
                    if isinstance(result, list):
                        result = result[0]
                    return {
                        "text": str(result),
                        "duration": round(time.time() - start, 2),
                        "backend": backend,
                        "model": model,
                    }

                else:
                    # For backends without warm models, delegate to their transcribe fn
                    info = BACKENDS.get(backend)
                    if not info:
                        return {"error": f"Unknown backend: {backend}"}
                    result = info["fn"](audio_path, model, language)
                    return result

            except Exception as e:
                return {"error": str(e), "backend": backend, "model": model, "traceback": traceback.format_exc()}


# ─── VAD (Voice Activity Detection) ─────────────────────────────────────────

_vad_model = None

def get_vad_model():
    """Lazy-load Silero VAD model."""
    global _vad_model
    if _vad_model is None:
        try:
            import torch
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=True,
            )
            _vad_model = (model, utils)
        except Exception:
            _vad_model = False  # Mark as unavailable
    return _vad_model if _vad_model else None

def run_vad(audio_path: str) -> dict:
    """Run VAD on audio file, return speech segments and whether speech was detected."""
    vad = get_vad_model()
    if not vad:
        return {"has_speech": True, "vad_available": False}  # Assume speech if no VAD

    try:
        model, utils = vad
        (get_speech_timestamps, _, read_audio, _, _) = utils
        wav = read_audio(audio_path, sampling_rate=16000)
        timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)
        has_speech = len(timestamps) > 0
        return {
            "has_speech": has_speech,
            "vad_available": True,
            "segments": len(timestamps),
            "speech_duration_ms": sum(t['end'] - t['start'] for t in timestamps) * 1000 // 16000 if timestamps else 0,
        }
    except Exception as e:
        return {"has_speech": True, "vad_available": False, "error": str(e)}


# ─── Server ──────────────────────────────────────────────────────────────────

class DaemonServer:
    def __init__(self, socket_path: str, backend: str | None = None, model: str | None = None):
        self.socket_path = socket_path
        self.cache = ModelCache()
        self.running = False
        self.last_activity = time.time()
        self.server_socket: socket.socket | None = None
        self._idle_thread: threading.Thread | None = None
        self.start_time = time.time()
        self.request_count = 0

        # Auto-load model if specified
        if backend and model:
            result = self.cache.load(backend, model)
            log(f"Pre-loaded model: {json.dumps(result)}")

    def handle_client(self, conn: socket.socket):
        """Handle a single client connection (one request-response per line)."""
        self.last_activity = time.time()
        buf = b""

        try:
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buf += chunk

                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line.strip():
                        continue

                    try:
                        req = json.loads(line.decode("utf-8"))
                    except json.JSONDecodeError as e:
                        response = {"error": f"Invalid JSON: {e}"}
                        conn.sendall((json.dumps(response) + "\n").encode())
                        continue

                    response = self.dispatch(req)
                    self.request_count += 1
                    self.last_activity = time.time()
                    conn.sendall((json.dumps(response) + "\n").encode())

                    # Shutdown command
                    if req.get("cmd") == "shutdown":
                        self.running = False
                        return

        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            conn.close()

    def dispatch(self, req: dict) -> dict:
        cmd = req.get("cmd", "")

        if cmd == "ping":
            return {"status": "ok", "pid": os.getpid()}

        elif cmd == "status":
            return {
                "status": "running",
                "pid": os.getpid(),
                "uptime": round(time.time() - self.start_time, 1),
                "requests": self.request_count,
                "idle": round(time.time() - self.last_activity, 1),
                "backend": self.cache.backend_name,
                "model": self.cache.model_name,
                "model_loaded": self.cache._model is not None,
            }

        elif cmd == "load":
            backend = req.get("backend") or detect_backend()
            model = req.get("model") or BACKENDS.get(backend, {}).get("default_model", "small")
            return self.cache.load(backend, model)

        elif cmd == "transcribe":
            audio = req.get("audio")
            if not audio or not os.path.exists(audio):
                return {"error": f"Audio file not found: {audio}"}

            language = req.get("language", "en")

            # Auto-load if not loaded yet
            if not self.cache.backend_name:
                backend = req.get("backend") or detect_backend()
                model = req.get("model") or BACKENDS.get(backend, {}).get("default_model", "small")
                load_result = self.cache.load(backend, model)
                if load_result.get("status") == "error":
                    return load_result

            # Optional VAD pre-check
            if req.get("vad", False):
                vad_result = run_vad(audio)
                if not vad_result.get("has_speech", True):
                    return {"text": "", "duration": 0, "vad": vad_result, "skipped": True}

            return self.cache.transcribe(audio, language)

        elif cmd == "vad":
            audio = req.get("audio")
            if not audio or not os.path.exists(audio):
                return {"error": f"Audio file not found: {audio}"}
            return run_vad(audio)

        elif cmd == "shutdown":
            return {"status": "shutting_down"}

        elif cmd == "backends":
            result = []
            for name, info in BACKENDS.items():
                available = info["available"]()
                result.append({
                    "name": name,
                    "available": available,
                    "type": info["type"],
                    "default_model": info["default_model"],
                    "models": info["models"],
                })
            return {"backends": result}

        else:
            return {"error": f"Unknown command: {cmd}"}

    def idle_watcher(self):
        """Background thread that shuts down the daemon after idle timeout."""
        while self.running:
            time.sleep(10)
            idle = time.time() - self.last_activity
            if idle > IDLE_TIMEOUT:
                log(f"Idle for {idle:.0f}s, shutting down")
                self.running = False
                # Connect to self to unblock accept()
                try:
                    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    s.connect(self.socket_path)
                    s.sendall(b'{"cmd":"shutdown"}\n')
                    s.close()
                except Exception:
                    pass
                break

    def start(self):
        # Clean up stale socket
        if os.path.exists(self.socket_path):
            try:
                # Check if another daemon is running
                test = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                test.settimeout(1)
                test.connect(self.socket_path)
                test.sendall(b'{"cmd":"ping"}\n')
                resp = test.recv(1024)
                test.close()
                if resp:
                    log(f"Another daemon is already running")
                    print(json.dumps({"error": "already_running", "socket": self.socket_path}))
                    sys.exit(1)
            except (ConnectionRefusedError, FileNotFoundError, OSError):
                os.unlink(self.socket_path)

        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.bind(self.socket_path)
        self.server_socket.listen(5)
        self.server_socket.settimeout(1.0)
        self.running = True

        # Write PID file
        pid_path = self.socket_path + ".pid"
        with open(pid_path, "w") as f:
            f.write(str(os.getpid()))

        # Start idle watcher
        self._idle_thread = threading.Thread(target=self.idle_watcher, daemon=True)
        self._idle_thread.start()

        log(f"Daemon started: pid={os.getpid()} socket={self.socket_path}")
        print(json.dumps({
            "status": "started",
            "pid": os.getpid(),
            "socket": self.socket_path,
        }), flush=True)

        try:
            while self.running:
                try:
                    conn, _ = self.server_socket.accept()
                    thread = threading.Thread(target=self.handle_client, args=(conn,), daemon=True)
                    thread.start()
                except socket.timeout:
                    continue
                except OSError:
                    break
        finally:
            self.cleanup()

    def cleanup(self):
        log("Daemon shutting down")
        self.running = False
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception:
                pass
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except Exception:
                pass
        pid_path = self.socket_path + ".pid"
        if os.path.exists(pid_path):
            try:
                os.unlink(pid_path)
            except Exception:
                pass


def log(msg: str):
    print(f"[pi-voice-daemon] {msg}", file=sys.stderr, flush=True)


# ─── Client helper (for testing / CLI) ───────────────────────────────────────

def send_command(socket_path: str, cmd: dict, timeout: float = 30) -> dict:
    """Send a command to the daemon and return the response."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(socket_path)
        sock.sendall((json.dumps(cmd) + "\n").encode())
        buf = b""
        while b"\n" not in buf:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf += chunk
        if buf:
            return json.loads(buf.decode("utf-8").strip())
        return {"error": "No response from daemon"}
    except ConnectionRefusedError:
        return {"error": "Daemon not running", "socket": socket_path}
    except FileNotFoundError:
        return {"error": "Daemon not running (socket not found)", "socket": socket_path}
    finally:
        sock.close()


def is_daemon_running(socket_path: str) -> bool:
    """Check if the daemon is running."""
    result = send_command(socket_path, {"cmd": "ping"}, timeout=2)
    return result.get("status") == "ok"


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="pi-voice STT daemon")
    sub = parser.add_subparsers(dest="action", help="Action to perform")

    # Start daemon
    start_p = sub.add_parser("start", help="Start the daemon")
    start_p.add_argument("--socket", default=DEFAULT_SOCKET, help="Unix socket path")
    start_p.add_argument("--backend", default=None, help="STT backend to pre-load")
    start_p.add_argument("--model", default=None, help="Model to pre-load")

    # Client commands
    status_p = sub.add_parser("status", help="Get daemon status")
    status_p.add_argument("--socket", default=DEFAULT_SOCKET)

    stop_p = sub.add_parser("stop", help="Stop the daemon")
    stop_p.add_argument("--socket", default=DEFAULT_SOCKET)

    ping_p = sub.add_parser("ping", help="Ping the daemon")
    ping_p.add_argument("--socket", default=DEFAULT_SOCKET)

    tx_p = sub.add_parser("transcribe", help="Transcribe audio file")
    tx_p.add_argument("audio", help="Path to audio file")
    tx_p.add_argument("--socket", default=DEFAULT_SOCKET)
    tx_p.add_argument("--language", default="en")
    tx_p.add_argument("--vad", action="store_true", help="Run VAD before transcription")

    load_p = sub.add_parser("load", help="Load a model")
    load_p.add_argument("--socket", default=DEFAULT_SOCKET)
    load_p.add_argument("--backend", default=None)
    load_p.add_argument("--model", default=None)

    args = parser.parse_args()

    if args.action == "start":
        # Handle signals
        server = DaemonServer(args.socket, args.backend, args.model)
        signal.signal(signal.SIGTERM, lambda *_: setattr(server, 'running', False))
        signal.signal(signal.SIGINT, lambda *_: setattr(server, 'running', False))
        server.start()

    elif args.action == "status":
        print(json.dumps(send_command(args.socket, {"cmd": "status"}), indent=2))

    elif args.action == "stop":
        print(json.dumps(send_command(args.socket, {"cmd": "shutdown"}), indent=2))

    elif args.action == "ping":
        result = send_command(args.socket, {"cmd": "ping"}, timeout=2)
        print(json.dumps(result, indent=2))
        sys.exit(0 if result.get("status") == "ok" else 1)

    elif args.action == "transcribe":
        result = send_command(args.socket, {
            "cmd": "transcribe",
            "audio": os.path.abspath(args.audio),
            "language": args.language,
            "vad": args.vad,
        })
        print(json.dumps(result, indent=2))

    elif args.action == "load":
        result = send_command(args.socket, {
            "cmd": "load",
            "backend": args.backend,
            "model": args.model,
        })
        print(json.dumps(result, indent=2))

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

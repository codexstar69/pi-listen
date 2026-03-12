#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

MODE="local"
BACKEND="faster-whisper"
SCOPE="global"
PROJECT_DIR="$PWD"
DEEPGRAM_KEY="${DEEPGRAM_API_KEY:-}"
SKIP_PI_INSTALL=0
PERSIST_DEEPGRAM_KEY=0

usage() {
  cat <<'EOF'
pi-listen macOS bootstrap

Usage:
  scripts/setup-macos.sh [options]

Options:
  --mode local|api              Setup mode (default: local)
  --backend NAME                Local backend: faster-whisper | moonshine | whisper-cpp | parakeet
  --scope global|project        Where to write settings and install the package (default: global)
  --project-dir PATH            Project directory used for --scope project (default: current directory)
  --deepgram-key KEY            Deepgram API key for api mode (defaults to current DEEPGRAM_API_KEY)
  --persist-deepgram-key        Append DEEPGRAM_API_KEY to ~/.zshrc when a key is available
  --skip-pi-install             Skip `pi install npm:pi-listen`
  -h, --help                    Show this help

Examples:
  scripts/setup-macos.sh --mode local --backend faster-whisper
  scripts/setup-macos.sh --mode local --backend whisper-cpp --scope project
  scripts/setup-macos.sh --mode api --deepgram-key dg_xxx --persist-deepgram-key
EOF
}

log() {
  printf '\n[%s] %s\n' "pi-listen" "$*"
}

warn() {
  printf '\n[%s] WARNING: %s\n' "pi-listen" "$*" >&2
}

have() {
  command -v "$1" >/dev/null 2>&1
}

append_if_missing() {
  local file="$1"
  local line="$2"
  mkdir -p "$(dirname "$file")"
  touch "$file"
  APPEND_FILE="$file" APPEND_LINE="$line" python3 - <<'PY'
from pathlib import Path
import os

path = Path(os.environ["APPEND_FILE"])
line = os.environ["APPEND_LINE"]
existing = path.read_text() if path.exists() else ""
if line not in existing:
    with path.open("a", encoding="utf8") as handle:
        handle.write("\n" + line + "\n")
PY
}

selected_backend() {
  if [[ "$MODE" == "api" ]]; then
    printf 'deepgram\n'
  else
    printf '%s\n' "$BACKEND"
  fi
}

selected_model() {
  case "$(selected_backend)" in
    faster-whisper) printf 'small\n' ;;
    moonshine) printf 'moonshine/base\n' ;;
    whisper-cpp) printf 'small\n' ;;
    parakeet) printf 'nvidia/parakeet-tdt-0.6b-v2\n' ;;
    deepgram) printf 'nova-3\n' ;;
    *)
      warn "No default model for backend $(selected_backend)"
      exit 1
      ;;
  esac
}

settings_path() {
  if [[ "$SCOPE" == "project" ]]; then
    printf '%s/.pi/settings.json\n' "$PROJECT_DIR"
  else
    printf '%s/.pi/agent/settings.json\n' "$HOME"
  fi
}

install_pi_package() {
  if (( SKIP_PI_INSTALL )); then
    return
  fi

  if ! have pi; then
    warn "The 'pi' command was not found, so package installation was skipped. Install Pi first, then run: pi install npm:pi-listen"
    return
  fi

  if [[ "$SCOPE" == "project" ]]; then
    log "Installing pi-listen into Pi project settings"
    (
      cd "$PROJECT_DIR"
      pi install -l npm:pi-listen
    )
  else
    log "Installing pi-listen into global Pi settings"
    pi install npm:pi-listen
  fi
}

write_voice_config() {
  local settings_file
  settings_file="$(settings_path)"
  local backend model
  backend="$(selected_backend)"
  model="$(selected_model)"

  mkdir -p "$(dirname "$settings_file")"
  SETTINGS_PATH="$settings_file" \
  VOICE_MODE="$MODE" \
  VOICE_BACKEND="$backend" \
  VOICE_MODEL="$model" \
  VOICE_SCOPE="$SCOPE" \
  python3 - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

settings_path = Path(os.environ["SETTINGS_PATH"])
if settings_path.exists():
    try:
        data = json.loads(settings_path.read_text())
    except Exception:
        data = {}
else:
    data = {}

timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
data["voice"] = {
    "version": 2,
    "enabled": True,
    "language": "en",
    "mode": os.environ["VOICE_MODE"],
    "backend": os.environ["VOICE_BACKEND"],
    "model": os.environ["VOICE_MODEL"],
    "scope": os.environ["VOICE_SCOPE"],
    "btwEnabled": True,
    "onboarding": {
        "completed": True,
        "schemaVersion": 2,
        "completedAt": timestamp,
        "lastValidatedAt": timestamp,
        "source": "setup-command"
    }
}
settings_path.write_text(json.dumps(data, indent=2) + "\n")
PY

  log "Wrote ready-to-use voice config to $settings_file"
}

smoke_test() {
  local backend model socket log_file err_file
  backend="$(selected_backend)"
  model="$(selected_model)"
  socket="$(mktemp -u /tmp/pi-listen-bootstrap-XXXXXX.sock)"
  log_file="$(mktemp /tmp/pi-listen-daemon-stdout-XXXXXX.log)"
  err_file="$(mktemp /tmp/pi-listen-daemon-stderr-XXXXXX.log)"

  log "Checking backend availability via transcribe.py"
  python3 "$PACKAGE_ROOT/transcribe.py" --list-backends | python3 -c '
import json
import sys
backend_name = sys.argv[1]
backends = json.load(sys.stdin)
selected = next((b for b in backends if b.get("name") == backend_name), None)
if not selected:
    raise SystemExit(f"Backend {backend_name} was not returned by transcribe.py --list-backends")
if not selected.get("available"):
    raise SystemExit(f"Backend {backend_name} is still not available after setup")
print(json.dumps({"backend": selected["name"], "available": selected["available"], "install_detection": selected.get("install_detection") or "unknown"}))
' "$backend"

  log "Starting daemon smoke test for backend=$backend model=$model"
  python3 "$PACKAGE_ROOT/daemon.py" start --socket "$socket" --backend "$backend" --model "$model" >"$log_file" 2>"$err_file" &
  local daemon_pid=$!

  cleanup() {
    python3 "$PACKAGE_ROOT/daemon.py" stop --socket "$socket" >/dev/null 2>&1 || true
    wait "$daemon_pid" >/dev/null 2>&1 || true
    rm -f "$socket" "$socket.pid" "$log_file" "$err_file"
  }
  trap cleanup RETURN

  local attempt
  for attempt in {1..120}; do
    if python3 "$PACKAGE_ROOT/daemon.py" ping --socket "$socket" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done

  if ! python3 "$PACKAGE_ROOT/daemon.py" ping --socket "$socket" >/dev/null 2>&1; then
    warn "Daemon failed to start cleanly"
    warn "Daemon stdout log: $log_file"
    warn "Daemon stderr log: $err_file"
    tail -n 40 "$err_file" >&2 || true
    return 1
  fi

  python3 "$PACKAGE_ROOT/daemon.py" status --socket "$socket" >/dev/null
  log "Daemon smoke test passed"
}

while (($#)); do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --backend)
      BACKEND="${2:-}"
      shift 2
      ;;
    --scope)
      SCOPE="${2:-}"
      shift 2
      ;;
    --project-dir)
      PROJECT_DIR="${2:-}"
      shift 2
      ;;
    --deepgram-key)
      DEEPGRAM_KEY="${2:-}"
      shift 2
      ;;
    --persist-deepgram-key)
      PERSIST_DEEPGRAM_KEY=1
      shift
      ;;
    --skip-pi-install)
      SKIP_PI_INSTALL=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      warn "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ "$MODE" != "local" && "$MODE" != "api" ]]; then
  warn "--mode must be local or api"
  exit 1
fi

if [[ "$SCOPE" != "global" && "$SCOPE" != "project" ]]; then
  warn "--scope must be global or project"
  exit 1
fi

PROJECT_DIR="$(cd "$PROJECT_DIR" && pwd)"

case "$BACKEND" in
  faster-whisper|moonshine|whisper-cpp|parakeet) ;;
  *)
    warn "Unsupported backend: $BACKEND"
    exit 1
    ;;
esac

if ! have brew; then
  warn "Homebrew is required for automatic macOS setup. Install it from https://brew.sh/ and rerun this script."
  exit 1
fi

if ! have python3; then
  log "Installing python3 via Homebrew"
  brew install python
fi

if ! have rec; then
  log "Installing SoX"
  brew install sox
else
  log "SoX already available"
fi

log "Ensuring pip tooling is ready"
python3 -m pip install --upgrade pip >/dev/null

if [[ "$MODE" == "local" ]]; then
  case "$BACKEND" in
    faster-whisper)
      log "Installing faster-whisper"
      python3 -m pip install faster-whisper
      ;;
    moonshine)
      log "Installing moonshine"
      python3 -m pip install 'useful-moonshine[onnx]'
      ;;
    whisper-cpp)
      log "Installing whisper-cpp"
      brew install whisper-cpp
      ;;
    parakeet)
      log "Installing parakeet"
      python3 -m pip install 'nemo_toolkit[asr]'
      ;;
  esac
else
  if [[ -z "$DEEPGRAM_KEY" ]]; then
    warn "API mode requires DEEPGRAM_API_KEY. Pass --deepgram-key or export DEEPGRAM_API_KEY first."
    exit 1
  fi
  export DEEPGRAM_API_KEY="$DEEPGRAM_KEY"
  log "Loaded DEEPGRAM_API_KEY into the current shell process"
  if (( PERSIST_DEEPGRAM_KEY )); then
    append_if_missing "$HOME/.zshrc" "export DEEPGRAM_API_KEY=$DEEPGRAM_KEY"
    log "Persisted DEEPGRAM_API_KEY to ~/.zshrc"
  fi
fi

install_pi_package
smoke_test
write_voice_config

cat <<EOF

Done.

Happy path result:
  - dependencies installed
  - selected backend validated
  - daemon smoke test passed
  - Pi voice settings written automatically

Selected configuration:
  mode: $MODE
  backend: $(selected_backend)
  model: $(selected_model)
  scope: $SCOPE

Next steps:
  1. Grant microphone permission once:
       System Settings -> Privacy & Security -> Microphone
       Allow your terminal app (Ghostty / Terminal / iTerm)

  2. Start Pi:
       $( [[ "$SCOPE" == "project" ]] && printf 'cd %q && ' "$PROJECT_DIR" )pi

  3. Try hold-to-talk in an empty editor.

You should not need to run /voice setup on the happy path.
If you want a quick verification inside Pi, run:
  /voice test
  /voice doctor
EOF

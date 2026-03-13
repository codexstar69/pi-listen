#!/usr/bin/env bash
set -eo pipefail

# When piped from curl, BASH_SOURCE is empty — detect and handle gracefully
if [[ -n "${BASH_SOURCE[0]:-}" && "${BASH_SOURCE[0]}" != "bash" ]]; then
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  PACKAGE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
else
  SCRIPT_DIR=""
  PACKAGE_ROOT=""
fi

set -u

SCOPE="global"
PROJECT_DIR="$PWD"
DEEPGRAM_KEY="${DEEPGRAM_API_KEY:-}"
SKIP_PI_INSTALL=0
PERSIST_DEEPGRAM_KEY=0

usage() {
  cat <<'EOF'
pi-listen macOS bootstrap — Deepgram streaming voice input for Pi

Usage:
  scripts/setup-macos.sh [options]

Options:
  --scope global|project        Where to write settings (default: global)
  --project-dir PATH            Project directory for --scope project (default: .)
  --deepgram-key KEY            Deepgram API key (defaults to DEEPGRAM_API_KEY env)
  --persist-deepgram-key        Append DEEPGRAM_API_KEY to ~/.zshrc
  --skip-pi-install             Skip `pi install npm:@codexstar/pi-listen`
  -h, --help                    Show this help

Prerequisites:
  - macOS with Homebrew
  - Pi CLI (`npm i -g @mariozechner/pi-coding-agent`)
  - SoX (brew install sox) — audio capture
  - Deepgram API key (free $200 credits at https://dpgr.am/pi-voice)

You should not need to run `/voice setup` on the happy path —
this script configures everything.
EOF
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --scope)          SCOPE="$2"; shift 2 ;;
    --project-dir)    PROJECT_DIR="$2"; shift 2 ;;
    --deepgram-key)   DEEPGRAM_KEY="$2"; shift 2 ;;
    --persist-deepgram-key) PERSIST_DEEPGRAM_KEY=1; shift ;;
    --skip-pi-install) SKIP_PI_INSTALL=1; shift ;;
    -h|--help)        usage; exit 0 ;;
    *)                echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

log()  { printf '  \033[1;32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[1;33m!\033[0m %s\n' "$*"; }
fail() { printf '  \033[1;31m✗\033[0m %s\n' "$*"; exit 1; }

# ── Step 1: Install SoX ──────────────────────────────────────────────────

if command -v rec &>/dev/null; then
  log "SoX already installed ($(rec --version 2>/dev/null | head -1 || echo 'unknown'))"
else
  log "Installing SoX via Homebrew…"
  brew install sox
  log "SoX installed"
fi

# ── Step 2: Install pi-listen extension ───────────────────────────────────

if [[ $SKIP_PI_INSTALL -eq 0 ]]; then
  log "Installing pi-listen extension…"
  pi install npm:@codexstar/pi-listen
  log "pi-listen installed"
else
  log "Skipping pi install (--skip-pi-install)"
fi

# ── Step 3: Deepgram API key ──────────────────────────────────────────────

if [[ -z "$DEEPGRAM_KEY" ]]; then
  warn "DEEPGRAM_API_KEY not set."
  warn "Get a free key → https://dpgr.am/pi-voice (\$200 credits, no card)"
  warn "Then: export DEEPGRAM_API_KEY=your_key_here"
  warn "Or run: scripts/setup-macos.sh --deepgram-key YOUR_KEY"
fi

if [[ $PERSIST_DEEPGRAM_KEY -eq 1 && -n "$DEEPGRAM_KEY" ]]; then
  ZSHRC="$HOME/.zshrc"
  if grep -q "DEEPGRAM_API_KEY" "$ZSHRC" 2>/dev/null; then
    sed -i '' "s|^export DEEPGRAM_API_KEY=.*|export DEEPGRAM_API_KEY=\"$DEEPGRAM_KEY\"|" "$ZSHRC"
    log "Updated DEEPGRAM_API_KEY in $ZSHRC"
  else
    echo "" >> "$ZSHRC"
    echo "export DEEPGRAM_API_KEY=\"$DEEPGRAM_KEY\"" >> "$ZSHRC"
    log "Added DEEPGRAM_API_KEY to $ZSHRC"
  fi
fi

# ── Step 4: Write voice settings ──────────────────────────────────────────

if [[ "$SCOPE" == "project" ]]; then
  SETTINGS_DIR="$PROJECT_DIR/.pi"
else
  SETTINGS_DIR="$HOME/.pi/agent"
fi

SETTINGS_FILE="$SETTINGS_DIR/settings.json"
mkdir -p "$SETTINGS_DIR"

# Merge voice key into existing settings (or create new)
if [[ -f "$SETTINGS_FILE" ]]; then
  # Use python to merge JSON safely
  python3 -c "
import json, sys
with open('$SETTINGS_FILE') as f:
    settings = json.load(f)
settings['voice'] = {
    'version': 2,
    'enabled': True,
    'language': 'en',
    'scope': '$SCOPE',
    'onboarding': {
        'completed': True,
        'schemaVersion': 2,
        'completedAt': __import__('datetime').datetime.utcnow().isoformat() + 'Z',
        'source': 'setup-script'
    }
}
if '$DEEPGRAM_KEY':
    settings['voice']['deepgramApiKey'] = '$DEEPGRAM_KEY'
with open('$SETTINGS_FILE', 'w') as f:
    json.dump(settings, f, indent=2)
    f.write('\n')
" 2>/dev/null || python3 -c "
import json
settings = {'voice': {
    'version': 2, 'enabled': True, 'language': 'en', 'scope': '$SCOPE',
    'onboarding': {'completed': True, 'schemaVersion': 2, 'source': 'setup-script'}
}}
with open('$SETTINGS_FILE', 'w') as f:
    json.dump(settings, f, indent=2)
    f.write('\n')
"
else
  python3 -c "
import json
settings = {'voice': {
    'version': 2, 'enabled': True, 'language': 'en', 'scope': '$SCOPE',
    'deepgramApiKey': '$DEEPGRAM_KEY' if '$DEEPGRAM_KEY' else None,
    'onboarding': {
        'completed': True, 'schemaVersion': 2,
        'completedAt': __import__('datetime').datetime.utcnow().isoformat() + 'Z',
        'source': 'setup-script'
    }
}}
with open('$SETTINGS_FILE', 'w') as f:
    json.dump(settings, f, indent=2)
    f.write('\n')
"
fi

log "Settings written to $SETTINGS_FILE"

# ── Step 5: Verify ───────────────────────────────────────────────────────

echo ""
echo "  Setup complete!"
echo ""
echo "  Next: open Pi and hold SPACE to talk."
if [[ -z "$DEEPGRAM_KEY" ]]; then
  echo ""
  echo "  ⚠  Don't forget: export DEEPGRAM_API_KEY=your_key"
fi
echo ""

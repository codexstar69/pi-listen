#!/usr/bin/env python3
"""
Local & cloud STT transcriber for pi-voice.

Backends (in priority order):
  faster-whisper  — CTranslate2 Whisper (pip install faster-whisper)
  moonshine       — Moonshine v2 ONNX (pip install useful-moonshine[onnx])
  whisper-cpp     — whisper.cpp CLI (brew install whisper-cpp)
  deepgram        — Deepgram Nova 3 API (DEEPGRAM_API_KEY env var)
  parakeet        — NVIDIA Parakeet v3 via NeMo (pip install nemo_toolkit[asr])

Usage:
  transcribe.py <audio_file>
  transcribe.py --backend faster-whisper --model small <audio_file>
  transcribe.py --backend deepgram <audio_file>
  transcribe.py --list-backends
  transcribe.py --list-models --backend faster-whisper

Output: JSON {"text": "...", "duration": 1.23, "backend": "...", "model": "..."}
"""
import sys
import json
import time
import argparse
import os
import shutil
from pathlib import Path


def _existing_dirs(paths: list[str]) -> list[Path]:
    result: list[Path] = []
    for p in paths:
        path = Path(os.path.expanduser(p))
        if path.exists() and path.is_dir():
            result.append(path)
    return result


def huggingface_cache_dirs() -> list[Path]:
    candidates = []
    if os.environ.get("HUGGINGFACE_HUB_CACHE"):
        candidates.append(os.environ["HUGGINGFACE_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        candidates.append(str(Path(os.environ["HF_HOME"]) / "hub"))
    if os.environ.get("TRANSFORMERS_CACHE"):
        candidates.append(os.environ["TRANSFORMERS_CACHE"])
    candidates.append("~/.cache/huggingface/hub")
    return _existing_dirs(candidates)


def repo_id_to_cache_dir(repo_id: str) -> str:
    return f"models--{repo_id.replace('/', '--')}"


def huggingface_repo_exists(repo_ids: list[str]) -> bool:
    for cache_dir in huggingface_cache_dirs():
        for repo_id in repo_ids:
            repo_dir = cache_dir / repo_id_to_cache_dir(repo_id)
            snapshots_dir = repo_dir / "snapshots"
            if snapshots_dir.exists() and any(snapshots_dir.iterdir()):
                return True
            if repo_dir.exists() and any(repo_dir.iterdir()):
                return True
    return False


def detect_installed_models(model_names: list[str], detector) -> list[str]:
    installed: list[str] = []
    for model_name in model_names:
        try:
            if detector(model_name):
                installed.append(model_name)
        except Exception:
            continue
    return installed


# ─── Backend: faster-whisper ──────────────────────────────────────────────────

FASTER_WHISPER_MODELS = [
    "tiny", "tiny.en",
    "base", "base.en",
    "small", "small.en",
    "medium", "medium.en",
    "large-v3", "large-v3-turbo",
    "distil-small.en", "distil-medium.en", "distil-large-v3",
]

def is_faster_whisper_available() -> bool:
    try:
        import faster_whisper
        return True
    except ImportError:
        return False


FASTER_WHISPER_MODEL_REPO_MAP = {
    "tiny": "Systran/faster-whisper-tiny",
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "base": "Systran/faster-whisper-base",
    "base.en": "Systran/faster-whisper-base.en",
    "small": "Systran/faster-whisper-small",
    "small.en": "Systran/faster-whisper-small.en",
    "medium": "Systran/faster-whisper-medium",
    "medium.en": "Systran/faster-whisper-medium.en",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "distil-small.en": "Systran/faster-distil-whisper-small.en",
    "distil-medium.en": "Systran/faster-distil-whisper-medium.en",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
}


def faster_whisper_repo_ids(model_name: str) -> list[str]:
    try:
        from faster_whisper.utils import _MODELS  # type: ignore
        repo_id = _MODELS.get(model_name)
        if repo_id:
            return [repo_id]
    except Exception:
        pass

    repo_id = FASTER_WHISPER_MODEL_REPO_MAP.get(model_name)
    if repo_id:
        return [repo_id]

    if model_name.startswith("distil-"):
        return [f"Systran/faster-{model_name.replace('distil-', 'distil-whisper-')}"]
    return [f"Systran/faster-whisper-{model_name}"]


def is_faster_whisper_model_installed(model_name: str) -> bool:
    return huggingface_repo_exists(faster_whisper_repo_ids(model_name))


def transcribe_faster_whisper(audio_path: str, model_name: str, language: str) -> dict:
    from faster_whisper import WhisperModel
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    start = time.time()
    segments, info = model.transcribe(
        audio_path,
        language=language if language != "auto" else None,
        beam_size=1,
        vad_filter=True,
    )
    text = " ".join(seg.text.strip() for seg in segments)
    return {
        "text": text,
        "duration": round(time.time() - start, 2),
        "backend": "faster-whisper",
        "model": model_name,
        "language": info.language,
    }


# ─── Backend: moonshine ──────────────────────────────────────────────────────

MOONSHINE_MODELS = ["moonshine/tiny", "moonshine/base"]

def is_moonshine_available() -> bool:
    try:
        import moonshine_onnx
        return True
    except ImportError:
        return shutil.which("moonshine") is not None


def moonshine_repo_ids(model_name: str) -> list[str]:
    slug = model_name.split("/", 1)[-1]
    return [
        f"UsefulSensors/moonshine-{slug}",
        f"UsefulSensors/moonshine-{slug}-onnx",
    ]


def is_moonshine_model_installed(model_name: str) -> bool:
    cache_hits = huggingface_repo_exists(moonshine_repo_ids(model_name))
    if cache_hits:
        return True
    local_candidates = [
        f"~/.cache/moonshine/{model_name.replace('/', '-')}",
        f"~/.cache/moonshine/{model_name.split('/', 1)[-1]}",
    ]
    return any(path.exists() for path in _existing_dirs(local_candidates))


def transcribe_moonshine(audio_path: str, model_name: str, language: str) -> dict:
    start = time.time()

    try:
        # Try Python API first
        from moonshine_onnx import transcribe as ms_transcribe
        import soundfile as sf
        audio, sr = sf.read(audio_path)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        tokens = ms_transcribe(audio, model=model_name)
        text = " ".join(tokens) if isinstance(tokens, list) else str(tokens)
    except ImportError:
        # Fall back to CLI
        import subprocess
        result = subprocess.run(
            ["moonshine", "transcribe", "--model", model_name, audio_path],
            capture_output=True, text=True, timeout=30
        )
        text = result.stdout.strip()

    return {
        "text": text,
        "duration": round(time.time() - start, 2),
        "backend": "moonshine",
        "model": model_name,
    }


# ─── Backend: whisper-cpp ─────────────────────────────────────────────────────

WHISPER_CPP_MODELS = ["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large"]

def whisper_cpp_model_candidates(model_name: str) -> list[str]:
    return [
        os.path.expanduser(f"~/.cache/whisper-cpp/ggml-{model_name}.bin"),
        f"/opt/homebrew/share/whisper-cpp/models/ggml-{model_name}.bin",
        f"/usr/local/share/whisper-cpp/models/ggml-{model_name}.bin",
    ]


def is_whisper_cpp_available() -> bool:
    return shutil.which("whisper-cpp") is not None or shutil.which("main") is not None


def is_whisper_cpp_model_installed(model_name: str) -> bool:
    return any(Path(candidate).exists() for candidate in whisper_cpp_model_candidates(model_name))


def transcribe_whisper_cpp(audio_path: str, model_name: str, language: str) -> dict:
    import subprocess
    cmd = "whisper-cpp" if shutil.which("whisper-cpp") else "main"
    start = time.time()

    # whisper.cpp expects model path, resolve from Homebrew or standard location
    model_path = model_name
    if not os.path.exists(model_path):
        candidates = whisper_cpp_model_candidates(model_name)
        for c in candidates:
            if os.path.exists(c):
                model_path = c
                break

    result = subprocess.run(
        [cmd, "-m", model_path, "-l", language, "--no-timestamps", "-f", audio_path],
        capture_output=True, text=True, timeout=60
    )
    text = result.stdout.strip()
    # whisper.cpp outputs with [timestamp] prefix, clean it
    lines = [l.strip() for l in text.split("\n") if l.strip() and not l.strip().startswith("[")]
    if not lines:
        lines = [l.split("]", 1)[-1].strip() for l in text.split("\n") if "]" in l]

    return {
        "text": " ".join(lines),
        "duration": round(time.time() - start, 2),
        "backend": "whisper-cpp",
        "model": model_name,
    }


# ─── Backend: deepgram (cloud) ───────────────────────────────────────────────

DEEPGRAM_MODELS = ["nova-3", "nova-2", "whisper-large", "whisper-medium", "whisper-small"]

def is_deepgram_available() -> bool:
    return bool(os.environ.get("DEEPGRAM_API_KEY"))

def transcribe_deepgram(audio_path: str, model_name: str, language: str) -> dict:
    import urllib.request

    api_key = os.environ["DEEPGRAM_API_KEY"]
    start = time.time()

    with open(audio_path, "rb") as f:
        audio_data = f.read()

    params = f"model={model_name}&language={language}&smart_format=true"
    url = f"https://api.deepgram.com/v1/listen?{params}"

    req = urllib.request.Request(
        url,
        data=audio_data,
        headers={
            "Authorization": f"Token {api_key}",
            "Content-Type": "audio/wav",
        },
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())

    text = result["results"]["channels"][0]["alternatives"][0]["transcript"]
    return {
        "text": text,
        "duration": round(time.time() - start, 2),
        "backend": "deepgram",
        "model": model_name,
    }


# ─── Backend: parakeet (NVIDIA NeMo) ─────────────────────────────────────────

PARAKEET_MODELS = [
    "nvidia/parakeet-tdt-0.6b-v2",
    "nvidia/parakeet-ctc-0.6b",
    "nvidia/parakeet-tdt-1.1b",
]

def is_parakeet_available() -> bool:
    try:
        import nemo.collections.asr
        return True
    except ImportError:
        return False


def parakeet_repo_ids(model_name: str) -> list[str]:
    return [model_name]


def is_parakeet_model_installed(model_name: str) -> bool:
    if huggingface_repo_exists(parakeet_repo_ids(model_name)):
        return True
    local_candidates = [
        f"~/.cache/torch/NeMo/{model_name.replace('/', '--')}",
        f"~/.cache/torch/NeMo/{model_name.split('/', 1)[-1]}",
    ]
    return any(path.exists() for path in _existing_dirs(local_candidates))


def transcribe_parakeet(audio_path: str, model_name: str, language: str) -> dict:
    import nemo.collections.asr as nemo_asr
    start = time.time()
    model = nemo_asr.models.ASRModel.from_pretrained(model_name)
    text = model.transcribe([audio_path])[0]
    if isinstance(text, list):
        text = text[0]
    return {
        "text": str(text),
        "duration": round(time.time() - start, 2),
        "backend": "parakeet",
        "model": model_name,
    }


# ─── Registry ────────────────────────────────────────────────────────────────

BACKENDS = {
    "faster-whisper": {
        "fn": transcribe_faster_whisper,
        "available": is_faster_whisper_available,
        "models": FASTER_WHISPER_MODELS,
        "installed_models": lambda: detect_installed_models(FASTER_WHISPER_MODELS, is_faster_whisper_model_installed),
        "default_model": "small",
        "install": "pip install faster-whisper",
        "install_detection": "huggingface-cache",
        "type": "local",
    },
    "moonshine": {
        "fn": transcribe_moonshine,
        "available": is_moonshine_available,
        "models": MOONSHINE_MODELS,
        "installed_models": lambda: detect_installed_models(MOONSHINE_MODELS, is_moonshine_model_installed),
        "default_model": "moonshine/base",
        "install": "pip install useful-moonshine[onnx]",
        "install_detection": "moonshine-cache-heuristic",
        "type": "local",
    },
    "whisper-cpp": {
        "fn": transcribe_whisper_cpp,
        "available": is_whisper_cpp_available,
        "models": WHISPER_CPP_MODELS,
        "installed_models": lambda: detect_installed_models(WHISPER_CPP_MODELS, is_whisper_cpp_model_installed),
        "default_model": "small",
        "install": "brew install whisper-cpp",
        "install_detection": "whisper-cpp-model-paths",
        "type": "local",
    },
    "deepgram": {
        "fn": transcribe_deepgram,
        "available": is_deepgram_available,
        "models": DEEPGRAM_MODELS,
        "installed_models": lambda: [],
        "default_model": "nova-3",
        "install": "Set DEEPGRAM_API_KEY env var (free: deepgram.com)",
        "install_detection": "api-key",
        "type": "cloud",
    },
    "parakeet": {
        "fn": transcribe_parakeet,
        "available": is_parakeet_available,
        "models": PARAKEET_MODELS,
        "installed_models": lambda: detect_installed_models(PARAKEET_MODELS, is_parakeet_model_installed),
        "default_model": "nvidia/parakeet-tdt-0.6b-v2",
        "install": "pip install nemo_toolkit[asr]",
        "install_detection": "huggingface-or-nemo-cache",
        "type": "local",
    },
}


def detect_backend() -> str:
    """Auto-detect best available backend (prefer local, fastest first)."""
    priority = ["faster-whisper", "moonshine", "whisper-cpp", "deepgram", "parakeet"]
    for name in priority:
        if BACKENDS[name]["available"]():
            return name
    return "none"


def main():
    parser = argparse.ArgumentParser(description="Local & cloud STT transcriber for pi-voice")
    parser.add_argument("audio_file", nargs="?", help="Path to audio file (WAV)")
    parser.add_argument("--backend", choices=list(BACKENDS.keys()), default=None)
    parser.add_argument("--model", default=None, help="Model name/size")
    parser.add_argument("--language", default="en", help="Language code (or 'auto')")
    parser.add_argument("--list-backends", action="store_true", help="List available backends")
    parser.add_argument("--list-models", action="store_true", help="List models for a backend")
    args = parser.parse_args()

    if args.list_backends:
        result = []
        for name, info in BACKENDS.items():
            available = info["available"]()
            installed_models = info["installed_models"]() if "installed_models" in info else []
            result.append({
                "name": name,
                "available": available,
                "type": info["type"],
                "default_model": info["default_model"],
                "install": info["install"] if not available else None,
                "install_detection": info.get("install_detection", "unknown"),
                "models": info["models"],
                "installed_models": installed_models,
            })
        print(json.dumps(result, indent=2))
        return

    if args.list_models:
        backend = args.backend or detect_backend()
        if backend == "none" or backend not in BACKENDS:
            print(json.dumps({"error": f"Unknown backend: {backend}"}))
            sys.exit(1)
        print(json.dumps({
            "backend": backend,
            "models": BACKENDS[backend]["models"],
            "default": BACKENDS[backend]["default_model"],
        }, indent=2))
        return

    if not args.audio_file:
        parser.print_help()
        sys.exit(1)

    backend = args.backend or detect_backend()
    if backend == "none":
        print(json.dumps({
            "error": "No STT backend found",
            "install_options": {name: info["install"] for name, info in BACKENDS.items()},
        }))
        sys.exit(1)

    info = BACKENDS[backend]
    model = args.model or info["default_model"]

    try:
        result = info["fn"](args.audio_file, model, args.language)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e), "backend": backend, "model": model}))
        sys.exit(1)


if __name__ == "__main__":
    main()

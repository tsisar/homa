#!/usr/bin/env bash
# One-time setup for the Ukrainian voice agent:
#   1. Python venv with the `piper-tts` package (bundles onnxruntime + espeak-ng;
#      the standalone macOS arm64 release binary is shipped without its dylibs,
#      so the pip package is the reliable path).
#   2. Download the Ukrainian voice uk_UA-ukrainian_tts-medium.
# Idempotent: re-running skips anything already present.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"
VOICES_DIR="$ROOT/voices"

VOICE_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/uk/uk_UA/ukrainian_tts/medium"
VOICE_ONNX="uk_UA-ukrainian_tts-medium.onnx"
VOICE_JSON="uk_UA-ukrainian_tts-medium.onnx.json"

# --- Python venv + piper-tts ---------------------------------------------
if [[ -x "$VENV/bin/piper" ]]; then
  echo "✓ piper already installed in venv"
else
  echo "↓ creating venv + installing piper-tts…"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q piper-tts
  echo "✓ piper installed: $VENV/bin/piper"
fi

# --- Ukrainian voice ------------------------------------------------------
mkdir -p "$VOICES_DIR"
if [[ -s "$VOICES_DIR/$VOICE_ONNX" ]]; then
  echo "✓ voice model present"
else
  echo "↓ downloading voice model (~77 MB)…"
  curl -L --fail -o "$VOICES_DIR/$VOICE_ONNX" "$VOICE_BASE/$VOICE_ONNX"
fi
if [[ -s "$VOICES_DIR/$VOICE_JSON" ]]; then
  echo "✓ voice config present"
else
  curl -L --fail -o "$VOICES_DIR/$VOICE_JSON" "$VOICE_BASE/$VOICE_JSON"
fi

echo
echo "Done. Smoke-test the voice:"
echo "  echo 'привіт світ' | $VENV/bin/piper -m $VOICES_DIR/$VOICE_ONNX -s 2 -f /tmp/t.wav && afplay /tmp/t.wav"

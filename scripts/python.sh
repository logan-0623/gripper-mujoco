#!/bin/bash
# Run project Python with process-local native libraries; never change ~/.zshrc.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
project_python="$project_root/.venv-lerobot/bin/python"
if [[ ! -x "$project_python" ]]; then
  echo "Missing .venv-lerobot Python; follow README.md local setup." >&2
  exit 1
fi
if [[ "$(uname -s)" == Darwin ]]; then
  if ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew is required for ffmpeg@8; see README.md." >&2
    exit 1
  fi
  project_ffmpeg="$(brew --prefix ffmpeg@8)"
  if [[ ! -x "$project_ffmpeg/bin/ffmpeg" ]]; then
    echo "Missing compatible FFmpeg; run: brew install ffmpeg@8" >&2
    exit 1
  fi
  export DYLD_LIBRARY_PATH="$project_ffmpeg/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
  export PATH="$project_ffmpeg/bin:$PATH"
fi
export PATH="$project_root/.venv-lerobot/bin:$PATH"
cd "$project_root"
exec "$project_python" "$@"

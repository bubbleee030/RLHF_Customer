#!/usr/bin/env bash
# Install visualization dependencies into the local uv virtual environment.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
UV_PYTHON="$PROJECT_DIR/uv/bin/python"

if [[ ! -x "$UV_PYTHON" ]]; then
  echo "Missing uv python: $UV_PYTHON" >&2
  echo "Create environment first: python3 -m venv uv" >&2
  exit 1
fi

if ! "$UV_PYTHON" -m pip --version >/dev/null 2>&1; then
  if "$UV_PYTHON" -m ensurepip --upgrade >/dev/null 2>&1; then
    :
  else
    tmp_get_pip="$(mktemp)"
    curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$tmp_get_pip"
    "$UV_PYTHON" "$tmp_get_pip"
    rm -f "$tmp_get_pip"
  fi
fi

"$UV_PYTHON" -m pip install --upgrade pip
"$UV_PYTHON" -m pip install matplotlib tensorboard

echo "Installed visualization dependencies into: $PROJECT_DIR/uv"

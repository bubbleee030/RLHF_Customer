#!/usr/bin/env bash
# Run visualization with the uv virtual environment.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
UV_PYTHON="$PROJECT_DIR/uv/bin/python"
VIS_SCRIPT="$SCRIPT_DIR/visualize_training.py"

if [[ ! -x "$UV_PYTHON" ]]; then
  echo "Missing uv python: $UV_PYTHON" >&2
  echo "Create environment first: python3 -m venv uv" >&2
  exit 1
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: ./scripts/visualize_training.sh <output_dir> [save_dir]" >&2
  exit 1
fi

OUTPUT_DIR="$1"
SAVE_DIR="${2:-$PROJECT_DIR/test/reports/plots}"

"$UV_PYTHON" "$VIS_SCRIPT" --output-dir "$OUTPUT_DIR" --save-dir "$SAVE_DIR"

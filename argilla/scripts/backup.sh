#!/usr/bin/env bash
# Wrapper for Argilla backup script

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON_BIN="$PROJECT_DIR/uv/bin/python"
BACKUP_SCRIPT="$SCRIPT_DIR/auto_backup.py"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

show_help() {
  cat <<'EOF'
Argilla Backup Manager

Usage:
  ./scripts/backup.sh backup
  ./scripts/backup.sh schedule <minutes>
  ./scripts/backup.sh list
  ./scripts/backup.sh fix-encoding [path]

Commands:
  backup                 Run backup once (default)
  schedule <minutes>     Run periodic backup
  list                   List existing backups
  fix-encoding [path]    Rewrite all JSON files as UTF-8 (default path: ./argilla)
EOF
}

cmd="${1:-backup}"

case "$cmd" in
  backup)
    "$PYTHON_BIN" "$BACKUP_SCRIPT" --backup-dir "$PROJECT_DIR/backups"
    ;;
  schedule)
    interval="${2:-120}"
    "$PYTHON_BIN" "$BACKUP_SCRIPT" --schedule "$interval" --backup-dir "$PROJECT_DIR/backups"
    ;;
  list)
    "$PYTHON_BIN" "$BACKUP_SCRIPT" --list --backup-dir "$PROJECT_DIR/backups"
    ;;
  fix-encoding)
    target="${2:-$PROJECT_DIR/argilla}"
    "$PYTHON_BIN" "$BACKUP_SCRIPT" --fix-existing "$target"
    ;;
  help|--help|-h)
    show_help
    ;;
  *)
    echo "Unknown command: $cmd" >&2
    show_help
    exit 1
    ;;
esac

#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"

python3 "$PROJECT_DIR/src/breedai/cli.py" "$@"


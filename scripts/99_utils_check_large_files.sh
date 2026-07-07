#!/usr/bin/env bash
# Check for large files before commit.
# Usage: ./99_utils_check_large_files.sh [MAX_MB] [--staged]
#   MAX_MB: max size in MB (default: 5). Exit 1 if any file exceeds it.
#   --staged: check only staged files. Default: check what `git add -A` would add.

set -euo pipefail
MAX_MB="5"
MODE="would-add"
for a in "$@"; do
  if [[ "$a" == "--staged" ]]; then MODE="staged"
  elif [[ "$a" =~ ^[0-9]+$ ]]; then MAX_MB="$a"
  fi
done

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if [[ "$MODE" == "staged" ]]; then
  TO_CHECK="$(git diff --cached --name-only --diff-filter=ACMR)"
else
  TO_CHECK="$(git add -A --dry-run 2>&1 | sed -n "s/^add '\(.*\)'\$/\1/p")"
fi

FAILED=0
FILES_OVER=()

while IFS= read -r f; do
  [[ -z "$f" ]] && continue
  [[ ! -f "$f" ]] && continue
  BYTES="$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null)"
  MB="$(( BYTES / 1048576 ))"
  if (( MB >= MAX_MB )); then
    FAILED=1
    FILES_OVER+=("${MB}M  $f")
  fi
done <<< "$TO_CHECK"

if (( FAILED )); then
  echo "⚠️  The following files exceed ${MAX_MB}MB:"
  printf '   %s\n' "${FILES_OVER[@]}"
  echo ""
  if [[ "$MODE" == "staged" ]]; then
    echo "Unstage them:  git restore --staged <file>"
  else
    echo "Avoid adding them, or add to .gitignore, then stage only desired files."
  fi
  exit 1
fi

echo "✅ No files over ${MAX_MB}MB."
exit 0

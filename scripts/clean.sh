#!/usr/bin/env bash
set -euo pipefail

# STUNIR Safe Clean
# Removes generated artifacts while preserving tracked documentation.
#
# Usage:
#   ./scripts/clean.sh           # safe clean (default)
#   ./scripts/clean.sh --deep    # also removes heavy native build caches (if present)

DEEP=0
for arg in "$@"; do
  case "$arg" in
    --deep) DEEP=1 ;;
    -h|--help)
      echo "Usage: $0 [--deep]"
      exit 0
      ;;
    *)
      echo "ERROR: unknown arg: $arg" >&2
      echo "Usage: $0 [--deep]" >&2
      exit 2
      ;;
  esac
done

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"
cd "$ROOT"

echo "STUNIR clean: repo root = $ROOT"

echo "Cleaning build artifacts..."
rm -rf -- build _verify_build bin

echo "Cleaning python caches..."
rm -rf -- tools/__pycache__ 2>/dev/null || true
find tools -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
find tools -type f -name '*.pyc' -delete 2>/dev/null || true

echo "Cleaning receipts (preserve receipts/README.md and any other tracked docs)..."
if [ -d receipts ]; then
  find receipts -maxdepth 1 -type f -name '*.json' -delete 2>/dev/null || true
fi

if [ "$DEEP" -eq 1 ]; then
  echo "Deep clean enabled: removing native build caches (if present)..."
  rm -rf -- tools/native/rust/stunir-native/target 2>/dev/null || true
fi

echo "Clean complete."

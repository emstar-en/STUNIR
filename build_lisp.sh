#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f "scripts/build.sh" ]]; then
  echo "Error: scripts/build.sh not found. Run this from the STUNIR repo root." >&2
  exit 2
fi

# Portable Common Lisp emission (no runtime required)
export STUNIR_OUTPUT_TARGETS="lisp"

exec bash scripts/build.sh

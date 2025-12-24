#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f "scripts/build.sh" ]]; then
  echo "Error: scripts/build.sh not found. Run this from the STUNIR repo root." >&2
  exit 2
fi

# SBCL-backed Common Lisp variant (requires accepted sbcl dependency receipt)
export STUNIR_OUTPUT_TARGETS="lisp_sbcl"
: "${STUNIR_REQUIRE_DEPS:=1}"
export STUNIR_REQUIRE_DEPS

exec bash scripts/build.sh

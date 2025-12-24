#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f "scripts/build.sh" ]]; then
  echo "Error: scripts/build.sh not found. Run this from the STUNIR repo root." >&2
  exit 2
fi

# Emit both portable Lisp and SBCL-backed Lisp.
# If you want to fail when SBCL is missing, set STUNIR_REQUIRE_DEPS=1 (default for this wrapper).
export STUNIR_OUTPUT_TARGETS="lisp,lisp_sbcl"
: "${STUNIR_REQUIRE_DEPS:=1}"
export STUNIR_REQUIRE_DEPS

exec bash scripts/build.sh

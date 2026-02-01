#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage: build_targets.sh <targets_csv> [--require-deps|--allow-missing-deps]

Examples of <targets_csv>:
  lisp
  lisp_sbcl
  lisp,lisp_sbcl
  python
  python_cpython
  smt
  smt_z3
  wasm,c

Notes:
- This wrapper must be run from the STUNIR repo root (scripts/build.sh must exist).
- --require-deps sets STUNIR_REQUIRE_DEPS=1 (fail if toolchain deps missing)
- --allow-missing-deps sets STUNIR_REQUIRE_DEPS=0 (skip runtime-backed targets if deps missing)
USAGE
}

if [[ ! -f "scripts/build.sh" ]]; then
  echo "Error: scripts/build.sh not found. Run this from the STUNIR repo root." >&2
  exit 2
fi

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

TARGETS="$1"
shift

# Default policy: do not fail hard on missing deps unless requested.
: "${STUNIR_REQUIRE_DEPS:=0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --require-deps)
      STUNIR_REQUIRE_DEPS=1
      shift
      ;;
    --allow-missing-deps)
      STUNIR_REQUIRE_DEPS=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

export STUNIR_OUTPUT_TARGETS="$TARGETS"
export STUNIR_REQUIRE_DEPS

exec bash scripts/build.sh

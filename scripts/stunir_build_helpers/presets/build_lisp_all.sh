#!/usr/bin/env bash
set -euo pipefail

# Preset: build_lisp_all
# Targets: lisp,lisp_sbcl

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELPER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

exec bash "$HELPER_DIR/build_targets.sh" "lisp,lisp_sbcl" --require-deps

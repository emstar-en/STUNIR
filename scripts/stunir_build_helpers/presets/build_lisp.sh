#!/usr/bin/env bash
set -euo pipefail

# Preset: build_lisp
# Targets: lisp

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELPER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

exec bash "$HELPER_DIR/build_targets.sh" "lisp"

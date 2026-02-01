#!/usr/bin/env bash
set -euo pipefail

# Preset: build_smt_z3
# Targets: smt_z3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELPER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

exec bash "$HELPER_DIR/build_targets.sh" "smt_z3" --require-deps

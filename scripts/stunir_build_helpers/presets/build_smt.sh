#!/usr/bin/env bash
set -euo pipefail

# Preset: build_smt
# Targets: smt

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELPER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

exec bash "$HELPER_DIR/build_targets.sh" "smt"

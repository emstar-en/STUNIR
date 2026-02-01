#!/usr/bin/env bash
set -euo pipefail

# Preset: build_default_wasm_c
# Targets: wasm,c

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELPER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

exec bash "$HELPER_DIR/build_targets.sh" "wasm,c"

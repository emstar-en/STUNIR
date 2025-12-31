#!/usr/bin/env bash
# STUNIR Shell-Native Runner
# Profile 3: Minimal Verification & Orchestration

set -e

STUNIR_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "$STUNIR_ROOT/scripts/lib/core.sh"
source "$STUNIR_ROOT/scripts/lib/manifest.sh"

# --- CLI Dispatch ---

COMMAND="${1:-help}"
shift || true

case "$COMMAND" in
    discover|init)
        stunir_log "Running Toolchain Discovery..."
        stunir_generate_lockfile
        ;;
    build)
        stunir_log "Starting Build..."
        # Check for lockfile
        if [ ! -f "$STUNIR_ROOT/build/local_toolchain.lock.json" ]; then
            stunir_warn "Toolchain lockfile not found. Running discovery first..."
            stunir_generate_lockfile
        fi
        stunir_log "Build logic pending..."
        ;;
    clean)
        stunir_log "Cleaning build artifacts..."
        rm -rf "$STUNIR_ROOT/build"
        stunir_ok "Clean complete."
        ;;
    help|*)
        echo "Usage: $0 {discover|build|clean}"
        exit 1
        ;;
esac

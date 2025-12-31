#!/bin/sh
# STUNIR Polyglot Build Entrypoint
# Automatically dispatches to the best available runtime.

set -u

# --- Configuration ---
# Allow override via env var: STUNIR_PROFILE=native|python|shell
PROFILE="${STUNIR_PROFILE:-auto}"
SPEC_ROOT="spec"
OUT_FILE="asm/spec_ir.txt"
LOCK_FILE="local_toolchain.lock.json"

log() { echo "[STUNIR] $1"; }

# --- Pre-flight ---
# Ensure spec directory exists so tools don't crash on empty repo
if [ ! -d "$SPEC_ROOT" ]; then
    log "Creating default spec directory..."
    mkdir -p "$SPEC_ROOT"
fi

# --- Detection ---

detect_runtime() {
    if [ "$PROFILE" = "native" ]; then
        echo "native"
        return
    elif [ "$PROFILE" = "shell" ]; then
        echo "shell"
        return
    fi

    # Auto-detection
    if command -v python3 >/dev/null 2>&1; then
        echo "python"
    else
        echo "shell"
    fi
}

# --- Dispatch ---

RUNTIME=$(detect_runtime)
log "Runtime selected: $RUNTIME"

# 1. Discovery Phase (Always run shell manifest first to generate lockfile)
# This ensures Python has a lockfile to verify against
log "Running Toolchain Discovery..."
chmod +x scripts/lib/*.sh 2>/dev/null
./scripts/lib/manifest.sh

case "$RUNTIME" in
    python)
        # Original Python path
        # We use -B to prevent __pycache__ clutter
        exec python3 -B tools/spec_to_ir.py \
            --spec-root "$SPEC_ROOT" \
            --out "$OUT_FILE" \
            --lockfile "$LOCK_FILE"
        ;;

    shell)
        # Shell-Native path
        exec ./scripts/lib/runner.sh
        ;;

    native)
        # Native binary path (future)
        if [ -x "bin/stunir-cli" ]; then
            exec ./bin/stunir-cli build --spec "$SPEC_ROOT" --out "$OUT_FILE"
        else
            log "Error: Native binary not found at bin/stunir-cli"
            exit 1
        fi
        ;;

    *)
        log "Error: Unknown runtime profile '$RUNTIME'"
        exit 1
        ;;
esac

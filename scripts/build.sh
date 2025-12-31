#!/bin/sh
# STUNIR Polyglot Build Entrypoint
# Automatically dispatches to the best available runtime.

set -u

# --- Configuration ---
# Allow override via env var: STUNIR_PROFILE=native|python|shell
PROFILE="${STUNIR_PROFILE:-auto}"

log() { echo "[STUNIR] $1"; }

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

case "$RUNTIME" in
    python)
        # Original Python path
        # We use -B to prevent __pycache__ clutter
        exec python3 -B tools/spec_to_ir.py "$@"
        ;;

    shell)
        # Shell-Native path
        # Ensure scripts are executable
        chmod +x scripts/lib/*.sh 2>/dev/null
        exec ./scripts/lib/runner.sh "$@"
        ;;

    native)
        # Native binary path (future)
        if [ -x "bin/stunir-cli" ]; then
            exec ./bin/stunir-cli "$@"
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

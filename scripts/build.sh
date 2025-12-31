#!/bin/sh
# STUNIR Polyglot Build Entrypoint
# Automatically dispatches to the best available runtime.

set -u

# --- Configuration ---
# Allow override via env var: STUNIR_PROFILE=native|python|shell
PROFILE="${STUNIR_PROFILE:-auto}"
SPEC_ROOT="spec"
OUT_IR="asm/spec_ir.json"
OUT_PY="asm/output.py"
LOCK_FILE="local_toolchain.lock.json"
NATIVE_BIN="tools/native/rust/stunir-native/target/release/stunir-native"

log() { echo "[STUNIR] $1"; }

# --- Pre-flight ---
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
    elif [ "$PROFILE" = "python" ]; then
        echo "python"
        return
    fi

    # Auto-detection priority: Native -> Python -> Shell
    if [ -x "$NATIVE_BIN" ]; then
        echo "native"
    elif command -v python3 >/dev/null 2>&1; then
        echo "python"
    else
        echo "shell"
    fi
}

# --- Dispatch ---
RUNTIME=$(detect_runtime)
log "Runtime selected: $RUNTIME"

# 1. Discovery Phase (Always run shell manifest first to generate lockfile)
log "Running Toolchain Discovery..."
chmod +x scripts/lib/*.sh 2>/dev/null || true
./scripts/lib/manifest.sh

case "$RUNTIME" in
    native)
        log "Using Native Core: $NATIVE_BIN"

        # 1. Spec -> IR
        # Note: Native currently takes a single file input, we might need to loop or aggregate
        # For now, we assume a single entry point spec or aggregate it.
        # TODO: Update native to handle directory scanning like Python does.
        # Temporary hack: Create a dummy aggregate spec if none exists
        AGG_SPEC="asm/aggregate_spec.json"
        mkdir -p asm
        echo '{"kind": "spec", "modules": []}' > "$AGG_SPEC"

        "$NATIVE_BIN" spec-to-ir --input "$AGG_SPEC" --output "$OUT_IR"

        # 2. IR -> Code
        "$NATIVE_BIN" emit --input "$OUT_IR" --target python --output "$OUT_PY"

        log "Build complete (Native)"
        ;;

    python)
        # Original Python path
        exec python3 -B tools/spec_to_ir.py             --spec-root "$SPEC_ROOT"             --out "$OUT_IR"             --lockfile "$LOCK_FILE"
        ;;

    shell)
        # Shell-Native path
        exec ./scripts/lib/runner.sh
        ;;

    *)
        log "Error: Unknown runtime profile '$RUNTIME'"
        exit 1
        ;;
esac

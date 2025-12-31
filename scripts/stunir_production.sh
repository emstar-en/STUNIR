#!/bin/bash
set -euo pipefail

# STUNIR Production Pipeline Orchestrator
# Dispatches to Native Core (Rust/Haskell) if available, else Python Fallback

# Paths
RUST_BIN="build/bin/stunir-native"
HASKELL_BIN="build/bin/stunir-haskell"
PYTHON_CANON="tools/ir_canonicalizer/canonicalize.py"
PYTHON_RECEIPT="tools/receipt_emitter/emit_receipt.py"

# Ensure build dir exists
mkdir -p build

# Priority: Rust > Haskell > Python
if [ -x "$RUST_BIN" ]; then
    echo "🦀 [STUNIR] Using Native Core (Rust)"
    MODE="RUST"
    EXEC_BIN="$RUST_BIN"
elif [ -x "$HASKELL_BIN" ]; then
    echo "λ  [STUNIR] Using Native Core (Haskell)"
    MODE="HASKELL"
    EXEC_BIN="$HASKELL_BIN"
else
    echo "⚠️ [STUNIR] Native Core not found. Falling back to Python."
    MODE="PYTHON"
fi

# Command Dispatch
CMD="${1:-help}"
shift || true

case "$MODE" in
    "RUST"|"HASKELL")
        # Pass through to binary
        exec "$EXEC_BIN" "$CMD" "$@"
        ;;
    "PYTHON")
        case "$CMD" in
            "spec-to-ir")
                echo "🐍 [Python] Running spec-to-ir fallback..."
                "$PYTHON_CANON" "$@"
                ;;
            "gen-receipt")
                echo "🐍 [Python] Running gen-receipt fallback..."
                "$PYTHON_RECEIPT" "$@"
                ;;
            *)
                echo "❌ Error: Command '$CMD' not implemented in Python fallback."
                exit 1
                ;;
        esac
        ;;
esac

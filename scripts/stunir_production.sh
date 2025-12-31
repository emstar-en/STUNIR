#!/bin/bash
set -euo pipefail

# STUNIR Production Pipeline Orchestrator
# Dispatches to Native Core (Haskell) if available, else Python Fallback

STUNIR_BIN="build/bin/stunir-native"
PYTHON_CANON="tools/ir_canonicalizer/canonicalize.py"
PYTHON_RECEIPT="tools/receipt_emitter/emit_receipt.py"

# Ensure build dir exists
mkdir -p build

# Check for Native Core
if [ -x "$STUNIR_BIN" ]; then
    echo "🚀 [STUNIR] Using Native Core (Haskell)"
    MODE="NATIVE"
else
    echo "⚠️ [STUNIR] Native Core not found. Falling back to Python."
    MODE="PYTHON"
fi

# Command Dispatch
CMD="${1:-help}"
shift || true

case "$MODE" in
    "NATIVE")
        # Pass through to Haskell binary
        exec "$STUNIR_BIN" "$CMD" "$@"
        ;;
    "PYTHON")
        case "$CMD" in
            "spec-to-ir")
                # Python implementation of spec-to-ir would go here or call a script
                # For now, we just use the canonicalizer as a proxy for the IR generation step
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

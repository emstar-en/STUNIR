#!/bin/bash
set -e

# Configuration
STUNIR_BIN="build/stunir_native"
SHELL_COMPILER="scripts/compile.sh"

# Inputs
SPEC="build/human_spec.json"
IR="build/pipeline.ir.json"
OUT_DIR="build/dist"
mkdir -p "$OUT_DIR"

echo "=== STUNIR Build Pipeline ==="

# 1. Compile (Spec -> IR)
if [ -f "$STUNIR_BIN" ]; then
    echo "[1/2] Compiling with Native Binary..."
    "$STUNIR_BIN" compile --in-spec "$SPEC" --out-ir "$IR"
elif [ -f "$SHELL_COMPILER" ]; then
    echo "[1/2] Compiling with Shell Fallback..."
    "$SHELL_COMPILER" "$SPEC" "$IR"
else
    echo "ERROR: No compiler found!"
    exit 1
fi

# 2. Emit (IR -> Code)
# For now, we require the binary for emission as shell-emission is complex
if [ -f "$STUNIR_BIN" ]; then
    echo "[2/2] Emitting Code (Polyglot)..."
    "$STUNIR_BIN" emit --in-ir "$IR" --target python --out-file "$OUT_DIR/app.py"
    "$STUNIR_BIN" emit --in-ir "$IR" --target bash --out-file "$OUT_DIR/app.sh"
    "$STUNIR_BIN" emit --in-ir "$IR" --target rust --out-file "$OUT_DIR/app.rs"
    
    echo "SUCCESS: Artifacts generated in $OUT_DIR"
    ls -l "$OUT_DIR"
else
    echo "WARNING: Skipping Emission (Binary missing). IR generated at $IR"
fi

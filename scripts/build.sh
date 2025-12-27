<<<<<<< HEAD
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
=======
#!/usr/bin/env bash
set -e

# Load Dispatcher
source scripts/lib/dispatch.sh

mkdir -p build

echo ">>> [0/6] Checking Toolchain..."
if [[ -f "build/local_toolchain.lock.json" ]]; then
    stunir_dispatch check_toolchain --lockfile build/local_toolchain.lock.json
else
    echo "WARNING: No toolchain lockfile found. Skipping check."
fi

echo ">>> [1/6] Determining Epoch..."
stunir_dispatch epoch --out-json build/epoch.json --print-epoch

echo ">>> [2/6] Importing Code..."
if [[ -d "src" ]]; then
    stunir_dispatch import_code --input-root src --out-spec build/spec.json
else
    echo "No 'src' directory found, skipping."
    echo '{ "kind": "spec", "modules": [] }' > build/spec.json
fi

echo ">>> [3/6] Generating IR..."
stunir_dispatch spec_to_ir --in-json build/spec.json --out-ir build/ir.json

echo ">>> [4/6] Generating Provenance..."
stunir_dispatch gen_provenance --in-ir build/ir.json --out-prov build/provenance.json

echo ">>> [5/6] Compiling Provenance..."
stunir_dispatch compile_provenance --in-prov build/provenance.json --out-bin build/provenance.bin

echo ">>> [6/6] Generating Receipt..."
stunir_dispatch receipt --in-bin build/provenance.bin --out-receipt build/receipt.json

echo ">>> Build Complete!"
>>>>>>> origin/rescue/main-pre-force

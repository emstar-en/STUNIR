#!/bin/bash
set -euo pipefail

# STUNIR Build Script v2 (Native-First Polyglot)
# ------------------------------------------------
# This script orchestrates the deterministic build pipeline.
# It delegates actual logic to the 'stunir-dispatch' function,
# which prefers compiled native tools over Python/Shell.

# 1. Setup Environment
export STUNIR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export BUILD_DIR="$STUNIR_ROOT/build"
export ARTIFACTS_DIR="$STUNIR_ROOT/artifacts"
mkdir -p "$BUILD_DIR" "$ARTIFACTS_DIR"

# Load Dispatcher
source "$STUNIR_ROOT/scripts/lib/dispatch.sh"

echo "=== STUNIR Build Pipeline v2 ==="
echo "Root: $STUNIR_ROOT"

# 2. Bootstrap / Toolchain Check
# Ensure we have a usable toolchain (Native or Python)
stunir_dispatch check-toolchain --lockfile "$STUNIR_ROOT/local_toolchain.lock.json"

# 3. Generate Epoch
# We need a fixed timestamp for this build run
EPOCH_JSON="$BUILD_DIR/epoch.json"
EPOCH=$(stunir_dispatch epoch --out-json "$EPOCH_JSON" --print-epoch)
echo "Build Epoch: $EPOCH"

# 4. Import Code (Source -> Spec)
# Scans source code and generates a canonical spec.json
echo "--- Step 1: Import Code ---"
SPEC_ROOT="$BUILD_DIR/spec"
mkdir -p "$SPEC_ROOT"
stunir_dispatch import-code \
    --input-root "$STUNIR_ROOT" \
    --out-spec "$SPEC_ROOT/spec.json"

# 5. Spec -> IR (Intermediate Reference)
# Converts spec to canonical IR
echo "--- Step 2: Generate IR ---"
IR_JSON="$BUILD_DIR/ir.json"
stunir_dispatch spec-to-ir \
    --spec-root "$SPEC_ROOT" \
    --out "$IR_JSON"

# 6. Provenance Generation
# Generates the C header and JSON for provenance injection
echo "--- Step 3: Generate Provenance ---"
PROV_JSON="$BUILD_DIR/provenance.json"
PROV_HEADER="$BUILD_DIR/provenance.h"
stunir_dispatch gen-provenance \
    --epoch "$EPOCH" \
    --spec-root "$SPEC_ROOT" \
    --asm-root "$BUILD_DIR/asm" \
    --out-json "$PROV_JSON" \
    --out-header "$PROV_HEADER"

# 7. Compilation (The "Build")
# In a real scenario, this would compile the user's code.
# For STUNIR self-hosting, we compile the provenance emitter.
echo "--- Step 4: Compile Target ---"
TARGET_BIN="$ARTIFACTS_DIR/provenance_emitter"
stunir_dispatch compile-provenance \
    --in-prov "$PROV_HEADER" \
    --out-bin "$TARGET_BIN"

# 8. Receipt Generation
# Generate a cryptographic receipt for the build artifact
echo "--- Step 5: Generate Receipt ---"
TARGET_HASH=$(sha256sum "$TARGET_BIN" | awk '{print $1}')
# We use the dispatcher itself as the "tool" for the receipt in this self-hosted context
TOOL_NAME="stunir-native"
TOOL_PATH=$(which stunir-native 2>/dev/null || echo "internal")
TOOL_HASH="0000000000000000000000000000000000000000000000000000000000000000" # Placeholder
TOOL_VER="0.5.0"

RECEIPT_JSON="$ARTIFACTS_DIR/receipt.json"
stunir_dispatch gen-receipt \
    "provenance_emitter" \
    "SUCCESS" \
    "$EPOCH" \
    "$TOOL_NAME" \
    "$TOOL_PATH" \
    "$TOOL_HASH" \
    "$TOOL_VER" \
    > "$RECEIPT_JSON"

echo "=== Build Complete ==="
echo "Artifact: $TARGET_BIN"
echo "Receipt:  $RECEIPT_JSON"

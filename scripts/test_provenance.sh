#!/usr/bin/env bash
set -e

echo "--- STUNIR Provenance Conformance Test ---"

# 1. Setup
mkdir -p build/test
IR_FILE="build/test/test.ir"
EPOCH_FILE="build/test/epoch.json"
RUST_PROV="build/test/rust.prov"
HASKELL_PROV="build/test/haskell.prov"

# Create dummy IR and Epoch
echo '{"version": "1.0.0", "functions": []}' > "$IR_FILE"
echo '{"timestamp": "2025-01-01T00:00:00Z", "builder": "test"}' > "$EPOCH_FILE"

# 2. Run Haskell Tool
echo "[*] Running Haskell Tool..."
HASKELL_CMD="cabal run stunir-native --"
(cd tools/native/haskell && $HASKELL_CMD gen-provenance --in-ir "../../../$IR_FILE" --epoch-json "../../../$EPOCH_FILE" --out-prov "../../../$HASKELL_PROV")

# 3. Run Rust Tool
echo "[*] Running Rust Tool..."
RUST_CMD="cargo run --manifest-path tools/native/rust/Cargo.toml --"
$RUST_CMD gen-provenance --in-ir "$IR_FILE" --epoch-json "$EPOCH_FILE" --out-prov "$RUST_PROV"

# 4. Compare
echo "[*] Comparing Outputs..."
if diff -q "$RUST_PROV" "$HASKELL_PROV"; then
    echo "✅ PROVENANCE CONFORMANCE VERIFIED!"
else
    echo "❌ PROVENANCE FAILED: Outputs differ."
    echo "--- Rust Output ---"
    cat "$RUST_PROV"
    echo "--- Haskell Output ---"
    cat "$HASKELL_PROV"
    exit 1
fi

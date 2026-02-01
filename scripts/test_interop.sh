#!/usr/bin/env bash
set -e

echo "--- STUNIR Conformance Test ---"

# 1. Setup
mkdir -p build/test
SPEC_FILE="build/test/spec.json"
RUST_IR="build/test/rust.ir"
HASKELL_IR="build/test/haskell.ir"

# Create dummy spec
echo '{"id": "test", "name": "test", "schema": "v1", "stages": [], "targets": [], "profile": "3"}' > "$SPEC_FILE"

# 2. Run Haskell Tool
echo "[*] Running Haskell Tool..."
# Assuming binary is in dist-newstyle or we run via cabal run
# For simplicity, we try to find the binary or use cabal run
HASKELL_CMD="cabal run stunir-native --"
(cd tools/native/haskell && $HASKELL_CMD spec-to-ir --in-json "../../../$SPEC_FILE" --out-ir "../../../$HASKELL_IR")

# 3. Run Rust Tool
echo "[*] Running Rust Tool..."
RUST_CMD="cargo run --manifest-path tools/native/rust/Cargo.toml --"
$RUST_CMD spec-to-ir --in-json "$SPEC_FILE" --out-ir "$RUST_IR"

# 4. Compare
echo "[*] Comparing Outputs..."
if diff -q "$RUST_IR" "$HASKELL_IR"; then
    echo "✅ CONFORMANCE VERIFIED: Rust and Haskell outputs match!"
else
    echo "❌ CONFORMANCE FAILED: Outputs differ."
    echo "--- Rust Output ---"
    cat "$RUST_IR"
    echo "--- Haskell Output ---"
    cat "$HASKELL_IR"
    exit 1
fi

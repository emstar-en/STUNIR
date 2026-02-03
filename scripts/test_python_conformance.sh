#!/usr/bin/env bash
set -e

echo "--- STUNIR Python Conformance Test ---"

# 1. Setup
mkdir -p build/test
SPEC_FILE="build/test/spec.json"
RUST_IR="build/test/rust.ir"
PYTHON_IR="build/test/python.ir"

# Create dummy spec
echo '{"id": "test", "name": "test", "schema": "v1", "stages": [], "targets": [], "profile": "3"}' > "$SPEC_FILE"

# 2. Run Rust Tool (Reference)
echo "[*] Running Rust Tool..."
RUST_CMD="cargo run --manifest-path tools/native/rust/Cargo.toml --"
$RUST_CMD spec-to-ir --in-json "$SPEC_FILE" --out-ir "$RUST_IR"

# 3. Run Python Minimal Tool
echo "[*] Running Python Minimal Tool..."
python3 tools/python/stunir_minimal.py spec-to-ir --in-json "$SPEC_FILE" --out-ir "$PYTHON_IR"

# 4. Compare
echo "[*] Comparing Outputs..."
if diff -q "$RUST_IR" "$PYTHON_IR"; then
    echo "✅ PYTHON CONFORMANCE VERIFIED!"
else
    echo "❌ PYTHON FAILED: Outputs differ."
    echo "--- Rust Output ---"
    cat "$RUST_IR"
    echo "--- Python Output ---"
    cat "$PYTHON_IR"
    exit 1
fi

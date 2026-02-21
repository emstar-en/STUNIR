#!/bin/bash
set -euo pipefail

echo "🔬 HASKELL ↔ RUST CONFLUENCE TEST"

# Build both pipelines
./scripts/build_haskell_pipeline.sh
./scripts/build_rust_pipeline.sh

# Test identical behavior
echo "🧪 Haskell:"
./stunir-haskell spec/input_0001.json > haskell_output.txt

echo "🧪 Rust (Haskell mirror):"  
./stunir-rust spec/input_0001.json > rust_output.txt

# Compare outputs
if diff -q haskell_output.txt rust_output.txt >/dev/null 2>&1; then
    echo "✅ CONFLUENCE: Haskell ≡ Rust (identical outputs)"
else
    echo "❌ CONFLUENCE: Haskell ≠ Rust"
    exit 1
fi

echo "🎉 RUST MIRROR: PERFECT HASKELL CONFORMANCE"

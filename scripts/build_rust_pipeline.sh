#!/bin/bash
set -euo pipefail

echo "🔨 BUILDING RUST MIRROR PIPELINE (Haskell-aligned)"
cd tools/native/rust

# Clean + build + strip
cargo clean
cargo build --release --bin stunir-rust

# Copy binary to root
cp target/release/stunir-rust ../../stunir-rust

cd ../..
echo "✅ RUST PIPELINE BINARY: stunir-rust"
echo "🚀 TEST: ./stunir-rust spec/input_0001.json"

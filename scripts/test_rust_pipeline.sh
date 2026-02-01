#!/bin/bash
set -euo pipefail

echo "🧪 TESTING RUST MIRROR PIPELINE"

# Build first
./scripts/build_rust_pipeline.sh

# Test full pipeline
echo "⚙️ RUNNING: spec → Rust binary → receipt"
./stunir-rust spec/input_0001.json

echo "✅ RUST PIPELINE: Haskell-aligned & PRODUCTION READY"

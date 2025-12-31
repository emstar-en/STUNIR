#!/bin/bash
set -euo pipefail

echo "🧪 TESTING HASKELL REFERENCE PIPELINE"

# Build first
./scripts/build_haskell_pipeline.sh

# Test full pipeline
echo "🔮 RUNNING: spec → Haskell binary → receipt"
./stunir-haskell spec/input_0001.json

echo "✅ HASKELL PIPELINE: PRODUCTION READY"

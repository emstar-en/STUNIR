#!/bin/bash
set -euo pipefail

echo "🚀 COMPLETE STUNIR PIPELINE TEST"

./test_harness/haskell_pipeline_test.sh
./scripts/align_rust_haskell.sh

echo "✅ FULL PIPELINE: HASKELL → RUST ALIGNED"
echo "📦 Production artifacts ATTESTED"

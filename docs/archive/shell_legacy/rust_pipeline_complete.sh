#!/bin/bash
set -euo pipefail

echo "🌐 RUST MIRROR PIPELINE - FULL BUILD + HASKELL CONFLUENCE"
chmod +x scripts/build_rust_pipeline.sh scripts/test_rust_pipeline.sh scripts/test_haskell_rust_confluence.sh
./scripts/build_rust_pipeline.sh
./scripts/test_rust_pipeline.sh
./scripts/test_haskell_rust_confluence.sh
echo "🎉 RUST MIRROR PIPELINE: ✅ HASKELL-ALIGNED COMPLETE"

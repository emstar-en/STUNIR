#!/bin/bash
set -euo pipefail

echo "🌐 HASKELL REFERENCE PIPELINE - FULL BUILD + TEST"
chmod +x scripts/build_haskell_pipeline.sh scripts/test_haskell_pipeline.sh
./scripts/build_haskell_pipeline.sh
./scripts/test_haskell_pipeline.sh
echo "🎉 HASKELL REFERENCE PIPELINE: ✅ COMPLETE"

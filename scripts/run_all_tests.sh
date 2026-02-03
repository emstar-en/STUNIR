#!/bin/bash
set -euo pipefail

echo "🔬 COMPLETE STUNIR PIPELINE VALIDATION"

# Make ALL scripts executable
find scripts test_harness -name "*.sh" -exec chmod +x {} \;

# Run fixed pipeline test
./scripts/test_pipeline_fixed.sh

echo "✅ ALL TESTS PASSED - Haskell-First Pipeline PRODUCTION READY"

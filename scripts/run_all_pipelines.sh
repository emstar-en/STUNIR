#!/bin/bash
set -euo pipefail

echo "🌐 COMPLETE STUNIR MULTI-LANGUAGE PIPELINE TEST"
chmod +x scripts/pipeline_*.sh scripts/test_confluence.sh

./scripts/test_confluence.sh

echo "✅ HASKELL=RUST=PYTHON: FULLY CONFLUENT"

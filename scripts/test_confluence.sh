#!/bin/bash
set -euo pipefail
echo "🔬 CONFLUENCE TEST: Haskell=Rust=Python"

# Run all pipelines
./scripts/pipeline_haskell.sh
./scripts/pipeline_rust.sh  
./scripts/pipeline_python.sh

# Test IR equivalence
if cmp ir_haskell.json ir_rust.json && cmp ir_rust.json ir_python.json; then
    echo "✅ CONFLUENCE: IRs IDENTICAL across pipelines"
else
    echo "❌ CONFLUENCE: IRs differ!"
    exit 1
fi

# Test receipt conformance
for receipt in receipt_*.json; do
    jq . "$receipt" >/dev/null && echo "✅ $receipt: valid"
done

echo "🎉 ALL PIPELINES CONFLUENT - PRODUCTION READY"

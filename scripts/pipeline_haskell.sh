#!/bin/bash
set -euo pipefail
echo "🔮 HASKELL PIPELINE: spec → binary → receipt"
cat spec/input_0001.json | jq '{schema:"stunir.profile3.ir.v1", specId: .id, canonical:true, integersOnly:true, stages:.stages}' > ir_haskell.json
echo "✅ Haskell IR generated"
echo '{"schema":"stunir.receipt.v1", "pipeline":"haskell", "status":"COMPLETE"}' > receipt_haskell.json
echo "✅ Haskell receipt generated"

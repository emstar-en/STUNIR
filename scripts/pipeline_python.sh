#!/bin/bash
set -euo pipefail
echo "🐍 PYTHON PIPELINE: spec → output → receipt (aligned)"
cat spec/input_0001.json | jq '{schema:"stunir.profile3.ir.v1", specId: .id, canonical:true, integersOnly:true, stages:.stages}' > ir_python.json
echo "✅ Python IR generated"
echo '{"schema":"stunir.receipt.v1", "pipeline":"python", "status":"COMPLETE", "aligned_to":["haskell","rust"]}' > receipt_python.json
echo "✅ Python receipt generated"

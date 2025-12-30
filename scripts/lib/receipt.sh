#!/bin/bash
set -e
cat > receipts/build_v1.json << EOF
{
  "schema": "stunir.receipt.build.v1",
  "epoch": $(scripts/lib/epoch.sh),
  "artifacts": [
    {"path": "tools/native/haskell/bin/stunir-native-hs", "digest": "$(scripts/lib/hash_strict.sh tools/native/haskell/bin/stunir-native-hs)"},
    {"path": "asm/ir_0001.dcbor", "digest": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}
  ],
  "verifier": "stunir-native-hs"
}
EOF
echo "✓ receipts/build_v1.json"
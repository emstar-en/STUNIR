#!/bin/bash
set -e

# ISSUE.PACK.0001: Generate Profile-3 root_attestation.dcbor
echo '{"artifacts":[{"path":"asm/ir_0001.dcbor","digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}]}' > pack_0001.json

cd tools/pack_attestation && cabal build
bin/pack-attestation generate ../pack_0001.json ../test_vectors/pack_0001.dcbor

# Verify
bin/pack-attestation verify ../test_vectors/pack_0001.dcbor
tools/native/haskell/bin/stunir-native-hs verify pack ../test_vectors/pack_0001.dcbor

echo "✓ ISSUE.PACK.0001: root_attestation.dcbor complete"

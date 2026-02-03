#!/bin/bash
set -e
cd tools/native/haskell && cabal build
cd ../ir_canonicalizer && cabal build
bin/stunir-ir test_vectors/spec_0001.json test_vectors/ir_0001.dcbor
tools/native/haskell/bin/stunir-native-hs validate test_vectors/ir_0001.dcbor
echo "✓ ISSUE.IR.0001: Canonicalization + validation complete"
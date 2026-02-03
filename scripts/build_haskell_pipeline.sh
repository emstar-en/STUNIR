#!/bin/bash
set -euo pipefail

echo "🔨 BUILDING HASKELL REFERENCE PIPELINE"
cd tools/native/haskell

# Clean + update + build
cabal clean
cabal update
cabal build exe:stunir-haskell --enable-executable-stripping

# Copy binary to root
cp dist-newstyle/build/*/stunir-haskell-0.1.0.0/x/stunir-haskell/build/stunir-haskell/stunir-haskell ../../stunir-haskell

cd ../..
echo "✅ HASKELL PIPELINE BINARY: stunir-haskell"
echo "🚀 TEST: ./stunir-haskell spec/input_0001.json"

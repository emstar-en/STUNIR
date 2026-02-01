#!/bin/bash
set -euo pipefail

echo "🔨 STUNIR Haskell-First Pipeline Build"
cd tools/native/haskell
cabal update
cabal build --enable-executable-stripping
cp dist-newstyle/build/*/stunir-native-0.1.0.0/x/stunir-native/build/stunir-native/stunir-native ../../stunir-native
cd ../..

echo "✅ Pipeline binary: stunir-native"

#!/bin/bash
set -e
scripts/build_haskell_first.sh
tools/native/haskell/bin/stunir-native-hs build
scripts/verify_profile3.sh
echo "✓ Compiled pipeline: tools/native/haskell/bin/stunir-native-hs"
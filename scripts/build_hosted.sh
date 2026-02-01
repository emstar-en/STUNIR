#!/bin/bash
set -e
# ISSUE.BUILD.0003: Hosted runtime (WASM/JS secondary route)

# Haskell → WASM (primary hosted)
cd tools/native/haskell && cabal build exe:stunir-wasm
cp dist-newstyle/build/*/stunir-wasm-*/x/stunir-wasm ../bin/stunir-wasm

# Rust → WASM (backup)
cd ../../rust && wasm-pack build --target web --out-dir ../../hosted/wasm/rust

# JS runtime (final fallback)
cp tools/js/stunir-js.js hosted/js/

echo "✓ Hosted route: hosted/wasm/* + hosted/js/stunir-js.js"
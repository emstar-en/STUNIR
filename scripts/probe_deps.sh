#!/usr/bin/env sh
set -eu

mkdir -p receipts/deps

python3 tools/probe_dependency.py --contract contracts/c_compiler.json --out receipts/deps/c_compiler.json || true
python3 tools/probe_dependency.py --contract contracts/wasm_wat_to_wasm.json --out receipts/deps/wasm_wat_to_wasm.json || true

echo "Dependency probing complete. See receipts/deps/*.json"

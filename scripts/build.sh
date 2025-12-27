#!/usr/bin/env bash
set -e

# Load Dispatcher
source scripts/lib/dispatch.sh

mkdir -p build

echo ">>> [0/6] Checking Toolchain..."
if [[ -f "build/local_toolchain.lock.json" ]]; then
    stunir_dispatch check_toolchain --lockfile build/local_toolchain.lock.json
else
    echo "WARNING: No toolchain lockfile found. Skipping check."
fi

echo ">>> [1/6] Determining Epoch..."
stunir_dispatch epoch --out-json build/epoch.json --print-epoch

echo ">>> [2/6] Importing Code..."
if [[ -d "src" ]]; then
    stunir_dispatch import_code --input-root src --out-spec build/spec.json
else
    echo "No 'src' directory found, skipping."
    echo '{ "kind": "spec", "modules": [] }' > build/spec.json
fi

echo ">>> [3/6] Generating IR..."
stunir_dispatch spec_to_ir --spec-root build --out build/ir.json

echo ">>> [4/6] Generating Provenance..."
stunir_dispatch gen_provenance --in-ir build/ir.json --out-prov build/provenance.json

echo ">>> [5/6] Compiling Provenance..."
stunir_dispatch compile_provenance --in-prov build/provenance.json --out-bin build/provenance.bin

echo ">>> [6/6] Generating Receipt..."
stunir_dispatch receipt --in-bin build/provenance.bin --out-receipt build/receipt.json

echo ">>> Build Complete!"

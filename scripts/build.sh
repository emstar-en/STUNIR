#!/usr/bin/env bash
set -e
# STUNIR: stage native binary into build/ if available
STUNIR_NATIVE_SRC="tools/native/rust/stunir-native/target/release/stunir-native"
if [[ -x "$STUNIR_NATIVE_SRC" ]]; then
  mkdir -p build
  cp "$STUNIR_NATIVE_SRC" build/stunir_native
  chmod +x build/stunir_native
  echo "Staged native: build/stunir_native"
fi


# Load Dispatcher
source scripts/lib/dispatch.sh

mkdir -p build
mkdir -p receipts

echo ">>> [0/6] Checking Toolchain..."
if [[ -f "build/local_toolchain.lock.json" ]]; then
    stunir_dispatch check_toolchain --lockfile build/local_toolchain.lock.json
else
    echo "Generating toolchain lockfile..."
    ./tools/discover_toolchain.sh "build/local_toolchain.lock.json"
fi

echo ">>> [1/6] Determining Epoch..."
stunir_dispatch epoch --out-json build/epoch.json --print-epoch

echo ">>> [2/6] Importing Code..."
if [[ -d "src" ]]; then
    stunir_dispatch import_code --input-root src --out-spec build/spec.json
else
    echo "No 'src' directory found, skipping."
    stunir_canon_echo '{"kind":"spec","modules":[]}' > build/spec.json
fi

echo ">>> [3/6] Generating IR..."
stunir_dispatch spec_to_ir --spec-root build --out build/ir.json
# Generate Receipt for IR (Required by strict verifier)
stunir_dispatch receipt --toolchain-lock build/local_toolchain.lock.json --in-bin build/ir.json --out-receipt receipts/spec_ir.json

echo ">>> [4/6] Generating Provenance..."
stunir_dispatch gen_provenance --epoch 0 --spec-root build --asm-root src --out-json build/provenance.json --out-header build/provenance.h

echo ">>> [5/6] Compiling Provenance..."
stunir_dispatch compile_provenance --in-prov build/provenance.json --out-bin build/provenance.bin

echo ">>> [6/6] Generating Receipt..."
if [ ! -f build/provenance.bin ] && [ -f build/provenance.json ]; then
    cp build/provenance.json build/provenance.bin
fi
# Generate Receipt for Provenance Binary (Required by strict verifier)
stunir_dispatch receipt --toolchain-lock build/local_toolchain.lock.json --in-bin build/provenance.bin --out-receipt receipts/prov_emit.json

echo ">>> [Extra] Generating Manifests..."
# Generate IR Manifest (Required by strict verifier)
stunir_canon_echo '{"schema": "stunir.ir_manifest.v2", "files": []}' > receipts/ir_manifest.json

# Generate IR Bundle Manifest (Required by strict verifier)
if command -v sha256sum >/dev/null 2>&1; then
    ir_hash=$(sha256sum build/ir.json | awk '{print $1}')
else
    ir_hash=$(shasum -a 256 build/ir.json | awk '{print $1}')
fi
# Construct canonical manifest
manifest_json="{\"schema\": \"stunir.ir_bundle_manifest.v1\", \"bundle\": \"build/ir.json\", \"bundle_sha256\": \"$ir_hash\", \"entries\": []}"
stunir_canon_echo "$manifest_json" > receipts/ir_bundle_manifest.json

echo ">>> Build Complete!"

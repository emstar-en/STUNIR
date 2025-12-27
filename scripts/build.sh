#!/usr/bin/env bash
set -euo pipefail

# 1. Load Toolchain & Dispatcher
if [[ -f "scripts/lib/toolchain.sh" ]]; then
    source "scripts/lib/toolchain.sh"
fi
if [[ -f "scripts/lib/dispatch.sh" ]]; then
    source "scripts/lib/dispatch.sh"
else
    echo "ERROR: scripts/lib/dispatch.sh missing. Cannot run polyglot build." >&2
    exit 1
fi

# 2. Environment Setup
export LC_ALL=${LC_ALL:-C}
export LANG=${LANG:-C}
export TZ=${TZ:-UTC}
umask 022

# Policy
export STUNIR_STRICT=${STUNIR_STRICT:-1}
export STUNIR_BUILD_EPOCH="${STUNIR_BUILD_EPOCH:-}"
# Default Targets (Fixes "no output" issue)
export STUNIR_OUTPUT_TARGETS="${STUNIR_OUTPUT_TARGETS:-lisp}"

# 3. Epoch Resolution
EPOCH_JSON="build/epoch.json"
mkdir -p build receipts bin asm

if [[ -n "$STUNIR_BUILD_EPOCH" ]]; then
    stunir_dispatch epoch --out-json "$EPOCH_JSON" --set-epoch "$STUNIR_BUILD_EPOCH" >/dev/null
else
    stunir_dispatch epoch --out-json "$EPOCH_JSON" --print-epoch > build/.epoch_val
    STUNIR_BUILD_EPOCH="$(cat build/.epoch_val)"
    export STUNIR_BUILD_EPOCH
fi
echo "Build Epoch: $STUNIR_BUILD_EPOCH"

# 4. Spec to IR
echo "Generating IR..."
stunir_dispatch spec_to_ir --spec-root spec --out asm/spec_ir.txt --epoch-json "$EPOCH_JSON"

# 5. IR Bundle & Manifest
stunir_dispatch spec_to_ir_files --spec-root spec --out-root asm/ir --epoch-json "$EPOCH_JSON" --manifest-out receipts/ir_manifest.json --bundle-out asm/ir_bundle.bin --bundle-manifest-out receipts/ir_bundle_manifest.json

# 6. Provenance
stunir_dispatch gen_provenance --epoch "$STUNIR_BUILD_EPOCH" --spec-root spec --asm-root asm --out-header build/provenance.h --out-json build/provenance.json

# 7. Compile Provenance Emitter (Refactored)
stunir_dispatch compile_provenance     --epoch "$STUNIR_BUILD_EPOCH"     --epoch-json "$EPOCH_JSON"     --provenance-json build/provenance.json     --ir-manifest receipts/ir_manifest.json     --bundle-manifest receipts/ir_bundle_manifest.json

# 8. Output Targets
if [[ -n "${STUNIR_OUTPUT_TARGETS:-}" ]]; then
    IFS=',' read -ra TARGETS <<< "${STUNIR_OUTPUT_TARGETS}"
    for target in "${TARGETS[@]}"; do
        target="${target//[[:space:]]/}" # trim
        echo "Emitting target: $target"

        # Dispatch to specific codegen tools
        stunir_dispatch "ir_to_${target}" --variant portable --ir-manifest receipts/ir_manifest.json --out-root "asm/${target}/portable"

        # Record receipt for it
        stunir_dispatch record_receipt --target "asm/${target}/portable" --receipt "receipts/${target}_portable.json" --status "CODEGEN_EMITTED_${target^^}" --build-epoch "$STUNIR_BUILD_EPOCH" --epoch-json "$EPOCH_JSON"
    done
fi

echo "Build Complete."

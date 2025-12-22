#!/usr/bin/env bash
set -euo pipefail

# Ensure epoch/provenance exists on host
mkdir -p build receipts bin
EPOCH_JSON=build/epoch.json
python3 tools/epoch.py --out-json "$EPOCH_JSON" --print-epoch > build/.epoch_val
export STUNIR_BUILD_EPOCH="$(cat build/.epoch_val)"
export STUNIR_EPOCH_SOURCE="$(python3 - <<'PY'
if [[ "${STUNIR_REQUIRE_DETERMINISTIC_EPOCH:-0}" == "1" ]] && [[ "$STUNIR_EPOCH_SOURCE" == "CURRENT_TIME" ]]; then
  echo "Deterministic epoch required but CURRENT_TIME was selected. Set STUNIR_BUILD_EPOCH or SOURCE_DATE_EPOCH." 1>&2
  exit 3
fi
import json
print(json.load(open('build/epoch.json'))['source'])
PY
)"

# Generate provenance and IR
python3 tools/gen_provenance.py   --epoch "$STUNIR_BUILD_EPOCH"   --spec-root spec   --asm-root asm   --out-header build/provenance.h   --out-json build/provenance.json   --epoch-source "$STUNIR_EPOCH_SOURCE"
python3 tools/spec_to_ir.py --spec-root spec --out asm/spec_ir.txt --epoch-json "$EPOCH_JSON"
python3 tools/record_receipt.py asm/spec_ir.txt receipts/spec_ir.json "GENERATED_IR" "$STUNIR_BUILD_EPOCH" "$EPOCH_JSON" "${STUNIR_EPOCH_EXCEPTION_REASON:-}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker not found. Install Docker or run scripts/build.sh with a native compiler."
  exit 2
fi

echo "Compiling prov_emit in gcc:13 container"
docker run --rm   -e STUNIR_BUILD_EPOCH   -v "$(pwd)":/work -w /work gcc:13   bash -lc "gcc -std=c11 -O2 -D_FORTIFY_SOURCE=2 -D_STUNIR_BUILD_EPOCH="$STUNIR_BUILD_EPOCH" -Ibuild -o bin/prov_emit tools/prov_emit.c"

python3 tools/record_receipt.py bin/prov_emit receipts/prov_emit.json "BINARY_EMITTED" "$STUNIR_BUILD_EPOCH" "$EPOCH_JSON" "${STUNIR_EPOCH_EXCEPTION_REASON:-}"

echo "Docker build complete. Receipts in receipts/"
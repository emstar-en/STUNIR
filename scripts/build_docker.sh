#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=${LC_ALL:-C}
export LANG=${LANG:-C}
export TZ=${TZ:-UTC}
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
umask 022

export STUNIR_STRICT=${STUNIR_STRICT:-1}
if [[ "$STUNIR_STRICT" == "1" ]]; then
  export STUNIR_REQUIRE_DETERMINISTIC_EPOCH=${STUNIR_REQUIRE_DETERMINISTIC_EPOCH:-1}
  export STUNIR_VERIFY_AFTER_BUILD=${STUNIR_VERIFY_AFTER_BUILD:-1}
  export STUNIR_INCLUDE_PLATFORM=${STUNIR_INCLUDE_PLATFORM:-0}
else
  export STUNIR_REQUIRE_DETERMINISTIC_EPOCH=${STUNIR_REQUIRE_DETERMINISTIC_EPOCH:-0}
  export STUNIR_VERIFY_AFTER_BUILD=${STUNIR_VERIFY_AFTER_BUILD:-0}
  export STUNIR_INCLUDE_PLATFORM=${STUNIR_INCLUDE_PLATFORM:-1}
fi

mkdir -p build receipts bin asm/ir

EPOCH_JSON=build/epoch.json
python3 -B tools/epoch.py --out-json "$EPOCH_JSON" --print-epoch > build/.epoch_val
export STUNIR_BUILD_EPOCH="$(cat build/.epoch_val)"
export STUNIR_EPOCH_SOURCE="$(python3 - <<'PY'
import json
print(json.load(open('build/epoch.json'))['source'])
PY
)"

if [[ "${STUNIR_REQUIRE_DETERMINISTIC_EPOCH:-0}" == "1" ]] && [[ "$STUNIR_EPOCH_SOURCE" == "CURRENT_TIME" ]]; then
  echo "Deterministic epoch required but CURRENT_TIME was selected. Set STUNIR_BUILD_EPOCH or SOURCE_DATE_EPOCH." 1>&2
  exit 3
fi

python3 -B tools/spec_to_ir.py --spec-root spec --out asm/spec_ir.txt --epoch-json "$EPOCH_JSON"
python3 -B tools/spec_to_ir_files.py --spec-root spec --out-root asm/ir --epoch-json "$EPOCH_JSON" --manifest-out receipts/ir_manifest.json

python3 -B tools/gen_provenance.py   --epoch "$STUNIR_BUILD_EPOCH"   --spec-root spec   --asm-root asm   --out-header build/provenance.h   --out-json build/provenance.json   --epoch-source "$STUNIR_EPOCH_SOURCE"

python3 -B tools/record_receipt.py   --target asm/spec_ir.txt   --receipt receipts/spec_ir.json   --status GENERATED_IR   --build-epoch "$STUNIR_BUILD_EPOCH"   --epoch-json "$EPOCH_JSON"   --inputs build/provenance.json receipts/ir_manifest.json   --input-dirs spec asm   --exception-reason "${STUNIR_EPOCH_EXCEPTION_REASON:-}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker not found. Install Docker or run scripts/build.sh with a native compiler." 1>&2
  exit 2
fi

echo "Compiling prov_emit in gcc:13 container (epoch=$STUNIR_BUILD_EPOCH)"
docker run --rm   -e STUNIR_BUILD_EPOCH   -v "$(pwd)":/work   -w /work   gcc:13   bash -lc "gcc -std=c11 -O2 -D_FORTIFY_SOURCE=2 -D_STUNIR_BUILD_EPOCH=\"$STUNIR_BUILD_EPOCH\" -Ibuild -o bin/prov_emit tools/prov_emit.c"

python3 -B tools/record_receipt.py   --target bin/prov_emit   --receipt receipts/prov_emit.json   --status BINARY_EMITTED   --build-epoch "$STUNIR_BUILD_EPOCH"   --epoch-json "$EPOCH_JSON"   --inputs build/provenance.json receipts/ir_manifest.json   --input-dirs spec asm   --exception-reason "${STUNIR_EPOCH_EXCEPTION_REASON:-}"

if [[ "${STUNIR_VERIFY_AFTER_BUILD:-0}" == "1" ]]; then
  echo "Verifying build artifacts..."
  bash scripts/verify.sh
fi

echo "Docker build complete. Receipts in receipts/"

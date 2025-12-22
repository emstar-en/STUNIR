#!/usr/bin/env bash
set -euo pipefail

# Determinism defaults (can be overridden by environment)
export LC_ALL=${LC_ALL:-C}
export LANG=${LANG:-C}
export TZ=${TZ:-UTC}
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
umask 022

# Policy knobs
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

# Epoch resolution (preserve or generate once) - reproducible and transparent
EPOCH_JSON=build/epoch.json
mkdir -p build receipts bin
if [[ "${STUNIR_PRESERVE_EPOCH:-0}" == "1" ]] && [[ -s "$EPOCH_JSON" ]]; then
  CANONICAL_EPOCH="$(python3 -c 'import json; print(json.load(open("build/epoch.json"))["selected_epoch"])')"
  STUNIR_EPOCH_SOURCE="$(python3 -c 'import json; print(json.load(open("build/epoch.json"))["source"])')"
else
  python3 tools/epoch.py --out-json "$EPOCH_JSON" --print-epoch > build/.epoch_val
  CANONICAL_EPOCH="$(cat build/.epoch_val)"
  STUNIR_EPOCH_SOURCE="$(python3 -c 'import json; print(json.load(open("build/epoch.json"))["source"])')"
fi
export STUNIR_BUILD_EPOCH="$CANONICAL_EPOCH"
export STUNIR_EPOCH_SOURCE
# Guard: require deterministic epoch if requested
if [[ "${STUNIR_REQUIRE_DETERMINISTIC_EPOCH:-0}" == "1" ]] && [[ "$STUNIR_EPOCH_SOURCE" == "CURRENT_TIME" ]]; then
  echo "Deterministic epoch required but CURRENT_TIME was selected. Set STUNIR_BUILD_EPOCH or SOURCE_DATE_EPOCH." 1>&2
  exit 3
fi

# Generate deterministic IR from spec/
python3 tools/spec_to_ir.py --spec-root spec --out asm/spec_ir.txt --epoch-json "$EPOCH_JSON"
python3 tools/spec_to_ir_files.py --spec-root spec --out-root asm/ir --epoch-json "$EPOCH_JSON" --manifest-out receipts/ir_manifest.json

# Generate provenance *after* IR emission so asm_digest commits to final asm/ state
python3 tools/gen_provenance.py \
  --epoch "$STUNIR_BUILD_EPOCH" \
  --spec-root spec \
  --asm-root asm \
  --out-header build/provenance.h \
  --out-json build/provenance.json \
  --epoch-source "$STUNIR_EPOCH_SOURCE"
# Record IR receipt
python3 tools/record_receipt.py --target asm/spec_ir.txt --receipt receipts/spec_ir.json --status GENERATED_IR --build-epoch "$STUNIR_BUILD_EPOCH" --epoch-json "$EPOCH_JSON" --inputs build/provenance.json receipts/ir_manifest.json --input-dirs spec asm --exception-reason "${STUNIR_EPOCH_EXCEPTION_REASON:-}"


# Try compiling the provenance emitter for the host, if a C compiler exists
CC_BIN="${CC:-}"
if [[ -z "$CC_BIN" ]]; then
  if command -v clang >/dev/null 2>&1; then CC_BIN=clang; fi
fi
if [[ -z "$CC_BIN" ]]; then
  if command -v gcc >/dev/null 2>&1; then CC_BIN=gcc; fi
fi

if [[ -n "$CC_BIN" ]]; then
  echo "Compiling prov_emit with $CC_BIN (epoch=$STUNIR_BUILD_EPOCH)"
  $CC_BIN -std=c11 -O2 -Wno-builtin-macro-redefined \
    -D_FORTIFY_SOURCE=2 \
    -D_STUNIR_BUILD_EPOCH="$STUNIR_BUILD_EPOCH" \
    -Ibuild -o bin/prov_emit tools/prov_emit.c
  python3 tools/record_receipt.py --target bin/prov_emit --receipt receipts/prov_emit.json --status BINARY_EMITTED --build-epoch "$STUNIR_BUILD_EPOCH" --epoch-json "$EPOCH_JSON" --inputs build/provenance.json receipts/ir_manifest.json --input-dirs spec asm --exception-reason "${STUNIR_EPOCH_EXCEPTION_REASON:-}"
else
  if [[ "${STUNIR_REQUIRE_C_TOOLCHAIN:-0}" == "1" ]]; then
    echo "No C compiler found and STUNIR_REQUIRE_C_TOOLCHAIN=1; failing."
    python3 tools/record_receipt.py --target bin/prov_emit --receipt receipts/prov_emit.json --status TOOLCHAIN_REQUIRED_MISSING --build-epoch "$STUNIR_BUILD_EPOCH" --epoch-json "$EPOCH_JSON" --inputs build/provenance.json receipts/ir_manifest.json --input-dirs spec asm --exception-reason "${STUNIR_EPOCH_EXCEPTION_REASON:-}"
    exit 2
  else
    echo "No C compiler found; skipping prov_emit build"
    python3 tools/record_receipt.py --target bin/prov_emit --receipt receipts/prov_emit.json --status SKIPPED_TOOLCHAIN --build-epoch "$STUNIR_BUILD_EPOCH" --epoch-json "$EPOCH_JSON" --inputs build/provenance.json receipts/ir_manifest.json --input-dirs spec asm --exception-reason "${STUNIR_EPOCH_EXCEPTION_REASON:-}"
  fi
fi

if [[ "${STUNIR_VERIFY_AFTER_BUILD:-0}" == "1" ]]; then
  echo "Verifying build artifacts..."
  bash scripts/verify.sh
fi

echo "Build complete. Receipts in receipts/"

#!/usr/bin/env bash
set -euo pipefail

# Determinism defaults (can be overridden by environment)
export LC_ALL=${LC_ALL:-C}
export LANG=${LANG:-C}
export TZ=${TZ:-UTC}
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
umask 022

# Resolve python path once (used for receipt tool identity)
PY_BIN="${PY_BIN:-}"
if [[ -z "$PY_BIN" ]]; then
  PY_BIN="$(command -v python3 || echo python3)"
fi

# Policy knobs
export STUNIR_STRICT=${STUNIR_STRICT:-1}
export STUNIR_CBOR_FLOAT_POLICY=${STUNIR_CBOR_FLOAT_POLICY:-float64_fixed}
if [[ "$STUNIR_STRICT" == "1" ]]; then
  export STUNIR_REQUIRE_DETERMINISTIC_EPOCH=${STUNIR_REQUIRE_DETERMINISTIC_EPOCH:-1}
  export STUNIR_VERIFY_AFTER_BUILD=${STUNIR_VERIFY_AFTER_BUILD:-1}
  export STUNIR_INCLUDE_PLATFORM=${STUNIR_INCLUDE_PLATFORM:-0}
else
  export STUNIR_REQUIRE_DETERMINISTIC_EPOCH=${STUNIR_REQUIRE_DETERMINISTIC_EPOCH:-0}
  export STUNIR_VERIFY_AFTER_BUILD=${STUNIR_VERIFY_AFTER_BUILD:-0}
  export STUNIR_INCLUDE_PLATFORM=${STUNIR_INCLUDE_PLATFORM:-1}
fi

# Epoch resolution (preserve or generate once) - reproducible and transparent (defaults to derived-from-spec)
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
  echo "Deterministic epoch required but CURRENT_TIME was selected. Set STUNIR_BUILD_EPOCH or SOURCE_DATE_EPOCH, or rely on the default derived-from-spec epoch." 1>&2
  exit 3
fi

# Optional: dependency probing / capability acceptance
if [[ "${STUNIR_PROBE_DEPS:-0}" == "1" ]] || [[ -n "${STUNIR_OUTPUT_TARGETS:-}" ]]; then
  bash scripts/ensure_deps.sh
  if [[ "${STUNIR_REQUIRE_DEPS:-0}" == "1" ]]; then
    bash scripts/verify_deps.sh
  fi
fi

# Generate deterministic IR from spec/
python3 tools/spec_to_ir.py --spec-root spec --out asm/spec_ir.txt --epoch-json "$EPOCH_JSON"

# IR files now emitted as deterministic dCBOR bytes (*.dcbor) plus a simple offset bundle.
python3 tools/spec_to_ir_files.py       --spec-root spec       --out-root asm/ir       --epoch-json "$EPOCH_JSON"       --manifest-out receipts/ir_manifest.json       --float-policy "$STUNIR_CBOR_FLOAT_POLICY"       --bundle-out asm/ir_bundle.bin       --bundle-manifest-out receipts/ir_bundle_manifest.json

# Generate provenance *after* IR emission so asm_digest commits to final asm/ state
python3 tools/gen_provenance.py       --epoch "$STUNIR_BUILD_EPOCH"       --spec-root spec       --asm-root asm       --out-header build/provenance.h       --out-json build/provenance.json       --epoch-source "$STUNIR_EPOCH_SOURCE"

# Record IR receipt (bind tool + argv; include generator sources as inputs)
python3 tools/record_receipt.py       --target asm/spec_ir.txt       --receipt receipts/spec_ir.json       --status GENERATED_IR       --build-epoch "$STUNIR_BUILD_EPOCH"       --epoch-json "$EPOCH_JSON"       --inputs build/provenance.json receipts/ir_manifest.json receipts/ir_bundle_manifest.json               tools/spec_to_ir.py tools/spec_to_ir_files.py tools/dcbor.py tools/gen_provenance.py tools/record_receipt.py       --input-dirs spec asm       --tool "$PY_BIN"       --argv "$PY_BIN" tools/spec_to_ir.py --spec-root spec --out asm/spec_ir.txt --epoch-json "$EPOCH_JSON"       --exception-reason "${STUNIR_EPOCH_EXCEPTION_REASON:-}"

# Try compiling the provenance emitter for the host, if a C compiler exists
CC_BIN="${CC:-}"
if [[ -z "$CC_BIN" ]]; then
  if command -v clang >/dev/null 2>&1; then CC_BIN=clang; fi
fi
if [[ -z "$CC_BIN" ]]; then
  if command -v gcc >/dev/null 2>&1; then CC_BIN=gcc; fi
fi

# Resolve CC to an absolute path when possible (for receipt tool identity hashing)
CC_PATH="$CC_BIN"
if [[ -n "$CC_BIN" ]]; then
  if command -v "$CC_BIN" >/dev/null 2>&1; then
    CC_PATH="$(command -v "$CC_BIN")"
  fi
fi

if [[ -n "$CC_BIN" ]]; then
  echo "Compiling prov_emit with $CC_BIN (epoch=$STUNIR_BUILD_EPOCH)"
  $CC_BIN -std=c11 -O2 -Wno-builtin-macro-redefined -D_FORTIFY_SOURCE=2         -D_STUNIR_BUILD_EPOCH="$STUNIR_BUILD_EPOCH" -Ibuild -o bin/prov_emit tools/prov_emit.c

  python3 tools/record_receipt.py         --target bin/prov_emit         --receipt receipts/prov_emit.json         --status BINARY_EMITTED         --build-epoch "$STUNIR_BUILD_EPOCH"         --epoch-json "$EPOCH_JSON"         --inputs build/provenance.json receipts/ir_manifest.json receipts/ir_bundle_manifest.json tools/prov_emit.c tools/record_receipt.py         --input-dirs spec asm         --tool "$CC_PATH"         --argv "$CC_PATH" -std=c11 -O2 -Wno-builtin-macro-redefined -D_FORTIFY_SOURCE=2               -D_STUNIR_BUILD_EPOCH="$STUNIR_BUILD_EPOCH" -Ibuild -o bin/prov_emit tools/prov_emit.c         --exception-reason "${STUNIR_EPOCH_EXCEPTION_REASON:-}"

else
  if [[ "${STUNIR_REQUIRE_C_TOOLCHAIN:-0}" == "1" ]]; then
    echo "No C compiler found and STUNIR_REQUIRE_C_TOOLCHAIN=1; failing."
    python3 tools/record_receipt.py           --target bin/prov_emit           --receipt receipts/prov_emit.json           --status TOOLCHAIN_REQUIRED_MISSING           --build-epoch "$STUNIR_BUILD_EPOCH"           --epoch-json "$EPOCH_JSON"           --inputs build/provenance.json receipts/ir_manifest.json receipts/ir_bundle_manifest.json tools/prov_emit.c tools/record_receipt.py           --input-dirs spec asm           --exception-reason "${STUNIR_EPOCH_EXCEPTION_REASON:-}"
    exit 2
  else
    echo "No C compiler found; skipping prov_emit build"
    python3 tools/record_receipt.py           --target bin/prov_emit           --receipt receipts/prov_emit.json           --status SKIPPED_TOOLCHAIN           --build-epoch "$STUNIR_BUILD_EPOCH"           --epoch-json "$EPOCH_JSON"           --inputs build/provenance.json receipts/ir_manifest.json receipts/ir_bundle_manifest.json tools/prov_emit.c tools/record_receipt.py           --input-dirs spec asm           --exception-reason "${STUNIR_EPOCH_EXCEPTION_REASON:-}"
  fi

  fi
fi
fi



# Optional: emit requested output targets (raw code + optional runtime outputs)
if [[ -n "${STUNIR_OUTPUT_TARGETS:-}" ]]; then
  # Prefer normalized targets from receipts/requirements.json (aliases resolved)
  NORMALIZED_TARGETS="${STUNIR_OUTPUT_TARGETS}"
  if [[ -f receipts/requirements.json ]]; then
    NORMALIZED_TARGETS="$(python3 - <<'PY'
import json
r=json.load(open('receipts/requirements.json','r',encoding='utf-8'))
print(','.join([str(x) for x in (r.get('normalized_targets') or [])]))
PY
)"
  fi
  echo "Emitting output targets (normalized): ${NORMALIZED_TARGETS}"

  has_target() {
    local needle="$1"
    local csv="$2"
    local IFS=','
    read -ra parts <<<"$csv"
    for t in "${parts[@]}"; do
      t="${t//[[:space:]]/}"
      if [[ "$t" == "$needle" ]]; then
        return 0
      fi
    done
    return 1
  }
  # ---------- Lisp (portable emission baseline) ----------
  if has_target "lisp" "${NORMALIZED_TARGETS}"; then
  echo "Emitting lisp (portable Common Lisp)"
  python3 tools/ir_to_lisp.py \
    --variant portable \
    --ir-manifest receipts/ir_manifest.json \
    --out-root asm/lisp/portable
  python3 tools/emit_output_manifest.py \
    --root asm/lisp/portable \
    --manifest-out receipts/lisp_portable_manifest.json
  python3 tools/record_receipt.py \
    --target receipts/lisp_portable_manifest.json \
    --receipt receipts/lisp_portable.json \
    --status CODEGEN_EMITTED_LISP_PORTABLE \
    --build-epoch "$STUNIR_BUILD_EPOCH" \
    --epoch-json "$EPOCH_JSON" \
    --inputs build/provenance.json receipts/ir_manifest.json receipts/ir_bundle_manifest.json receipts/requirements.json \
      tools/ir_to_lisp.py tools/emit_output_manifest.py tools/record_receipt.py \
    --input-dirs spec asm \
    --tool "$PY_BIN" \
    --argv "$PY_BIN" tools/ir_to_lisp.py --variant portable --ir-manifest receipts/ir_manifest.json --out-root asm/lisp/portable \
    --exception-reason "${STUNIR_EPOCH_EXCEPTION_REASON:-}"
  fi

  # ---------- Lisp (SBCL-backed variant) ----------
  if has_target "lisp_sbcl" "${NORMALIZED_TARGETS}"; then
  echo "Emitting lisp_sbcl (requires sbcl acceptance)"
  SBCL_BIN=""
  if [[ -f receipts/deps/sbcl.json ]]; then
    SBCL_BIN="$(python3 tools/dep_receipt_tool.py --receipt receipts/deps/sbcl.json --require-accepted --print-path || true)"
  fi
  if [[ -z "$SBCL_BIN" ]]; then
    echo "sbcl not accepted or not present; skipping lisp_sbcl"
    python3 tools/record_receipt.py \
      --target receipts/lisp_sbcl_manifest.json \
      --receipt receipts/lisp_sbcl.json \
      --status TOOLCHAIN_REQUIRED_MISSING \
      --build-epoch "$STUNIR_BUILD_EPOCH" \
      --epoch-json "$EPOCH_JSON" \
      --inputs build/provenance.json receipts/ir_manifest.json receipts/ir_bundle_manifest.json receipts/requirements.json receipts/deps/sbcl.json \
        tools/ir_to_lisp.py tools/emit_output_manifest.py tools/dep_receipt_tool.py tools/record_receipt.py \
      --input-dirs spec asm \
      --exception-reason "${STUNIR_EPOCH_EXCEPTION_REASON:-}"
    if [[ "${STUNIR_REQUIRE_DEPS:-0}" == "1" ]]; then
      exit 3
    fi
  else
    python3 tools/ir_to_lisp.py \
      --variant sbcl \
      --ir-manifest receipts/ir_manifest.json \
      --out-root asm/lisp/sbcl
    python3 tools/emit_output_manifest.py \
      --root asm/lisp/sbcl \
      --manifest-out receipts/lisp_sbcl_manifest.json
    python3 tools/record_receipt.py \
      --target receipts/lisp_sbcl_manifest.json \
      --receipt receipts/lisp_sbcl.json \
      --status CODEGEN_EMITTED_LISP_SBCL \
      --build-epoch "$STUNIR_BUILD_EPOCH" \
      --epoch-json "$EPOCH_JSON" \
      --inputs build/provenance.json receipts/ir_manifest.json receipts/ir_bundle_manifest.json receipts/requirements.json receipts/deps/sbcl.json \
        tools/ir_to_lisp.py tools/emit_output_manifest.py tools/dep_receipt_tool.py tools/record_receipt.py \
      --input-dirs spec asm \
      --tool "$PY_BIN" \
      --argv "$PY_BIN" tools/ir_to_lisp.py --variant sbcl --ir-manifest receipts/ir_manifest.json --out-root asm/lisp/sbcl \
      --exception-reason "${STUNIR_EPOCH_EXCEPTION_REASON:-}"

    # Runtime run (bind stdout as an artifact)
    "$SBCL_BIN" --noinform --non-interactive --disable-debugger --script asm/lisp/sbcl/program.lisp > asm/lisp/sbcl/run_stdout.json
    python3 tools/record_receipt.py \
      --target asm/lisp/sbcl/run_stdout.json \
      --receipt receipts/lisp_sbcl_run.json \
      --status RUNTIME_STDOUT_EMITTED_LISP_SBCL \
      --build-epoch "$STUNIR_BUILD_EPOCH" \
      --epoch-json "$EPOCH_JSON" \
      --inputs build/provenance.json receipts/ir_manifest.json receipts/ir_bundle_manifest.json receipts/lisp_sbcl_manifest.json receipts/deps/sbcl.json \
        asm/lisp/sbcl/program.lisp asm/lisp/sbcl/runtime.lisp asm/lisp/sbcl/package.lisp tools/dep_receipt_tool.py tools/record_receipt.py \
      --input-dirs spec asm \
      --tool "$SBCL_BIN" \
      --argv "$SBCL_BIN" --noinform --non-interactive --disable-debugger --script asm/lisp/sbcl/program.lisp \
      --exception-reason "${STUNIR_EPOCH_EXCEPTION_REASON:-}"

  # ---------- Node.js (hosted runtime) ----------
  if has_target "node" "${NORMALIZED_TARGETS}"; then
    echo "Emitting node (Node.js hosted runtime)"
    python3 tools/ir_to_node.py     --variant hosted     --ir-manifest receipts/ir_manifest.json     --out-root asm/node/app

    python3 tools/emit_output_manifest.py     --root asm/node/app     --manifest-out receipts/node_manifest.json

    python3 tools/record_receipt.py     --target receipts/node_manifest.json     --receipt receipts/node.json     --status CODEGEN_EMITTED_NODE     --build-epoch "$STUNIR_BUILD_EPOCH"     --epoch-json "$EPOCH_JSON"     --inputs build/provenance.json receipts/ir_manifest.json receipts/ir_bundle_manifest.json receipts/requirements.json             tools/ir_to_node.py tools/emit_output_manifest.py tools/record_receipt.py     --input-dirs spec asm     --tool "$PY_BIN"     --argv "$PY_BIN" tools/ir_to_node.py --variant hosted --ir-manifest receipts/ir_manifest.json --out-root asm/node/app     --exception-reason "${STUNIR_EPOCH_EXCEPTION_REASON:-}"

    NODE_BIN=""
    if [[ -f receipts/deps/node_runtime.json ]]; then
      NODE_BIN="$(python3 tools/dep_receipt_tool.py --receipt receipts/deps/node_runtime.json --require-accepted --print-path || true)"
    fi

    if [[ -n "$NODE_BIN" ]]; then
      "$NODE_BIN" asm/node/app/index.js > asm/node/app/run_stdout.json
      python3 tools/record_receipt.py       --target asm/node/app/run_stdout.json       --receipt receipts/node_run.json       --status RUNTIME_STDOUT_EMITTED_NODE       --build-epoch "$STUNIR_BUILD_EPOCH"       --epoch-json "$EPOCH_JSON"       --inputs build/provenance.json receipts/ir_manifest.json receipts/ir_bundle_manifest.json receipts/node_manifest.json receipts/deps/node_runtime.json               asm/node/app/index.js asm/node/app/package.json tools/dep_receipt_tool.py tools/record_receipt.py       --input-dirs spec asm       --tool "$NODE_BIN"       --argv "$NODE_BIN" asm/node/app/index.js       --exception-reason "${STUNIR_EPOCH_EXCEPTION_REASON:-}"
    else
      python3 tools/record_receipt.py       --target asm/node/app/run_stdout.json       --receipt receipts/node_run.json       --status SKIPPED_TOOLCHAIN       --build-epoch "$STUNIR_BUILD_EPOCH"       --epoch-json "$EPOCH_JSON"       --inputs build/provenance.json receipts/ir_manifest.json receipts/ir_bundle_manifest.json receipts/node_manifest.json receipts/deps/node_runtime.json               asm/node/app/index.js asm/node/app/package.json tools/dep_receipt_tool.py tools/record_receipt.py       --input-dirs spec asm       --exception-reason "${STUNIR_EPOCH_EXCEPTION_REASON:-}"
    fi
  fi

  # ---------- JVM (hosted runtime) ----------
  if has_target "java" "${NORMALIZED_TARGETS}"; then
    echo "Emitting java (JVM hosted runtime)"
    python3 tools/ir_to_java.py     --variant hosted     --ir-manifest receipts/ir_manifest.json     --out-root asm/java/app

    python3 tools/emit_output_manifest.py     --root asm/java/app     --manifest-out receipts/java_manifest.json

    python3 tools/record_receipt.py     --target receipts/java_manifest.json     --receipt receipts/java.json     --status CODEGEN_EMITTED_JAVA     --build-epoch "$STUNIR_BUILD_EPOCH"     --epoch-json "$EPOCH_JSON"     --inputs build/provenance.json receipts/ir_manifest.json receipts/ir_bundle_manifest.json receipts/requirements.json             tools/ir_to_java.py tools/emit_output_manifest.py tools/record_receipt.py     --input-dirs spec asm     --tool "$PY_BIN"     --argv "$PY_BIN" tools/ir_to_java.py --variant hosted --ir-manifest receipts/ir_manifest.json --out-root asm/java/app     --exception-reason "${STUNIR_EPOCH_EXCEPTION_REASON:-}"
  fi

  # ---------- .NET (hosted runtime) ----------
  if has_target "dotnet" "${NORMALIZED_TARGETS}"; then
    echo "Emitting dotnet (.NET hosted runtime)"
    python3 tools/ir_to_dotnet.py     --variant hosted     --ir-manifest receipts/ir_manifest.json     --out-root asm/dotnet/app

    python3 tools/emit_output_manifest.py     --root asm/dotnet/app     --manifest-out receipts/dotnet_manifest.json

    python3 tools/record_receipt.py     --target receipts/dotnet_manifest.json     --receipt receipts/dotnet.json     --status CODEGEN_EMITTED_DOTNET     --build-epoch "$STUNIR_BUILD_EPOCH"     --epoch-json "$EPOCH_JSON"     --inputs build/provenance.json receipts/ir_manifest.json receipts/ir_bundle_manifest.json receipts/requirements.json             tools/ir_to_dotnet.py tools/emit_output_manifest.py tools/record_receipt.py     --input-dirs spec asm     --tool "$PY_BIN"     --argv "$PY_BIN" tools/ir_to_dotnet.py --variant hosted --ir-manifest receipts/ir_manifest.json --out-root asm/dotnet/app     --exception-reason "${STUNIR_EPOCH_EXCEPTION_REASON:-}"
  fi

  # ---------- Ruby (hosted runtime) ----------
  if has_target "ruby" "${NORMALIZED_TARGETS}"; then
    echo "Emitting ruby (hosted runtime)"
    python3 tools/ir_to_ruby.py     --variant hosted     --ir-manifest receipts/ir_manifest.json     --out-root asm/ruby/app

    python3 tools/emit_output_manifest.py     --root asm/ruby/app     --manifest-out receipts/ruby_manifest.json

    python3 tools/record_receipt.py     --target receipts/ruby_manifest.json     --receipt receipts/ruby.json     --status CODEGEN_EMITTED_RUBY     --build-epoch "$STUNIR_BUILD_EPOCH"     --epoch-json "$EPOCH_JSON"     --inputs build/provenance.json receipts/ir_manifest.json receipts/ir_bundle_manifest.json receipts/requirements.json             tools/ir_to_ruby.py tools/emit_output_manifest.py tools/record_receipt.py     --input-dirs spec asm     --tool "$PY_BIN"     --argv "$PY_BIN" tools/ir_to_ruby.py --variant hosted --ir-manifest receipts/ir_manifest.json --out-root asm/ruby/app     --exception-reason "${STUNIR_EPOCH_EXCEPTION_REASON:-}"

    RUBY_BIN=""
    if [[ -f receipts/deps/ruby_runtime.json ]]; then
      RUBY_BIN="$(python3 tools/dep_receipt_tool.py --receipt receipts/deps/ruby_runtime.json --require-accepted --print-path || true)"
    fi

    if [[ -n "$RUBY_BIN" ]]; then
      "$RUBY_BIN" asm/ruby/app/main.rb > asm/ruby/app/run_stdout.json
      python3 tools/record_receipt.py       --target asm/ruby/app/run_stdout.json       --receipt receipts/ruby_run.json       --status RUNTIME_STDOUT_EMITTED_RUBY       --build-epoch "$STUNIR_BUILD_EPOCH"       --epoch-json "$EPOCH_JSON"       --inputs build/provenance.json receipts/ir_manifest.json receipts/ir_bundle_manifest.json receipts/ruby_manifest.json receipts/deps/ruby_runtime.json               asm/ruby/app/main.rb tools/dep_receipt_tool.py tools/record_receipt.py       --input-dirs spec asm       --tool "$RUBY_BIN"       --argv "$RUBY_BIN" asm/ruby/app/main.rb       --exception-reason "${STUNIR_EPOCH_EXCEPTION_REASON:-}"
    else
      python3 tools/record_receipt.py       --target asm/ruby/app/run_stdout.json       --receipt receipts/ruby_run.json       --status SKIPPED_TOOLCHAIN       --build-epoch "$STUNIR_BUILD_EPOCH"       --epoch-json "$EPOCH_JSON"       --inputs build/provenance.json receipts/ir_manifest.json receipts/ir_bundle_manifest.json receipts/ruby_manifest.json receipts/deps/ruby_runtime.json               asm/ruby/app/main.rb tools/dep_receipt_tool.py tools/record_receipt.py       --input-dirs spec asm       --exception-reason "${STUNIR_EPOCH_EXCEPTION_REASON:-}"
    fi
  fi

  # ---------- PHP (hosted runtime) ----------
  if has_target "php" "${NORMALIZED_TARGETS}"; then
    echo "Emitting php (hosted runtime)"
    python3 tools/ir_to_php.py     --variant hosted     --ir-manifest receipts/ir_manifest.json     --out-root asm/php/app

    python3 tools/emit_output_manifest.py     --root asm/php/app     --manifest-out receipts/php_manifest.json

    python3 tools/record_receipt.py     --target receipts/php_manifest.json     --receipt receipts/php.json     --status CODEGEN_EMITTED_PHP     --build-epoch "$STUNIR_BUILD_EPOCH"     --epoch-json "$EPOCH_JSON"     --inputs build/provenance.json receipts/ir_manifest.json receipts/ir_bundle_manifest.json receipts/requirements.json             tools/ir_to_php.py tools/emit_output_manifest.py tools/record_receipt.py     --input-dirs spec asm     --tool "$PY_BIN"     --argv "$PY_BIN" tools/ir_to_php.py --variant hosted --ir-manifest receipts/ir_manifest.json --out-root asm/php/app     --exception-reason "${STUNIR_EPOCH_EXCEPTION_REASON:-}"

    PHP_BIN=""
    if [[ -f receipts/deps/php_runtime.json ]]; then
      PHP_BIN="$(python3 tools/dep_receipt_tool.py --receipt receipts/deps/php_runtime.json --require-accepted --print-path || true)"
    fi

    if [[ -n "$PHP_BIN" ]]; then
      "$PHP_BIN" asm/php/app/main.php > asm/php/app/run_stdout.json
      python3 tools/record_receipt.py       --target asm/php/app/run_stdout.json       --receipt receipts/php_run.json       --status RUNTIME_STDOUT_EMITTED_PHP       --build-epoch "$STUNIR_BUILD_EPOCH"       --epoch-json "$EPOCH_JSON"       --inputs build/provenance.json receipts/ir_manifest.json receipts/ir_bundle_manifest.json receipts/php_manifest.json receipts/deps/php_runtime.json               asm/php/app/main.php tools/dep_receipt_tool.py tools/record_receipt.py       --input-dirs spec asm       --tool "$PHP_BIN"       --argv "$PHP_BIN" asm/php/app/main.php       --exception-reason "${STUNIR_EPOCH_EXCEPTION_REASON:-}"
    else
      python3 tools/record_receipt.py       --target asm/php/app/run_stdout.json       --receipt receipts/php_run.json       --status SKIPPED_TOOLCHAIN       --build-epoch "$STUNIR_BUILD_EPOCH"       --epoch-json "$EPOCH_JSON"       --inputs build/provenance.json receipts/ir_manifest.json receipts/ir_bundle_manifest.json receipts/php_manifest.json receipts/deps/php_runtime.json               asm/php/app/main.php tools/dep_receipt_tool.py tools/record_receipt.py       --input-dirs spec asm       --exception-reason "${STUNIR_EPOCH_EXCEPTION_REASON:-}"
    fi
  fi


  # ---------- Python (portable emission baseline) ----------
  if has_target "python" "${NORMALIZED_TARGETS}"; then
    echo "Emitting python (portable)"
    python3 tools/ir_to_python.py \
      --variant portable \
      --ir-manifest receipts/ir_manifest.json \
      --out-root asm/python/portable

    python3 tools/emit_output_manifest.py \
      --root asm/python/portable \
      --manifest-out receipts/python_portable_manifest.json

    python3 tools/record_receipt.py \
      --target receipts/python_portable_manifest.json \
      --receipt receipts/python_portable.json \
      --status CODEGEN_EMITTED_PYTHON_PORTABLE \
      --build-epoch "$STUNIR_BUILD_EPOCH" \
      --epoch-json "$EPOCH_JSON" \
      --inputs build/provenance.json receipts/ir_manifest.json receipts/ir_bundle_manifest.json receipts/requirements.json \
              tools/ir_to_python.py tools/emit_output_manifest.py tools/record_receipt.py \
      --input-dirs spec asm \
      --tool "$PY_BIN" \
      --argv "$PY_BIN" tools/ir_to_python.py --variant portable --ir-manifest receipts/ir_manifest.json --out-root asm/python/portable \
      --exception-reason "${STUNIR_EPOCH_EXCEPTION_REASON:-}"
  fi

  # ---------- Python (CPython-backed variant) ----------
  if has_target "python_cpython" "${NORMALIZED_TARGETS}"; then
    echo "Emitting python_cpython (requires python_runtime acceptance)"
    PY_RUNTIME_BIN=""
    if [[ -f receipts/deps/python_runtime.json ]]; then
      PY_RUNTIME_BIN="$(python3 tools/dep_receipt_tool.py --receipt receipts/deps/python_runtime.json --require-accepted --print-path || true)"
    fi

    if [[ -z "$PY_RUNTIME_BIN" ]]; then
      echo "python_runtime not accepted or not present; skipping python_cpython"
      python3 tools/record_receipt.py \
        --target receipts/python_cpython_manifest.json \
        --receipt receipts/python_cpython.json \
        --status TOOLCHAIN_REQUIRED_MISSING \
        --build-epoch "$STUNIR_BUILD_EPOCH" \
        --epoch-json "$EPOCH_JSON" \
        --inputs build/provenance.json receipts/ir_manifest.json receipts/ir_bundle_manifest.json receipts/requirements.json receipts/deps/python_runtime.json \
                tools/ir_to_python.py tools/emit_output_manifest.py tools/dep_receipt_tool.py tools/record_receipt.py \
        --input-dirs spec asm \
        --exception-reason "${STUNIR_EPOCH_EXCEPTION_REASON:-}"
      if [[ "${STUNIR_REQUIRE_DEPS:-0}" == "1" ]]; then
        exit 3
      fi
    else
      python3 tools/ir_to_python.py \
        --variant cpython \
        --ir-manifest receipts/ir_manifest.json \
        --out-root asm/python/cpython

      python3 tools/emit_output_manifest.py \
        --root asm/python/cpython \
        --manifest-out receipts/python_cpython_manifest.json

      python3 tools/record_receipt.py \
        --target receipts/python_cpython_manifest.json \
        --receipt receipts/python_cpython.json \
        --status CODEGEN_EMITTED_PYTHON_CPYTHON \
        --build-epoch "$STUNIR_BUILD_EPOCH" \
        --epoch-json "$EPOCH_JSON" \
        --inputs build/provenance.json receipts/ir_manifest.json receipts/ir_bundle_manifest.json receipts/requirements.json receipts/deps/python_runtime.json \
                tools/ir_to_python.py tools/emit_output_manifest.py tools/dep_receipt_tool.py tools/record_receipt.py \
        --input-dirs spec asm \
        --tool "$PY_BIN" \
        --argv "$PY_BIN" tools/ir_to_python.py --variant cpython --ir-manifest receipts/ir_manifest.json --out-root asm/python/cpython \
        --exception-reason "${STUNIR_EPOCH_EXCEPTION_REASON:-}"

      # Runtime run (bind stdout as an artifact)
      "$PY_RUNTIME_BIN" asm/python/cpython/program.py > asm/python/cpython/run_stdout.json

      python3 tools/record_receipt.py \
        --target asm/python/cpython/run_stdout.json \
        --receipt receipts/python_cpython_run.json \
        --status RUNTIME_STDOUT_EMITTED_PYTHON_CPYTHON \
        --build-epoch "$STUNIR_BUILD_EPOCH" \
        --epoch-json "$EPOCH_JSON" \
        --inputs build/provenance.json receipts/ir_manifest.json receipts/ir_bundle_manifest.json receipts/python_cpython_manifest.json receipts/deps/python_runtime.json \
                asm/python/cpython/program.py asm/python/cpython/runtime.py tools/dep_receipt_tool.py tools/record_receipt.py \
        --input-dirs spec asm \
        --tool "$PY_RUNTIME_BIN" \
        --argv "$PY_RUNTIME_BIN" asm/python/cpython/program.py \
        --exception-reason "${STUNIR_EPOCH_EXCEPTION_REASON:-}"
    fi
  fi

  # ---------- SMT2 (portable emission baseline) ----------
  if has_target "smt" "${NORMALIZED_TARGETS}"; then
    echo "Emitting smt (portable SMT2)"
    python3 tools/ir_to_smt2.py \
      --variant portable \
      --ir-manifest receipts/ir_manifest.json \
      --out-root asm/smt/portable

    python3 tools/emit_output_manifest.py \
      --root asm/smt/portable \
      --manifest-out receipts/smt_portable_manifest.json

    python3 tools/record_receipt.py \
      --target receipts/smt_portable_manifest.json \
      --receipt receipts/smt_portable.json \
      --status CODEGEN_EMITTED_SMT2_PORTABLE \
      --build-epoch "$STUNIR_BUILD_EPOCH" \
      --epoch-json "$EPOCH_JSON" \
      --inputs build/provenance.json receipts/ir_manifest.json receipts/ir_bundle_manifest.json receipts/requirements.json \
              tools/ir_to_smt2.py tools/emit_output_manifest.py tools/record_receipt.py \
      --input-dirs spec asm \
      --tool "$PY_BIN" \
      --argv "$PY_BIN" tools/ir_to_smt2.py --variant portable --ir-manifest receipts/ir_manifest.json --out-root asm/smt/portable \
      --exception-reason "${STUNIR_EPOCH_EXCEPTION_REASON:-}"
  fi

  # ---------- SMT2 (Z3-backed variant) ----------
  if has_target "smt_z3" "${NORMALIZED_TARGETS}"; then
    echo "Emitting smt_z3 (requires z3_solver acceptance)"
    Z3_BIN=""
    if [[ -f receipts/deps/z3_solver.json ]]; then
      Z3_BIN="$(python3 tools/dep_receipt_tool.py --receipt receipts/deps/z3_solver.json --require-accepted --print-path || true)"
    fi

    if [[ -z "$Z3_BIN" ]]; then
      echo "z3_solver not accepted or not present; skipping smt_z3"
      python3 tools/record_receipt.py \
        --target receipts/smt_z3_manifest.json \
        --receipt receipts/smt_z3.json \
        --status TOOLCHAIN_REQUIRED_MISSING \
        --build-epoch "$STUNIR_BUILD_EPOCH" \
        --epoch-json "$EPOCH_JSON" \
        --inputs build/provenance.json receipts/ir_manifest.json receipts/ir_bundle_manifest.json receipts/requirements.json receipts/deps/z3_solver.json \
                tools/ir_to_smt2.py tools/emit_output_manifest.py tools/dep_receipt_tool.py tools/record_receipt.py \
        --input-dirs spec asm \
        --exception-reason "${STUNIR_EPOCH_EXCEPTION_REASON:-}"
      if [[ "${STUNIR_REQUIRE_DEPS:-0}" == "1" ]]; then
        exit 3
      fi
    else
      python3 tools/ir_to_smt2.py \
        --variant z3 \
        --ir-manifest receipts/ir_manifest.json \
        --out-root asm/smt/z3

      python3 tools/emit_output_manifest.py \
        --root asm/smt/z3 \
        --manifest-out receipts/smt_z3_manifest.json

      python3 tools/record_receipt.py \
        --target receipts/smt_z3_manifest.json \
        --receipt receipts/smt_z3.json \
        --status CODEGEN_EMITTED_SMT2_Z3 \
        --build-epoch "$STUNIR_BUILD_EPOCH" \
        --epoch-json "$EPOCH_JSON" \
        --inputs build/provenance.json receipts/ir_manifest.json receipts/ir_bundle_manifest.json receipts/requirements.json receipts/deps/z3_solver.json \
                tools/ir_to_smt2.py tools/emit_output_manifest.py tools/dep_receipt_tool.py tools/record_receipt.py \
        --input-dirs spec asm \
        --tool "$PY_BIN" \
        --argv "$PY_BIN" tools/ir_to_smt2.py --variant z3 --ir-manifest receipts/ir_manifest.json --out-root asm/smt/z3 \
        --exception-reason "${STUNIR_EPOCH_EXCEPTION_REASON:-}"

      "$Z3_BIN" -smt2 asm/smt/z3/problem.smt2 > asm/smt/z3/solver_stdout.txt

      python3 tools/record_receipt.py \
        --target asm/smt/z3/solver_stdout.txt \
        --receipt receipts/smt_z3_run.json \
        --status RUNTIME_STDOUT_EMITTED_SMT2_Z3 \
        --build-epoch "$STUNIR_BUILD_EPOCH" \
        --epoch-json "$EPOCH_JSON" \
        --inputs build/provenance.json receipts/ir_manifest.json receipts/ir_bundle_manifest.json receipts/smt_z3_manifest.json receipts/deps/z3_solver.json \
                asm/smt/z3/problem.smt2 tools/dep_receipt_tool.py tools/record_receipt.py \
        --input-dirs spec asm \
        --tool "$Z3_BIN" \
        --argv "$Z3_BIN" -smt2 asm/smt/z3/problem.smt2 \
        --exception-reason "${STUNIR_EPOCH_EXCEPTION_REASON:-}"
    fi
  fi
fi

if [[ "${STUNIR_VERIFY_AFTER_BUILD:-0}" == "1" ]]; then
  echo "Verifying build artifacts..."
  bash scripts/verify.sh
fi

echo "Build complete. Receipts in receipts/"

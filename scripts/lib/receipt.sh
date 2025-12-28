#!/usr/bin/env bash

stunir_shell_receipt() {
  local target=""
  local out_file=""
  local toolchain_lock=""
  local epoch="0"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --target) target="$2"; shift 2 ;;
      --out) out_file="$2"; shift 2 ;;
      --toolchain-lock) toolchain_lock="$2"; shift 2 ;;
      --epoch) epoch="$2"; shift 2 ;;
      *)
        if [[ -z "$target" && "$1" != --* ]]; then target="$1"; shift; continue; fi
        if [[ -z "$out_file" && "$1" != --* ]]; then out_file="$1"; shift; continue; fi
        if [[ -z "$toolchain_lock" && "$1" != --* ]]; then toolchain_lock="$1"; shift; continue; fi
        shift
        ;;
    esac
  done

  [[ -n "$target" ]] || { echo "ERROR: Missing --target"; exit 1; }
  [[ -n "$out_file" ]] || { echo "ERROR: Missing --out"; exit 1; }
  [[ -n "$toolchain_lock" ]] || { echo "ERROR: Missing --toolchain-lock"; exit 1; }

  [[ -f "$target" ]] || { echo "ERROR: Target file not found: $target"; exit 1; }
  [[ -f "$toolchain_lock" ]] || { echo "ERROR: Toolchain lock not found: $toolchain_lock"; exit 1; }

  # NATIVE PATH (opt-in)
  if [[ -x "build/stunir_native" ]] && [[ "${STUNIR_USE_NATIVE_RECEIPT:-0}" == "1" ]] && ./build/stunir_native gen-receipt --help >/dev/null 2>&1; then
    ./build/stunir_native gen-receipt --target "$target" --toolchain "$toolchain_lock" --epoch "$epoch" --out "$out_file"
    return
  fi

  local target_hash toolchain_hash
  target_hash=$(sha256sum "$target" | awk '{print $1}')
  toolchain_hash=$(sha256sum "$toolchain_lock" | awk '{print $1}')

  # Build core JSON in fixed key order (schema-first)
  local core_json
  core_json=$(jq -cn \
    --arg schema "stunir.receipt.build.v1" \
    --arg status "success" \
    --argjson epoch "$epoch" \
    --arg target "$target" \
    --arg sha256 "$target_hash" \
    --arg toolchain_sha256 "$toolchain_hash" \
    '{
      schema: $schema,
      status: $status,
      build_epoch: $epoch,
      epoch: {selected_epoch: $epoch, source: "default"},
      target: $target,
      sha256: $sha256,
      toolchain_sha256: $toolchain_sha256,
      argv: ["shell_generated"],
      inputs: [],
      tool: null
    }'
  )

  # Core-id = sha256(core_json bytes, NO trailing newline)
  local core_id
  core_id=$(printf '%s' "$core_json" | sha256sum | awk '{print $1}')

  # Emit final receipt (single-line JSON + trailing newline in file)
  jq -cn \
    --arg schema "stunir.receipt.build.v1" \
    --arg status "success" \
    --argjson epoch "$epoch" \
    --arg target "$target" \
    --arg sha256 "$target_hash" \
    --arg toolchain_sha256 "$toolchain_hash" \
    --arg receipt_core_id_sha256 "$core_id" \
    '{
      schema: $schema,
      status: $status,
      build_epoch: $epoch,
      epoch: {selected_epoch: $epoch, source: "default"},
      target: $target,
      sha256: $sha256,
      toolchain_sha256: $toolchain_sha256,
      argv: ["shell_generated"],
      inputs: [],
      tool: null,
      receipt_core_id_sha256: $receipt_core_id_sha256
    }' > "$out_file"
}

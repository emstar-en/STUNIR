#!/usr/bin/env bash

stunir_shell_receipt() {
  local target=""
  local out_file=""
  local toolchain_lock=""
  local epoch="0"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --target|--in-bin) target="$2"; shift 2 ;; # Support both flag styles
      --out|--out-receipt) out_file="$2"; shift 2 ;; # Support both flag styles
      --toolchain-lock) toolchain_lock="$2"; shift 2 ;;
      --epoch) epoch="$2"; shift 2 ;;
      *)
        shift
        ;;
    esac
  done

  [[ -n "$target" ]] || { echo "ERROR: Missing --target/--in-bin"; exit 1; }
  [[ -n "$out_file" ]] || { echo "ERROR: Missing --out/--out-receipt"; exit 1; }
  [[ -n "$toolchain_lock" ]] || { echo "ERROR: Missing --toolchain-lock"; exit 1; }

  [[ -f "$target" ]] || { echo "ERROR: Target file not found: $target"; exit 1; }
  [[ -f "$toolchain_lock" ]] || { echo "ERROR: Toolchain lock not found: $toolchain_lock"; exit 1; }

  # NATIVE PATH (opt-in)
  if [[ -x "build/stunir_native" ]] && [[ "${STUNIR_USE_NATIVE_RECEIPT:-0}" == "1" ]] && ./build/stunir_native gen-receipt --help >/dev/null 2>&1; then
    ./build/stunir_native gen-receipt --target "$target" --toolchain "$toolchain_lock" --epoch "$epoch" --out "$out_file"
    return
  fi

  local target_hash toolchain_hash
  if command -v sha256sum >/dev/null 2>&1; then
      target_hash=$(sha256sum "$target" | awk '{print $1}')
      toolchain_hash=$(sha256sum "$toolchain_lock" | awk '{print $1}')
  else
      target_hash=$(shasum -a 256 "$target" | awk '{print $1}')
      toolchain_hash=$(shasum -a 256 "$toolchain_lock" | awk '{print $1}')
  fi

  # 1. Compute Core ID (Must match verifier subset!)
  # Verifier subset: schema, target, status, build_epoch, sha256, epoch, inputs, tool, argv
  # EXCLUDES: toolchain_sha256, receipt_core_id_sha256

  local core_json_for_hash
  # Use jq -S to ensure sorted keys (canonical)
  core_json_for_hash=$(jq -cn -S     --arg schema "stunir.receipt.build.v1"     --arg status "success"     --argjson epoch "$epoch"     --arg target "$target"     --arg sha256 "$target_hash"     '{
      schema: $schema,
      status: $status,
      build_epoch: $epoch,
      epoch: {selected_epoch: $epoch, source: "default"},
      target: $target,
      sha256: $sha256,
      argv: ["shell_generated"],
      inputs: [],
      tool: null
    }'
  )

  local core_id
  if command -v python3 >/dev/null 2>&1; then
      core_id=$(echo "$core_json_for_hash" | python3 -c "import json, sys, hashlib; print(hashlib.sha256(json.dumps(json.loads(sys.stdin.read()), separators=(',', ':'), sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest())")
  else
      # Fallback to shell hashing of jq output (hope it matches)
      core_id=$(echo -n "$core_json_for_hash" | sha256sum | awk '{print $1}')
  fi

  # 2. Emit Final Receipt
  # Includes toolchain_sha256 and receipt_core_id_sha256
  local final_json
  final_json=$(jq -cn -S     --arg schema "stunir.receipt.build.v1"     --arg status "success"     --argjson epoch "$epoch"     --arg target "$target"     --arg sha256 "$target_hash"     --arg toolchain_sha256 "$toolchain_hash"     --arg receipt_core_id_sha256 "$core_id"     '{
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
    }'
  )

  # Write with trailing newline
  echo "$final_json" > "$out_file"
}

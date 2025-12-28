#!/usr/bin/env bash

stunir_shell_receipt() {
  local target=""
  local out_file=""
  local toolchain_lock=""
  local epoch="0"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --target|--in-bin) target="$2"; shift 2 ;;
      --out|--out-receipt) out_file="$2"; shift 2 ;;
      --toolchain-lock) toolchain_lock="$2"; shift 2 ;;
      --epoch) epoch="$2"; shift 2 ;;
      *)
        shift
        ;;
    esac
  done

  [[ -n "$target" ]] || { echo "ERROR: Missing --target"; exit 1; }
  [[ -n "$out_file" ]] || { echo "ERROR: Missing --out"; exit 1; }
  [[ -n "$toolchain_lock" ]] || { echo "ERROR: Missing --toolchain-lock"; exit 1; }

  # Force shell path - do not check for native binary here since we want to guarantee compliance

  local target_hash toolchain_hash
  if command -v sha256sum >/dev/null 2>&1; then
      target_hash=$(sha256sum "$target" | awk '{print $1}')
      toolchain_hash=$(sha256sum "$toolchain_lock" | awk '{print $1}')
  else
      target_hash=$(shasum -a 256 "$target" | awk '{print $1}')
      toolchain_hash=$(shasum -a 256 "$toolchain_lock" | awk '{print $1}')
  fi

  # 1. Compute Core ID
  # We use python3 if available to guarantee exact match with verifier logic
  local core_id
  if command -v python3 >/dev/null 2>&1; then
      core_id=$(python3 -c "
import json, hashlib, sys
data = {
    'schema': 'stunir.receipt.build.v1',
    'status': 'success',
    'build_epoch': int('$epoch'),
    'epoch': {'selected_epoch': int('$epoch'), 'source': 'default'},
    'target': '$target',
    'sha256': '$target_hash',
    'argv': ['shell_generated'],
    'inputs': [],
    'tool': None
}
# Canonicalize: no spaces, sorted keys, ensure_ascii=False
c14n = json.dumps(data, separators=(',', ':'), sort_keys=True, ensure_ascii=False).encode('utf-8')
print(hashlib.sha256(c14n).hexdigest())
")
  else
      echo "ERROR: Python3 required for strict receipt generation (core-id calculation)"
      exit 1
  fi

  # 2. Emit Final Receipt
  if command -v python3 >/dev/null 2>&1; then
      python3 -c "
import json, sys
data = {
    'schema': 'stunir.receipt.build.v1',
    'status': 'success',
    'build_epoch': int('$epoch'),
    'epoch': {'selected_epoch': int('$epoch'), 'source': 'default'},
    'target': '$target',
    'sha256': '$target_hash',
    'toolchain_sha256': '$toolchain_hash',
    'argv': ['shell_generated'],
    'inputs': [],
    'tool': None,
    'receipt_core_id_sha256': '$core_id'
}
print(json.dumps(data, separators=(',', ':'), sort_keys=True, ensure_ascii=False))
" > "$out_file"
  else
      echo "ERROR: Python3 required for strict receipt generation"
      exit 1
  fi
}

#!/usr/bin/env bash
set -euo pipefail

# STUNIR verify wrapper.
#
# Modes:
#   - Local mode (default): verifies receipts/manifests produced by scripts/build.sh
#   - DSSE mode: verifies a DSSE envelope + in-toto Statement receipt (requires --trust-key)
#
# Auto-mode selection:
#   - If --local is given: local
#   - Else if --dsse is given: dsse
#   - Else if first positional arg is an existing .json file that looks like a DSSE envelope: dsse
#   - Else: local

MODE=""
REPO="."
TMP_DIR="_verify_build"
FIXTURE_ROOT=""   # aka snapshot root

print_help() {
  cat <<'EOF'
Usage:
  # Local verification (default)
  ./scripts/verify.sh
  ./scripts/verify.sh --local

  # Verify a snapshot/fixture produced by scripts/snapshot_receipts.sh
  ./scripts/verify.sh --root fixtures/receipts/TAG
  ./scripts/verify.sh --snapshot fixtures/receipts/TAG

  # DSSE verification
  ./scripts/verify.sh receipt.dsse.json --trust-key KEYID=keys/pubkey.pem [--required-algs sha256,sha512]
  ./scripts/verify.sh --dsse receipt.dsse.json --trust-key KEYID=keys/pubkey.pem

Wrapper flags:
  --local            Force local mode
  --dsse             Force DSSE mode
  --repo PATH        Repo root (default: .)
  --tmp-dir PATH     Temp dir for verifier (default: _verify_build)
  --root PATH        Alias for --snapshot
  --snapshot PATH    Fixture root containing receipts/ and build/ (and optionally bin/)
  -h, --help         Show this help

All remaining args are forwarded to tools/verify_build.py.
EOF
}

# Parse wrapper flags (leave the rest to the Python verifier)
FORWARD=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --local)
      MODE="local"; shift 1 ;;
    --dsse)
      MODE="dsse"; shift 1 ;;
    --repo)
      REPO="$2"; shift 2 ;;
    --tmp-dir)
      TMP_DIR="$2"; shift 2 ;;
    --snapshot)
      FIXTURE_ROOT="$2"; shift 2 ;;
    --root)
      FIXTURE_ROOT="$2"; shift 2 ;;
    -h|--help)
      print_help; exit 0 ;;
    *)
      FORWARD+=("$1"); shift 1 ;;
  esac
done

# Extract possible envelope path from first positional argument (compat with historical usage)
ENVELOPE=""
REST=()
if [[ ${#FORWARD[@]} -gt 0 ]]; then
  if [[ "${FORWARD[0]}" == *.json ]]; then
    ENVELOPE="${FORWARD[0]}"
    REST=("${FORWARD[@]:1}")
  else
    REST=("${FORWARD[@]}")
  fi
fi

looks_like_dsse() {
  local p="$1"
  python3 - <<'PY' "$p" >/dev/null 2>&1
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
try:
    obj = json.loads(p.read_text(encoding='utf-8'))
except Exception:
    sys.exit(1)
# Minimal DSSE v1 JSON envelope shape
ok = (
    isinstance(obj, dict)
    and isinstance(obj.get('payloadType'), str)
    and isinstance(obj.get('payload'), str)
    and isinstance(obj.get('signatures'), list)
)
sys.exit(0 if ok else 1)
PY
}

contains_dsse_only_flags() {
  # If the user passed DSSE-specific flags, do not silently fall back to local.
  for a in "${REST[@]}"; do
    case "$a" in
      --trust-key|--required-algs|--no-require-canonical-payload)
        return 0 ;;
    esac
  done
  return 1
}

# Auto select mode if not forced
if [[ -z "$MODE" ]]; then
  if [[ -n "$ENVELOPE" ]] && [[ -f "$ENVELOPE" ]] && looks_like_dsse "$ENVELOPE"; then
    MODE="dsse"
  else
    MODE="local"
    # If user *intended* DSSE (passed DSSE flags), keep DSSE and fail loudly.
    if [[ -n "$ENVELOPE" ]] && ! [[ -f "$ENVELOPE" ]] && contains_dsse_only_flags; then
      MODE="dsse"
    fi
  fi
fi

mkdir -p "$TMP_DIR"

if [[ "$MODE" == "dsse" ]]; then
  if [[ -z "$ENVELOPE" ]]; then
    ENVELOPE="receipt.dsse.json"
  fi
  if [[ ! -f "$ENVELOPE" ]]; then
    echo "ERROR: DSSE envelope not found: $ENVELOPE" 1>&2
    exit 2
  fi
  exec python3 -B tools/verify_build.py \
    --envelope "$ENVELOPE" \
    --repo "$REPO" \
    --tmp-dir "$TMP_DIR" \
    "${REST[@]}"
fi

# Local mode
if [[ -n "$ENVELOPE" ]] && [[ ! -f "$ENVELOPE" ]]; then
  echo "NOTE: '$ENVELOPE' not found; running local verification instead." 1>&2
fi

LOCAL_ARGS=("--local" "--repo" "$REPO" "--tmp-dir" "$TMP_DIR" "--strict")
if [[ -n "$FIXTURE_ROOT" ]]; then
  LOCAL_ARGS+=("--root" "$FIXTURE_ROOT")
fi

exec python3 -B tools/verify_build.py \
  "${LOCAL_ARGS[@]}" \
  "${REST[@]}"

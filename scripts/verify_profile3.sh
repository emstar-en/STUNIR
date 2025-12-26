#!/usr/bin/env sh
set -eu

ROOT="."
ATTEST="root_attestation.txt"
PUBKEY=""
MANIFEST_PATH="pack_manifest.tsv"

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

# shellcheck disable=SC1091
. "$SCRIPT_DIR/profile3_lib.sh"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --root) ROOT="$2"; shift 2;;
    --attestation) ATTEST="$2"; shift 2;;
    --pubkey) PUBKEY="$2"; shift 2;;
    --manifest) MANIFEST_PATH="$2"; shift 2;;
    -h|--help)
      echo "usage: verify_profile3.sh [--root DIR] [--attestation FILE] [--pubkey PUBKEY] [--manifest RELPATH]";
      exit 0;;
    *)
      echo "unknown arg: $1" 1>&2; exit 2;;
  esac
done

# 1) Always run minimal integrity checks first.
"$SCRIPT_DIR/verify_minimal.sh" --root "$ROOT" --attestation "$ATTEST" --pubkey "$PUBKEY" >/dev/null

ATTEST_PATH="$ROOT/$ATTEST"

if [ ! -f "$ATTEST_PATH" ]; then
  echo "missing $ATTEST_PATH" 1>&2
  exit 1
fi

expected_manifest_hex=""
expected_manifest_logical_path=""

# 2) Find manifest binding in root_attestation.txt (artifact kind=manifest logical_path=...)
first_seen=0
while IFS= read -r line || [ -n "$line" ]; do
  line=$(printf "%s" "$line" | sed 's/\r$//')
  case "$line" in
    ""|"#"*) continue;;
  esac

  if [ "$first_seen" -eq 0 ]; then
    # version line already validated by verify_minimal; still consume it.
    first_seen=1
    continue
  fi

  set -- $line
  rtype="$1"
  case "$rtype" in
    artifact)
      if [ "$#" -lt 3 ]; then
        echo "malformed line: $line" 1>&2
        exit 1
      fi
      digest="$2"
      if [ "${digest#sha256:}" = "$digest" ]; then
        continue
      fi
      hex=${digest#sha256:}
      if ! stunir_is_hex64 "$hex"; then
        continue
      fi
      shift 2
      kind=""
      logical_path=""
      for tok in "$@"; do
        case "$tok" in
          kind=*) kind=${tok#kind=};;
          logical_path=*) logical_path=${tok#logical_path=};;
        esac
      done
      if [ "$kind" = "manifest" ] && [ -n "$logical_path" ]; then
        expected_manifest_hex="$hex"
        expected_manifest_logical_path="$logical_path"
        break
      fi
      ;;
  esac

done < "$ATTEST_PATH"

# 3) Profile 3 STRICT requirement: the pack MUST be closed-world.
# Therefore, root_attestation.txt MUST bind pack_manifest.tsv via an artifact sha256 digest.
# Expected binding line shape (tokens may be reordered):
#   artifact sha256:<hex64> kind=manifest logical_path=pack_manifest.tsv
if [ -z "$expected_manifest_hex" ]; then
  echo "FATAL: Profile 3 requires a bound manifest, but no manifest binding was found in root_attestation.txt." >&2
  echo "FATAL: This violates the Closed World assumption for strict Profile 3 verification." >&2
  exit 40
fi

# If a manifest is bound, enforce it.
if [ "$expected_manifest_logical_path" != "$MANIFEST_PATH" ]; then
  # The attestation points to a different logical_path; verify that instead.
  MANIFEST_PATH="$expected_manifest_logical_path"
fi

if ! stunir_is_safe_relpath "$MANIFEST_PATH"; then
  echo "unsafe_filename: $MANIFEST_PATH" 1>&2
  exit 1
fi

MANIFEST_FILE="$ROOT/$MANIFEST_PATH"
if [ ! -f "$MANIFEST_FILE" ]; then
  echo "missing manifest file: $MANIFEST_FILE" 1>&2
  exit 1
fi

# CRLF policy for manifest (strict).
if stunir_manifest_has_crlf "$MANIFEST_FILE"; then
  echo "crlf_detected_in_manifest: $MANIFEST_PATH" 1>&2
  exit 1
fi

actual_manifest_hex=$(stunir_hash_file_sha256 "$MANIFEST_FILE")
if [ "$actual_manifest_hex" != "$expected_manifest_hex" ]; then
  echo "manifest_target_hash_mismatch: $MANIFEST_PATH" 1>&2
  echo "expected $expected_manifest_hex" 1>&2
  echo "actual $actual_manifest_hex" 1>&2
  exit 1
fi

if ! stunir_manifest_is_sorted "$MANIFEST_FILE"; then
  echo "manifest_unsorted: $MANIFEST_PATH" 1>&2
  exit 1
fi

# 4) Verify each manifest entry.
while IFS= read -r mline || [ -n "$mline" ]; do
  case "$mline" in
    ""|"#"*) continue;;
  esac

  # Expect: <hex64><TAB><path>
  # Paths are restricted to safe subset, so whitespace splitting is OK.
  set -- $mline
  if [ "$#" -ne 2 ]; then
    echo "manifest_line_malformed: $mline" 1>&2
    exit 1
  fi
  hex="$1"
  rel="$2"

  if ! stunir_is_hex64 "$hex"; then
    echo "bad digest in manifest: $hex" 1>&2
    exit 1
  fi
  if ! stunir_is_safe_relpath "$rel"; then
    echo "unsafe_filename: $rel" 1>&2
    exit 1
  fi
  if [ "$rel" = "$MANIFEST_PATH" ]; then
    echo "manifest_line_malformed: manifest must not include itself" 1>&2
    exit 1
  fi
  case "$rel" in
    objects/sha256/*)
      echo "manifest_line_malformed: manifest must exclude objects/sha256" 1>&2
      exit 1
      ;;
  esac

  f="$ROOT/$rel"
  if [ ! -f "$f" ]; then
    echo "manifest_target_missing: $rel" 1>&2
    exit 1
  fi

  actual=$(stunir_hash_file_sha256 "$f")
  if [ "$actual" != "$hex" ]; then
    echo "manifest_target_hash_mismatch: $rel" 1>&2
    echo "expected $hex" 1>&2
    echo "actual $actual" 1>&2
    exit 1
  fi

done < "$MANIFEST_FILE"

echo "OK (integrity + manifest)"

\
#!/usr/bin/env sh
set -eu
ROOT="."
ATTEST="root_attestation.txt"
PUBKEY=""
while [ "$#" -gt 0 ]; do
case "$1" in
--root) ROOT="$2"; shift 2;;
--attestation) ATTEST="$2"; shift 2;;
--pubkey) PUBKEY="$2"; shift 2;;
-h|--help)
echo "usage: verify_minimal.sh [--root DIR] [--attestation FILE] [--pubkey PUBKEY]";
exit 0;;
*)
echo "unknown arg: $1" 1>&2; exit 2;;
esac
done
ATTEST_PATH="$ROOT/$ATTEST"
OBJ_DIR="$ROOT/objects/sha256"
if [ ! -f "$ATTEST_PATH" ]; then
echo "missing $ATTEST_PATH" 1>&2
exit 1
fi
if [ ! -d "$OBJ_DIR" ]; then
echo "missing $OBJ_DIR" 1>&2
exit 1
fi
hash_file() {
  f="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$f" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$f" | awk '{print $1}'
  elif command -v openssl >/dev/null 2>&1; then
    openssl dgst -sha256 "$f" | awk '{print $NF}'
  else
    echo "no sha256 tool found (need sha256sum, shasum, or openssl)" 1>&2
    exit 3
  fi
}
is_sha256_digest() {
  d="$1"
  echo "$d" | awk 'BEGIN{ok=0} /^sha256:[0-9a-f]{64}$/{ok=1} END{exit ok?0:1}'
}
first_seen=0
ir_count=0
crlf_seen=0
# Parse root_attestation.txt (text v0). For parsing convenience, accept CRLF by stripping trailing \r.
while IFS= read -r line || [ -n "$line" ]; do
  case "$line" in
    *"$(printf '\r')") crlf_seen=1;;
  esac
  line=$(printf "%s" "$line" | sed 's/\r$//')
  case "$line" in
    ""|"#"*) continue;;
  esac
  if [ "$first_seen" -eq 0 ]; then
    if [ "$line" != "stunir.pack.root_attestation_text.v0" ]; then
      echo "bad version line: $line" 1>&2
      exit 1
    fi
    first_seen=1
    continue
  fi
  # Tokenize by ASCII whitespace.
  set -- $line
  rtype="$1"
  case "$rtype" in
    epoch)
      # epoch <opaque>
      continue
      ;;
    ir|receipt)
      if [ "$#" -lt 2 ]; then
        echo "malformed line: $line" 1>&2
        exit 1
      fi
      digest="$2"
      ;;
    input|artifact)
      if [ "$#" -lt 3 ]; then
        echo "malformed line: $line" 1>&2
        exit 1
      fi
      digest="$2"
      ;;
    *)
      echo "unknown record type: $rtype" 1>&2
      exit 1
      ;;
  esac
  if ! is_sha256_digest "$digest"; then
    echo "bad digest: $digest" 1>&2
    exit 1
  fi
  hex=${digest#sha256:}
  obj="$OBJ_DIR/$hex"
  if [ ! -f "$obj" ]; then
    echo "missing object: $obj" 1>&2
    exit 1
  fi
  actual=$(hash_file "$obj")
  if [ "$actual" != "$hex" ]; then
    echo "hash mismatch for $obj" 1>&2
    echo "expected $hex" 1>&2
    echo "actual $actual" 1>&2
    exit 1
  fi
  if [ "$rtype" = "ir" ]; then
    ir_count=$((ir_count+1))
  fi
done < "$ATTEST_PATH"
if [ "$first_seen" -eq 0 ]; then
  echo "no version line found" 1>&2
  exit 1
fi
if [ "$ir_count" -ne 1 ]; then
  echo "expected exactly 1 ir record, got $ir_count" 1>&2
  exit 1
fi
SIG_PATH="$ROOT/root_attestation.txt.sig"
if [ -f "$SIG_PATH" ] && [ -n "$PUBKEY" ]; then
  if command -v openssl >/dev/null 2>&1; then
    if openssl dgst -sha256 -verify "$PUBKEY" -signature "$SIG_PATH" "$ATTEST_PATH" >/dev/null 2>&1; then
      if [ "$crlf_seen" -eq 1 ]; then
        echo "OK (integrity + signature; note: CRLF accepted in root_attestation.txt)"
      else
        echo "OK (integrity + signature)"
      fi
      exit 0
    else
      echo "signature verification failed" 1>&2
      exit 1
    fi
  else
    echo "note: signature present but openssl not available; integrity OK" 1>&2
  fi
fi
if [ "$crlf_seen" -eq 1 ]; then
  echo "OK (integrity; note: CRLF accepted in root_attestation.txt)"
else
  echo "OK (integrity)"
fi

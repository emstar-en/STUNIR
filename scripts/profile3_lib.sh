#!/usr/bin/env sh
# Library for Profile 3 strict verification helpers.
# Intended to be sourced by verify_profile3.sh.

stunir_hash_file_sha256() {
  f="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$f" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$f" | awk '{print $1}'
  elif command -v openssl >/dev/null 2>&1; then
    openssl dgst -sha256 "$f" | awk '{print $NF}'
  else
    echo "no sha256 tool found (need sha256sum, shasum, or openssl)" 1>&2
    return 3
  fi
}

stunir_is_hex64() {
  x="$1"
  echo "$x" | awk 'BEGIN{ok=0} /^[0-9a-f]{64}$/{ok=1} END{exit ok?0:1}'
}

stunir_is_safe_relpath() {
  p="$1"
  # Reject empties and dangerous prefixes.
  case "$p" in
    "") return 1;;
    /*) return 1;;
    ./*) return 1;;
    ..|../*|*/..|*/../*) return 1;;
    *"//"*) return 1;;
    *"\\"*) return 1;;
    *" "*|*"\t"*|*"\r"*|*"\n"*) return 1;;
    -*|*/-*) return 1;;
    */|*/./*|*/.) return 1;;
    *[!A-Za-z0-9._/-]*) return 1;;
  esac
  return 0
}

stunir_manifest_has_crlf() {
  f="$1"
  # exit 0 if CR found, 1 otherwise
  awk 'index($0, "\r"){exit 0} END{exit 1}' "$f"
}

stunir_manifest_is_sorted() {
  f="$1"
  tmp="${TMPDIR:-/tmp}/stunir_manifest_sorted.$$"
  LC_ALL=C sort "$f" > "$tmp"
  if command -v cmp >/dev/null 2>&1; then
    cmp -s "$tmp" "$f"
    rc=$?
  else
    # fallback
    diff -q "$tmp" "$f" >/dev/null 2>&1
    rc=$?
  fi
  rm -f "$tmp" >/dev/null 2>&1 || true
  return $rc
}

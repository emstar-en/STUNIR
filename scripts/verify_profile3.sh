\
#!/usr/bin/env sh
set -eu

ROOT="."
ATTEST="root_attestation.txt"
PUBKEY=""
STRICT_COMPLETENESS=1

while [ "$#" -gt 0 ]; do
  case "$1" in
    --root) ROOT="$2"; shift 2;;
    --attestation) ATTEST="$2"; shift 2;;
    --pubkey) PUBKEY="$2"; shift 2;;
    --no-completeness) STRICT_COMPLETENESS=0; shift 1;;
    -h|--help)
      echo "usage: verify_profile3.sh [--root DIR] [--attestation FILE] [--pubkey PUBKEY] [--no-completeness]";
      exit 0;;
    *)
      echo "unknown arg: $1" 1>&2; exit 2;;
  esac
done

SCRIPT_DIR=$(dirname "$0")

# shellcheck disable=SC1090
. "$SCRIPT_DIR/profile3_lib.sh"

ATTEST_PATH="$ROOT/$ATTEST"
MANIFEST_PATH="$ROOT/pack_manifest.tsv"

# 1) Minimal integrity first (object store + exactly one IR)
if [ -n "$PUBKEY" ]; then
  "$SCRIPT_DIR/verify_minimal.sh" --root "$ROOT" --attestation "$ATTEST" --pubkey "$PUBKEY"
else
  "$SCRIPT_DIR/verify_minimal.sh" --root "$ROOT" --attestation "$ATTEST"
fi

# 2) Strict binding: root_attestation MUST bind pack_manifest.tsv
if [ ! -f "$ATTEST_PATH" ]; then
  p3_die "missing $ATTEST_PATH"
fi

manifest_hex=$(p3_extract_manifest_binding_digest "$ATTEST_PATH") || p3_die "missing or ambiguous manifest binding in $ATTEST_PATH (need artifact sha256:... kind=manifest logical_path=pack_manifest.tsv)"

# 3) Verify pack_manifest.tsv exists and is hash-matched to the attested digest
if [ ! -f "$MANIFEST_PATH" ]; then
  p3_die "missing $MANIFEST_PATH"
fi

actual_manifest_hex=$(p3_hash_file "$MANIFEST_PATH")
if [ "$actual_manifest_hex" != "$manifest_hex" ]; then
  p3_die "pack_manifest.tsv hash mismatch (expected $manifest_hex, got $actual_manifest_hex)"
fi

# 4) Require LF-only for pack_manifest.tsv
if p3_file_contains_cr "$MANIFEST_PATH"; then
  p3_die "CRLF/CR detected in pack_manifest.tsv (LF-only required)"
fi

# 5) Verify sorted by path under LC_ALL=C
p3_require_cmd awk
p3_require_cmd sort

paths_tmp=$(p3_tmpfile)
man_paths_tmp="$paths_tmp.man"
fs_paths_tmp="$paths_tmp.fs"

trap 'rm -f "$paths_tmp" "$paths_tmp.chk" "$paths_tmp.fs" "$paths_tmp.man" "$paths_tmp.man.chk"' EXIT

LC_ALL=C awk '
  /^[[:space:]]*#/ {next}
  /^[[:space:]]*$/ {next}
  {
    if (NF != 2) { exit 1 }
    print $2
  }' "$MANIFEST_PATH" > "$paths_tmp" || p3_die "malformed pack_manifest.tsv (expected exactly 2 fields per non-comment line)"

if sort -c "$paths_tmp" >/dev/null 2>&1; then
  :
else
  LC_ALL=C awk 'NR==1{prev=$0;next} { if ($0 < prev) exit 1; prev=$0 } END{exit 0}' "$paths_tmp" || p3_die "pack_manifest.tsv not sorted by path (LC_ALL=C)"
fi

# 6) Verify each manifest entry
> "$man_paths_tmp"

cr=$(printf '\r')
while IFS= read -r line || [ -n "$line" ]; do
  case "$line" in
    *"$cr"*) p3_die "CR detected in pack_manifest.tsv";;
  esac

  case "$line" in
    ""|"#"*) continue;;
  esac

  set -- $line
  if [ "$#" -ne 2 ]; then
    p3_die "malformed manifest line (expected 2 fields): $line"
  fi
  hex="$1"
  path="$2"

  if ! p3_is_hex64_lower "$hex"; then
    p3_die "bad hex digest in manifest: $hex"
  fi
  if ! p3_is_safe_manifest_path "$path"; then
    p3_die "unsafe_filename in manifest path: $path"
  fi

  case "$path" in
    objects/sha256/*) p3_die "manifest_scope_violation: objects store must be excluded: $path";;
    pack_manifest.tsv) p3_die "manifest_scope_violation: manifest must exclude itself";;
  esac

  file_path="$ROOT/$path"
  if [ ! -f "$file_path" ]; then
    p3_die "manifest_file_missing: $file_path"
  fi

  actual_hex=$(p3_hash_file "$file_path")
  if [ "$actual_hex" != "$hex" ]; then
    p3_die "manifest_file_hash_mismatch: $path (expected $hex, got $actual_hex)"
  fi

  printf "%s\n" "$path" >> "$man_paths_tmp"
done < "$MANIFEST_PATH"

# 7) Completeness check (strict): filesystem regular files excluding objects/sha256/** and pack_manifest.tsv
if [ "$STRICT_COMPLETENESS" -eq 1 ]; then
  p3_require_cmd find
  p3_require_cmd sed

  # Enumerate all regular files, then filter with awk to avoid non-POSIX find predicates.
  (cd "$ROOT" && \
    find . -type f -print \
      | sed 's|^\./||' \
      | awk '$0 != "pack_manifest.tsv" && $0 !~ /^objects\/sha256\// { print }' \
      | LC_ALL=C sort) > "$fs_paths_tmp" || p3_die "failed to enumerate pack-root files for completeness check"

  LC_ALL=C sort "$man_paths_tmp" > "$man_paths_tmp.chk"

  awk '
    FNR==NR { man[$0]=1; next }
    { fs[$0]=1 }
    END {
      missing=0; extra=0
      for (p in fs) if (!(p in man)) { extra=1 }
      for (p in man) if (!(p in fs)) { missing=1 }
      if (missing || extra) exit 1
      exit 0
    }' "$man_paths_tmp.chk" "$fs_paths_tmp" || p3_die "manifest_incomplete_or_extra_files"
fi

echo "OK (integrity + strict manifest)"

\
#!/usr/bin/env sh
set -eu

p3_die() {
  echo "ERROR: $*" 1>&2
  exit 1
}

p3_die_usage() {
  echo "ERROR: $*" 1>&2
  exit 2
}

p3_die_missing_tool() {
  echo "ERROR: missing required tool: $*" 1>&2
  exit 3
}

p3_note() {
  echo "note: $*" 1>&2
}

p3_require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    p3_die_missing_tool "$1"
  fi
}

p3_hash_file() {
  f="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$f" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$f" | awk '{print $1}'
  elif command -v openssl >/dev/null 2>&1; then
    openssl dgst -sha256 "$f" | awk '{print $NF}'
  else
    p3_die_missing_tool "sha256sum OR shasum OR openssl"
  fi
}

p3_is_hex64_lower() {
  h="$1"
  echo "$h" | awk 'BEGIN{ok=0} /^[0-9a-f]{64}$/{ok=1} END{exit ok?0:1}'
}

p3_is_safe_manifest_path() {
  p="$1"

  [ -n "$p" ] || return 1

  # No leading slash
  case "$p" in
    /*) return 1;;
  esac

  # Must not start with ./
  case "$p" in
    ./*) return 1;;
  esac

  # No backslashes
  case "$p" in
    *\\*) return 1;;
  esac

  # No ASCII whitespace (space or tab). Newlines are impossible once read as a line.
  tab=$(printf '\t')
  case "$p" in
    *" "*|*"$tab"*) return 1;;
  esac

  # Allowed chars only
  echo "$p" | awk 'BEGIN{ok=0} /^[A-Za-z0-9._\/-]+$/{ok=1} END{exit ok?0:1}' || return 1

  # No empty segments, no .. segments, and no segment begins with '-'
  echo "$p" | awk -F'/' '
    function bad(){ exit 1 }
    {
      for (i=1; i<=NF; i++) {
        if ($i == "") bad()
        if ($i == "..") bad()
        if (substr($i,1,1) == "-") bad()
      }
      exit 0
    }'
}

p3_file_contains_cr() {
  f="$1"
  cr=$(printf '\r')
  if command -v grep >/dev/null 2>&1; then
    grep -q "$cr" "$f"
  else
    awk 'index($0, "\r") { found=1 } END{ exit found?0:1 }' "$f"
  fi
}

p3_extract_manifest_binding_digest() {
  attest_path="$1"
  awk '
    function is_digest(d){ return (d ~ /^sha256:[0-9a-f]{64}$/) }
    function has_token(tok, n, a, i){ for(i=1;i<=n;i++) if(a[i]==tok) return 1; return 0 }
    BEGIN{count=0; hex=""}
    {
      line=$0
      sub(/\r$/, "", line)
      if (line ~ /^#/ || line ~ /^$/) next
      if (line == "stunir.pack.root_attestation_text.v0") next
      n=split(line, a, /[[:space:]]+/)
      if (n < 3) next
      if (a[1] != "artifact") next
      if (!is_digest(a[2])) next
      if (!has_token("kind=manifest", n, a)) next
      if (!has_token("logical_path=pack_manifest.tsv", n, a)) next
      count++
      hex = substr(a[2], 8)
    }
    END{
      if (count==1) { print hex; exit 0 }
      exit 1
    }' "$attest_path"
}

p3_tmpfile() {
  if command -v mktemp >/dev/null 2>&1; then
    mktemp 2>/dev/null || mktemp -t stunir_p3 2>/dev/null || echo "./.stunir_p3_tmp_$$"
  else
    echo "./.stunir_p3_tmp_$$"
  fi
}

#!/usr/bin/env sh
set -eu

# Snapshot current build receipts into fixtures/ for optional committing.
# Default tag: UTC timestamp (standard snapshot labeling).
# Override with:
#   STUNIR_SNAPSHOT_TAG=... scripts/snapshot_receipts.sh
# or
#   scripts/snapshot_receipts.sh --tag <TAG>
# Options:
#   --force           overwrite destination if it already exists
#   --include-bin     also copy bin/ (may include native binaries)

TAG="${STUNIR_SNAPSHOT_TAG:-}"
FORCE=0
INCLUDE_BIN=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --tag)
      TAG="$2"; shift 2 ;;
    --force)
      FORCE=1; shift 1 ;;
    --include-bin)
      INCLUDE_BIN=1; shift 1 ;;
    *)
      echo "Unknown arg: $1" 1>&2
      exit 2
      ;;
  esac
done

if [ "$TAG" = "" ]; then
  TAG="$(date -u +%Y%m%d_%H%M%SZ)"
fi

DST="fixtures/receipts/${TAG}"

if [ ! -d "receipts" ]; then
  echo "Missing receipts/. Run scripts/build.sh first." 1>&2
  exit 3
fi

if [ -e "${DST}" ]; then
  if [ "$FORCE" = "1" ]; then
    rm -rf "${DST}"
  else
    echo "Destination exists: ${DST} (use --force to overwrite)" 1>&2
    exit 4
  fi
fi

mkdir -p "${DST}"

mkdir -p "${DST}/receipts" "${DST}/build"
cp -R receipts/. "${DST}/receipts/"
if [ -d "build" ]; then
  cp -R build/. "${DST}/build/" || true
fi

if [ "$INCLUDE_BIN" = "1" ] && [ -d "bin" ]; then
  mkdir -p "${DST}/bin"
  cp -R bin/. "${DST}/bin/" || true
fi

echo "OK: snapshot created at ${DST}"

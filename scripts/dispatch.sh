#!/bin/bash
set -e

# ISSUE.HASKELL.0001: Model dispatch router
ISSUE_ID=${1:?Usage: dispatch.sh ISSUE.HASKELL.0001}

case $ISSUE_ID in
    ISSUE.HASKELL.*)      scripts/build_haskell_first.sh ;;
    ISSUE.IR.*)           scripts/canonicalize.sh ;;
    ISSUE.CONFORMANCE.*)  scripts/test_conformance.sh ;;
    ISSUE.PACK.*)         scripts/verify_pack.sh ;;
    *) echo "Unknown dispatch: $ISSUE_ID" >&2; exit 1 ;;
esac

echo "✓ Dispatch $ISSUE_ID complete"
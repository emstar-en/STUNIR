#!/bin/bash
set -e

# V1: Original master dispatch
ISSUE_ID=${1:-"all"}

dispatch() {
    local id=$1
    echo "Dispatching $id..."
    case $id in
        ISSUE.HASKELL.*) scripts/build_haskell_first.sh ;;
        ISSUE.IR.*) scripts/canonicalize.sh ;;
        ISSUE.CONFORMANCE.*) scripts/test_conformance.sh ;;
        ISSUE.PACK.*) tools/pack_attestation/generate_root.sh ;;
        *) echo "Unknown issue: $id" >&2; return 1 ;;
    esac
}

if [ "$ISSUE_ID" = "all" ]; then
    for issue in issues/*.machine.json issues/*/*.machine.json; do
        id=$(grep '"id"' "$issue" | cut -d'"' -f4)
        [ "$id" != "" ] && dispatch "$id"
    done
else
    dispatch "$ISSUE_ID"
fi

echo "✓ V1 Master dispatch complete"
    #!/usr/bin/env sh
    set -eu

    : "${STUNIR_OUTPUT_TARGETS:=wasm,c}"
    : "${STUNIR_REQUIRE_DEPS:=0}"

    mkdir -p receipts/deps receipts

    python3 tools/resolve_requirements.py --targets "${STUNIR_OUTPUT_TARGETS}" --out receipts/requirements.json

    REQUIRED=$(python3 - <<'PY'
import json
r=json.load(open('receipts/requirements.json','r',encoding='utf-8'))
print(' '.join([x['contract_name'] for x in r.get('required_contracts',[])]))
PY
    )

    ATTEST_ARGS=""
    if [ "${STUNIR_ATTESTATIONS:-}" != "" ]; then
      for a in ${STUNIR_ATTESTATIONS}; do
        ATTEST_ARGS="$ATTEST_ARGS --attestation $a"
      done
    fi

    for C in ${REQUIRED}; do
      TOOLVAR="STUNIR_TOOL_${C}"
      eval TOOLVAL="\${$TOOLVAR-}"
      TOOLARGS=""
      if [ "${TOOLVAL}" != "" ]; then
        TOOLARGS="--tool ${TOOLVAL}"
      fi

      REQFLAG=""
      if [ "${STUNIR_REQUIRE_DEPS}" = "1" ]; then
        REQFLAG="--require"
      fi

      # shellcheck disable=SC2086
      python3 tools/probe_dependency.py --contract "contracts/${C}.json" --out "receipts/deps/${C}.json" ${TOOLARGS} ${ATTEST_ARGS} ${REQFLAG} || {
        if [ "${STUNIR_REQUIRE_DEPS}" = "1" ]; then
          exit 3
        fi
        true
      }
    done

    echo "OK: dependency probing attempted. See receipts/requirements.json and receipts/deps/*.json"

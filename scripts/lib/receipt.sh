#!/usr/bin/env bash
# scripts/lib/receipt.sh
# Shell implementation of receipt generation

source scripts/lib/json_canon.sh

# cmd_gen_receipt <target> <status> <epoch> <tool_name> <tool_path> <tool_hash> <tool_ver>
cmd_gen_receipt() {
    local target="$1"
    local status="$2"
    local epoch="$3"
    local t_name="$4"
    local t_path="$5"
    local t_hash="$6"
    local t_ver="$7"
    shift 7
    # argv is remaining args

    # 1. Construct Tool Info (Sorted Keys: tool_name, tool_path, tool_sha256, tool_version)
    # Note: 'tool_name' comes before 'tool_path'? 
    # Alphabetical: tool_name, tool_path, tool_sha256, tool_version
    # n, p, s, v. Correct.

    # We need to construct the inner object string first
    local tool_json="{\"tool_name\":\"$t_name\",\"tool_path\":\"$t_path\",\"tool_sha256\":\"$t_hash\",\"tool_version\":\"$t_ver\"}"

    # 2. Construct Receipt (Sorted Keys)
    # receipt_argv
    # receipt_build_epoch
    # receipt_epoch_json
    # receipt_inputs
    # receipt_schema
    # receipt_status
    # receipt_target
    # receipt_tool

    # Argv array construction
    local argv_json="["
    local first=1
    for arg in "$@"; do
        if [[ $first -eq 0 ]]; then argv_json+=","; fi
        argv_json+="\"$arg\""
        first=0
    done
    argv_json+="]"

    # Inputs (empty for now)
    local inputs_json="{}"

    # Assemble final JSON
    # Note: We manually ensure alphabetical order of keys here.
    echo -n "{"
    echo -n "\"receipt_argv\":$argv_json,"
    echo -n "\"receipt_build_epoch\":$epoch,"
    echo -n "\"receipt_epoch_json\":\"build/epoch.json\","
    echo -n "\"receipt_inputs\":$inputs_json,"
    echo -n "\"receipt_schema\":\"stunir.receipt.build.v1\","
    echo -n "\"receipt_status\":\"$status\","
    echo -n "\"receipt_target\":\"$target\","
    echo -n "\"receipt_tool\":$tool_json"
    echo "}"
}

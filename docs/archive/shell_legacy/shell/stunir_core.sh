#!/bin/bash
# STUNIR Shell-Native Core (Profile 3)
# v7: Compatibility Mode (No --argjson, explicit file checks)

set -e
set -o pipefail

# --- COMPILER LOGIC ---
compile_task_stream() {
    local task_json="$1"
    local type=$(echo "$task_json" | jq -r '.type // empty')
    
    case "$type" in
        "say")
            local msg=$(echo "$task_json" | jq -r '.message // ""')
            jq -n --arg msg "$msg" '{op: "print", args: [$msg], body: []}'
            ;;
        "calc")
            local var=$(echo "$task_json" | jq -r '.var // "x"')
            local expr=$(echo "$task_json" | jq -r '.expr // "0"')
            if [[ "$expr" =~ ^([0-9]+)\ \+\ ([0-9]+)$ ]]; then
                local lhs="${BASH_REMATCH[1]}"
                local rhs="${BASH_REMATCH[2]}"
                jq -n --arg var "$var" --arg lhs "$lhs" --arg rhs "$rhs" \
                    '{op: "add", args: [$var, $lhs, $rhs], body: []}'
                jq -n --arg var "$var" \
                    '{op: "print_var", args: [$var], body: []}'
            else
                jq -n --arg var "$var" --arg expr "$expr" \
                    '{op: "var_def", args: [$var, $expr], body: []}'
            fi
            ;;
        "repeat")
            local count=$(echo "$task_json" | jq -r '.count // 1')
            
            local subtasks_json=$(echo "$task_json" | jq -c '.tasks // []')
            
            local body_content=$(
                echo "$subtasks_json" | jq -c '.[]' | while read -r subtask; do
                    compile_task_stream "$subtask"
                done | jq -s '.'
            )
            
            if [ -z "$body_content" ] || [ "$body_content" == "null" ]; then
                body_content="[]"
            fi
            
            # Use --arg + fromjson instead of --argjson for compatibility
            jq -n --arg count "$count" --arg body "$body_content" \
                '{op: "loop", args: [$count], body: ($body | fromjson)}'
            ;;
    esac
}

run_compiler() {
    local in_spec="$1"
    local out_ir="$2"
    
    echo "DEBUG: Compiling $in_spec..."
    if [ ! -f "$in_spec" ]; then echo "Error: Spec file '$in_spec' not found"; exit 1; fi
    
    local main_body=$(
        jq -c '(.tasks // [])[]' "$in_spec" | while read -r task; do
            compile_task_stream "$task"
        done | jq -s '.'
    )
    
    if [ -z "$main_body" ] || [ "$main_body" == "null" ]; then
        main_body="[]"
    fi
    
    jq -n --arg body "$main_body" \
        '{functions: [{name: "main", body: ($body | fromjson)}]}' > "$out_ir"
        
    echo "DEBUG: IR generated at $out_ir"
}

# --- EMITTER LOGIC (BASH) ---
emit_body() {
    local instrs_json="$1"
    local indent="$2"
    
    if [ -z "$instrs_json" ] || [ "$instrs_json" == "null" ]; then return; fi

    echo "$instrs_json" | jq -c '.[]?' | while read -r instr; do
        local op=$(echo "$instr" | jq -r '.op // empty')
        case "$op" in
            "print")
                local msg=$(echo "$instr" | jq -r '.args[0] // ""')
                echo "${indent}echo \"$msg\""
                ;;
            "print_var")
                local var=$(echo "$instr" | jq -r '.args[0] // ""')
                echo "${indent}echo \"\$$var\""
                ;;
            "var_def")
                local var=$(echo "$instr" | jq -r '.args[0] // ""')
                local val=$(echo "$instr" | jq -r '.args[1] // ""')
                echo "${indent}$var=$val"
                ;;
            "add")
                local target=$(echo "$instr" | jq -r '.args[0] // ""')
                local lhs=$(echo "$instr" | jq -r '.args[1] // "0"')
                local rhs=$(echo "$instr" | jq -r '.args[2] // "0"')
                echo "${indent}$target=\$(( $lhs + $rhs ))"
                ;;
            "loop")
                local count=$(echo "$instr" | jq -r '.args[0] // "1"')
                echo "${indent}for ((i=0; i<$count; i++)); do"
                local body=$(echo "$instr" | jq -c '.body // []')
                emit_body "$body" "${indent}    "
                echo "${indent}done"
                ;;
        esac
    done
}

run_emitter() {
    local in_ir="$1"
    local target="$2"
    local out_file="$3"
    
    echo "DEBUG: Emitting to $out_file..."
    if [ ! -f "$in_ir" ]; then echo "Error: IR file '$in_ir' not found"; exit 1; fi
    
    if [ "$target" != "bash" ]; then
        echo "Error: Only bash target supported"
        exit 1
    fi
    
    echo "#!/bin/bash" > "$out_file"
    echo "set -e" >> "$out_file"
    echo "" >> "$out_file"
    
    local main_body=$(jq -c '.functions[0].body // []' "$in_ir")
    
    echo "main() {" >> "$out_file"
    emit_body "$main_body" "    " >> "$out_file"
    echo "}" >> "$out_file"
    echo "" >> "$out_file"
    echo "main" >> "$out_file"
    
    chmod +x "$out_file"
}

# --- MAIN ---
CMD="$1"
shift

case "$CMD" in
    "compile")
        IN_SPEC=""
        OUT_IR=""
        while [[ $# -gt 0 ]]; do
            case $1 in
                --in-spec) IN_SPEC="$2"; shift ;;
                --out-ir) OUT_IR="$2"; shift ;;
            esac
            shift
        done
        run_compiler "$IN_SPEC" "$OUT_IR"
        ;;
    "emit")
        IN_IR=""
        TARGET=""
        OUT_FILE=""
        while [[ $# -gt 0 ]]; do
            case $1 in
                --in-ir) IN_IR="$2"; shift ;;
                --target) TARGET="$2"; shift ;;
                --out-file) OUT_FILE="$2"; shift ;;
            esac
            shift
        done
        run_emitter "$IN_IR" "$TARGET" "$OUT_FILE"
        ;;
    *)
        echo "Usage: $0 <compile|emit> [args]"
        exit 1
        ;;
esac

import sys
import json
import hashlib
import os

# --- 1. Canonicalization (RFC 8785-ish) ---
def canonicalize_json(data):
    """
    Returns a byte string of the canonical JSON representation.
    Rules:
    - Keys sorted lexicographically.
    - No whitespace.
    - Floats are NOT supported in this profile (integers only).
    """
    if isinstance(data, dict):
        sorted_items = sorted(data.items())
        parts = []
        for k, v in sorted_items:
            parts.append(json.dumps(k) + ":" + canonicalize_json(v).decode('utf-8'))
        return ("{" + ",".join(parts) + "}").encode('utf-8')
    elif isinstance(data, list):
        parts = [canonicalize_json(item).decode('utf-8') for item in data]
        return ("[" + ",".join(parts) + "]").encode('utf-8')
    elif isinstance(data, int):
        return str(data).encode('utf-8')
    elif isinstance(data, str):
        return json.dumps(data).encode('utf-8')
    elif data is None:
        return b"null"
    elif data is True:
        return b"true"
    elif data is False:
        return b"false"
    else:
        raise ValueError(f"Unsupported type for canonicalization: {type(data)}")

# --- 2. Compiler (Spec -> IR) ---
def compile_spec(spec_path, ir_path):
    with open(spec_path, 'r') as f:
        spec = json.load(f)
    
    ir_functions = []
    main_body = []

    for task in spec.get("tasks", []):
        t_type = task.get("type")
        if t_type == "say":
            main_body.append({
                "op": "print",
                "args": [task.get("message", "")],
                "body": []
            })
        elif t_type == "calc":
            var = task.get("var", "x")
            expr = task.get("expr", "0")
            parts = expr.split()
            if len(parts) == 3 and parts[1] == "+":
                main_body.append({
                    "op": "add",
                    "args": [var, parts[0], parts[2]],
                    "body": []
                })
                main_body.append({
                    "op": "print_var",
                    "args": [var],
                    "body": []
                })
            else:
                main_body.append({
                    "op": "var_def",
                    "args": [var, expr],
                    "body": []
                })
        elif t_type == "repeat":
            count = str(task.get("count", 1))
            subtasks = []
            for sub in task.get("tasks", []):
                # Simple recursion for demo
                if sub.get("type") == "say":
                    subtasks.append({"op": "print", "args": [sub.get("message", "")], "body": []})
            
            main_body.append({
                "op": "loop",
                "args": [count],
                "body": subtasks
            })

    ir = {"functions": [{"name": "main", "body": main_body}]}
    
    # Write Canonical IR
    with open(ir_path, 'wb') as f:
        f.write(canonicalize_json(ir))

# --- 3. Emitter (IR -> Bash) ---
def emit_bash(ir_path, out_path):
    with open(ir_path, 'r') as f:
        ir = json.load(f)
    
    lines = ["#!/bin/bash", "set -e", ""]

    def emit_block(instrs, indent="    "):
        block_lines = []
        for instr in instrs:
            op = instr["op"]
            args = instr["args"]
            if op == "print":
                block_lines.append(f'{indent}echo "{args[0]}"')
            elif op == "print_var":
                block_lines.append(f'{indent}echo "${args[0]}"')
            elif op == "var_def":
                block_lines.append(f'{indent}{args[0]}={args[1]}')
            elif op == "add":
                block_lines.append(f'{indent}{args[0]}=$(( {args[1]} + {args[2]} ))')
            elif op == "loop":
                block_lines.append(f'{indent}for ((i=0; i<{args[0]}; i++)); do')
                block_lines.extend(emit_block(instr["body"], indent + "    "))
                block_lines.append(f'{indent}done')
        return block_lines

    for func in ir.get("functions", []):
        lines.append(f"{func['name']}() {{")
        lines.extend(emit_block(func["body"]))
        lines.append("}")
        lines.append("")

    lines.append("main")
    
    with open(out_path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(out_path, 0o755)

# --- Main Dispatch ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: stunir_core.py [compile|emit] ...")
        sys.exit(1)
    
    cmd = sys.argv[1]
    if cmd == "compile":
        # --in-spec X --out-ir Y
        spec = sys.argv[sys.argv.index("--in-spec") + 1]
        ir = sys.argv[sys.argv.index("--out-ir") + 1]
        compile_spec(spec, ir)
    elif cmd == "emit":
        # --in-ir X --out-file Y
        ir = sys.argv[sys.argv.index("--in-ir") + 1]
        out = sys.argv[sys.argv.index("--out-file") + 1]
        emit_bash(ir, out)

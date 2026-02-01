#!/usr/bin/env python3
"""
Apply v0.8.2 multi-level nesting patch to stunir_json_utils.adb
This script replaces the old single-level flattening code with recursive flattening.
"""

import re

def apply_patch():
    file_path = "tools/spark/src/stunir_json_utils.adb"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the start of body parsing section (line ~287)
    # This is where we have: -- Parse body statements (v0.8.0: with control flow support - parse only, flattening TODO)
    start_marker = "-- Parse body statements (v0.8.0: with control flow support - parse only, flattening TODO)"
    
    # Find the end of the old body parsing (before line ~797)
    # This is where we have the end of the while/for handling
    # We need to find: "Stmt_Pos := Stmt_End + 1;"  followed by "end loop;" followed by "end;" followed by "end if;" for Body_Pos check
    
    # Let's use a more specific approach - find the section between the two markers
    pattern = r'(-- Parse body statements \(v0\.8\.0.*?\n.*?if Body_Pos > 0 then\n.*?declare\n.*?Stmt_Pos : Natural.*?\n.*?Stmt_Start, Stmt_End : Natural;\n.*?Func_Idx : constant Positive.*?\n.*?begin\n)(.*?)(end;\s+end if;\s+end if;\s+end;\s+\n\s+Func_Pos := Obj_End \+ 1;)'
    
    # This is too complex. Let me use line numbers instead
    lines = content.split('\n')
    
    # Find line with "-- Parse body statements (v0.8.0"
    start_line = None
    for i, line in enumerate(lines):
        if "Parse body statements (v0.8.0" in line or "Parse body statements (v0.8" in line:
            start_line = i
            break
    
    if start_line is None:
        print("ERROR: Could not find start marker")
        return False
    
    # Find the end - look for "Func_Pos := Obj_End + 1;" after a long block
    # We want to find the end of the "if Body_Pos > 0" block
    # This should be around line 797
    end_line = None
    for i in range(start_line + 400, start_line + 600):
        if i < len(lines) and "Func_Pos := Obj_End + 1;" in lines[i]:
            end_line = i
            break
    
    if end_line is None:
        print("ERROR: Could not find end marker")
        return False
    
    print(f"Found section to replace: lines {start_line+1} to {end_line}")
    print(f"Old code: {end_line - start_line} lines")
    
    # Read the new code from our snippet file
    with open("recursive_flatten_snippet.ada", 'r') as f:
        new_code = f.read()
    
    # Build the replacement
    # We need to keep the comment and the "if Body_Pos > 0 then" check
    # But replace everything inside with the new Flatten_Block procedure + call
    
    # Extract the indentation from the original
    body_check_line = None
    for i in range(start_line, end_line):
        if "if Body_Pos > 0 then" in lines[i]:
            body_check_line = i
            break
    
    if body_check_line is None:
        print("ERROR: Could not find 'if Body_Pos > 0 then'")
        return False
    
    base_indent = len(lines[body_check_line]) - len(lines[body_check_line].lstrip())
    print(f"Base indentation: {base_indent} spaces")
    
    # Create the new section
    new_section_lines = [
        " " * (base_indent - 6) + "-- v0.8.2: Parse body statements with recursive multi-level nesting support",
        " " * (base_indent - 6) + "if Body_Pos > 0 then",
        " " * (base_indent) + "declare",
        " " * (base_indent + 3) + "Func_Idx : constant Positive := Module.Func_Cnt;",
        " " * (base_indent + 3) + "",
        " " * (base_indent + 3) + "-- v0.8.2: Recursive procedure to flatten nested statements",
    ]
    
    # Add the Flatten_Block procedure (indented properly)
    flatten_lines = new_code.split('\n')
    # Skip the first few comment lines and add the procedure
    in_procedure = False
    for line in flatten_lines:
        if line.strip().startswith('procedure Flatten_Block'):
            in_procedure = True
        if in_procedure:
            if line.strip() and not line.strip().startswith('--'):
                # Add proper indentation
                new_section_lines.append(" " * (base_indent + 3) + line)
            elif line.strip().startswith('--') and 'Then call it' not in line and 'Flatten_Block (Func_JSON' not in line:
                new_section_lines.append(" " * (base_indent + 3) + line)
            if line.strip().startswith('end Flatten_Block;'):
                break
    
    # Add the call to Flatten_Block
    new_section_lines.extend([
        " " * (base_indent + 3) + "",
        " " * (base_indent) + "begin",
        " " * (base_indent + 3) + "-- Call recursive flattening for function body",
        " " * (base_indent + 3) + "Flatten_Block (Func_JSON, Body_Pos);",
        " " * (base_indent) + "end;",
        " " * (base_indent - 6) + "end if;",
    ])
    
    # Replace the old section with the new one
    new_lines = lines[:start_line] + new_section_lines + lines[end_line:]
    
    new_content = '\n'.join(new_lines)
    
    # Write back
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    print(f"New code: {len(new_section_lines)} lines")
    print(f"Saved changes to {file_path}")
    print("SUCCESS: Patch applied!")
    return True

if __name__ == "__main__":
    success = apply_patch()
    exit(0 if success else 1)

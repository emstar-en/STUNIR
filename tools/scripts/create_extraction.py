import re
import json

# Read the original file
with open('ardupilot_crc.cpp', 'r') as f:
    content = f.read()

# Pattern to match function signatures
functions = []
for line_num, line in enumerate(content.split('\n'), 1):
    # Look for function definitions (not calls)
    match = re.match(r'^\s*(uint8_t|uint16_t|uint32_t|uint64_t|void|static|const)\s+(\w+)\s*\(([^)]*)\)', line)
    if match:
        # Skip if it's a variable declaration or function call
        if '=' in line and not '{' in line:
            continue
        # Skip if it's inside a function body (indented)
        if line.startswith('    ') or line.startswith('\t'):
            continue
        
        return_type = match.group(1)
        func_name = match.group(2)
        params_str = match.group(3).strip()
        
        # Parse parameters
        args = []
        if params_str and params_str != 'void':
            # Split by comma, but handle complex types
            param_parts = []
            depth = 0
            current = ""
            for char in params_str:
                if char == '(' or char == '<':
                    depth += 1
                    current += char
                elif char == ')' or char == '>':
                    depth -= 1
                    current += char
                elif char == ',' and depth == 0:
                    param_parts.append(current.strip())
                    current = ""
                else:
                    current += char
            if current.strip():
                param_parts.append(current.strip())
            
            for param in param_parts:
                param = param.strip()
                if param:
                    # Extract type and name
                    # Handle const, pointers, etc.
                    parts = param.split()
                    if len(parts) >= 2:
                        # Last part is usually the name
                        name = parts[-1].replace('*', '').replace('&', '')
                        type_name = ' '.join(parts[:-1])
                        args.append({"name": name, "type": type_name})
                    elif len(parts) == 1:
                        # Just a type (unnamed parameter)
                        args.append({"name": "arg" + str(len(args)), "type": parts[0]})
        
        functions.append({
            "name": func_name,
            "type": "function",
            "signature": f"{return_type} {func_name}({params_str})",
            "return_type": return_type,
            "args": args
        })

# Create extraction.json
extraction = {
    "elements": functions
}

output_path = 'stunir_runs/ardupilot_full/extraction.json'
with open(output_path, 'w') as f:
    json.dump(extraction, f, indent=2)

print(f"Created {output_path} with {len(functions)} functions")
for func in functions:
    print(f"  - {func['signature']}")
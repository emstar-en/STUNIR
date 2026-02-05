import re
import json
import os
from pathlib import Path


def parse_param_decl(param_str, index=0):
    """
    Parse a C/C++ parameter declaration into type and name.

    Handles:
    - Simple types: int x, char c
    - Pointers: const char *str, char** argv
    - Arrays: int arr[10], char *argv[]
    - Function pointers: int (*cb)(int), void (* const handler)(int)
    - References: int &ref
    - Unnamed parameters: int -> type='int', name='argN'
    - Varargs: ... -> type='...', name='args'

    Args:
        param_str: The parameter declaration string
        index: Index for generating fallback argN names

    Returns:
        dict with 'type' and 'name' keys
    """
    param_str = param_str.strip()

    if not param_str:
        return {"type": "void", "name": f"arg{index}"}

    # Handle varargs
    if param_str == "...":
        return {"type": "...", "name": "args"}

    # Extract array suffixes
    array_suffix = ""
    array_match = re.search(r'(\s*\[[^\]]*\])+\s*$', param_str)
    if array_match:
        array_suffix = array_match.group(0)
        param_str = param_str[:array_match.start()]

    # Handle function pointers with qualifiers: type (* const name)(params)
    func_ptr_match = re.search(r'\((\*|&)\s*(const\s+|volatile\s+)*([A-Za-z_][A-Za-z0-9_]*)\s*\)', param_str)
    if func_ptr_match:
        name = func_ptr_match.group(3)
        ptr_char = func_ptr_match.group(1)
        qualifiers = func_ptr_match.group(2) or ""
        before = param_str[:func_ptr_match.start()]
        after = param_str[func_ptr_match.end():]
        if qualifiers:
            qualifiers = qualifiers.strip()
            type_str = before + f"({ptr_char} {qualifiers})" + after + array_suffix
        else:
            type_str = before + f"({ptr_char})" + after + array_suffix
        type_str = re.sub(r'\s+', ' ', type_str).strip()
        return {"type": type_str, "name": name}

    # Find the rightmost identifier
    id_match = re.search(r'(?<![\w])([A-Za-z_][A-Za-z0-9_]*)\s*$', param_str)

    if id_match:
        name = id_match.group(1)
        type_str = param_str[:id_match.start()].strip()

        # Check if this is a type-only parameter (unnamed)
        simple_types = {'int', 'char', 'float', 'double', 'void', 'short', 'long',
                       'signed', 'unsigned', 'bool', 'size_t', 'ssize_t', 'ptrdiff_t',
                       'uint8_t', 'uint16_t', 'uint32_t', 'uint64_t',
                       'int8_t', 'int16_t', 'int32_t', 'int64_t'}

        if name in simple_types and not type_str:
            return {"type": name, "name": f"arg{index}"}

        # Normalize type spacing
        type_str = re.sub(r'\s+', ' ', type_str)
        type_str = type_str.strip()

        # Add array suffix back to type with proper spacing
        if array_suffix:
            array_suffix = array_suffix.strip()
            if type_str and not type_str.endswith(' '):
                type_str = type_str + ' '
            type_str = type_str + array_suffix

        return {"type": type_str, "name": name}

    # No identifier found - treat as unnamed parameter
    type_str = param_str + array_suffix
    type_str = re.sub(r'\s+', ' ', type_str).strip()
    return {"type": type_str, "name": f"arg{index}"}


def extract_c_functions(filepath):
    """Extract function signatures from C source file"""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    functions = []
    
    # Pattern for C function definitions
    # Matches: return_type function_name(params) {
    # Handles: static, const, pointers, etc.
    pattern = r'^(static\s+|const\s+|inline\s+)*([a-zA-Z_][a-zA-Z0-9_]*\s*\*?\s+)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)\s*\{'
    
    for match in re.finditer(pattern, content, re.MULTILINE):
        qualifiers = match.group(1) or ""
        return_type = (qualifiers + match.group(2)).strip()
        func_name = match.group(3)
        params_str = match.group(4).strip()
        
        # Parse parameters
        parameters = []
        if params_str and params_str != 'void':
            # Split by comma, handling nested parens
            param_parts = []
            depth = 0
            current = ""
            for char in params_str:
                if char in '(<[':
                    depth += 1
                    current += char
                elif char in ')>]':
                    depth -= 1
                    current += char
                elif char == ',' and depth == 0:
                    param_parts.append(current.strip())
                    current = ""
                else:
                    current += char
            if current.strip():
                param_parts.append(current.strip())
            
            for i, param in enumerate(param_parts):
                param = param.strip()
                if param:
                    # Use robust parameter parsing
                    parsed = parse_param_decl(param, i)
                    parameters.append(parsed)
        
        functions.append({
            'name': func_name,
            'return_type': return_type,
            'parameters': parameters,
            'source_file': str(filepath)
        })
    
    return functions

def create_extraction_json(source_files, output_file):
    """Create extraction.json from multiple source files"""
    all_functions = []
    
    for source_file in source_files:
        if Path(source_file).exists():
            funcs = extract_c_functions(source_file)
            all_functions.extend(funcs)
            print(f"Extracted {len(funcs)} functions from {source_file}")
        else:
            print(f"Warning: {source_file} not found")
    
    extraction = {
        'source_files': source_files,
        'total_functions': len(all_functions),
        'functions': all_functions
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(extraction, f, indent=2)
    
    print(f"\nCreated {output_file} with {len(all_functions)} total functions")
    return extraction

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python extract_bc_functions.py <output.json> <source1.c> [source2.c] ...")
        sys.exit(1)
    
    output_file = sys.argv[1]
    source_files = sys.argv[2:]
    
    create_extraction_json(source_files, output_file)

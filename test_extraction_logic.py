#!/usr/bin/env python3
"""Unit tests for parameter parsing logic in extract_bc_functions.py"""

import re
import sys

# Test cases from param_parsing_fix_plan.json
TEST_CASES = [
    {
        "case": "Pointer with space",
        "example": "const char *str",
        "expected": {"type": "const char *", "name": "str"}
    },
    {
        "case": "Pointer without space",
        "example": "char*str",
        "expected": {"type": "char*", "name": "str"}
    },
    {
        "case": "Reference",
        "example": "int &ref",
        "expected": {"type": "int &", "name": "ref"}
    },
    {
        "case": "Double pointer",
        "example": "char **argv",
        "expected": {"type": "char **", "name": "argv"}
    },
    {
        "case": "Const pointer to const",
        "example": "const char * const p",
        "expected": {"type": "const char * const", "name": "p"}
    },
    {
        "case": "Struct pointer",
        "example": "struct my_struct *ptr",
        "expected": {"type": "struct my_struct *", "name": "ptr"}
    },
    {
        "case": "Array parameter",
        "example": "int vals[10]",
        "expected": {"type": "int [10]", "name": "vals"}
    },
    {
        "case": "Array of pointers",
        "example": "char *argv[]",
        "expected": {"type": "char * []", "name": "argv"}
    },
    {
        "case": "Multi-dimensional array",
        "example": "int matrix[3][3]",
        "expected": {"type": "int [3][3]", "name": "matrix"}
    },
    {
        "case": "Function pointer",
        "example": "int (*cb)(int)",
        "expected": {"type": "int (*)(int)", "name": "cb"}
    },
    {
        "case": "Function pointer with qualifiers",
        "example": "void (* const handler)(int)",
        "expected": {"type": "void (* const)(int)", "name": "handler"}
    },
    {
        "case": "Pointer to array",
        "example": "int (*arr)[10]",
        "expected": {"type": "int (*)[10]", "name": "arr"}
    },
    {
        "case": "Unnamed parameter",
        "example": "int",
        "expected": {"type": "int", "name": "arg12"}
    },
    {
        "case": "Void types",
        "example": "void",
        "expected": {"type": "void", "name": "arg13"}
    },
    {
        "case": "Variable arguments",
        "example": "...",
        "expected": {"type": "...", "name": "args"}
    },
    {
        "case": "Typedef name",
        "example": "size_t n",
        "expected": {"type": "size_t", "name": "n"}
    },
    {
        "case": "Pointer with macro-like identifier",
        "example": "MYTYPE *x",
        "expected": {"type": "MYTYPE *", "name": "x"}
    }
]


def parse_param_decl(param_str, index=0):
    """
    Parse a C/C++ parameter declaration into type and name.

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
    # Pattern: (* followed by optional qualifiers and then the name)
    func_ptr_match = re.search(r'\((\*|&)\s*(const\s+|volatile\s+)*([A-Za-z_][A-Za-z0-9_]*)\s*\)', param_str)
    if func_ptr_match:
        name = func_ptr_match.group(3)
        ptr_char = func_ptr_match.group(1)
        qualifiers = func_ptr_match.group(2) or ""
        # Build type: everything before ( + (* + qualifiers + ) + everything after
        before = param_str[:func_ptr_match.start()]
        after = param_str[func_ptr_match.end():]
        # Normalize qualifiers spacing
        if qualifiers:
            qualifiers = qualifiers.strip()
            type_str = before + f"({ptr_char} {qualifiers})" + after + array_suffix
        else:
            type_str = before + f"({ptr_char})" + after + array_suffix
        # Normalize spaces
        type_str = re.sub(r'\s+', ' ', type_str).strip()
        return {"type": type_str, "name": name}

    # Find the rightmost identifier
    # Look for a valid C identifier at the end (after removing arrays)
    id_match = re.search(r'(?<![\w])([A-Za-z_][A-Za-z0-9_]*)\s*$', param_str)

    if id_match:
        name = id_match.group(1)
        type_str = param_str[:id_match.start()].strip()

        # Check if this is a type-only parameter (common C types without name)
        # If the "name" is actually a type keyword and there's no other type info
        simple_types = {'int', 'char', 'float', 'double', 'void', 'short', 'long',
                       'signed', 'unsigned', 'bool', 'size_t', 'ssize_t', 'ptrdiff_t'}

        if name in simple_types and not type_str:
            # This is an unnamed parameter like just "int"
            return {"type": name, "name": f"arg{index}"}

        # Normalize type spacing - preserve original style for pointers
        # Don't normalize if it's a simple pointer pattern
        if '*' in type_str or '&' in type_str:
            # For pointers, we want to keep them attached to the type
            # Just clean up excessive whitespace
            type_str = re.sub(r'\s+', ' ', type_str)
        else:
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
    
    # Find the rightmost identifier
    # Look for a valid C identifier at the end (after removing arrays)
    id_match = re.search(r'(?<![\w])([A-Za-z_][A-Za-z0-9_]*)\s*$', param_str)
    
    if id_match:
        name = id_match.group(1)
        type_str = param_str[:id_match.start()].strip()
        
        # Normalize type spacing
        type_str = re.sub(r'\s+', ' ', type_str)
        type_str = re.sub(r'\s*\*\s*', '*', type_str)  # Remove spaces around *
        type_str = re.sub(r'\*', ' *', type_str)  # Add single space before *
        type_str = type_str.strip()
        
        # Add array suffix back to type
        if array_suffix:
            type_str = type_str + array_suffix
        
        return {"type": type_str, "name": name}
    
    # No identifier found - treat as unnamed parameter
    type_str = param_str + array_suffix
    return {"type": type_str.strip(), "name": f"arg{index}"}


def run_tests():
    """Run all test cases and report results"""
    passed = 0
    failed = 0
    
    print("=" * 70)
    print("PARAMETER PARSING UNIT TESTS")
    print("=" * 70)
    
    for i, test in enumerate(TEST_CASES):
        result = parse_param_decl(test["example"], i)
        expected = test["expected"].copy()
        
        # Handle argN naming for unnamed params
        if expected["name"] == "argN":
            expected["name"] = f"arg{i}"
        
        match = (result["type"] == expected["type"] and 
                 result["name"] == expected["name"])
        
        status = "PASS" if match else "FAIL"
        
        print(f"\n[{status}] {test['case']}")
        print(f"  Input:    '{test['example']}'")
        print(f"  Expected: type='{expected['type']}', name='{expected['name']}'")
        print(f"  Got:      type='{result['type']}', name='{result['name']}'")
        
        if match:
            passed += 1
        else:
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(TEST_CASES)} tests")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

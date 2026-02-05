import re
import json

def extract_signatures_from_cpp(filepath):
    """Extract function signatures from C++ file"""
    signatures = {}
    with open(filepath, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()
        # Look for function definitions
        if re.match(r'^(uint8_t|uint16_t|uint32_t|uint64_t|void|static|const)\s+\w+\s*\(', stripped):
            # Skip if it's a variable declaration or function call
            if '=' in stripped and '{' not in stripped:
                continue
            # Skip if it's inside a function body (indented)
            if line.startswith('    ') or line.startswith('\t'):
                continue
            
            # Extract function signature
            match = re.match(r'^(uint8_t|uint16_t|uint32_t|uint64_t|void)\s+(\w+)\s*\(([^)]*)\)', stripped)
            if match:
                return_type = match.group(1)
                func_name = match.group(2)
                params = match.group(3).strip()
                signatures[func_name] = {
                    'return_type': return_type,
                    'params': params,
                    'line': line_num
                }
    return signatures

# Extract from original
original_sigs = extract_signatures_from_cpp('ardupilot_crc.cpp')

# Extract from generated
generated_sigs = extract_signatures_from_cpp('stunir_runs/ardupilot_full/generated_final.cpp')

print("=" * 70)
print("STUNIR PIPELINE COMPARISON REPORT")
print("=" * 70)
print()

print(f"Original ardupilot_crc.cpp: {len(original_sigs)} functions")
print(f"Generated generated_final.cpp: {len(generated_sigs)} functions")
print()

# Find missing functions
missing_in_generated = set(original_sigs.keys()) - set(generated_sigs.keys())
extra_in_generated = set(generated_sigs.keys()) - set(original_sigs.keys())

if missing_in_generated:
    print("MISSING in generated.cpp:")
    for func in sorted(missing_in_generated):
        sig = original_sigs[func]
        print(f"  - {sig['return_type']} {func}({sig['params']})")
    print()

if extra_in_generated:
    print("EXTRA in generated.cpp:")
    for func in sorted(extra_in_generated):
        sig = generated_sigs[func]
        print(f"  - {sig['return_type']} {func}({sig['params']})")
    print()

# Compare signatures
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Total functions in original:    {len(original_sigs)}")
print(f"Total functions in generated:   {len(generated_sigs)}")
print(f"Common functions:               {len(set(original_sigs.keys()) & set(generated_sigs.keys()))}")
print(f"Missing in generated:           {len(missing_in_generated)}")
print(f"Extra in generated:             {len(extra_in_generated)}")
print()

if not missing_in_generated and len(original_sigs) == len(generated_sigs):
    print("SUCCESS: All function signatures match!")
else:
    print(f"NOTE: {len(missing_in_generated)} functions missing")
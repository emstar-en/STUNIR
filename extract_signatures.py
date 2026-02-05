import re

# Read the original file
with open('ardupilot_crc.cpp', 'r') as f:
    content = f.read()

# Pattern to match function signatures
# Matches: return_type function_name(params)
pattern = r'^\s*(uint8_t|uint16_t|uint32_t|uint64_t|void|static|const)\s+\w+\s*\([^)]*\)\s*\{?'

functions = []
for line_num, line in enumerate(content.split('\n'), 1):
    # Look for function definitions (not calls)
    if re.match(r'^\s*(uint8_t|uint16_t|uint32_t|uint64_t|void|static|const)\s+\w+\s*\(', line):
        # Skip if it's a variable declaration or function call
        if '=' in line and not '{' in line:
            continue
        # Skip if it's inside a function body (indented)
        if line.startswith('    ') or line.startswith('\t'):
            continue
        # Clean up the line
        func_sig = line.strip()
        if func_sig.endswith('{'):
            func_sig = func_sig[:-1].strip()
        functions.append((line_num, func_sig))

print("=== Functions found in ardupilot_crc.cpp ===")
print(f"Total: {len(functions)}\n")
for line_num, sig in functions:
    print(f"Line {line_num}: {sig}")

# Save to file for reference
with open('stunir_runs/ardupilot_full/baseline_signatures.txt', 'w') as f:
    f.write("=== Functions found in ardupilot_crc.cpp ===\n")
    f.write(f"Total: {len(functions)}\n\n")
    for line_num, sig in functions:
        f.write(f"Line {line_num}: {sig}\n")

print(f"\nSaved to stunir_runs/ardupilot_full/baseline_signatures.txt")
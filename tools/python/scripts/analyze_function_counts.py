import os
import re
from pathlib import Path

def count_functions_in_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Simple regex for C function definitions:
    # return_type name(params) {
    # It's not perfect but good enough for estimation
    # Excludes static functions if needed, but we probably want to count them for now
    # We look for: start of line, type, space, name, space?, (, ..., ), space?, {
    
    # 1. Type: roughly [a-zA-Z0-9_*]+ (including pointers)
    # 2. Name: [a-zA-Z0-9_]+
    # 3. Params: (...)
    # 4. Start block: {
    
    # This regex looks for:
    # ^ - start of line (assuming standard formatting)
    # (static\s+)? - optional static
    # [a-zA-Z0-9_]+\s+[\*]* - return type
    # [a-zA-Z0-9_]+ - function name
    # \s*\( - open paren
    # [^;]+ - args (no semicolon inside args usually, avoiding prototypes)
    # \)\s*\{ - close paren and start brace
    
    pattern = r'^(?:static\s+|const\s+|inline\s+)*[a-zA-Z0-9_]+\s+\*?([a-zA-Z0-9_]+)\s*\([^;]*\)\s*\{'
    # Less strict:
    # pattern = r'^\s*(?:[\w\*]+\s+)+(\w+)\s*\([^\)]*\)\s*\{'
    
    matches = re.findall(pattern, content, re.MULTILINE)
    return len(matches), matches

def analyze_directory(name, path, limit=100):
    print(f"Analyzing {name} in {path}...")
    total_funcs = 0
    files_map = {}
    
    path_obj = Path(path)
    if not path_obj.exists():
        print(f"  Error: Path {path} does not exist.")
        return

    # Handle directory or single file (like sqlite amalgamation)
    files = []
    if path_obj.is_file():
        files = [path_obj]
    else:
        files = list(path_obj.glob('**/*.c'))

    for p in files:
        count, funcs = count_functions_in_file(p)
        if count > 0:
            files_map[str(p)] = count
            total_funcs += count
            # print(f"  {p.name}: {count}")

    print(f"  Total Functions: {total_funcs}")
    print(f"  Total Files: {len(files)}")
    
    batches_needed = (total_funcs + limit - 1) // limit
    print(f"  Estimated Batches needed (limit {limit}): {batches_needed}")
    print("-" * 40)
    return total_funcs

if __name__ == "__main__":
    base = Path("stunir_execution_workspace/sources")
    
    analyze_directory("GNU bc", base / "bc-1.07.1")
    analyze_directory("Lua 5.4", base / "lua-5.4.6/src")
    analyze_directory("SQLite", base / "sqlite-src/sqlite-amalgamation-3450000/sqlite3.c")

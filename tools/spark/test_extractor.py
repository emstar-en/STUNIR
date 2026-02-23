#!/usr/bin/env python3
"""Test runner for SPARK extractor with output capture."""

import subprocess
import os
import sys
import json

def main():
    spark_dir = r"C:\Users\MSTAR\AppData\Roaming\AbacusAI\Agent Workspaces\stunir\tools\spark"
    exe = os.path.join(spark_dir, "bin", "spark_extract_main.exe")
    input_file = os.path.join(spark_dir, "test_data", "golden_test.ads")
    output_file = os.path.join(spark_dir, "test_data", "golden_test_extraction.json")
    
    print("=== SPARK Extractor Test ===")
    print(f"Exe: {exe}")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print()
    
    # Check files exist
    if not os.path.exists(exe):
        print(f"ERROR: Exe not found: {exe}")
        return 1
    if not os.path.exists(input_file):
        print(f"ERROR: Input not found: {input_file}")
        return 1
    
    # Remove old output
    for suffix in ["", ".started.txt", ".ok.txt", ".error.txt"]:
        path = output_file + suffix
        if os.path.exists(path):
            os.remove(path)
    
    # Run extractor
    print("=== Running Extractor ===")
    result = subprocess.run(
        [exe, "-i", input_file, "-o", output_file, "--lang", "spark"],
        capture_output=True,
        text=True,
        cwd=spark_dir
    )
    
    print(f"Exit Code: {result.returncode}")
    print(f"\nSTDOUT:\n{result.stdout}")
    print(f"\nSTDERR:\n{result.stderr}")
    
    # Check output files
    print("\n=== Output Files ===")
    for suffix in ["", ".started.txt", ".ok.txt", ".error.txt"]:
        path = output_file + suffix
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"\n{os.path.basename(path)} ({size} bytes):")
            with open(path, 'r') as f:
                content = f.read()
                print(content)
        else:
            print(f"\n{suffix if suffix else 'output'}: NOT FOUND")
    
    # Validate JSON
    if os.path.exists(output_file):
        print("\n=== JSON Validation ===")
        try:
            with open(output_file, 'r') as f:
                data = json.load(f)
            print(f"Schema: {data.get('schema', 'N/A')}")
            print(f"Module: {data.get('module_name', 'N/A')}")
            print(f"Language: {data.get('language', 'N/A')}")
            funcs = data.get('functions', [])
            print(f"Functions: {len(funcs)}")
            for func in funcs:
                print(f"  - {func.get('name', 'N/A')}: {func.get('return_type', 'void')}")
        except json.JSONDecodeError as e:
            print(f"JSON Error: {e}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

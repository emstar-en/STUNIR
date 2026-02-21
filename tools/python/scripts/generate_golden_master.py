import subprocess
import os
from pathlib import Path

def run_command(cmd, cwd=None, input_data=None):
    """Run a command and return stdout, stderr, returncode"""
    result = subprocess.run(
        cmd,
        cwd=cwd,
        input=input_data,
        capture_output=True,
        text=True,
        shell=True
    )
    return result.stdout, result.stderr, result.returncode

def build_bc(bc_dir):
    """Build bc from source"""
    print(f"Building bc in {bc_dir}...")
    
    # Configure
    stdout, stderr, rc = run_command("./configure", cwd=bc_dir)
    if rc != 0:
        print(f"Configure failed: {stderr}")
        return None
    
    # Make
    stdout, stderr, rc = run_command("make", cwd=bc_dir)
    if rc != 0:
        print(f"Make failed: {stderr}")
        return None
    
    bc_exe = Path(bc_dir) / "bc" / "bc.exe"
    if bc_exe.exists():
        print(f"Built bc: {bc_exe}")
        return str(bc_exe)
    
    # Try alternative location
    bc_exe = Path(bc_dir) / "bc.exe"
    if bc_exe.exists():
        print(f"Built bc: {bc_exe}")
        return str(bc_exe)
    
    print("Could not find bc.exe after build")
    return None

def generate_golden_outputs(bc_exe, test_dir, output_dir):
    """Generate golden master outputs from bc tests"""
    os.makedirs(output_dir, exist_ok=True)
    
    test_files = [
        "array.b", "arrayp.b", "aryprm.b", "atan.b", "checklib.b",
        "div.b", "exp.b", "fact.b", "jn.b", "ln.b", "mul.b",
        "raise.b", "sine.b", "sqrt.b", "sqrt1.b", "sqrt2.b", "testfn.b"
    ]
    
    results = {}
    for test_file in test_files:
        test_path = Path(test_dir) / test_file
        if not test_path.exists():
            print(f"  Skipping {test_file} (not found)")
            continue
        
        with open(test_path, 'r') as f:
            test_input = f.read()
        
        stdout, stderr, rc = run_command(f'"{bc_exe}" -l', input_data=test_input)
        
        output_file = Path(output_dir) / f"{test_file}.out"
        with open(output_file, 'w') as f:
            f.write(stdout)
            if stderr:
                f.write("\n--- STDERR ---\n")
                f.write(stderr)
        
        results[test_file] = {
            'output': output_file.name,
            'returncode': rc,
            'has_output': len(stdout) > 0
        }
        print(f"  Generated {output_file.name}")
    
    return results

def main():
    base_dir = Path("stunir_execution_workspace/sources/bc-1.07.1")
    test_dir = base_dir / "Test"
    output_dir = Path("stunir_execution_workspace/golden/bc")
    
    # Build bc
    bc_exe = build_bc(str(base_dir))
    if not bc_exe:
        print("Failed to build bc")
        return
    
    # Generate golden outputs
    print(f"\nGenerating golden outputs to {output_dir}...")
    results = generate_golden_outputs(bc_exe, str(test_dir), str(output_dir))
    
    # Summary
    print("\n=== Golden Master Summary ===")
    for test, info in results.items():
        status = "✓" if info['has_output'] else "○"
        print(f"  {status} {test}: {info['output']}")

if __name__ == "__main__":
    main()

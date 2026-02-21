#!/usr/bin/env python3
"""Simple test of the pipeline without rich dependencies."""

import json
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from bridge_spec_assemble import assemble_spec
from bridge_spec_to_ir import convert_spec_to_ir
from bridge_ir_to_code import generate_code, get_file_extension

def main():
    input_file = Path("stunir_execution_workspace/gnu_bc/batch_01/extraction.json")
    output_dir = Path("stunir_execution_workspace/gnu_bc/batch_01/output")
    targets = ["cpp", "c", "python"]
    
    print("STUNIR Pipeline Test (Simple)")
    print("=" * 60)
    
    # Phase 1
    print("\nPhase 1: Spec Assembly")
    with open(input_file, 'r') as f:
        extraction_data = json.load(f)
    spec_data = assemble_spec(extraction_data, "execute")
    spec_file = output_dir / "spec.json"
    spec_file.parent.mkdir(parents=True, exist_ok=True)
    with open(spec_file, 'w') as f:
        json.dump(spec_data, f, indent=2)
    print(f"  Created: {spec_file}")
    print(f"  Functions: {len(spec_data.get('functions', []))}")
    
    # Phase 2
    print("\nPhase 2: IR Conversion")
    ir_data = convert_spec_to_ir(spec_data, "execute")
    ir_file = output_dir / "ir.json"
    with open(ir_file, 'w') as f:
        json.dump(ir_data, f, indent=2)
    print(f"  Created: {ir_file}")
    print(f"  Functions: {len(ir_data.get('functions', []))}")
    
    # Phase 3
    print("\nPhase 3: Code Emission")
    for target in targets:
        code = generate_code(ir_data, target)
        ext = get_file_extension(target)
        output_file = output_dir / f"execute{ext}"
        with open(output_file, 'w') as f:
            f.write(code)
        print(f"  Created: {output_file}")
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print(f"Output directory: {output_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

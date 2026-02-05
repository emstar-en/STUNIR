#!/usr/bin/env python3
"""
STUNIR Hardened Pipeline - No interactive features, minimal dependencies
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

# Force stdout to be unbuffered
sys.stdout.reconfigure(line_buffering=True)


def log(msg: str):
    """Simple logging that flushes immediately."""
    print(msg, flush=True)


def run_phase1(extraction_file: Path, spec_file: Path, module_name: str) -> bool:
    """Run Phase 1: extraction.json -> spec.json"""
    log("PHASE 1: Spec Assembly")
    
    try:
        from bridge_spec_assemble import assemble_spec
        
        with open(extraction_file, 'r') as f:
            extraction_data = json.load(f)
            
        spec_data = assemble_spec(extraction_data, module_name)
        
        spec_file.parent.mkdir(parents=True, exist_ok=True)
        with open(spec_file, 'w') as f:
            json.dump(spec_data, f, indent=2)
                
        log(f"  OK: Generated spec with {len(spec_data.get('functions', []))} functions")
        return True
        
    except Exception as e:
        log(f"  FAIL: Phase 1 error: {e}")
        return False


def run_phase2(spec_file: Path, ir_file: Path, module_name: str) -> bool:
    """Run Phase 2: spec.json -> ir.json"""
    log("PHASE 2: IR Conversion")
    
    try:
        from bridge_spec_to_ir import convert_spec_to_ir
        
        with open(spec_file, 'r') as f:
            spec_data = json.load(f)
            
        ir_data = convert_spec_to_ir(spec_data, module_name)
        
        ir_file.parent.mkdir(parents=True, exist_ok=True)
        with open(ir_file, 'w') as f:
            json.dump(ir_data, f, indent=2)
                
        log(f"  OK: Generated IR with {len(ir_data.get('functions', []))} functions")
        return True
        
    except Exception as e:
        log(f"  FAIL: Phase 2 error: {e}")
        return False


def run_phase3(ir_file: Path, output_dir: Path, targets: List[str]) -> bool:
    """Run Phase 3: ir.json -> generated code"""
    log("PHASE 3: Code Emission")
    
    try:
        from bridge_ir_to_code import generate_code, get_file_extension
        
        with open(ir_file, 'r') as f:
            ir_data = json.load(f)
            
        generated_count = 0
        
        for target in targets:
            try:
                code = generate_code(ir_data, target)
                ext = get_file_extension(target)
                
                module_name = ir_data.get("module", "generated")
                output_file = output_dir / f"{module_name}{ext}"
                
                output_dir.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'w') as f:
                    f.write(code)
                    
                generated_count += 1
                log(f"  OK: Generated {target} -> {output_file.name}")
            except Exception as e:
                log(f"  WARN: Failed to generate {target}: {e}")
                
        log(f"  OK: Generated {generated_count}/{len(targets)} target files")
        return generated_count > 0
        
    except Exception as e:
        log(f"  FAIL: Phase 3 error: {e}")
        return False


def run_pipeline(extraction_file: Path, output_dir: Path, 
                 targets: List[str], module_name: Optional[str] = None) -> bool:
    """Run the complete pipeline."""
    
    log("")
    log("=" * 60)
    log("STUNIR Hardened Pipeline v2.1.0")
    log("=" * 60)
    
    if not extraction_file.exists():
        log(f"FAIL: Extraction file not found: {extraction_file}")
        return False
        
    if module_name is None:
        module_name = extraction_file.stem.replace("_extraction", "").replace("extraction", "module")
        
    spec_file = output_dir / "spec.json"
    ir_file = output_dir / "ir.json"
    
    log(f"Input: {extraction_file}")
    log(f"Output: {output_dir}")
    log(f"Targets: {', '.join(targets)}")
    log(f"Module: {module_name}")
    log("")
    
    # Run phases
    if not run_phase1(extraction_file, spec_file, module_name):
        return False
        
    if not run_phase2(spec_file, ir_file, module_name):
        return False
        
    if not run_phase3(ir_file, output_dir, targets):
        return False
            
    log("")
    log("=" * 60)
    log("PIPELINE COMPLETED SUCCESSFULLY")
    log("=" * 60)
    log("")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="STUNIR Hardened Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("-i", "--input", required=True, type=Path,
                        help="Input extraction.json file")
    parser.add_argument("-o", "--output", required=True, type=Path,
                        help="Output directory for generated files")
    parser.add_argument("-t", "--targets", nargs="+", 
                        default=["cpp"],
                        help="Target language(s) for code generation")
    parser.add_argument("-m", "--module", type=str,
                        help="Module name (default: derived from input file)")
    
    args = parser.parse_args()
    
    success = run_pipeline(
        extraction_file=args.input,
        output_dir=args.output,
        targets=args.targets,
        module_name=args.module
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

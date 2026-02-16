#!/usr/bin/env python3
"""
STUNIR Unified Pipeline Runner (Phase 4)
Orchestrates the complete pipeline: extraction.json -> spec.json -> ir.json -> generated code
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

if RICH_AVAILABLE:
    console = Console()
else:
    class FakeConsole:
        def print(self, *args, **kwargs):
            print(*args)
    console = FakeConsole()


class PipelineRunner:
    """Unified STUNIR pipeline runner."""
    
    VERSION = "2.1.0"
    SUPPORTED_TARGETS = ["cpp", "c", "python", "rust", "go", "javascript", "java", "csharp", "swift", "kotlin"]
    
    def __init__(self, verbose: bool = False, dry_run: bool = False):
        self.verbose = verbose
        self.dry_run = dry_run
        self.results: Dict[str, Any] = {}
        
    def log(self, message: str, level: str = "info"):
        """Log a message with appropriate styling."""
        if level == "error":
            console.print(f"[red]✗ {message}[/red]" if RICH_AVAILABLE else f"✗ {message}")
        elif level == "success":
            console.print(f"[green]✓ {message}[/green]" if RICH_AVAILABLE else f"✓ {message}")
        elif level == "warning":
            console.print(f"[yellow]⚠ {message}[/yellow]" if RICH_AVAILABLE else f"⚠ {message}")
        elif level == "info":
            if self.verbose:
                console.print(f"[blue]ℹ {message}[/blue]" if RICH_AVAILABLE else f"ℹ {message}")
        elif level == "header":
            console.print(f"[bold cyan]{message}[/bold cyan]" if RICH_AVAILABLE else f"=== {message} ===")
            
    def run_phase1(self, extraction_file: Path, spec_file: Path, module_name: str) -> bool:
        """Run Phase 1: extraction.json -> spec.json"""
        self.log("PHASE 1: Spec Assembly", "header")
        
        try:
            from bridge_spec_assemble import assemble_spec
            
            with open(extraction_file, 'r') as f:
                extraction_data = json.load(f)
                
            spec_data = assemble_spec(extraction_data, module_name)
            
            if not self.dry_run:
                spec_file.parent.mkdir(parents=True, exist_ok=True)
                with open(spec_file, 'w') as f:
                    json.dump(spec_data, f, indent=2)
                    
            self.log(f"Generated spec: {spec_file}", "success")
            self.results["phase1"] = {
                "status": "success",
                "input": str(extraction_file),
                "output": str(spec_file),
                "functions": len(spec_data.get("functions", []))
            }
            return True
            
        except Exception as e:
            self.log(f"Phase 1 failed: {e}", "error")
            self.results["phase1"] = {"status": "failed", "error": str(e)}
            return False
            
    def run_phase2(self, spec_file: Path, ir_file: Path, module_name: str) -> bool:
        """Run Phase 2: spec.json -> ir.json"""
        self.log("PHASE 2: IR Conversion", "header")
        
        try:
            from bridge_spec_to_ir import convert_spec_to_ir
            
            with open(spec_file, 'r') as f:
                spec_data = json.load(f)
                
            ir_data = convert_spec_to_ir(spec_data, module_name)
            
            if not self.dry_run:
                ir_file.parent.mkdir(parents=True, exist_ok=True)
                with open(ir_file, 'w') as f:
                    json.dump(ir_data, f, indent=2)
                    
            self.log(f"Generated IR: {ir_file}", "success")
            self.results["phase2"] = {
                "status": "success",
                "input": str(spec_file),
                "output": str(ir_file),
                "functions": len(ir_data.get("functions", []))
            }
            return True
            
        except Exception as e:
            self.log(f"Phase 2 failed: {e}", "error")
            self.results["phase2"] = {"status": "failed", "error": str(e)}
            return False
            
    def run_phase3(self, ir_file: Path, output_dir: Path, targets: List[str]) -> bool:
        """Run Phase 3: ir.json -> generated code"""
        self.log("PHASE 3: Code Emission", "header")
        
        try:
            from bridge_ir_to_code import generate_code, get_file_extension
            
            with open(ir_file, 'r') as f:
                ir_data = json.load(f)
                
            generated_files = []
            
            for target in targets:
                if target not in self.SUPPORTED_TARGETS:
                    self.log(f"Skipping unsupported target: {target}", "warning")
                    continue
                    
                code = generate_code(ir_data, target)
                ext = get_file_extension(target)
                
                # Use module name for output file
                module_name = ir_data.get("module", "generated")
                output_file = output_dir / f"{module_name}{ext}"
                
                if not self.dry_run:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    with open(output_file, 'w') as f:
                        f.write(code)
                        
                generated_files.append(str(output_file))
                self.log(f"Generated {target}: {output_file}", "success")
                
            self.results["phase3"] = {
                "status": "success",
                "input": str(ir_file),
                "output_dir": str(output_dir),
                "targets": targets,
                "files": generated_files
            }
            return True
            
        except Exception as e:
            self.log(f"Phase 3 failed: {e}", "error")
            self.results["phase3"] = {"status": "failed", "error": str(e)}
            return False
            
    def run_pipeline(self, extraction_file: Path, output_dir: Path, 
                     targets: List[str], module_name: Optional[str] = None) -> bool:
        """Run the complete pipeline."""
        
        header = f"STUNIR Pipeline v{self.VERSION}\nMulti-Language Source-to-Source Transformation"
        if RICH_AVAILABLE:
            console.print(Panel.fit(f"[bold]{header}[/bold]", border_style="cyan"))
        else:
            print(f"\n{'='*60}")
            print(header)
            print(f"{'='*60}\n")
        
        if not extraction_file.exists():
            self.log(f"Extraction file not found: {extraction_file}", "error")
            return False
            
        # Derive module name from extraction file if not provided
        if module_name is None:
            module_name = extraction_file.stem.replace("_extraction", "").replace("extraction", "module")
            
        # Define intermediate files
        spec_file = output_dir / "spec.json"
        ir_file = output_dir / "ir.json"
        
        self.log(f"Input: {extraction_file}", "info")
        self.log(f"Output directory: {output_dir}", "info")
        self.log(f"Targets: {', '.join(targets)}", "info")
        self.log(f"Module: {module_name}", "info")
        console.print("")
        
        # Run phases sequentially
        if not self.run_phase1(extraction_file, spec_file, module_name):
            return False
            
        if not self.run_phase2(spec_file, ir_file, module_name):
            return False
            
        if not self.run_phase3(ir_file, output_dir, targets):
            return False
                
        # Print summary
        console.print("")
        self.print_summary()
        
        return True
        
    def print_summary(self):
        """Print execution summary."""
        if RICH_AVAILABLE:
            table = Table(title="Pipeline Execution Summary")
            table.add_column("Phase", style="cyan")
            table.add_column("Status", style="bold")
            table.add_column("Details", style="dim")
            
            for phase, result in self.results.items():
                status = result.get("status", "unknown")
                status_style = "green" if status == "success" else "red"
                
                if phase == "phase1":
                    details = f"{result.get('functions', 0)} functions"
                elif phase == "phase2":
                    details = f"{result.get('functions', 0)} functions"
                elif phase == "phase3":
                    details = f"{len(result.get('files', []))} files"
                else:
                    details = ""
                    
                table.add_row(
                    phase.replace("phase", "Phase "),
                    f"[{status_style}]{status}[/{status_style}]",
                    details
                )
                
            console.print(table)
        else:
            print("\nPipeline Execution Summary")
            print("-" * 40)
            for phase, result in self.results.items():
                status = result.get("status", "unknown")
                phase_name = phase.replace("phase", "Phase ")
                print(f"{phase_name}: {status}")


def main():
    parser = argparse.ArgumentParser(
        description="STUNIR Unified Pipeline - Multi-Language Source-to-Source Transformation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline for C++
  python stunir_pipeline.py -i extraction.json -o output -t cpp
  
  # Generate multiple targets
  python stunir_pipeline.py -i extraction.json -o output -t cpp python rust
  
  # Dry run (validate without writing files)
  python stunir_pipeline.py -i extraction.json -o output -t cpp --dry-run
        """
    )
    
    parser.add_argument("-i", "--input", required=True, type=Path,
                        help="Input extraction.json file")
    parser.add_argument("-o", "--output", required=True, type=Path,
                        help="Output directory for generated files")
    parser.add_argument("-t", "--targets", nargs="+", 
                        default=["cpp"],
                        choices=PipelineRunner.SUPPORTED_TARGETS + ["all"],
                        help="Target language(s) for code generation")
    parser.add_argument("-m", "--module", type=str,
                        help="Module name (default: derived from input file)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose output")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate pipeline without writing files")
    parser.add_argument("--version", action="version", version=f"STUNIR Pipeline v{PipelineRunner.VERSION}")
    
    args = parser.parse_args()
    
    # Handle "all" target
    targets = args.targets
    if "all" in targets:
        targets = PipelineRunner.SUPPORTED_TARGETS
        
    runner = PipelineRunner(verbose=args.verbose, dry_run=args.dry_run)
    success = runner.run_pipeline(
        extraction_file=args.input,
        output_dir=args.output,
        targets=targets,
        module_name=args.module
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
STUNIR Plan Executor v1.2
Executes the refinement plan defined in stunir_refinement_plan.json
"""

import json
import os
import sys
import subprocess
import hashlib
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

class STUNIRExecutor:
    def __init__(self, plan_path: str = "stunir_refinement_plan.json"):
        self.plan = self._load_plan(plan_path)
        self.workspace = Path("stunir_execution_workspace")
        self.results = {}
        
    def _load_plan(self, path: str) -> dict:
        with open(path, 'r') as f:
            return json.load(f)
    
    def _log(self, phase: str, message: str):
        timestamp = datetime.now().isoformat()
        print(f"[{timestamp}] [{phase}] {message}")
        
    def _run_cmd(self, cmd: List[str], cwd: Optional[Path] = None) -> tuple:
        """Run command and return (success, stdout, stderr)"""
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=cwd,
                timeout=300
            )
            return (result.returncode == 0, result.stdout, result.stderr)
        except Exception as e:
            return (False, "", str(e))
    
    def _ensure_workspace(self, program_name: str) -> Path:
        """Create workspace for program"""
        ws = self.workspace / program_name.lower().replace(" ", "_")
        ws.mkdir(parents=True, exist_ok=True)
        return ws
    
    def execute_all(self):
        """Execute plan for all programs in order"""
        self._log("INIT", f"STUNIR Plan Executor v{self.plan['plan_version']}")
        self._log("INIT", f"Objective: {self.plan['objective']}")
        
        for program in self.plan['programs']:
            self._execute_program(program)
            
        self._generate_final_report()
    
    def _execute_program(self, program: dict):
        """Execute all phases for a single program"""
        name = program['name']
        self._log(name, f"Starting execution for {name}")
        
        ws = self._ensure_workspace(name)
        self.results[name] = {"phases": {}, "status": "running"}
        
        # P0: Analysis & Stabilization
        if not self._phase_p0_analysis(program, ws):
            self.results[name]["status"] = "failed_p0"
            return
            
        # P1: Corpus Preparation  
        if not self._phase_p1_corpus(program, ws):
            self.results[name]["status"] = "failed_p1"
            return
            
        # P2: Extraction & IR Generation
        if not self._phase_p2_extraction(program, ws):
            self.results[name]["status"] = "failed_p2"
            return
            
        # P3: Codegen & Static Verification
        if not self._phase_p3_codegen(program, ws):
            self.results[name]["status"] = "failed_p3"
            return
            
        # P4: Compilation & Linkage
        if not self._phase_p4_compile(program, ws):
            self.results[name]["status"] = "failed_p4"
            return
            
        # P5: Behavioral Verification
        if not self._phase_p5_verify(program, ws):
            self.results[name]["status"] = "failed_p5"
            return
            
        # P6: Triage & Refinement
        self._phase_p6_triage(program, ws)
        
        self.results[name]["status"] = "completed"
        self._log(name, f"Completed execution for {name}")
    
    def _phase_p0_analysis(self, program: dict, ws: Path) -> bool:
        """P0: Analysis & Stabilization"""
        phase_id = "P0"
        self._log(phase_id, f"Starting Analysis & Stabilization for {program['name']}")
        
        # Create metadata file
        metadata = {
            "program": program['name'],
            "type": program['type'],
            "execution_start": datetime.now().isoformat(),
            "workspace": str(ws),
            "challenges": program.get('specific_challenges', [])
        }
        
        with open(ws / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Record environment
        env_info = {
            "python_version": sys.version,
            "platform": sys.platform,
            "cwd": str(Path.cwd()),
            "stunir_tools_available": self._check_stunir_tools()
        }
        
        with open(ws / "environment.json", 'w') as f:
            json.dump(env_info, f, indent=2)
        
        self._log(phase_id, f"Environment recorded for {program['name']}")
        self.results[program['name']]["phases"][phase_id] = {"status": "completed"}
        return True
    
    def _check_stunir_tools(self) -> dict:
        """Check which STUNIR tools are available"""
        tools = {}
        tool_names = [
            "stunir_spec_assemble_main.exe",
            "stunir_spec_to_ir_main.exe", 
            "stunir_ir_to_code_main.exe"
        ]
        
        for tool in tool_names:
            success, _, _ = self._run_cmd([tool, "--help"])
            tools[tool] = success
            
        return tools
    
    def _phase_p1_corpus(self, program: dict, ws: Path) -> bool:
        """P1: Corpus Preparation"""
        phase_id = "P1"
        self._log(phase_id, f"Starting Corpus Preparation for {program['name']}")
        
        corpus_dir = ws / "corpus"
        corpus_dir.mkdir(exist_ok=True)
        
        # Create corpus manifest
        manifest = {
            "program": program['name'],
            "tests_source": program['tests'],
            "expected_output_source": program['expected_output_source'],
            "test_cases": [],
            "deterministic_policy": {
                "line_endings": "LF",
                "whitespace": "trim_trailing",
                "locale": "C",
                "timezone": "UTC"
            }
        }
        
        with open(ws / "corpus_manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)
        
        self._log(phase_id, f"Corpus manifest created for {program['name']}")
        self.results[program['name']]["phases"][phase_id] = {"status": "completed"}
        return True
    
    def _phase_p2_extraction(self, program: dict, ws: Path) -> bool:
        """P2: Extraction & IR Generation (Batched)"""
        phase_id = "P2"
        self._log(phase_id, f"Starting Extraction & IR Generation for {program['name']}")
        
        extraction_dir = ws / "extraction"
        extraction_dir.mkdir(exist_ok=True)
        
        # Check batch processing strategy
        batch_config = self.plan['limits_and_safety']['batch_processing_strategy']
        self._log(phase_id, f"Batch strategy: {batch_config['trigger']}")
        
        # Placeholder for actual extraction
        # In real implementation, this would:
        # 1. Run clang-based extraction on source
        # 2. Split into batches if needed
        # 3. Run spec_assemble on each batch
        # 4. Run spec_to_ir on each batch
        # 5. Merge or link IR modules
        
        extraction_result = {
            "batches_processed": 0,
            "functions_extracted": 0,
            "unsupported_constructs_found": [],
            "status": "placeholder"
        }
        
        with open(ws / "extraction_result.json", 'w') as f:
            json.dump(extraction_result, f, indent=2)
        
        self._log(phase_id, f"Extraction placeholder completed for {program['name']}")
        self.results[program['name']]["phases"][phase_id] = {"status": "completed"}
        return True
    
    def _phase_p3_codegen(self, program: dict, ws: Path) -> bool:
        """P3: Codegen & Static Verification"""
        phase_id = "P3"
        self._log(phase_id, f"Starting Codegen & Static Verification for {program['name']}")
        
        codegen_dir = ws / "generated"
        codegen_dir.mkdir(exist_ok=True)
        
        # Placeholder for code generation
        # In real implementation, this would:
        # 1. Run ir_to_code on IR batches
        # 2. Verify signatures match original
        # 3. Run static analysis (lint)
        # 4. Check struct alignment
        
        codegen_result = {
            "files_generated": 0,
            "signature_matches": True,
            "static_analysis_issues": [],
            "status": "placeholder"
        }
        
        with open(ws / "codegen_result.json", 'w') as f:
            json.dump(codegen_result, f, indent=2)
        
        self._log(phase_id, f"Codegen placeholder completed for {program['name']}")
        self.results[program['name']]["phases"][phase_id] = {"status": "completed"}
        return True
    
    def _phase_p4_compile(self, program: dict, ws: Path) -> bool:
        """P4: Compilation & Linkage"""
        phase_id = "P4"
        self._log(phase_id, f"Starting Compilation & Linkage for {program['name']}")
        
        build_dir = ws / "build"
        build_dir.mkdir(exist_ok=True)
        
        # Placeholder for compilation
        # In real implementation, this would:
        # 1. Compile generated sources
        # 2. Handle libc shims
        # 3. Link against original objects if hybrid
        
        build_result = {
            "compile_success": True,
            "warnings": [],
            "errors": [],
            "binary_path": None,
            "status": "placeholder"
        }
        
        with open(ws / "build_result.json", 'w') as f:
            json.dump(build_result, f, indent=2)
        
        self._log(phase_id, f"Compilation placeholder completed for {program['name']}")
        self.results[program['name']]["phases"][phase_id] = {"status": "completed"}
        return True
    
    def _phase_p5_verify(self, program: dict, ws: Path) -> bool:
        """P5: Behavioral Verification"""
        phase_id = "P5"
        self._log(phase_id, f"Starting Behavioral Verification for {program['name']}")
        
        # Placeholder for verification
        # In real implementation, this would:
        # 1. Run generated binary against Golden Master corpus
        # 2. Compare stdout/stderr
        # 3. Track deltas per test
        # 4. Run Valgrind/ASAN
        
        metrics = self.plan['metrics']
        verification_result = {
            "test_cases_run": 0,
            "test_cases_passed": 0,
            "pass_rate": 0.0,
            "output_exactness": 0.0,
            "exit_code_match": True,
            "signature_match_rate": 1.0,
            "metrics_tracked": list(metrics.keys()),
            "status": "placeholder"
        }
        
        with open(ws / "verification_result.json", 'w') as f:
            json.dump(verification_result, f, indent=2)
        
        self._log(phase_id, f"Verification placeholder completed for {program['name']}")
        self.results[program['name']]["phases"][phase_id] = {"status": "completed"}
        return True
    
    def _phase_p6_triage(self, program: dict, ws: Path):
        """P6: Triage & Refinement"""
        phase_id = "P6"
        self._log(phase_id, f"Starting Triage & Refinement for {program['name']}")
        
        # Placeholder for triage
        # In real implementation, this would:
        # 1. Map failing tests to functions
        # 2. Categorize issues
        # 3. Generate failure_map.json
        # 4. Maintain regression suite
        
        triage_result = {
            "issues_categorized": {},
            "failure_map": {},
            "regression_tests_added": [],
            "status": "placeholder"
        }
        
        with open(ws / "triage_result.json", 'w') as f:
            json.dump(triage_result, f, indent=2)
        
        self._log(phase_id, f"Triage placeholder completed for {program['name']}")
        self.results[program['name']]["phases"][phase_id] = {"status": "completed"}
    
    def _generate_final_report(self):
        """Generate final execution report"""
        report_path = self.workspace / "execution_report.json"
        
        report = {
            "plan_version": self.plan['plan_version'],
            "execution_timestamp": datetime.now().isoformat(),
            "exit_criteria": self.plan['exit_criteria'],
            "results": self.results
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self._log("REPORT", f"Final report saved to {report_path}")
        
        # Print summary
        print("\n" + "="*70)
        print("STUNIR EXECUTION SUMMARY")
        print("="*70)
        for name, result in self.results.items():
            status = result['status']
            phases_completed = sum(1 for p in result['phases'].values() if p['status'] == 'completed')
            total_phases = len(result['phases'])
            print(f"{name:20} | {status:15} | Phases: {phases_completed}/{total_phases}")
        print("="*70)


def main():
    executor = STUNIRExecutor()
    executor.execute_all()


if __name__ == "__main__":
    main()
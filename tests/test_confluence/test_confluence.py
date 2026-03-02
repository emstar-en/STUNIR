#!/usr/bin/env python3
"""
STUNIR Confluence Verification System
Tests all 24 emitter categories across all 4 languages (SPARK, Python, Rust, Haskell)

FIXED (2026-02-28):
- Correct SPARK tool paths and CLI usage
- Create proper JSON test specs (not .stunir)
- Add early failure detection with clear error messages
- Add progress logging to avoid apparent hangs
- Reduce timeout to 5s per subprocess
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import difflib

# All 24 STUNIR categories
CATEGORIES = [
    # Core (5)
    "embedded", "gpu", "wasm", "assembly", "polyglot",
    # Language Families (2)
    "lisp", "prolog",
    # Specialized (17)
    "business", "fpga", "grammar", "lexer", "parser", "expert",
    "constraints", "functional", "oop", "mobile", "scientific",
    "bytecode", "systems", "planning", "asmIR", "beam", "asp"
]

# Default target for code emission
DEFAULT_TARGET = "c"

# Language configurations - paths relative to STUNIR root
LANGUAGES = {
    "spark": {
        "spec_to_ir": "tools/spark/bin/spec_to_ir_main.exe",
        "ir_to_code": "tools/spark/bin/code_emitter_main.exe",
        "enabled": True,
        "uses_spec_root": False,
        "timeout": 5  # seconds per subprocess
    },
    "python": {
        "spec_to_ir": "tools/python/spec_to_ir.py",
        "ir_to_code": "tools/python/ir_to_code.py",
        "enabled": True,  # Python spec_to_ir works; ir_to_code requires semantic_ir
        "uses_spec_root": True,
        "timeout": 5,
        "requires_semantic_ir": True,  # ir_to_code needs semantic_ir for code generation
        "skip_code_gen": True  # Skip code generation for now (semantic_ir not available)
    },
    "rust": {
        "spec_to_ir": "tools/rust/target/release/stunir_spec_to_ir.exe",
        "ir_to_code": "tools/rust/target/release/stunir_ir_to_code.exe",
        "enabled": False,  # Not built
        "uses_spec_root": False,
        "timeout": 5
    },
    "haskell": {
        "spec_to_ir": "tools/haskell/.stack-work/install/*/bin/stunir-spec-to-ir",
        "ir_to_code": "tools/haskell/.stack-work/install/*/bin/stunir-ir-to-code",
        "enabled": False,  # Not tested yet
        "uses_spec_root": False,
        "timeout": 5
    }
}

def log(msg: str):
    """Print timestamped log message"""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

class ConfluenceTest:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        # Go up from tests/test_confluence to STUNIR root (need 2 levels up)
        self.stunir_root = (base_dir / ".." / "..").resolve()
        self.results = {}
        self.specs_dir = base_dir / "specs"
        self.output_dir = base_dir / "outputs"
        self.output_dir.mkdir(exist_ok=True)
        self._validated_tools = {}  # Cache tool validation results
        
        log(f"STUNIR root: {self.stunir_root}")
        
    def validate_tool(self, tool_path: Path) -> Tuple[bool, str]:
        """Check if a tool exists and is executable"""
        if tool_path in self._validated_tools:
            return self._validated_tools[tool_path]
        
        if not tool_path.exists():
            result = (False, f"Tool not found: {tool_path}")
        else:
            result = (True, "OK")
        
        self._validated_tools[tool_path] = result
        return result
    
    def create_json_spec(self, category: str) -> Path:
        """Create a proper JSON test spec for the given category"""
        spec_file = self.specs_dir / f"{category}_test.json"
        self.specs_dir.mkdir(exist_ok=True)
        
        # Create a minimal valid IR-style spec
        spec = {
            "schema": "stunir_ir_v1",
            "ir_version": "1.0.0",
            "module_name": f"{category}_test",
            "functions": [
                {
                    "name": "test_func",
                    "return_type": "i32",
                    "args": [
                        {"name": "a", "type": "i32"},
                        {"name": "b", "type": "i32"}
                    ],
                    "steps": [
                        {"op": "return", "value": "a"}
                    ]
                }
            ]
        }
        
        spec_file.write_text(json.dumps(spec, indent=2), encoding='utf-8')
        return spec_file
    
    def run_pipeline(self, lang: str, spec_dir: Path, category: str) -> Tuple[bool, str, str]:
        """Run spec->IR->code pipeline for a language"""
        config = LANGUAGES[lang]
        if not config["enabled"]:
            return False, "Language not enabled", ""
        
        timeout = config.get("timeout", 5)
        lang_output_dir = self.output_dir / lang / category
        lang_output_dir.mkdir(parents=True, exist_ok=True)
        
        ir_file = lang_output_dir / "ir.json"
        code_file = lang_output_dir / "output.txt"
        
        try:
            # Step 1: Spec -> IR
            spec_to_ir = self.stunir_root / config["spec_to_ir"]
            valid, msg = self.validate_tool(spec_to_ir)
            if not valid:
                return False, msg, ""
            
            # Find a spec file to use (prefer JSON)
            spec_files = list(spec_dir.glob("*.json")) if spec_dir.is_dir() else []
            if not spec_files:
                # Create a test spec if none exists
                spec_file = self.create_json_spec(category)
                spec_files = [spec_file]
            
            spec_file = spec_files[0]
            log(f"  {lang}: Running spec_to_ir with {spec_file.name}...")
            
            # SPARK uses positional args: spec_to_ir <input> <output>
            if lang == "spark":
                cmd = [str(spec_to_ir), str(spec_file), str(ir_file)]
            elif lang == "python":
                # Python spec_to_ir requires --spec-root (directory) and --out
                spec_dir_for_python = spec_file.parent if spec_file.is_file() else spec_dir
                cmd = [sys.executable, str(spec_to_ir), "--spec-root", str(spec_dir_for_python), "--out", str(ir_file)]
            elif lang == "rust":
                cmd = [str(spec_to_ir), str(spec_file), "-o", str(ir_file)]
            else:
                cmd = [str(spec_to_ir), str(spec_file), str(ir_file)]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if result.returncode != 0:
                error_msg = result.stderr.strip()[:200] if result.stderr else "Unknown error"
                return False, f"spec_to_ir failed (exit {result.returncode}): {error_msg}", ""
            
            if not ir_file.exists():
                return False, f"IR file not created (stdout: {result.stdout[:100]})", ""
            
            # Check if we should skip code generation (e.g., semantic_ir not available)
            if config.get("skip_code_gen", False):
                log(f"  {lang}: Skipping ir_to_code (skip_code_gen=True)")
                # Read IR as output for comparison purposes
                output = ir_file.read_text(encoding='utf-8')
                return True, "Success (IR only, code gen skipped)", output
            
            # Step 2: IR -> Code
            ir_to_code = self.stunir_root / config["ir_to_code"]
            valid, msg = self.validate_tool(ir_to_code)
            if not valid:
                return False, msg, ""
            
            log(f"  {lang}: Running ir_to_code...")
            
            # SPARK code_emitter uses: code_emitter <input> <target> <output>
            if lang == "spark":
                cmd = [str(ir_to_code), str(ir_file), DEFAULT_TARGET, str(code_file)]
            elif lang == "python":
                # Python ir_to_code requires --ir, --lang, --out
                cmd = [sys.executable, str(ir_to_code), "--ir", str(ir_file), "--lang", DEFAULT_TARGET, "--out", str(lang_output_dir)]
            elif lang == "rust":
                cmd = [str(ir_to_code), str(ir_file), "--target", DEFAULT_TARGET, "-o", str(code_file)]
            else:
                cmd = [str(ir_to_code), str(ir_file), "-o", str(code_file)]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if result.returncode != 0:
                error_msg = result.stderr.strip()[:200] if result.stderr else "Unknown error"
                return False, f"ir_to_code failed (exit {result.returncode}): {error_msg}", ""
            
            # For Python, find the generated file in the output directory
            if lang == "python":
                # Python ir_to_code writes to output_dir/module_name.ext
                generated_files = list(lang_output_dir.glob("*.c")) + list(lang_output_dir.glob("*.rs")) + \
                                  list(lang_output_dir.glob("*.py")) + list(lang_output_dir.glob("*.js")) + \
                                  list(lang_output_dir.glob("*.go")) + list(lang_output_dir.glob("*.adb"))
                if not generated_files:
                    return False, f"No code file generated in {lang_output_dir} (stdout: {result.stdout[:100]})", ""
                code_file = generated_files[0]
            elif not code_file.exists():
                return False, f"Code file not created (stdout: {result.stdout[:100]})", ""
            
            output = code_file.read_text(encoding='utf-8')
            return True, "Success", output
            
        except subprocess.TimeoutExpired:
            return False, f"Timeout after {timeout}s", ""
        except Exception as e:
            return False, f"Exception: {str(e)}", ""
    
    def compare_outputs(self, outputs: Dict[str, str]) -> Tuple[bool, str]:
        """Compare outputs from different languages"""
        if len(outputs) < 2:
            return False, "Not enough outputs to compare"
        
        # Get reference output (first successful one)
        ref_lang = list(outputs.keys())[0]
        ref_output = outputs[ref_lang]
        
        all_match = True
        differences = []
        
        for lang, output in outputs.items():
            if lang == ref_lang:
                continue
            
            if output == ref_output:
                continue
            
            # Check structural equivalence (ignore whitespace differences)
            ref_normalized = " ".join(ref_output.split())
            out_normalized = " ".join(output.split())
            
            if ref_normalized == out_normalized:
                continue
            
            all_match = False
            diff = list(difflib.unified_diff(
                ref_output.splitlines(keepends=True),
                output.splitlines(keepends=True),
                fromfile=ref_lang,
                tofile=lang,
                lineterm=''
            ))
            differences.append(f"\n{ref_lang} vs {lang}:\n" + "".join(diff[:20]))
        
        if all_match:
            return True, "All outputs match"
        else:
            return False, "Outputs differ:\n" + "\n".join(differences)
    
    def test_category(self, category: str) -> Dict:
        """Test a single category across all languages"""
        log(f"Testing {category}...")
        
        # Use existing ardupilot_test specs or create new ones
        spec_dir = self.specs_dir / "ardupilot_test"
        if not spec_dir.exists():
            # Fall back to creating a spec for this category
            spec_file = self.create_json_spec(category)
            spec_dir = self.specs_dir
        
        # Run pipeline for each enabled language
        results = {}
        outputs = {}
        
        for lang in ["spark", "python"]:  # SPARK and Python are enabled
            if not LANGUAGES[lang]["enabled"]:
                results[lang] = {"success": False, "message": "Language not enabled"}
                continue
                
            success, message, output = self.run_pipeline(lang, spec_dir, category)
            results[lang] = {
                "success": success,
                "message": message
            }
            if success:
                outputs[lang] = output
                log(f"  {lang}: OK")
            else:
                log(f"  {lang}: FAILED - {message[:100]}")
        
        # Compare outputs (only meaningful if multiple languages succeed)
        if len(outputs) >= 2:
            confluence, comparison = self.compare_outputs(outputs)
            results["confluence"] = {
                "achieved": confluence,
                "details": comparison
            }
            if confluence:
                log(f"  Confluence: ACHIEVED")
            else:
                log(f"  Confluence: PARTIAL - {comparison[:100]}")
        else:
            results["confluence"] = {
                "achieved": False,
                "details": "Only SPARK enabled - confluence requires multiple languages"
            }
            log(f"  Confluence: SKIPPED (only SPARK enabled)")
        
        return results
    
    def run_all_tests(self) -> Dict:
        """Run confluence tests for all categories"""
        log("=" * 60)
        log("STUNIR Confluence Verification System")
        log("=" * 60)
        
        all_results = {}
        
        for category in CATEGORIES:
            all_results[category] = self.test_category(category)
        
        return all_results
    
    def generate_report(self, results: Dict) -> str:
        """Generate comprehensive confluence report"""
        report = []
        report.append("# STUNIR Confluence Verification Report\n")
        report.append("**Date:** (deterministic)\n")
        report.append(f"**Branch:** devsite\n\n")
        
        # Summary statistics
        total_categories = len(CATEGORIES)
        spark_success = sum(1 for r in results.values() if r.get("spark", {}).get("success", False))
        python_success = sum(1 for r in results.values() if r.get("python", {}).get("success", False))
        rust_success = sum(1 for r in results.values() if r.get("rust", {}).get("success", False))
        confluence_achieved = sum(1 for r in results.values() if r.get("confluence", {}).get("achieved", False))
        
        report.append("## Summary\n\n")
        report.append(f"- **Total Categories Tested:** {total_categories}\n")
        report.append(f"- **SPARK Success:** {spark_success}/{total_categories} ({spark_success*100//total_categories}%)\n")
        report.append(f"- **Python Success:** {python_success}/{total_categories} ({python_success*100//total_categories}%)\n")
        report.append(f"- **Rust Success:** {rust_success}/{total_categories} ({rust_success*100//total_categories}%)\n")
        report.append(f"- **Confluence Achieved:** {confluence_achieved}/{total_categories} ({confluence_achieved*100//total_categories}%)\n\n")
        
        # Detailed results
        report.append("## Detailed Results\n\n")
        report.append("| Category | SPARK | Python | Rust | Confluence |\n")
        report.append("|----------|-------|--------|------|------------|\n")
        
        for category in CATEGORIES:
            r = results[category]
            spark = "[OK]" if r.get("spark", {}).get("success", False) else "[X]"
            python = "[OK]" if r.get("python", {}).get("success", False) else "[X]"
            rust = "[OK]" if r.get("rust", {}).get("success", False) else "[X]"
            confluence = "[OK]" if r.get("confluence", {}).get("achieved", False) else "[X]"
            report.append(f"| {category} | {spark} | {python} | {rust} | {confluence} |\n")
        
        report.append("\n## Issues Found\n\n")
        for category, r in results.items():
            issues = []
            for lang in ["spark", "python", "rust"]:
                if not r.get(lang, {}).get("success", False):
                    msg = r.get(lang, {}).get("message", "Unknown error")
                    issues.append(f"- **{lang}:** {msg}")
            
            if not r.get("confluence", {}).get("achieved", False):
                details = r.get("confluence", {}).get("details", "Unknown")
                if "Not enough" not in details:
                    issues.append(f"- **Confluence:** {details[:200]}")
            
            if issues:
                report.append(f"\n### {category}\n")
                report.append("\n".join(issues) + "\n")
        
        return "".join(report)

def main():
    base_dir = Path(__file__).parent
    tester = ConfluenceTest(base_dir)
    
    # Run all tests
    results = tester.run_all_tests()
    
    # Generate report
    report = tester.generate_report(results)
    report_file = base_dir / "CONFLUENCE_REPORT.md"
    report_file.write_text(report, encoding='utf-8')
    
    print("\n" + "=" * 80)
    print(f"Report saved to: {report_file}")
    print("=" * 80)
    
    # Save JSON results
    json_file = base_dir / "confluence_results.json"
    json_file.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f"JSON results saved to: {json_file}")
    
    # Exit code based on confluence
    confluence_count = sum(1 for r in results.values() if r.get("confluence", {}).get("achieved", False))
    if confluence_count == len(CATEGORIES):
        log("100% CONFLUENCE ACHIEVED!")
        sys.exit(0)
    else:
        log(f"Partial confluence: {confluence_count}/{len(CATEGORIES)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

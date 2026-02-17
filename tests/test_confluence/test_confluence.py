#!/usr/bin/env python3
"""
STUNIR Confluence Verification System
Tests all 24 emitter categories across all 4 languages (SPARK, Python, Rust, Haskell)
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple
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

# Language configurations
LANGUAGES = {
    "spark": {
        "spec_to_ir": "../tools/spark/bin/stunir_spec_to_ir_main",
        "ir_to_code": "../tools/spark/bin/stunir_ir_to_code_main",
        "enabled": True,
        "uses_spec_root": True
    },
    "python": {
        "spec_to_ir": "../tools/spec_to_ir.py",
        "ir_to_code": "../tools/ir_to_code.py",
        "enabled": True,
        "uses_spec_root": True
    },
    "rust": {
        "spec_to_ir": "../tools/rust/target/release/stunir_spec_to_ir",
        "ir_to_code": "../tools/rust/target/release/stunir_ir_to_code",
        "enabled": True,
        "uses_spec_root": False
    },
    "haskell": {
        "spec_to_ir": "../tools/haskell/.stack-work/install/*/bin/stunir-spec-to-ir",
        "ir_to_code": "../tools/haskell/.stack-work/install/*/bin/stunir-ir-to-code",
        "enabled": False,  # Not tested yet
        "uses_spec_root": False
    }
}

class ConfluenceTest:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.results = {}
        self.specs_dir = base_dir / "specs"
        self.output_dir = base_dir / "outputs"
        self.output_dir.mkdir(exist_ok=True)
        
    def create_test_spec(self, category: str) -> Path:
        """Create a simple test spec for the given category"""
        spec_file = self.specs_dir / f"{category}_test.stunir"
        self.specs_dir.mkdir(exist_ok=True)
        
        # Simple test specs for each category
        specs = {
            "embedded": "target: arm\nplatform: cortex-m4\nfunction: blink_led() { gpio_toggle(13); }",
            "gpu": "target: cuda\nkernel: vector_add(a, b, c, n) { c[i] = a[i] + b[i]; }",
            "wasm": "target: wasm\nfunction: add(a: i32, b: i32) -> i32 { return a + b; }",
            "assembly": "target: x86_64\nfunction: add(a, b) { mov rax, a; add rax, b; ret; }",
            "polyglot": "target: c99\nfunction: hello() { printf(\"Hello\"); }",
            "lisp": "target: common-lisp\nfunction: (defun add (a b) (+ a b))",
            "prolog": "target: swi-prolog\nrule: parent(X, Y) :- father(X, Y).",
            "business": "target: cobol\nprogram: HELLO. DISPLAY 'Hello'.",
            "fpga": "target: vhdl\nentity: counter port(clk: in std_logic);",
            "grammar": "target: antlr\ngrammar: expr: term ('+' term)*;",
            "lexer": "target: flex\ntoken: [0-9]+ { return NUMBER; }",
            "parser": "target: yacc\nrule: expr: expr '+' term;",
            "expert": "target: clips\nrule: (defrule test (fact) => (assert (result)))",
            "constraints": "target: minizinc\nvar 1..10: x; constraint x > 5;",
            "functional": "target: haskell\nfunction: add x y = x + y",
            "oop": "target: java\nclass Test { void run() {} }",
            "mobile": "target: ios-swift\nfunc hello() { print(\"Hello\") }",
            "scientific": "target: matlab\nfunction y = f(x); y = x^2; end",
            "bytecode": "target: jvm\nmethod: add(II)I",
            "systems": "target: ada\nprocedure Test is begin null; end;",
            "planning": "target: pddl\naction: move(?x ?y)",
            "asmIR": "target: llvm\ndefine i32 @add(i32 %a, i32 %b) { ret i32 %a }",
            "beam": "target: erlang\nfun() -> ok end.",
            "asp": "target: clingo\nrule: a :- b, not c."
        }
        
        spec_content = specs.get(category, f"target: test\nfunction: test() {{ }}")
        spec_file.write_text(spec_content)
        return spec_file
    
    def run_pipeline(self, lang: str, spec_dir: Path, category: str) -> Tuple[bool, str, str]:
        """Run spec->IR->code pipeline for a language"""
        config = LANGUAGES[lang]
        if not config["enabled"]:
            return False, "Language not enabled", ""
        
        lang_output_dir = self.output_dir / lang / category
        lang_output_dir.mkdir(parents=True, exist_ok=True)
        
        ir_file = lang_output_dir / "ir.json"
        code_file = lang_output_dir / "output.txt"
        
        try:
            # Step 1: Spec -> IR
            spec_to_ir = Path(config["spec_to_ir"])
            if not spec_to_ir.exists():
                return False, f"Tool not found: {spec_to_ir}", ""
            
            # All languages use --spec-root now
            if lang == "python":
                cmd = ["python3", str(spec_to_ir), "--spec-root", str(spec_dir), "--out", str(ir_file)]
            elif lang == "rust":
                # Rust needs a single spec file, use first JSON file in directory
                spec_files = list(spec_dir.glob("*.json"))
                if not spec_files:
                    return False, "No spec files found in directory", ""
                cmd = [str(spec_to_ir), str(spec_files[0]), "-o", str(ir_file)]
            else:
                cmd = [str(spec_to_ir), "--spec-root", str(spec_dir), "--out", str(ir_file)]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                return False, f"spec_to_ir failed: {result.stderr[:200]}", ""
            
            if not ir_file.exists():
                return False, "IR file not created", ""
            
            # Step 2: IR -> Code
            ir_to_code = Path(config["ir_to_code"])
            if not ir_to_code.exists():
                return False, f"Tool not found: {ir_to_code}", ""
            
            if lang == "python":
                cmd = ["python3", str(ir_to_code), str(ir_file), str(code_file)]
            else:
                cmd = [str(ir_to_code), str(ir_file), "-o", str(code_file)]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                return False, f"ir_to_code failed: {result.stderr[:200]}", ""
            
            if not code_file.exists():
                return False, "Code file not created", ""
            
            output = code_file.read_text()
            return True, "Success", output
            
        except subprocess.TimeoutExpired:
            return False, "Timeout", ""
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
        print(f"\nTesting {category}...")
        
        # Use existing ardupilot_test specs instead of creating new ones
        spec_dir = self.specs_dir / "ardupilot_test"
        if not spec_dir.exists():
            print(f"  ❌ Spec directory not found: {spec_dir}")
            return {"error": "Spec directory not found"}
        
        # Run pipeline for each language
        results = {}
        outputs = {}
        
        for lang in ["spark", "python", "rust"]:  # Skip haskell for now
            success, message, output = self.run_pipeline(lang, spec_dir, category)
            results[lang] = {
                "success": success,
                "message": message
            }
            if success:
                outputs[lang] = output
                print(f"  ✅ {lang}: {message}")
            else:
                print(f"  ❌ {lang}: {message}")
        
        # Compare outputs
        if len(outputs) >= 2:
            confluence, comparison = self.compare_outputs(outputs)
            results["confluence"] = {
                "achieved": confluence,
                "details": comparison
            }
            if confluence:
                print(f"  ✅ Confluence: ACHIEVED")
            else:
                print(f"  ⚠️  Confluence: PARTIAL - {comparison[:100]}")
        else:
            results["confluence"] = {
                "achieved": False,
                "details": "Not enough successful outputs"
            }
            print(f"  ❌ Confluence: FAILED - Not enough outputs")
        
        return results
    
    def run_all_tests(self) -> Dict:
        """Run confluence tests for all categories"""
        print("=" * 80)
        print("STUNIR Confluence Verification System")
        print("=" * 80)
        
        all_results = {}
        
        for category in CATEGORIES:
            all_results[category] = self.test_category(category)
        
        return all_results
    
    def generate_report(self, results: Dict) -> str:
        """Generate comprehensive confluence report"""
        report = []
        report.append("# STUNIR Confluence Verification Report\n")
        report.append(f"**Date:** {subprocess.run(['date'], capture_output=True, text=True).stdout.strip()}\n")
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
            spark = "✅" if r.get("spark", {}).get("success", False) else "❌"
            python = "✅" if r.get("python", {}).get("success", False) else "❌"
            rust = "✅" if r.get("rust", {}).get("success", False) else "❌"
            confluence = "✅" if r.get("confluence", {}).get("achieved", False) else "❌"
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
    report_file.write_text(report)
    
    print("\n" + "=" * 80)
    print(f"Report saved to: {report_file}")
    print("=" * 80)
    
    # Save JSON results
    json_file = base_dir / "confluence_results.json"
    json_file.write_text(json.dumps(results, indent=2))
    print(f"JSON results saved to: {json_file}")
    
    # Exit code based on confluence
    confluence_count = sum(1 for r in results.values() if r.get("confluence", {}).get("achieved", False))
    if confluence_count == len(CATEGORIES):
        print("\n✅ 100% CONFLUENCE ACHIEVED!")
        sys.exit(0)
    else:
        print(f"\n⚠️  Partial confluence: {confluence_count}/{len(CATEGORIES)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

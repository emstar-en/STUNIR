#!/usr/bin/env python3
"""
STUNIR Mutation Testing Runner
==============================

Runs mutation testing for critical modules and reports scores.

Usage:
    python tests/mutation/run_mutation_tests.py [--module MODULE] [--threshold SCORE]
"""

import subprocess
import sys
import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Minimum mutation scores by module (as percentages)
MUTATION_THRESHOLDS: Dict[str, float] = {
    "tools/ir_emitter": 80.0,
    "tools/security/validation.py": 85.0,
    "manifests/base.py": 75.0,
    "tools/ir_canonicalizer": 75.0,
    "tools/emitters": 70.0,
}

CRITICAL_MODULES = [
    "tools/ir_emitter/emit_ir.py",
    "tools/security/validation.py",
    "manifests/base.py",
    "tools/ir_canonicalizer/canonicalize.py",
]


def check_mutmut_installed() -> bool:
    """Check if mutmut is installed."""
    try:
        subprocess.run(["mutmut", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def run_mutmut(module_path: str, timeout: int = 300) -> Tuple[int, int, int]:
    """Run mutmut on a specific module.
    
    Returns: (killed, survived, total) mutation counts
    """
    repo_root = Path(__file__).parent.parent.parent
    os.chdir(repo_root)
    
    # Clear previous results
    cache_dir = repo_root / ".mutmut-cache"
    if cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir)
    
    # Run mutation testing
    cmd = [
        "mutmut", "run",
        f"--paths-to-mutate={module_path}",
        "--no-progress",
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=repo_root
        )
    except subprocess.TimeoutExpired:
        print(f"  Timeout after {timeout}s")
        return (0, 0, 0)
    
    # Get results
    results_cmd = ["mutmut", "results"]
    results = subprocess.run(results_cmd, capture_output=True, text=True, cwd=repo_root)
    
    # Parse results
    killed = 0
    survived = 0
    total = 0
    
    for line in results.stdout.split("\n"):
        if "killed" in line.lower():
            try:
                killed = int(line.split(":")[1].strip())
            except (IndexError, ValueError):
                pass
        elif "survived" in line.lower():
            try:
                survived = int(line.split(":")[1].strip())
            except (IndexError, ValueError):
                pass
    
    total = killed + survived
    return (killed, survived, total)


def calculate_mutation_score(killed: int, total: int) -> float:
    """Calculate mutation score as percentage."""
    if total == 0:
        return 100.0
    return (killed / total) * 100.0


def run_all_mutation_tests(threshold: Optional[float] = None) -> Dict[str, dict]:
    """Run mutation tests on all critical modules."""
    results = {}
    
    for module in CRITICAL_MODULES:
        module_key = "/".join(module.split("/")[:2])
        min_threshold = threshold or MUTATION_THRESHOLDS.get(module_key, 70.0)
        
        print(f"\n🧬 Testing: {module}")
        print(f"   Threshold: {min_threshold}%")
        
        killed, survived, total = run_mutmut(module)
        score = calculate_mutation_score(killed, total)
        passed = score >= min_threshold
        
        results[module] = {
            "killed": killed,
            "survived": survived,
            "total": total,
            "score": score,
            "threshold": min_threshold,
            "passed": passed,
        }
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   Result: {killed}/{total} killed ({score:.1f}%) {status}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run STUNIR mutation tests")
    parser.add_argument("--module", help="Specific module to test")
    parser.add_argument("--threshold", type=float, help="Override threshold")
    parser.add_argument("--json", action="store_true", help="Output JSON results")
    args = parser.parse_args()
    
    print("="*60)
    print("STUNIR Mutation Testing")
    print("="*60)
    
    if not check_mutmut_installed():
        print("\n⚠️  mutmut not installed. Install with: pip install mutmut")
        print("    Skipping Python mutation tests.")
        return 0
    
    if args.module:
        print(f"\n🧬 Testing: {args.module}")
        killed, survived, total = run_mutmut(args.module)
        score = calculate_mutation_score(killed, total)
        threshold = args.threshold or 70.0
        passed = score >= threshold
        
        print(f"   Killed: {killed}")
        print(f"   Survived: {survived}")
        print(f"   Score: {score:.1f}%")
        print(f"   Status: {'PASS' if passed else 'FAIL'}")
        
        return 0 if passed else 1
    
    results = run_all_mutation_tests(args.threshold)
    
    if args.json:
        print(json.dumps(results, indent=2))
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    all_passed = all(r["passed"] for r in results.values())
    total_killed = sum(r["killed"] for r in results.values())
    total_mutations = sum(r["total"] for r in results.values())
    overall_score = calculate_mutation_score(total_killed, total_mutations)
    
    print(f"\nOverall Score: {overall_score:.1f}%")
    print(f"Total Killed: {total_killed}/{total_mutations}")
    print(f"Status: {'✅ ALL PASSED' if all_passed else '❌ SOME FAILED'}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

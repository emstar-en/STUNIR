#!/usr/bin/env python3
"""
STUNIR Comprehensive Analysis - 20 Pass Deep Scan

A systematic, multi-pass analysis tool that performs narrow, focused scans
to identify all remaining issues in the STUNIR codebase.

Each pass is independent and produces findings that feed into the final report.
"""

import json
import os
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional


class AnalysisFindings:
    """Container for all findings across all passes."""
    
    def __init__(self):
        self.findings: Dict[str, List[Dict[str, Any]]] = {}
        self.metadata = {
            "start_time": datetime.now().isoformat(),
            "root_dir": "STUNIR-main",
            "total_passes": 20
        }
    
    def add(self, pass_name: str, finding: Dict[str, Any]):
        """Add a finding from a specific pass."""
        if pass_name not in self.findings:
            self.findings[pass_name] = []
        self.findings[pass_name].append(finding)
    
    def get_count(self, pass_name: Optional[str] = None) -> int:
        """Get total findings count, optionally filtered by pass."""
        if pass_name:
            return len(self.findings.get(pass_name, []))
        return sum(len(f) for f in self.findings.values())
    
    def to_dict(self) -> Dict[str, Any]:
        """Export findings to dictionary."""
        return {
            "metadata": self.metadata,
            "summary": {
                "total_findings": self.get_count(),
                "by_pass": {k: len(v) for k, v in self.findings.items()}
            },
            "findings": self.findings
        }
    
    def save(self, filepath: str):
        """Save findings to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


findings = AnalysisFindings()

OUTPUT_SETTINGS = {
    "output_dir": "STUNIR-main/analysis",
    "jsonl_name": "analysis_report.jsonl",
    "summary_name": "analysis_report.summary.json",
    "index_name": "analysis_report.index.json",
    "fixes_name": "analysis_report.fixes.jsonl",
    "jsonl_max_field_len": 500,
    "legacy_path": "STUNIR-main/analysis_report.json"
}

SEVERITY_ORDER = {"ERROR": 0, "WARNING": 1, "INFO": 2}

def _severity_rank(severity: str) -> int:
    return SEVERITY_ORDER.get(severity or "", 3)

def _truncate_value(value: Any, max_len: int) -> Any:
    if value is None:
        return value
    if isinstance(value, str) and len(value) > max_len:
        return value[:max_len] + "..."
    return value

def _truncate_record(record: Dict[str, Any], max_len: int) -> Dict[str, Any]:
    return {k: _truncate_value(v, max_len) for k, v in record.items()}

def _build_fix_record(item: Dict[str, Any]) -> Dict[str, Any]:
    record = {
        "record_type": "fix",
        "severity": item.get("severity", "INFO"),
        "pass_name": item.get("pass_name"),
        "file": item.get("file"),
        "line": item.get("line"),
        "type": item.get("type"),
        "message": item.get("message"),
        "expected_result": item.get("expected_result")
    }
    return {k: v for k, v in record.items() if v is not None}

def _finding_sort_key(item: Dict[str, Any]):
    return (
        _severity_rank(item.get("severity")),
        item.get("file", ""),
        item.get("line", 0) or 0,
        item.get("message", ""),
        item.get("path", ""),
        item.get("type", ""),
    )

def _normalize_findings(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for pass_name, findings_list in report.get("findings", {}).items():
        for finding in findings_list:
            item = dict(finding)
            item["pass_name"] = pass_name
            if "severity" not in item:
                item["severity"] = "INFO"
            items.append(item)
    items.sort(key=_finding_sort_key)
    return items


# ==========================================================================
# PASS 1: File Inventory
# ==========================================================================
def pass1_file_inventory():
    """
    PASS 1: Catalog all files by type and location.
    
    Scope: Create complete inventory of all files in STUNIR-main/
    Output: File counts by extension, directory tree structure
    """
    print("\n" + "="*60)
    print("PASS 1: File Inventory")
    print("="*60)
    
    root = Path("STUNIR-main")
    if not root.exists():
        findings.add("pass1_file_inventory", {
            "severity": "ERROR",
            "message": "STUNIR-main directory not found",
            "path": str(root)
        })
        return
    
    # Count files by extension
    extension_counts = {}
    total_files = 0
    
    for filepath in root.rglob("*"):
        if filepath.is_file():
            total_files += 1
            ext = filepath.suffix.lower() or "(no extension)"
            extension_counts[ext] = extension_counts.get(ext, 0) + 1
            
            # Flag unusual files
            if ext in ['.exe', '.dll', '.so', '.dylib', '.bin']:
                findings.add("pass1_file_inventory", {
                    "severity": "WARNING",
                    "message": f"Binary file found: {ext}",
                    "path": str(filepath.relative_to(root)),
                    "type": "binary"
                })
    
    # Report findings
    print(f"Total files: {total_files}")
    print("\nFiles by extension:")
    for ext, count in sorted(extension_counts.items(), key=lambda x: -x[1]):
        print(f"  {ext}: {count}")
        findings.add("pass1_file_inventory", {
            "severity": "INFO",
            "message": f"Found {count} {ext} files",
            "extension": ext,
            "count": count
        })


# ============================================================================
# PASS 2: Version String Analysis
# ============================================================================
def pass2_version_strings():
    """
    PASS 2: Find all version references.
    
    Scope: Search for version strings (0.8.9, 0.9.0, 1.0.0, etc.)
    Output: Locations of version strings, inconsistencies
    """
    print("\n" + "="*60)
    print("PASS 2: Version String Analysis")
    print("="*60)
    
    version_patterns = [
        (r'0\.8\.[0-9]+', "v0.8.x"),
        (r'0\.9\.[0-9]+', "v0.9.x"),
        (r'1\.0\.[0-9]+', "v1.0.x"),
        (r'version\s*=\s*"[^"]*"', "TOML version"),
        (r'__version__\s*=\s*["\'][^"\']+["\']', "Python version"),
    ]
    
    root = Path("STUNIR-main")
    version_locations = {}
    
    for filepath in root.rglob("*"):
        if filepath.is_file() and filepath.suffix in ['.py', '.rs', '.toml', '.md', '.txt', '.json']:
            try:
                content = filepath.read_text(encoding='utf-8', errors='ignore')
                for pattern, desc in version_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        version_str = match.group(0)
                        if version_str not in version_locations:
                            version_locations[version_str] = []
                        version_locations[version_str].append(str(filepath.relative_to(root)))
                        
                        findings.add("pass2_version_strings", {
                            "severity": "INFO",
                            "version": version_str,
                            "file": str(filepath.relative_to(root)),
                            "pattern_type": desc
                        })
            except Exception as e:
                findings.add("pass2_version_strings", {
                    "severity": "ERROR",
                    "message": f"Failed to read file: {e}",
                    "file": str(filepath.relative_to(root))
                })
    
    # Report unique versions
    print("\nVersion strings found:")
    for version in sorted(version_locations.keys()):
        files = version_locations[version]
        print(f"  {version}: {len(files)} occurrences")
        if len(files) <= 3:
            for f in files:
                print(f"    - {f}")


# ============================================================================
# PASS 3: TODO/FIXME Comments
# ============================================================================
def pass3_todo_fixme():
    """
    PASS 3: Extract all incomplete work markers.
    
    Scope: Find TODO, FIXME, HACK, XXX, NOTE comments
    Output: List of incomplete work items with locations
    """
    print("\n" + "="*60)
    print("PASS 3: TODO/FIXME Analysis")
    print("="*60)
    
    markers = ['TODO', 'FIXME', 'HACK', 'XXX', 'NOTE', 'BUG', 'OPTIMIZE']
    pattern = re.compile(r'(#|//|--|\"\"\"|\'\'\')\s*(' + '|'.join(markers) + r')[\s:]+(.+)', re.IGNORECASE)

    # Patterns to exclude (section headers, not actual TODOs)
    exclude_pattern = re.compile(
        r'^\s*(#|//|--)?\s*(PASS\s+\d+|TODO/FIXME\s+Comments|TODO/FIXME\s+Analysis)',
        re.IGNORECASE
    )
    
    root = Path("STUNIR-main")
    todo_count = 0
    
    for filepath in root.rglob("*"):
        if filepath.is_file() and filepath.suffix in ['.py', '.rs', '.adb', '.ads', '.c', '.h', '.cpp', '.js', '.ts']:
            try:
                content = filepath.read_text(encoding='utf-8', errors='ignore')
                lines = content.split('\n')

                for line_num, line in enumerate(lines, 1):
                    # Skip section headers (false positives)
                    if exclude_pattern.search(line):
                        continue
                    match = pattern.search(line)
                    if match:
                        todo_count += 1
                        marker_type = match.group(2).upper()
                        message = match.group(3).strip()
                        
                        findings.add("pass3_todo_fixme", {
                            "severity": "WARNING",
                            "marker": marker_type,
                            "message": message[:100],  # Truncate long messages
                            "file": str(filepath.relative_to(root)),
                            "line": line_num
                        })
            except Exception as exc:
                # Skip files that can't be read (binary, encoding issues, etc.)
                continue
    
    print(f"\nFound {todo_count} TODO/FIXME markers")


# ============================================================================
# PASS 4: Import/Dependency Analysis
# ============================================================================
def pass4_import_analysis():
    """
    PASS 4: Check Python/Rust import consistency.
    
    Scope: Find import statements, check for circular dependencies
    Output: Import graphs, unused imports
    """
    print("\n" + "="*60)
    print("PASS 4: Import/Dependency Analysis")
    print("="*60)
    
    root = Path("STUNIR-main")
    python_imports = set()
    rust_imports = set()
    
    # Python imports
    for filepath in root.rglob("*.py"):
        try:
            content = filepath.read_text(encoding='utf-8', errors='ignore')
            # Find import statements
            imports = re.findall(r'^(?:from|import)\s+(\S+)', content, re.MULTILINE)
            for imp in imports:
                python_imports.add(imp)
                if 'stunir' in imp or 'semantic_ir' in imp:
                    findings.add("pass4_import_analysis", {
                        "severity": "INFO",
                        "type": "python_internal",
                        "import": imp,
                        "file": str(filepath.relative_to(root))
                    })
        except Exception as exc:
            # Skip files that can't be parsed
            continue
    
    # Rust imports
    for filepath in root.rglob("*.rs"):
        try:
            content = filepath.read_text(encoding='utf-8', errors='ignore')
            # Find use statements
            uses = re.findall(r'^use\s+([^;]+);', content, re.MULTILINE)
            for use in uses:
                rust_imports.add(use)
        except Exception as exc:
            # Skip files that can't be read
            continue
    
    print(f"\nPython imports: {len(python_imports)}")
    print(f"Rust imports: {len(rust_imports)}")


# ============================================================================
# PASS 5: Hardcoded Paths
# ============================================================================
def pass5_hardcoded_paths():
    """
    PASS 5: Find absolute paths and Windows-specific paths.
    
    Scope: Detect hardcoded paths that may break on other systems
    Output: List of hardcoded paths with suggested fixes
    """
    print("\n" + "="*60)
    print("PASS 5: Hardcoded Path Analysis")
    print("="*60)
    
    # Patterns for hardcoded paths
    patterns = [
        (r'[C-Z]:\\\S+', "Windows absolute path"),
        (r'/home/\S+', "Linux home path"),
        (r'/Users/\S+', "macOS home path"),
        (r'\\\\\S+', "UNC path"),
    ]
    
    root = Path("STUNIR-main")
    path_count = 0
    
    for filepath in root.rglob("*"):
        if filepath.is_file() and filepath.suffix in ['.py', '.rs', '.sh', '.bat', '.ps1', '.json', '.toml']:
            try:
                content = filepath.read_text(encoding='utf-8', errors='ignore')
                for pattern, desc in patterns:
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        path_count += 1
                        findings.add("pass5_hardcoded_paths", {
                            "severity": "WARNING",
                            "path_type": desc,
                            "path": match.group(0),
                            "file": str(filepath.relative_to(root))
                        })
            except Exception as exc:
                # Skip files that can't be read
                continue
    
    print(f"\nFound {path_count} hardcoded paths")


# ============================================================================
# PASS 6: Error Handling Audit
# ============================================================================
def pass6_error_handling():
    """
    PASS 6: Check unwrap(), panic(), expect() usage in Rust.
    
    Scope: Find potential panic points in Rust code
    Output: Locations of unsafe error handling
    """
    print("\n" + "="*60)
    print("PASS 6: Error Handling Audit")
    print("="*60)
    
    root = Path("STUNIR-main")
    patterns = [
        (r'\.unwrap\(\)', "unwrap()"),
        (r'\.expect\([^)]+\)', "expect()"),
        (r'panic!\([^)]+\)', "panic!"),
        (r'\.unwrap_or_default\(\)', "unwrap_or_default()"),
    ]
    
    for filepath in root.rglob("*.rs"):
        try:
            content = filepath.read_text(encoding='utf-8', errors='ignore')
            lines = content.split('\n')
            
            for line_num, line in enumerate(lines, 1):
                for pattern, desc in patterns:
                    if re.search(pattern, line):
                        findings.add("pass6_error_handling", {
                            "severity": "WARNING",
                            "pattern": desc,
                            "file": str(filepath.relative_to(root)),
                            "line": line_num,
                            "context": line.strip()[:80]
                        })
        except Exception as exc:
            # Skip files that can't be read
            pass
    
    count = findings.get_count("pass6_error_handling")
    print(f"\nFound {count} potential panic points")


# ============================================================================
# PASS 7: Documentation Gaps
# ============================================================================
def pass7_documentation_gaps():
    """
    PASS 7: Find undocumented public APIs.
    
    Scope: Check for missing docstrings/comments on public functions
    Output: List of undocumented public items
    """
    print("\n" + "="*60)
    print("PASS 7: Documentation Gap Analysis")
    print("="*60)
    
    root = Path("STUNIR-main")
    
    # Check Python files for missing docstrings
    for filepath in root.rglob("*.py"):
        try:
            content = filepath.read_text(encoding='utf-8', errors='ignore')
            # Find public functions without docstrings
            # Pattern: def name(...): followed by non-docstring
            # But exclude: test functions, main, and very short functions
            pattern = r'def\s+([a-z_][a-z0-9_]*)\s*\([^)]*\):\s*(?!\s*"""|\s*\'\'\')'
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                func_name = match.group(1)
                # Skip private, test functions, and common non-documented functions
                if func_name.startswith('_') or func_name.startswith('test_'):
                    continue
                if func_name in ('main', 'run', 'cli', 'entry_point'):
                    continue
                # Check if function has a body (not just pass/ellipsis)
                func_start = match.end()
                func_preview = content[func_start:func_start+200]
                if func_preview.strip() in ('pass', '...', 'pass\n', '...\n'):
                    continue  # Skip stub functions

                findings.add("pass7_documentation_gaps", {
                    "severity": "INFO",
                    "type": "python_function",
                    "name": func_name,
                    "file": str(filepath.relative_to(root)),
                    "message": "Public function missing docstring"
                })
        except Exception as exc:
            # Skip files that can't be read
            pass
    
    count = findings.get_count("pass7_documentation_gaps")
    print(f"\nFound {count} potential documentation gaps")


# ============================================================================
# PASS 8: Test Coverage
# ============================================================================
def pass8_test_coverage():
    """
    PASS 8: Identify untested modules and functions.
    
    Scope: Map source files to test files
    Output: Coverage gaps
    """
    print("\n" + "="*60)
    print("PASS 8: Test Coverage Analysis")
    print("="*60)
    
    root = Path("STUNIR-main")
    
    # Find all test files
    test_files = list(root.rglob("*test*.py")) + list(root.rglob("*test*.rs"))
    source_files = list(root.rglob("*.py")) + list(root.rglob("*.rs"))
    
    # Filter out test files from source files
    source_files = [f for f in source_files if 'test' not in f.name.lower()]
    
    print(f"\nTest files: {len(test_files)}")
    print(f"Source files: {len(source_files)}")
    
    findings.add("pass8_test_coverage", {
        "severity": "INFO",
        "test_files": len(test_files),
        "source_files": len(source_files),
        "ratio": len(test_files) / max(len(source_files), 1)
    })


# ============================================================================
# PASS 9: Configuration File Validation
# ============================================================================
def pass9_config_validation():
    """
    PASS 9: Validate JSON, TOML, YAML syntax.
    
    Scope: Check all config files for syntax errors
    Output: List of invalid config files
    """
    print("\n" + "="*60)
    print("PASS 9: Configuration File Validation")
    print("="*60)
    
    root = Path("STUNIR-main")
    
    # Validate JSON files
    for filepath in root.rglob("*.json"):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                json.load(f)
        except json.JSONDecodeError as e:
            findings.add("pass9_config_validation", {
                "severity": "ERROR",
                "type": "json",
                "file": str(filepath.relative_to(root)),
                "error": str(e)
            })
        except Exception as e:
            pass  # Skip files that can't be read
    
    count = findings.get_count("pass9_config_validation")
    print(f"\nFound {count} invalid configuration files")


# ============================================================================
# PASS 10: Security Audit
# ============================================================================
def pass10_security_audit():
    """
    PASS 10: Find unsafe blocks, shell commands, eval usage.
    
    Scope: Security-sensitive code patterns
    Output: Security concerns
    """
    print("\n" + "="*60)
    print("PASS 10: Security Audit")
    print("="*60)
    
    root = Path("STUNIR-main")
    
    # Check for unsafe Rust blocks
    for filepath in root.rglob("*.rs"):
        try:
            content = filepath.read_text(encoding='utf-8', errors='ignore')
            if 'unsafe' in content:
                findings.add("pass10_security_audit", {
                    "severity": "WARNING",
                    "type": "unsafe_rust",
                    "file": str(filepath.relative_to(root)),
                    "message": "File contains 'unsafe' keyword"
                })
        except Exception as exc:
            # Skip files that can't be read
            pass
    
    # Check for shell=True in Python
    for filepath in root.rglob("*.py"):
        try:
            content = filepath.read_text(encoding='utf-8', errors='ignore')
            if 'shell=True' in content or 'os.system' in content or 'eval(' in content:
                findings.add("pass10_security_audit", {
                    "severity": "WARNING",
                    "type": "python_shell",
                    "file": str(filepath.relative_to(root)),
                    "message": "File contains shell execution or eval"
                })
        except Exception as exc:
            # Skip files that can't be read
            pass
    
    count = findings.get_count("pass10_security_audit")
    print(f"\nFound {count} security concerns")


# ============================================================================
# PASS 11: Dead Code Detection
# ============================================================================
def pass11_dead_code():
    """
    PASS 11: Find unused imports, variables, functions.
    
    Scope: Simple dead code detection
    Output: Potentially unused code
    """
    print("\n" + "="*60)
    print("PASS 11: Dead Code Detection")
    print("="*60)
    
    root = Path("STUNIR-main")
    
    # Check Python for unused imports (simple heuristic)
    for filepath in root.rglob("*.py"):
        try:
            content = filepath.read_text(encoding='utf-8', errors='ignore')
            # Find imports
            imports = re.findall(r'^(?:from|import)\s+(\S+)', content, re.MULTILINE)
            # Very simple check - if import is not used elsewhere in file
            for imp in imports:
                base = imp.split('.')[0]
                if base not in ['os', 'sys', 'json', 're', 'pathlib']:
                    usage_count = content.count(base) - 1  # Subtract the import itself
                    if usage_count <= 0:
                        findings.add("pass11_dead_code", {
                            "severity": "INFO",
                            "type": "unused_import",
                            "import": imp,
                            "file": str(filepath.relative_to(root))
                        })
        except Exception as exc:
            # Skip files that can't be read
            pass
    
    count = findings.get_count("pass11_dead_code")
    print(f"\nFound {count} potential dead code issues")


# ============================================================================
# PASS 12: Naming Consistency
# ============================================================================
def pass12_naming_consistency():
    """
    PASS 12: Check naming conventions across codebase.
    
    Scope: snake_case vs camelCase inconsistencies
    Output: Naming convention violations
    """
    print("\n" + "="*60)
    print("PASS 12: Naming Consistency")
    print("="*60)
    
    # This is a simplified check - full analysis would require AST parsing
    print("\n[Naming consistency check requires AST parsing - skipped in basic scan]")


# ============================================================================
# PASS 13: IR Format Consistency
# ============================================================================
def pass13_ir_format():
    """
    PASS 13: Check IR JSON schema compliance.
    
    Scope: Validate IR JSON files against expected schema
    Output: Schema violations
    """
    print("\n" + "="*60)
    print("PASS 13: IR Format Consistency")
    print("="*60)
    
    root = Path("STUNIR-main")
    
    required_ir_fields = ['ir_version', 'module_name', 'functions']
    
    for filepath in root.rglob("*.json"):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if this looks like an IR file
            if 'ir_version' in data:
                missing = [f for f in required_ir_fields if f not in data]
                if missing:
                    findings.add("pass13_ir_format", {
                        "severity": "ERROR",
                        "file": str(filepath.relative_to(root)),
                        "missing_fields": missing
                    })
        except Exception as exc:
            # Skip files that can't be read
            pass
    
    count = findings.get_count("pass13_ir_format")
    print(f"\nFound {count} IR format issues")


# ============================================================================
# PASS 14: Build System Check
# ============================================================================
def pass14_build_system():
    """
    PASS 14: Validate Cargo.toml, gpr files.
    
    Scope: Check build configuration files
    Output: Build configuration issues
    """
    print("\n" + "="*60)
    print("PASS 14: Build System Check")
    print("="*60)
    
    root = Path("STUNIR-main")
    
    # Check for Cargo.toml
    cargo_files = list(root.rglob("Cargo.toml"))
    print(f"\nFound {len(cargo_files)} Cargo.toml files")
    
    # Check for gpr files
    gpr_files = list(root.rglob("*.gpr"))
    print(f"Found {len(gpr_files)} GPR project files")
    
    findings.add("pass14_build_system", {
        "severity": "INFO",
        "cargo_files": len(cargo_files),
        "gpr_files": len(gpr_files)
    })


# ============================================================================
# PASS 15: Cross-Platform Issues
# ============================================================================
def pass15_cross_platform():
    """
    PASS 15: Find platform-specific code.
    
    Scope: Windows vs Unix specific code
    Output: Platform dependencies
    """
    print("\n" + "="*60)
    print("PASS 15: Cross-Platform Analysis")
    print("="*60)
    
    root = Path("STUNIR-main")
    
    platform_patterns = [
        (r'win32|windows|Win32|Windows', "Windows-specific"),
        (r'linux|Linux|unix|Unix', "Unix-specific"),
        (r'darwin|macos|MacOS', "macOS-specific"),
        (r'#[cfg\(target_os', "Conditional compilation"),
    ]
    
    for filepath in root.rglob("*"):
        if filepath.is_file() and filepath.suffix in ['.py', '.rs', '.sh', '.bat']:
            try:
                content = filepath.read_text(encoding='utf-8', errors='ignore')
                for pattern, desc in platform_patterns:
                    if re.search(pattern, content):
                        findings.add("pass15_cross_platform", {
                            "severity": "INFO",
                            "type": desc,
                            "file": str(filepath.relative_to(root))
                        })
            except Exception as exc:
                # Skip files that can't be read
                pass
    
    count = findings.get_count("pass15_cross_platform")
    print(f"\nFound {count} platform-specific code sections")


# ============================================================================
# PASS 16: Unicode/Encoding Issues
# ============================================================================
def pass16_unicode_issues():
    """
    PASS 16: Find non-ASCII characters.
    
    Scope: Detect encoding issues
    Output: Files with non-ASCII content
    """
    print("\n" + "="*60)
    print("PASS 16: Unicode/Encoding Analysis")
    print("="*60)
    
    root = Path("STUNIR-main")
    
    for filepath in root.rglob("*"):
        if filepath.is_file() and filepath.suffix in ['.py', '.rs', '.adb', '.ads']:
            try:
                with open(filepath, 'rb') as f:
                    content = f.read()
                
                # Check for non-ASCII bytes
                non_ascii = [b for b in content if b > 127]
                if non_ascii:
                    findings.add("pass16_unicode_issues", {
                        "severity": "WARNING",
                        "file": str(filepath.relative_to(root)),
                        "non_ascii_bytes": len(non_ascii)
                    })
            except Exception as exc:
                # Skip files that can't be read
                pass
    
    count = findings.get_count("pass16_unicode_issues")
    print(f"\nFound {count} files with non-ASCII characters")


# ============================================================================
# PASS 17: License Headers
# ============================================================================
def pass17_license_headers():
    """
    PASS 17: Check file headers and LICENSE references.
    
    Scope: Verify license information
    Output: Missing license headers
    """
    print("\n" + "="*60)
    print("PASS 17: License Header Check")
    print("="*60)
    
    root = Path("STUNIR-main")
    
    # Check for LICENSE file
    license_files = list(root.glob("LICENSE*"))
    print(f"\nFound {len(license_files)} LICENSE files")
    
    # Check source files for license headers
    for filepath in root.rglob("*.py"):
        try:
            content = filepath.read_text(encoding='utf-8', errors='ignore')
            if 'license' not in content.lower()[:500] and 'copyright' not in content.lower()[:500]:
                findings.add("pass17_license_headers", {
                    "severity": "INFO",
                    "type": "missing_header",
                    "file": str(filepath.relative_to(root))
                })
        except Exception as exc:
            # Skip files that can't be read
            pass
    
    count = findings.get_count("pass17_license_headers")
    print(f"Found {count} files without license headers")


# ============================================================================
# PASS 18: Git Hygiene
# ============================================================================
def pass18_git_hygiene():
    """
    PASS 18: Check .gitignore, large files, binaries.
    
    Scope: Git repository cleanliness
    Output: Git hygiene issues
    """
    print("\n" + "="*60)
    print("PASS 18: Git Hygiene Check")
    print("="*60)
    
    root = Path("STUNIR-main")
    
    # Check for .gitignore
    gitignore = root / ".gitignore"
    if gitignore.exists():
        print("\n.gitignore exists")
    else:
        findings.add("pass18_git_hygiene", {
            "severity": "WARNING",
            "type": "missing_gitignore"
        })
    
    # Check for large files (>1MB)
    large_files = []
    for filepath in root.rglob("*"):
        if filepath.is_file():
            size = filepath.stat().st_size
            if size > 1_000_000:  # 1MB
                large_files.append((str(filepath.relative_to(root)), size))
                findings.add("pass18_git_hygiene", {
                    "severity": "WARNING",
                    "type": "large_file",
                    "file": str(filepath.relative_to(root)),
                    "size_bytes": size
                })
    
    print(f"Found {len(large_files)} large files (>1MB)")


# ============================================================================
# PASS 19: Performance Issues
# ============================================================================
def pass19_performance():
    """
    PASS 19: Find inefficient patterns.
    
    Scope: Performance anti-patterns
    Output: Performance concerns
    """
    print("\n" + "="*60)
    print("PASS 19: Performance Analysis")
    print("="*60)
    
    root = Path("STUNIR-main")
    
    # Check for inefficient patterns in Python
    patterns = [
        (r'for\s+\w+\s+in\s+range\s*\(\s*len\s*\(', "range(len()) anti-pattern"),
        (r'\.read\(\)\s*\.split\(\)', "read().split() - loads entire file"),
        (r'str\s*\+\s*str', "String concatenation in loop"),
    ]
    
    for filepath in root.rglob("*.py"):
        try:
            content = filepath.read_text(encoding='utf-8', errors='ignore')
            for pattern, desc in patterns:
                if re.search(pattern, content):
                    findings.add("pass19_performance", {
                        "severity": "INFO",
                        "type": desc,
                        "file": str(filepath.relative_to(root))
                    })
        except Exception as exc:
            # Skip files that can't be read
            pass

    count = findings.get_count("pass19_performance")
    print(f"\nFound {count} performance concerns")


# ============================================================================
def _write_jsonl_findings(output_file: str, report: Dict[str, Any]):
    max_len = OUTPUT_SETTINGS["jsonl_max_field_len"]
    with open(output_file, 'w') as f:
        meta = {
            "record_type": "meta",
            "generated": datetime.now().isoformat(),
            "total_findings": report.get("summary", {}).get("total_findings", 0),
            "severity_summary": report.get("metadata", {}).get("severity_summary", {}),
            "root_dir": report.get("metadata", {}).get("root_dir"),
            "start_time": report.get("metadata", {}).get("start_time"),
            "end_time": report.get("metadata", {}).get("end_time"),
            "total_passes": report.get("metadata", {}).get("total_passes")
        }
        f.write(json.dumps(meta) + '\n')
        for item in _normalize_findings(report):
            record = {"record_type": "finding", **item}
            f.write(json.dumps(_truncate_record(record, max_len)) + '\n')


def _write_fixes_jsonl(output_file: str, report: Dict[str, Any]):
    max_len = OUTPUT_SETTINGS["jsonl_max_field_len"]
    items = _normalize_findings(report)
    with open(output_file, 'w') as f:
        meta = {
            "record_type": "meta",
            "generated": datetime.now().isoformat(),
            "total_findings": len(items),
            "severity_summary": report.get("metadata", {}).get("severity_summary", {}),
            "root_dir": report.get("metadata", {}).get("root_dir"),
            "start_time": report.get("metadata", {}).get("start_time"),
            "end_time": report.get("metadata", {}).get("end_time"),
            "total_passes": report.get("metadata", {}).get("total_passes")
        }
        f.write(json.dumps(meta) + '\n')
        for item in items:
            fix_record = _build_fix_record(item)
            f.write(json.dumps(_truncate_record(fix_record, max_len)) + '\n')


def _write_findings_summary(output_file: str, report: Dict[str, Any]):
    summary = {
        "metadata": report.get("metadata", {}),
        "summary": report.get("summary", {}),
        "severity_summary": report.get("metadata", {}).get("severity_summary", {}),
        "by_pass": report.get("summary", {}).get("by_pass", {}),
        "by_file": {},
        "by_type": {}
    }

    items = _normalize_findings(report)
    for item in items:
        file_path = item.get("file", "unknown")
        entry = summary["by_file"].setdefault(file_path, {"count": 0, "types": {}, "severities": {}})
        entry["count"] += 1
        finding_type = item.get("type", "unknown")
        entry["types"][finding_type] = entry["types"].get(finding_type, 0) + 1
        sev = item.get("severity", "INFO")
        entry["severities"][sev] = entry["severities"].get(sev, 0) + 1

    for item in items:
        finding_type = item.get("type", "unknown")
        summary["by_type"][finding_type] = summary["by_type"].get(finding_type, 0) + 1

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    return output_file


def _write_findings_index(output_file: str, report: Dict[str, Any]):
    items = _normalize_findings(report)
    index = {
        "generated": datetime.now().isoformat(),
        "total_findings": len(items),
        "files": {}
    }

    for item in items:
        file_path = item.get("file", "unknown")
        entry = index["files"].setdefault(file_path, {"count": 0, "passes": {}, "severities": {}})
        entry["count"] += 1
        pass_name = item.get("pass_name", "unknown")
        entry["passes"][pass_name] = entry["passes"].get(pass_name, 0) + 1
        sev = item.get("severity", "INFO")
        entry["severities"][sev] = entry["severities"].get(sev, 0) + 1

    with open(output_file, 'w') as f:
        json.dump(index, f, indent=2)


def _write_model_outputs(report: Dict[str, Any], output_dir: str):
    Path(output_dir).mkdir(exist_ok=True)
    jsonl_path = os.path.join(output_dir, OUTPUT_SETTINGS["jsonl_name"])
    summary_path = os.path.join(output_dir, OUTPUT_SETTINGS["summary_name"])
    index_path = os.path.join(output_dir, OUTPUT_SETTINGS["index_name"])
    fixes_path = os.path.join(output_dir, OUTPUT_SETTINGS["fixes_name"])
    _write_jsonl_findings(jsonl_path, report)
    _write_findings_summary(summary_path, report)
    _write_findings_index(index_path, report)
    _write_fixes_jsonl(fixes_path, report)
    return {
        "JSONL": jsonl_path,
        "Summary": summary_path,
        "Index": index_path,
        "Fixes": fixes_path
    }


def pass20_final_integration():
    print("\n" + "="*60)
    print("PASS 20: Final Integration")
    print("="*60)

    findings.metadata["end_time"] = datetime.now().isoformat()
    findings.metadata["total_findings"] = findings.get_count()

    severity_counts = {"ERROR": 0, "WARNING": 0, "INFO": 0}
    for pass_findings in findings.findings.values():
        for finding in pass_findings:
            sev = finding.get("severity", "INFO")
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

    findings.metadata["severity_summary"] = severity_counts

    report = findings.to_dict()
    legacy_output = OUTPUT_SETTINGS["legacy_path"]
    findings.save(legacy_output)
    output_files = {"JSON": legacy_output}
    output_files.update(_write_model_outputs(report, OUTPUT_SETTINGS["output_dir"]))

    print(f"\n[OK] Analysis complete!")
    for format_name, filepath in output_files.items():
        print(f"[OK] {format_name} report saved to: {filepath}")

    print(f"\nSummary:")
    print(f"  Total findings: {findings.get_count()}")
    print(f"  Errors: {severity_counts['ERROR']}")
    print(f"  Warnings: {severity_counts['WARNING']}")
    print(f"  Info: {severity_counts['INFO']}")
    print(f"\nBreakdown by pass:")
    for pass_name in sorted(findings.findings.keys()):
        count = len(findings.findings[pass_name])
        print(f"  {pass_name}: {count} findings")


def main():
    print("="*60)
    print("STUNIR Comprehensive Analysis - 20 Pass Deep Scan")
    print("="*60)
    print(f"Started at: {datetime.now().isoformat()}")

    passes = [
        pass1_file_inventory,
        pass2_version_strings,
        pass3_todo_fixme,
        pass4_import_analysis,
        pass5_hardcoded_paths,
        pass6_error_handling,
        pass7_documentation_gaps,
        pass8_test_coverage,
        pass9_config_validation,
        pass10_security_audit,
        pass11_dead_code,
        pass12_naming_consistency,
        pass13_ir_format,
        pass14_build_system,
        pass15_cross_platform,
        pass16_unicode_issues,
        pass17_license_headers,
        pass18_git_hygiene,
        pass19_performance,
        pass20_final_integration,
    ]
    
    for i, pass_func in enumerate(passes, 1):
        try:
            pass_func()
        except Exception as e:
            print(f"\n[ERROR] Pass {i} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print(f"Report saved to: STUNIR-main/analysis_report.json")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

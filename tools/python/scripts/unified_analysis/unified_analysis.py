import json
import os
import re
import sys
import subprocess
import shutil
import argparse
import fnmatch
from pathlib import Path
from datetime import datetime

SEVERITY_ORDER = {"ERROR": 0, "WARNING": 1, "INFO": 2}

def _severity_rank(severity): return SEVERITY_ORDER.get(severity or "", 3)

def _severity_allowed(severity, threshold):
    return _severity_rank(severity) <= _severity_rank(threshold)

class AnalysisFindings:
    def __init__(self, root_dir, severity_threshold):
        self.findings = {}
        self.metadata = {"start_time": datetime.now().isoformat(), "root_dir": root_dir, "severity_threshold": severity_threshold}
        self.severity_threshold = severity_threshold
    def add(self, pass_name, finding):
        severity = finding.get("severity", "INFO")
        if not _severity_allowed(severity, self.severity_threshold):
            return
        self.findings.setdefault(pass_name, []).append(finding)
    def get_count(self, pass_name=None):
        return len(self.findings.get(pass_name, [])) if pass_name else sum(len(v) for v in self.findings.values())
    def to_dict(self):
        return {"metadata": self.metadata, "summary": {"total_findings": self.get_count(), "by_pass": {k: len(v) for k, v in self.findings.items()}}, "findings": self.findings}

def _truncate_value(value, max_len):
    if value is None: return value
    if isinstance(value, str) and len(value) > max_len: return value[:max_len] + "..."
    return value

def _truncate_record(record, max_len): return {k: _truncate_value(v, max_len) for k, v in record.items()}

def _default_config():
    return {
        "root_dir": str(Path(__file__).resolve().parents[2]),
        "output": {
            "output_dir": "analysis/unified",
            "jsonl_name": "unified_analysis.jsonl",
            "summary_name": "unified_analysis.summary.json",
            "index_name": "unified_analysis.index.json",
            "fixes_name": "unified_analysis.fixes.jsonl",
            "jsonl_max_field_len": 500
        },
        "severity_threshold": "INFO",
        "exclude_dirs": [".git", "target", "dist", "build", "node_modules", "test_output", "test_fixtures", "__pycache__", "analysis", "test_python_pipeline", "test_outputs", "meta", "test_spark_pipeline", "test_vectors"],
        "include_globs": [],
        "exclude_globs": [],
        "pipelines": {"rust": True, "python": True, "spark": True, "shared": True},
        "commands": {
            "rust": {"cargo_check": ["cargo", "check"], "cargo_clippy": ["cargo", "clippy"], "cargo_test": ["cargo", "test"]},
            "python": {"compileall": [sys.executable, "-m", "compileall", "-q"]},
            "spark": {"verifier": ["tools/spark/bin/stunir_verifier_main", "--help"]}
        },
        "patterns": {
            "rust_error_handling": [
                {"pattern": r"\.unwrap\(\)", "type": "unwrap()", "expected_result": "Replace unwrap with proper error handling", "severity": "WARNING", "mode": "lines"},
                {"pattern": r"\.expect\([^)]+\)", "type": "expect()", "expected_result": "Replace expect with proper error handling", "severity": "WARNING", "mode": "lines"},
                {"pattern": r"panic!\([^)]+\)", "type": "panic!", "expected_result": "Avoid panic in production code", "severity": "WARNING", "mode": "lines"}
            ],
            "python_error_handling": [
                {"pattern": r"^\s*except:\s*$", "type": "bare_except", "expected_result": "Handle specific exceptions (e.g., except ValueError:)", "severity": "WARNING", "mode": "lines"},
                {"pattern": r"^\s*pass\s*$", "type": "pass_statement", "expected_result": "Remove dead code or implement logic (exclude abstract methods, exception stubs)", "severity": "INFO", "mode": "lines", "context_exclude": ["@abstractmethod", "class.*Exception", "class.*Error"]}
            ],
            "python_docs": [
                {"pattern": r"def\s+(?!test_|main|run|cli|entry_point)[a-z_][a-z0-9_]*\s*\([^)]*\):\s*(?!\s*\"\"\"|\s*\'\'\')", "type": "missing_docstring", "expected_result": "Add docstring to public function", "severity": "INFO", "mode": "content"}
            ],
            "spark_todos": [
                {"pattern": r"TODO|FIXME|XXX|HACK", "type": "todo_fixme", "expected_result": "Address or remove TODO/FIXME", "severity": "WARNING", "mode": "lines"}
            ],
            "shared_todos": [
                {"pattern": r"TODO|FIXME|XXX|HACK", "type": "todo_fixme", "expected_result": "Address or remove TODO/FIXME", "severity": "WARNING", "mode": "lines"}
            ],
            "shared_paths": [
                {"pattern": r"[C-Z]:\\\\S+", "type": "windows_path", "expected_result": "Use PathBuf/pathlib for cross-platform paths", "severity": "INFO", "mode": "content"},
                {"pattern": r"/home/\S+", "type": "linux_home_path", "expected_result": "Avoid hardcoded home paths", "severity": "INFO", "mode": "content"},
                {"pattern": r"/Users/\S+", "type": "macos_home_path", "expected_result": "Avoid hardcoded home paths", "severity": "INFO", "mode": "content"}
            ]
        },
        "exts": {
            "rust": [".rs"],
            "python": [".py"],
            "spark": [".adb", ".ads"],
            "shared": [".rs", ".py", ".adb", ".ads", ".sh", ".ps1"]
        }
    }

def _load_config(path):
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _merge_dicts(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_dicts(base[key], value)
        else:
            base[key] = value
    return base

def _parse_args(argv):
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--enable", default=None)
    parser.add_argument("--disable", default=None)
    parser.add_argument("--include-glob", action="append", default=None)
    parser.add_argument("--exclude-glob", action="append", default=None)
    parser.add_argument("--exclude-dir", action="append", default=None)
    parser.add_argument("--severity-threshold", default=None)
    parser.add_argument("--jsonl-max-field-len", type=int, default=None)
    return parser.parse_args(argv)

def _apply_cli_overrides(config, args):
    if args.root:
        config["root_dir"] = args.root
    if args.output_dir:
        config["output"]["output_dir"] = args.output_dir
    if args.jsonl_max_field_len is not None:
        config["output"]["jsonl_max_field_len"] = args.jsonl_max_field_len
    if args.severity_threshold:
        config["severity_threshold"] = args.severity_threshold
    if args.include_glob is not None:
        config["include_globs"] = args.include_glob
    if args.exclude_glob is not None:
        config["exclude_globs"] = args.exclude_glob
    if args.exclude_dir is not None:
        config["exclude_dirs"] = args.exclude_dir
    if args.enable:
        for name in args.enable.split(","):
            config["pipelines"][name.strip()] = True
    if args.disable:
        for name in args.disable.split(","):
            config["pipelines"][name.strip()] = False
    return config

def _build_fix_record(item):
    record = {"record_type": "fix", "severity": item.get("severity", "INFO"), "pass_name": item.get("pass_name"), "file": item.get("file"), "line": item.get("line"), "type": item.get("type"), "message": item.get("message"), "expected_result": item.get("expected_result")}
    return {k: v for k, v in record.items() if v is not None}

def _finding_sort_key(item):
    return (_severity_rank(item.get("severity")), item.get("file", ""), item.get("line", 0) or 0, item.get("message", ""), item.get("path", ""), item.get("type", ""))

def _normalize_findings(report):
    items = []
    for pass_name, findings_list in report.get("findings", {}).items():
        for finding in findings_list:
            item = dict(finding)
            item["pass_name"] = pass_name
            if "severity" not in item: item["severity"] = "INFO"
            items.append(item)
    items.sort(key=_finding_sort_key)
    return items

def _write_jsonl_findings(output_file, report, max_len):
    with open(output_file, "w") as f:
        meta = {"record_type": "meta", "generated": datetime.now().isoformat(), "total_findings": report.get("summary", {}).get("total_findings", 0), "severity_summary": report.get("metadata", {}).get("severity_summary", {}), "root_dir": report.get("metadata", {}).get("root_dir"), "start_time": report.get("metadata", {}).get("start_time"), "end_time": report.get("metadata", {}).get("end_time")}
        f.write(json.dumps(meta) + "\n")
        for item in _normalize_findings(report):
            record = {"record_type": "finding", **item}
            f.write(json.dumps(_truncate_record(record, max_len)) + "\n")

def _write_fixes_jsonl(output_file, report, max_len):
    items = _normalize_findings(report)
    with open(output_file, "w") as f:
        meta = {"record_type": "meta", "generated": datetime.now().isoformat(), "total_findings": len(items), "severity_summary": report.get("metadata", {}).get("severity_summary", {}), "root_dir": report.get("metadata", {}).get("root_dir"), "start_time": report.get("metadata", {}).get("start_time"), "end_time": report.get("metadata", {}).get("end_time")}
        f.write(json.dumps(meta) + "\n")
        for item in items:
            fix_record = _build_fix_record(item)
            f.write(json.dumps(_truncate_record(fix_record, max_len)) + "\n")

def _write_findings_summary(output_file, report):
    summary = {"metadata": report.get("metadata", {}), "summary": report.get("summary", {}), "severity_summary": report.get("metadata", {}).get("severity_summary", {}), "by_pass": report.get("summary", {}).get("by_pass", {}), "by_file": {}, "by_type": {}}
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
    with open(output_file, "w") as f: json.dump(summary, f, indent=2)

def _write_findings_index(output_file, report):
    items = _normalize_findings(report)
    index = {"generated": datetime.now().isoformat(), "total_findings": len(items), "files": {}}
    for item in items:
        file_path = item.get("file", "unknown")
        entry = index["files"].setdefault(file_path, {"count": 0, "passes": {}, "severities": {}})
        entry["count"] += 1
        pass_name = item.get("pass_name", "unknown")
        entry["passes"][pass_name] = entry["passes"].get(pass_name, 0) + 1
        sev = item.get("severity", "INFO")
        entry["severities"][sev] = entry["severities"].get(sev, 0) + 1
    with open(output_file, "w") as f: json.dump(index, f, indent=2)

def _write_model_outputs(report, output_settings):
    output_dir = output_settings["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    jsonl_path = os.path.join(output_dir, output_settings["jsonl_name"])
    summary_path = os.path.join(output_dir, output_settings["summary_name"])
    index_path = os.path.join(output_dir, output_settings["index_name"])
    fixes_path = os.path.join(output_dir, output_settings["fixes_name"])
    max_len = output_settings["jsonl_max_field_len"]
    _write_jsonl_findings(jsonl_path, report, max_len)
    _write_findings_summary(summary_path, report)
    _write_findings_index(index_path, report)
    _write_fixes_jsonl(fixes_path, report, max_len)
    return {"JSONL": jsonl_path, "Summary": summary_path, "Index": index_path, "Fixes": fixes_path}

def _load_config(path):
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _merge_dicts(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_dicts(base[key], value)
        else:
            base[key] = value
    return base

def _parse_args(argv):
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--enable", default=None)
    parser.add_argument("--disable", default=None)
    parser.add_argument("--include-glob", action="append", default=None)
    parser.add_argument("--exclude-glob", action="append", default=None)
    parser.add_argument("--exclude-dir", action="append", default=None)
    parser.add_argument("--severity-threshold", default=None)
    parser.add_argument("--jsonl-max-field-len", type=int, default=None)
    return parser.parse_args(argv)

def _apply_cli_overrides(config, args):
    if args.root:
        config["root_dir"] = args.root
    if args.output_dir:
        config["output"]["output_dir"] = args.output_dir
    if args.jsonl_max_field_len is not None:
        config["output"]["jsonl_max_field_len"] = args.jsonl_max_field_len
    if args.severity_threshold:
        config["severity_threshold"] = args.severity_threshold
    if args.include_glob is not None:
        config["include_globs"] = args.include_glob
    if args.exclude_glob is not None:
        config["exclude_globs"] = args.exclude_glob
    if args.exclude_dir is not None:
        config["exclude_dirs"] = args.exclude_dir
    if args.enable:
        for name in args.enable.split(","):
            config["pipelines"][name.strip()] = True
    if args.disable:
        for name in args.disable.split(","):
            config["pipelines"][name.strip()] = False
    return config

def _iter_files(root, exts, exclude_dirs, include_globs=None, exclude_globs=None):
    for path in sorted(root.rglob("*")):
        if not path.is_file(): continue
        if exts and path.suffix.lower() not in exts: continue
        if any(part in exclude_dirs for part in path.parts): continue
        rel_path = str(path.relative_to(root))
        if include_globs and not any(fnmatch.fnmatch(rel_path, pattern) for pattern in include_globs):
            continue
        if exclude_globs and any(fnmatch.fnmatch(rel_path, pattern) for pattern in exclude_globs):
            continue
        yield path

def _scan_regex_lines(findings, pass_name, root, exts, patterns, pipeline, exclude_dirs, include_globs, exclude_globs):
    for filepath in _iter_files(root, exts, exclude_dirs, include_globs, exclude_globs):
        try:
            content = filepath.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        lines = content.split("\n")
        for line_num, line in enumerate(lines, 1):
            for pattern in patterns:
                if re.search(pattern["pattern"], line):
                    skip_finding = False

                    # Context-aware exclusions
                    if pattern.get("type") == "pass_statement":
                        # Skip if previous non-empty line is a class definition (exception classes)
                        prev_lines = [l.strip() for l in lines[:line_num-1] if l.strip()]
                        if prev_lines:
                            prev = prev_lines[-1]
                            if prev.startswith("class ") and "Error" in prev:
                                skip_finding = True
                            else:
                                # Also skip if inside a class with Error in name
                                for prev_line in reversed(prev_lines):
                                    if prev_line.startswith("class "):
                                        if "Error" in prev_line:
                                            skip_finding = True
                                        break

                    # TODO/FIXME exclusions
                    if pattern.get("type") == "todo_fixme" and not skip_finding:
                        # Skip section headers like "# PASS 3: TODO/FIXME Comments"
                        if re.search(r'^\s*(#|//|--)?\s*(PASS\s+\d+|TODO[/\s]*FIXME\s+(Comments|Analysis|Check))', line, re.IGNORECASE):
                            skip_finding = True
                        # Skip docstring mentions of TODO/FIXME
                        elif re.search(r'["\'].*TODO.*FIXME.*["\']', line):
                            skip_finding = True

                    if not skip_finding:
                        findings.add(pass_name, {"pipeline": pipeline, "severity": pattern.get("severity", "WARNING"), "type": pattern.get("type"), "file": str(filepath.relative_to(root)), "line": line_num, "context": line.strip()[:120], "expected_result": pattern.get("expected_result")})

def _scan_regex_content(findings, pass_name, root, exts, patterns, pipeline, exclude_dirs, include_globs, exclude_globs):
    for filepath in _iter_files(root, exts, exclude_dirs, include_globs, exclude_globs):
        try:
            content = filepath.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for pattern in patterns:
            if re.search(pattern["pattern"], content):
                findings.add(pass_name, {"pipeline": pipeline, "severity": pattern.get("severity", "INFO"), "type": pattern.get("type"), "file": str(filepath.relative_to(root)), "expected_result": pattern.get("expected_result")})

def _run_cmd(findings, pass_name, cmd, cwd, pipeline, root):
    if not cmd:
        return
    resolved = [str(root / cmd[0])] + cmd[1:] if not os.path.isabs(cmd[0]) and not shutil.which(cmd[0]) and (root / cmd[0]).exists() else cmd
    if not shutil.which(resolved[0]) and not os.path.exists(resolved[0]):
        findings.add(pass_name, {"pipeline": pipeline, "severity": "WARNING", "type": "tool_missing", "message": f"Tool not found: {resolved[0]}", "expected_result": f"Install {resolved[0]} or remove from checks"})
        return
    try:
        result = subprocess.run(resolved, cwd=cwd, capture_output=True, text=True, timeout=900)
        if result.returncode != 0:
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            output = (stdout + stderr)[-1000:]
            findings.add(pass_name, {"pipeline": pipeline, "severity": "ERROR", "type": "command_failed", "message": "Command failed", "command": " ".join(resolved), "file": str(cwd), "output": output, "expected_result": "Command exits with status 0"})
    except Exception as exc:
        findings.add(pass_name, {"pipeline": pipeline, "severity": "ERROR", "type": "command_error", "message": str(exc), "command": " ".join(resolved), "file": str(cwd), "expected_result": "Command runs without exception"})

def _run_pattern_set(findings, name, root, exts, patterns, pipeline, exclude_dirs, include_globs, exclude_globs):
    if not patterns:
        return
    if any(p.get("mode") == "content" for p in patterns):
        content_patterns = [p for p in patterns if p.get("mode") == "content"]
        if content_patterns:
            _scan_regex_content(findings, name, root, exts, content_patterns, pipeline, exclude_dirs, include_globs, exclude_globs)
    line_patterns = [p for p in patterns if p.get("mode", "lines") == "lines"]
    if line_patterns:
        _scan_regex_lines(findings, name, root, exts, line_patterns, pipeline, exclude_dirs, include_globs, exclude_globs)

def run_rust_pipeline(findings, root, config):
    rust_root = root / "tools" / "rust"
    exclude_dirs = set(config.get("exclude_dirs", []))
    include_globs = config.get("include_globs", [])
    exclude_globs = config.get("exclude_globs", [])
    commands = config.get("commands", {}).get("rust", {})
    if rust_root.exists():
        _run_cmd(findings, "rust_cargo_check", commands.get("cargo_check"), rust_root, "rust", root)
        _run_cmd(findings, "rust_cargo_clippy", commands.get("cargo_clippy"), rust_root, "rust", root)
        _run_cmd(findings, "rust_cargo_test", commands.get("cargo_test"), rust_root, "rust", root)
    patterns = config.get("patterns", {}).get("rust_error_handling", [])
    _run_pattern_set(findings, "rust_error_handling", root, set(config.get("exts", {}).get("rust", [".rs"])), patterns, "rust", exclude_dirs, include_globs, exclude_globs)

def run_python_pipeline(findings, root, config):
    exclude_dirs = set(config.get("exclude_dirs", []))
    include_globs = config.get("include_globs", [])
    exclude_globs = config.get("exclude_globs", [])
    commands = config.get("commands", {}).get("python", {})
    compile_cmd = commands.get("compileall", [])
    if compile_cmd:
        _run_cmd(findings, "python_compileall", compile_cmd + [str(root)], root, "python", root)
    patterns = config.get("patterns", {}).get("python_error_handling", [])
    _run_pattern_set(findings, "python_error_handling", root, set(config.get("exts", {}).get("python", [".py"])), patterns, "python", exclude_dirs, include_globs, exclude_globs)
    doc_patterns = config.get("patterns", {}).get("python_docs", [])
    _run_pattern_set(findings, "python_docs", root, set(config.get("exts", {}).get("python", [".py"])), doc_patterns, "python", exclude_dirs, include_globs, exclude_globs)

def run_spark_pipeline(findings, root, config):
    exclude_dirs = set(config.get("exclude_dirs", []))
    include_globs = config.get("include_globs", [])
    exclude_globs = config.get("exclude_globs", [])
    commands = config.get("commands", {}).get("spark", {})
    verifier_cmd = commands.get("verifier", [])
    if verifier_cmd:
        _run_cmd(findings, "spark_verifier_help", verifier_cmd, root, "spark", root)
    patterns = config.get("patterns", {}).get("spark_todos", [])
    _run_pattern_set(findings, "spark_todos", root, set(config.get("exts", {}).get("spark", [".adb", ".ads"])), patterns, "spark", exclude_dirs, include_globs, exclude_globs)

def run_shared_scans(findings, root, config):
    exclude_dirs = set(config.get("exclude_dirs", []))
    include_globs = config.get("include_globs", [])
    exclude_globs = config.get("exclude_globs", [])
    shared_exts = set(config.get("exts", {}).get("shared", [".rs", ".py", ".adb", ".ads", ".sh", ".ps1"]))
    todo_patterns = config.get("patterns", {}).get("shared_todos", [])
    _run_pattern_set(findings, "shared_todos", root, shared_exts, todo_patterns, "shared", exclude_dirs, include_globs, exclude_globs)
    path_patterns = config.get("patterns", {}).get("shared_paths", [])
    _run_pattern_set(findings, "shared_paths", root, shared_exts, path_patterns, "shared", exclude_dirs, include_globs, exclude_globs)

def main():
    args = _parse_args(sys.argv[1:])
    config = _default_config()
    config_override = _load_config(args.config)
    if config_override:
        _merge_dicts(config, config_override)
    _apply_cli_overrides(config, args)
    root = Path(config.get("root_dir")).resolve()
    findings = AnalysisFindings(str(root), config.get("severity_threshold", "INFO"))
    pipelines = config.get("pipelines", {})
    if pipelines.get("shared", True):
        run_shared_scans(findings, root, config)
    if pipelines.get("rust", True):
        run_rust_pipeline(findings, root, config)
    if pipelines.get("python", True):
        run_python_pipeline(findings, root, config)
    if pipelines.get("spark", True):
        run_spark_pipeline(findings, root, config)
    findings.metadata["end_time"] = datetime.now().isoformat()
    findings.metadata["total_findings"] = findings.get_count()
    severity_counts = {"ERROR": 0, "WARNING": 0, "INFO": 0}
    for pass_findings in findings.findings.values():
        for finding in pass_findings:
            sev = finding.get("severity", "INFO")
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
    findings.metadata["severity_summary"] = severity_counts
    report = findings.to_dict()
    _write_model_outputs(report, config.get("output", {}))
    return 0

if __name__ == "__main__":
    sys.exit(main())

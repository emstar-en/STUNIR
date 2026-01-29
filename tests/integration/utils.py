"""STUNIR Integration Test Utilities

Helper functions for integration testing.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
TOOLS_DIR = PROJECT_ROOT / "tools"
MANIFESTS_DIR = PROJECT_ROOT / "manifests"


def compute_sha256(content: str) -> str:
    """Compute SHA-256 hash of a string."""
    return hashlib.sha256(content.encode()).hexdigest()


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def canonical_json(data: Any) -> str:
    """Generate canonical JSON (sorted keys, no extra whitespace)."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def load_json_file(path: Path) -> Dict[str, Any]:
    """Load JSON from a file."""
    with open(path, "r") as f:
        return json.load(f)


def save_json_file(path: Path, data: Dict[str, Any], pretty: bool = True) -> None:
    """Save JSON to a file."""
    with open(path, "w") as f:
        if pretty:
            json.dump(data, f, indent=2, sort_keys=True)
        else:
            json.dump(data, f, sort_keys=True)


def check_tool_availability() -> Dict[str, bool]:
    """Check which STUNIR tools are available."""
    tools = {
        "python": True,  # We're running Python
        "ir_emitter": (TOOLS_DIR / "ir_emitter" / "emit_ir.py").exists(),
        "manifest_generator": (MANIFESTS_DIR / "base.py").exists(),
        "rust_native": False,
        "haskell_native": False,
    }
    
    # Check Rust tool
    rust_path = TOOLS_DIR / "native" / "rust" / "stunir-native"
    if (rust_path / "Cargo.toml").exists():
        try:
            result = subprocess.run(
                ["cargo", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            tools["rust_native"] = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    
    # Check Haskell tool
    haskell_path = TOOLS_DIR / "native" / "haskell" / "stunir-native"
    if (haskell_path / "stunir-native.cabal").exists():
        try:
            result = subprocess.run(
                ["cabal", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            tools["haskell_native"] = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    
    return tools


def run_python_tool(script_path: Path, args: List[str], cwd: Optional[Path] = None) -> Tuple[int, str, str]:
    """Run a Python tool and return exit code, stdout, stderr."""
    cmd = [sys.executable, str(script_path)] + args
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=cwd or PROJECT_ROOT,
        timeout=60
    )
    return result.returncode, result.stdout, result.stderr


def run_shell_command(cmd: List[str], cwd: Optional[Path] = None) -> Tuple[int, str, str]:
    """Run a shell command and return exit code, stdout, stderr."""
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=cwd or PROJECT_ROOT,
        timeout=60
    )
    return result.returncode, result.stdout, result.stderr


def verify_manifest_structure(manifest: Dict[str, Any]) -> List[str]:
    """Verify a manifest has required fields. Returns list of errors."""
    errors = []
    
    required_fields = ["schema", "entries"]
    for field in required_fields:
        if field not in manifest:
            errors.append(f"Missing required field: {field}")
    
    if "schema" in manifest:
        if not manifest["schema"].startswith("stunir.manifest."):
            errors.append(f"Invalid schema format: {manifest['schema']}")
    
    if "entries" in manifest:
        if not isinstance(manifest["entries"], list):
            errors.append("entries must be a list")
    
    return errors


def verify_ir_structure(ir: Dict[str, Any]) -> List[str]:
    """Verify an IR has required fields. Returns list of errors."""
    errors = []
    
    required_fields = ["kind", "functions"]
    for field in required_fields:
        if field not in ir:
            errors.append(f"Missing required field: {field}")
    
    if ir.get("kind") != "ir":
        errors.append(f"Invalid kind: expected 'ir', got '{ir.get('kind')}'")
    
    if "functions" in ir and not isinstance(ir["functions"], list):
        errors.append("functions must be a list")
    
    return errors


def assert_deterministic(func, *args, iterations: int = 3, **kwargs) -> Any:
    """Assert that a function produces deterministic output."""
    results = [func(*args, **kwargs) for _ in range(iterations)]
    
    first = results[0]
    for i, result in enumerate(results[1:], 2):
        if result != first:
            raise AssertionError(
                f"Non-deterministic output: iteration 1 != iteration {i}\n"
                f"First: {first}\nGot: {result}"
            )
    
    return first


def create_test_spec(name: str = "test", modules: int = 1) -> Dict[str, Any]:
    """Create a test spec with specified number of modules."""
    return {
        "kind": "spec",
        "modules": [
            {
                "name": f"module_{i}",
                "source": f"print('Module {i}')",
                "lang": "python"
            }
            for i in range(modules)
        ],
        "metadata": {
            "name": name,
            "version": "1.0.0"
        }
    }


def create_test_ir(module_name: str = "main", functions: int = 1) -> Dict[str, Any]:
    """Create a test IR with specified number of functions."""
    return {
        "kind": "ir",
        "generator": "stunir-test",
        "ir_version": "v1",
        "module_name": module_name,
        "functions": [
            {
                "name": f"func_{i}",
                "body": [{"op": "nop", "args": []}]
            }
            for i in range(functions)
        ],
        "modules": [],
        "metadata": {
            "original_spec_kind": "spec",
            "source_modules": []
        }
    }

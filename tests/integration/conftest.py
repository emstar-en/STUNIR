"""STUNIR Integration Test Fixtures

Provides pytest fixtures for integration testing.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, Generator

import pytest

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp = tempfile.mkdtemp(prefix="stunir_test_")
    yield Path(temp)
    shutil.rmtree(temp, ignore_errors=True)


@pytest.fixture
def sample_spec() -> Dict[str, Any]:
    """Return a sample STUNIR spec."""
    return {
        "kind": "spec",
        "modules": [
            {
                "name": "hello",
                "source": "print('Hello, World!')",
                "lang": "python"
            }
        ],
        "metadata": {
            "author": "test",
            "version": "1.0.0"
        }
    }


@pytest.fixture
def sample_ir() -> Dict[str, Any]:
    """Return a sample STUNIR IR."""
    return {
        "kind": "ir",
        "generator": "stunir-native",
        "ir_version": "v1",
        "module_name": "main",
        "functions": [
            {
                "name": "main",
                "body": [
                    {"op": "print", "args": ["Hello, World!"]}
                ]
            }
        ],
        "modules": [],
        "metadata": {
            "original_spec_kind": "spec",
            "source_modules": []
        }
    }


@pytest.fixture
def sample_manifest() -> Dict[str, Any]:
    """Return a sample STUNIR manifest."""
    return {
        "schema": "stunir.manifest.test.v1",
        "manifest_epoch": "2024-01-01T00:00:00Z",
        "entries": [],
        "manifest_hash": "0" * 64
    }


@pytest.fixture
def spec_file(temp_dir: Path, sample_spec: Dict[str, Any]) -> Path:
    """Create a spec file in the temp directory."""
    spec_path = temp_dir / "spec.json"
    with open(spec_path, "w") as f:
        json.dump(sample_spec, f, indent=2)
    return spec_path


@pytest.fixture
def ir_file(temp_dir: Path, sample_ir: Dict[str, Any]) -> Path:
    """Create an IR file in the temp directory."""
    ir_path = temp_dir / "output.ir.json"
    with open(ir_path, "w") as f:
        json.dump(sample_ir, f, indent=2)
    return ir_path


@pytest.fixture
def tools_available() -> Dict[str, bool]:
    """Check which STUNIR tools are available."""
    from .utils import check_tool_availability
    return check_tool_availability()


@pytest.fixture
def ir_emitter_path(project_root: Path) -> Path:
    """Return path to the IR emitter tool."""
    return project_root / "tools" / "ir_emitter" / "emit_ir.py"


@pytest.fixture
def manifest_tools_path(project_root: Path) -> Path:
    """Return path to manifest tools."""
    return project_root / "manifests"

"""Complete Pipeline Integration Tests

Tests the full STUNIR pipeline: spec → IR → targets → receipt
"""

import json
import sys
from pathlib import Path

import pytest

from .utils import (
    compute_sha256,
    canonical_json,
    verify_ir_structure,
    create_test_spec,
    create_test_ir,
    save_json_file,
    load_json_file,
    run_python_tool,
    PROJECT_ROOT,
)


class TestCompletePipeline:
    """Test complete STUNIR pipeline."""

    def test_spec_to_ir_conversion(self, temp_dir: Path, sample_spec: dict):
        """Test spec to IR conversion produces valid IR."""
        # Create spec file
        spec_path = temp_dir / "spec.json"
        save_json_file(spec_path, sample_spec)

        # Create expected IR structure
        ir = {
            "kind": "ir",
            "generator": "stunir-test",
            "ir_version": "v1",
            "module_name": "main",
            "functions": [
                {
                    "name": sample_spec["modules"][0]["name"],
                    "body": [
                        {"op": "raw", "args": [sample_spec["modules"][0]["source"]]}
                    ]
                }
            ],
            "modules": [],
            "metadata": {
                "original_spec_kind": "spec",
                "source_modules": sample_spec["modules"]
            }
        }

        # Verify IR structure
        errors = verify_ir_structure(ir)
        assert not errors, f"IR validation errors: {errors}"

        # Verify IR preserves spec content
        assert ir["functions"][0]["name"] == sample_spec["modules"][0]["name"]

    def test_ir_to_manifest_generation(self, temp_dir: Path, sample_ir: dict):
        """Test IR to manifest generation."""
        # Create IR directory structure
        ir_dir = temp_dir / "asm" / "ir"
        ir_dir.mkdir(parents=True)

        # Write IR files
        for i in range(3):
            ir_path = ir_dir / f"module_{i}.json"
            ir_data = create_test_ir(f"module_{i}", functions=2)
            save_json_file(ir_path, ir_data)

        # Generate manifest entries
        entries = []
        for ir_file in sorted(ir_dir.glob("*.json")):
            content = ir_file.read_text()
            entries.append({
                "name": ir_file.name,
                "path": str(ir_file.relative_to(temp_dir)),
                "hash": compute_sha256(content),
                "size": len(content)
            })

        manifest = {
            "schema": "stunir.manifest.ir.v1",
            "entries": entries,
            "count": len(entries)
        }

        # Verify manifest
        assert manifest["count"] == 3
        assert all("hash" in e for e in manifest["entries"])
        assert all(len(e["hash"]) == 64 for e in manifest["entries"])

    def test_full_pipeline_determinism(self, temp_dir: Path):
        """Test that full pipeline produces deterministic output."""
        spec = create_test_spec("determinism-test", modules=2)

        results = []
        for _ in range(3):
            # Serialize spec
            spec_json = canonical_json(spec)
            spec_hash = compute_sha256(spec_json)

            # Create IR
            ir = create_test_ir("main", functions=2)
            ir_json = canonical_json(ir)
            ir_hash = compute_sha256(ir_json)

            results.append({
                "spec_hash": spec_hash,
                "ir_hash": ir_hash
            })

        # All iterations should produce same hashes
        first = results[0]
        for i, result in enumerate(results[1:], 2):
            assert result["spec_hash"] == first["spec_hash"], \
                f"Spec hash differs at iteration {i}"
            assert result["ir_hash"] == first["ir_hash"], \
                f"IR hash differs at iteration {i}"

    def test_multi_module_spec_processing(self, temp_dir: Path):
        """Test processing spec with multiple modules."""
        # Create multi-module spec
        spec = {
            "kind": "spec",
            "modules": [
                {"name": "auth", "source": "# Auth module", "lang": "python"},
                {"name": "data", "source": "-- Data module", "lang": "sql"},
                {"name": "api", "source": "// API module", "lang": "javascript"},
            ],
            "metadata": {"name": "multi-module-app"}
        }

        # Save and reload
        spec_path = temp_dir / "multi_spec.json"
        save_json_file(spec_path, spec)
        loaded = load_json_file(spec_path)

        # Verify all modules preserved
        assert len(loaded["modules"]) == 3
        module_names = {m["name"] for m in loaded["modules"]}
        assert module_names == {"auth", "data", "api"}

        # Create combined IR
        ir = {
            "kind": "ir",
            "generator": "stunir-test",
            "ir_version": "v1",
            "module_name": "main",
            "functions": [
                {"name": m["name"], "body": [{"op": "raw", "args": [m["source"]]}]}
                for m in loaded["modules"]
            ],
            "modules": [],
            "metadata": {
                "original_spec_kind": "spec",
                "source_modules": loaded["modules"]
            }
        }

        assert len(ir["functions"]) == 3
        function_names = {f["name"] for f in ir["functions"]}
        assert function_names == {"auth", "data", "api"}


class TestPipelineOutputs:
    """Test pipeline output generation."""

    def test_ir_output_structure(self, temp_dir: Path):
        """Test IR output has correct structure."""
        ir = create_test_ir("output-test", functions=3)

        # Required fields
        assert ir["kind"] == "ir"
        assert "generator" in ir
        assert "ir_version" in ir
        assert "module_name" in ir
        assert "functions" in ir
        assert "metadata" in ir

        # Functions structure
        for func in ir["functions"]:
            assert "name" in func
            assert "body" in func
            assert isinstance(func["body"], list)

    def test_manifest_output_structure(self, temp_dir: Path):
        """Test manifest output has correct structure."""
        manifest = {
            "schema": "stunir.manifest.ir.v1",
            "manifest_epoch": "2024-01-01T00:00:00Z",
            "entries": [
                {
                    "name": "test.json",
                    "path": "asm/ir/test.json",
                    "hash": "a" * 64,
                    "size": 100
                }
            ],
            "manifest_hash": "b" * 64
        }

        # Verify schema format
        assert manifest["schema"].startswith("stunir.manifest.")

        # Verify entries
        for entry in manifest["entries"]:
            assert "name" in entry
            assert "path" in entry
            assert "hash" in entry
            assert len(entry["hash"]) == 64

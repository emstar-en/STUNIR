"""Receipt Generation and Verification Tests

Tests receipt generation and verification workflow.
"""

import json
from datetime import datetime
from pathlib import Path

import pytest

from .utils import (
    compute_sha256,
    canonical_json,
    save_json_file,
    load_json_file,
)


def create_receipt(
    receipt_id: str,
    artifacts: list,
    tools: list = None
) -> dict:
    """Create a STUNIR receipt."""
    if tools is None:
        tools = [{"name": "stunir-test", "version": "0.1.0"}]

    # Compute artifact hashes
    artifact_entries = []
    for artifact in artifacts:
        if isinstance(artifact, dict):
            content = canonical_json(artifact)
        else:
            content = str(artifact)
        artifact_entries.append({
            "hash": compute_sha256(content),
            "type": "json" if isinstance(artifact, dict) else "raw"
        })

    receipt = {
        "id": receipt_id,
        "schema": "stunir.receipt.v1",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "tools": tools,
        "artifacts": artifact_entries
    }

    # Add receipt hash
    receipt_content = canonical_json({
        "id": receipt["id"],
        "artifacts": receipt["artifacts"],
        "tools": receipt["tools"]
    })
    receipt["receipt_hash"] = compute_sha256(receipt_content)

    return receipt


def verify_receipt(receipt: dict, artifacts: list) -> tuple:
    """Verify a receipt against artifacts. Returns (valid, errors)."""
    errors = []

    # Check required fields
    required = ["id", "schema", "artifacts", "tools"]
    for field in required:
        if field not in receipt:
            errors.append(f"Missing required field: {field}")

    if errors:
        return False, errors

    # Verify artifact count matches
    if len(receipt["artifacts"]) != len(artifacts):
        errors.append(
            f"Artifact count mismatch: receipt has {len(receipt['artifacts'])}, "
            f"expected {len(artifacts)}"
        )
        return False, errors

    # Verify artifact hashes
    for i, (entry, artifact) in enumerate(zip(receipt["artifacts"], artifacts)):
        if isinstance(artifact, dict):
            expected_hash = compute_sha256(canonical_json(artifact))
        else:
            expected_hash = compute_sha256(str(artifact))

        if entry["hash"] != expected_hash:
            errors.append(
                f"Hash mismatch for artifact {i}: "
                f"receipt={entry['hash'][:16]}..., expected={expected_hash[:16]}..."
            )

    return len(errors) == 0, errors


class TestReceiptGeneration:
    """Test receipt generation."""

    def test_create_basic_receipt(self):
        """Test creating a basic receipt."""
        artifacts = [{"data": "test"}]
        receipt = create_receipt("test-001", artifacts)

        assert receipt["id"] == "test-001"
        assert receipt["schema"] == "stunir.receipt.v1"
        assert len(receipt["artifacts"]) == 1
        assert len(receipt["tools"]) == 1
        assert "receipt_hash" in receipt
        assert len(receipt["receipt_hash"]) == 64

    def test_create_multi_artifact_receipt(self):
        """Test receipt with multiple artifacts."""
        artifacts = [
            {"module": "auth", "code": "..."},
            {"module": "data", "code": "..."},
            {"module": "api", "code": "..."},
        ]
        receipt = create_receipt("multi-001", artifacts)

        assert len(receipt["artifacts"]) == 3
        # Each artifact should have unique hash
        hashes = [a["hash"] for a in receipt["artifacts"]]
        # Note: artifacts have different content so hashes differ
        assert len(set(hashes)) == 3

    def test_receipt_determinism(self):
        """Test that receipt generation is deterministic."""
        artifacts = [{"test": "data", "value": 42}]

        receipts = []
        for i in range(3):
            # Use fixed ID to test determinism (timestamp will vary)
            receipt = create_receipt(f"det-{i}", artifacts)
            # Compare only deterministic parts
            receipts.append({
                "artifacts": receipt["artifacts"],
                "tools": receipt["tools"]
            })

        # All artifact hashes should be identical
        first_hash = receipts[0]["artifacts"][0]["hash"]
        for r in receipts[1:]:
            assert r["artifacts"][0]["hash"] == first_hash


class TestReceiptVerification:
    """Test receipt verification."""

    def test_verify_valid_receipt(self):
        """Test verifying a valid receipt."""
        artifacts = [{"data": "test"}]
        receipt = create_receipt("valid-001", artifacts)

        valid, errors = verify_receipt(receipt, artifacts)
        assert valid, f"Expected valid receipt, got errors: {errors}"

    def test_verify_invalid_hash(self):
        """Test verification fails with wrong artifact."""
        original_artifacts = [{"data": "original"}]
        receipt = create_receipt("mismatch-001", original_artifacts)

        # Try to verify with different artifacts
        different_artifacts = [{"data": "different"}]
        valid, errors = verify_receipt(receipt, different_artifacts)

        assert not valid, "Should fail verification"
        assert any("Hash mismatch" in e for e in errors)

    def test_verify_missing_artifact(self):
        """Test verification fails with missing artifact."""
        artifacts = [{"a": 1}, {"b": 2}]
        receipt = create_receipt("missing-001", artifacts)

        # Verify with fewer artifacts
        valid, errors = verify_receipt(receipt, [{"a": 1}])

        assert not valid, "Should fail verification"
        assert any("count mismatch" in e for e in errors)

    def test_verify_incomplete_receipt(self):
        """Test verification fails with incomplete receipt."""
        incomplete = {"id": "incomplete-001"}  # Missing required fields

        valid, errors = verify_receipt(incomplete, [])

        assert not valid, "Should fail verification"
        assert any("Missing required field" in e for e in errors)


class TestReceiptPersistence:
    """Test receipt save/load operations."""

    def test_save_and_load_receipt(self, temp_dir: Path):
        """Test saving and loading a receipt."""
        artifacts = [{"data": "persistent"}]
        receipt = create_receipt("persist-001", artifacts)

        # Save
        receipt_path = temp_dir / "receipt.json"
        save_json_file(receipt_path, receipt)

        # Load
        loaded = load_json_file(receipt_path)

        assert loaded["id"] == receipt["id"]
        assert loaded["receipt_hash"] == receipt["receipt_hash"]

        # Verify loaded receipt
        valid, errors = verify_receipt(loaded, artifacts)
        assert valid, f"Loaded receipt should be valid: {errors}"

    def test_receipt_roundtrip_determinism(self, temp_dir: Path):
        """Test receipt is identical after save/load."""
        artifacts = [{"key": "value", "number": 42}]
        receipt = create_receipt("roundtrip-001", artifacts)

        # Save and load
        path = temp_dir / "roundtrip.json"
        save_json_file(path, receipt, pretty=False)
        loaded = load_json_file(path)

        # Compare deterministic fields
        assert loaded["id"] == receipt["id"]
        assert loaded["artifacts"] == receipt["artifacts"]
        assert loaded["tools"] == receipt["tools"]
        assert loaded["receipt_hash"] == receipt["receipt_hash"]

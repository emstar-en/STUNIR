#!/usr/bin/env python3
"""STUNIR Test Vectors Base Module.

Shared utilities for all test vector generators and validators.
Part of Phase 5: Test Vectors pipeline stages.
"""

import json
import hashlib
import os
import random
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod


# Fixed epochs for determinism
DEFAULT_EPOCH = 1735500000
TEST_SEED = 42


def canonical_json(data: Any) -> str:
    """Generate RFC 8785 / JCS subset canonical JSON.
    
    Args:
        data: Data to serialize
        
    Returns:
        Canonical JSON string with sorted keys and no whitespace
    """
    return json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def canonical_json_pretty(data: Any) -> str:
    """Generate human-readable canonical JSON.
    
    Args:
        data: Data to serialize
        
    Returns:
        Pretty-printed JSON string with sorted keys
    """
    return json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False)


def compute_sha256(data: Any) -> str:
    """Compute SHA-256 hash of data.
    
    Args:
        data: String or bytes to hash
        
    Returns:
        Lowercase hex digest
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def compute_file_hash(filepath: str) -> str:
    """Compute SHA-256 hash of a file.
    
    Args:
        filepath: Path to file
        
    Returns:
        Lowercase hex digest
    """
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def seeded_rng(seed: int = TEST_SEED) -> random.Random:
    """Create a seeded random number generator for determinism.
    
    Args:
        seed: Random seed value
        
    Returns:
        Seeded Random instance
    """
    return random.Random(seed)


def generate_test_id(category: str, index: int) -> str:
    """Generate a unique test vector ID.
    
    Args:
        category: Test category (e.g., 'contracts', 'native')
        index: Test index number
        
    Returns:
        Unique test ID string
    """
    return f"tv_{category}_{index:03d}"


def validate_schema(data: Dict, required_fields: List[str]) -> Tuple[bool, List[str]]:
    """Validate that data contains required fields.
    
    Args:
        data: Dictionary to validate
        required_fields: List of required field names
        
    Returns:
        Tuple of (is_valid, missing_fields)
    """
    missing = [f for f in required_fields if f not in data]
    return len(missing) == 0, missing


class BaseTestVectorGenerator(ABC):
    """Abstract base class for test vector generators."""
    
    SCHEMA_VERSION = "stunir.test_vector.v1"
    
    def __init__(self, category: str, output_dir: str = None):
        """Initialize generator.
        
        Args:
            category: Test vector category name
            output_dir: Output directory (defaults to test_vectors/<category>)
        """
        self.category = category
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'test_vectors', category
        )
        self.rng = seeded_rng()
        self.vectors = []
    
    def create_vector(self, index: int, name: str, description: str,
                     test_input: Any, expected_output: Any,
                     tags: List[str] = None) -> Dict:
        """Create a test vector with standard structure.
        
        Args:
            index: Test index
            name: Human-readable test name
            description: What this test verifies
            test_input: Input data for the test
            expected_output: Expected result
            tags: Optional list of tags
            
        Returns:
            Test vector dictionary
        """
        expected_canonical = canonical_json(expected_output)
        expected_hash = compute_sha256(expected_canonical)
        
        return {
            "schema": f"stunir.test_vector.{self.category}.v1",
            "id": generate_test_id(self.category, index),
            "name": name,
            "description": description,
            "input": test_input,
            "expected_output": expected_output,
            "expected_hash": expected_hash,
            "tags": tags or ["unit", "determinism"],
            "created_epoch": DEFAULT_EPOCH
        }
    
    @abstractmethod
    def _generate_vectors(self) -> List[Dict]:
        """Generate category-specific test vectors.
        
        Returns:
            List of test vector dictionaries
        """
        pass
    
    def generate(self) -> List[Dict]:
        """Generate all test vectors.
        
        Returns:
            List of generated test vectors
        """
        self.vectors = self._generate_vectors()
        return self.vectors
    
    def write_vectors(self) -> Tuple[int, str]:
        """Write generated vectors to files.
        
        Returns:
            Tuple of (count, manifest_hash)
        """
        if not self.vectors:
            self.generate()
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        manifest_entries = []
        
        for vector in self.vectors:
            filename = f"{vector['id']}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w') as f:
                f.write(canonical_json_pretty(vector))
            
            manifest_entries.append({
                "id": vector['id'],
                "name": vector['name'],
                "file": filename,
                "hash": compute_sha256(canonical_json(vector))
            })
        
        # Write manifest
        manifest = {
            "schema": f"stunir.test_vectors.{self.category}.manifest.v1",
            "category": self.category,
            "count": len(self.vectors),
            "epoch": DEFAULT_EPOCH,
            "entries": manifest_entries
        }
        
        manifest_path = os.path.join(self.output_dir, 'manifest.json')
        with open(manifest_path, 'w') as f:
            f.write(canonical_json_pretty(manifest))
        
        manifest_hash = compute_sha256(canonical_json(manifest))
        
        return len(self.vectors), manifest_hash


class BaseTestVectorValidator(ABC):
    """Abstract base class for test vector validators."""
    
    def __init__(self, category: str, vectors_dir: str = None):
        """Initialize validator.
        
        Args:
            category: Test vector category name
            vectors_dir: Directory containing test vectors
        """
        self.category = category
        self.vectors_dir = vectors_dir or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'test_vectors', category
        )
        self.results = []
    
    def load_vectors(self) -> List[Dict]:
        """Load all test vectors from directory.
        
        Returns:
            List of test vector dictionaries
        """
        vectors = []
        
        if not os.path.isdir(self.vectors_dir):
            return vectors
        
        for filename in sorted(os.listdir(self.vectors_dir)):
            if filename.startswith('tv_') and filename.endswith('.json'):
                filepath = os.path.join(self.vectors_dir, filename)
                with open(filepath, 'r') as f:
                    vectors.append(json.load(f))
        
        return vectors
    
    @abstractmethod
    def _validate_vector(self, vector: Dict) -> Tuple[bool, str]:
        """Validate a single test vector.
        
        Args:
            vector: Test vector dictionary
            
        Returns:
            Tuple of (is_valid, message)
        """
        pass
    
    def validate_hash_integrity(self, vector: Dict) -> Tuple[bool, str]:
        """Validate expected_hash matches expected_output.
        
        Args:
            vector: Test vector dictionary
            
        Returns:
            Tuple of (is_valid, message)
        """
        expected_canonical = canonical_json(vector.get('expected_output', {}))
        computed_hash = compute_sha256(expected_canonical)
        expected_hash = vector.get('expected_hash', '')
        
        if computed_hash == expected_hash:
            return True, "Hash integrity verified"
        else:
            return False, f"Hash mismatch: expected {expected_hash}, got {computed_hash}"
    
    def validate_all(self) -> Tuple[int, int, List[Dict]]:
        """Validate all test vectors in directory.
        
        Returns:
            Tuple of (passed_count, failed_count, results)
        """
        vectors = self.load_vectors()
        passed = 0
        failed = 0
        self.results = []
        
        for vector in vectors:
            # Check hash integrity
            hash_valid, hash_msg = self.validate_hash_integrity(vector)
            
            # Check category-specific validation
            vector_valid, vector_msg = self._validate_vector(vector)
            
            is_valid = hash_valid and vector_valid
            
            result = {
                "id": vector.get('id', 'unknown'),
                "valid": is_valid,
                "hash_check": hash_msg,
                "validation": vector_msg
            }
            
            self.results.append(result)
            
            if is_valid:
                passed += 1
            else:
                failed += 1
        
        return passed, failed, self.results

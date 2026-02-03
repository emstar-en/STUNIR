#!/usr/bin/env python3
"""
STUNIR Load Testing with Locust
===============================

Load tests for critical STUNIR operations.

Run with:
    locust -f tests/load/locustfile.py --headless -u 10 -r 2 -t 60s
"""

import sys
import os
import time
import json
import tempfile
import random
import string

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from locust import User, task, between, events
    from locust.runners import MasterRunner, WorkerRunner
    LOCUST_AVAILABLE = True
except ImportError:
    LOCUST_AVAILABLE = False
    # Stub for when locust not installed
    class User:
        wait_time = None
    def task(weight=1):
        return lambda f: f
    def between(min_val, max_val):
        return None

# Import STUNIR modules
try:
    from manifests.base import canonical_json, compute_sha256, compute_file_hash
    MANIFESTS_AVAILABLE = True
except ImportError:
    MANIFESTS_AVAILABLE = False
    def canonical_json(data): return json.dumps(data, sort_keys=True)
    def compute_sha256(data): 
        import hashlib
        return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()
    def compute_file_hash(path):
        import hashlib
        with open(path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()

try:
    from tools.ir_emitter.emit_ir import spec_to_ir
    IR_EMITTER_AVAILABLE = True
except ImportError:
    IR_EMITTER_AVAILABLE = False


# ============================================================================
# Test Data Generation
# ============================================================================

def generate_random_spec(num_functions: int = 5) -> dict:
    """Generate a random STUNIR spec for testing."""
    def random_name():
        return ''.join(random.choices(string.ascii_lowercase, k=8))
    
    functions = []
    for _ in range(num_functions):
        functions.append({
            "name": random_name(),
            "params": [
                {"name": random_name(), "type": random.choice(["i32", "f64", "bool"])}
                for _ in range(random.randint(0, 3))
            ],
            "return_type": random.choice(["i32", "f64", "void"]),
            "body": [
                {"op": "return", "value": random.randint(0, 100)}
            ]
        })
    
    return {
        "module": random_name(),
        "version": "1.0",
        "functions": functions,
        "types": [],
        "imports": [],
        "exports": [f["name"] for f in functions[:2]]
    }


def generate_random_manifest(num_entries: int = 20) -> dict:
    """Generate a random manifest for testing."""
    entries = []
    for i in range(num_entries):
        entries.append({
            "name": f"entry_{i}",
            "path": f"path/to/file_{i}.json",
            "hash": compute_sha256(f"content_{i}_{random.random()}"),
            "size": random.randint(100, 10000)
        })
    
    return {
        "manifest_schema": "stunir.manifest.v1",
        "manifest_epoch": int(time.time()),
        "entries": entries
    }


# ============================================================================
# Load Test User
# ============================================================================

if LOCUST_AVAILABLE:
    class STUNIRUser(User):
        """Simulates a STUNIR user performing various operations."""
        
        wait_time = between(0.1, 0.5)  # 100-500ms between tasks
        
        def on_start(self):
            """Setup for each user."""
            self.specs = [generate_random_spec(random.randint(1, 10)) for _ in range(5)]
            self.manifests = [generate_random_manifest(random.randint(5, 50)) for _ in range(5)]
            self.temp_dir = tempfile.mkdtemp()
        
        @task(weight=10)
        def compute_hash(self):
            """Test SHA256 hash computation."""
            start = time.perf_counter()
            data = json.dumps(random.choice(self.specs))
            result = compute_sha256(data)
            elapsed = (time.perf_counter() - start) * 1000
            
            events.request.fire(
                request_type="HASH",
                name="compute_sha256",
                response_time=elapsed,
                response_length=len(result),
                exception=None,
                context={}
            )
        
        @task(weight=8)
        def canonical_json_task(self):
            """Test canonical JSON generation."""
            start = time.perf_counter()
            data = random.choice(self.manifests)
            result = canonical_json(data)
            elapsed = (time.perf_counter() - start) * 1000
            
            events.request.fire(
                request_type="JSON",
                name="canonical_json",
                response_time=elapsed,
                response_length=len(result),
                exception=None,
                context={}
            )
        
        @task(weight=5)
        def emit_ir_task(self):
            """Test IR emission (if available)."""
            if not IR_EMITTER_AVAILABLE:
                return
            
            start = time.perf_counter()
            spec = random.choice(self.specs)
            try:
                result = spec_to_ir(spec)
                elapsed = (time.perf_counter() - start) * 1000
                events.request.fire(
                    request_type="IR",
                    name="emit_ir",
                    response_time=elapsed,
                    response_length=len(json.dumps(result)),
                    exception=None,
                    context={}
                )
            except Exception as e:
                elapsed = (time.perf_counter() - start) * 1000
                events.request.fire(
                    request_type="IR",
                    name="emit_ir",
                    response_time=elapsed,
                    response_length=0,
                    exception=e,
                    context={}
                )
        
        @task(weight=3)
        def large_manifest_task(self):
            """Test processing a large manifest."""
            start = time.perf_counter()
            large_manifest = generate_random_manifest(100)
            result = canonical_json(large_manifest)
            hash_result = compute_sha256(result)
            elapsed = (time.perf_counter() - start) * 1000
            
            events.request.fire(
                request_type="MANIFEST",
                name="large_manifest_processing",
                response_time=elapsed,
                response_length=len(result),
                exception=None,
                context={}
            )
        
        @task(weight=2)
        def file_hash_task(self):
            """Test file hashing."""
            # Create temp file
            content = json.dumps(random.choice(self.specs))
            filepath = os.path.join(self.temp_dir, f"test_{random.randint(0, 1000)}.json")
            with open(filepath, 'w') as f:
                f.write(content)
            
            start = time.perf_counter()
            result = compute_file_hash(filepath)
            elapsed = (time.perf_counter() - start) * 1000
            
            events.request.fire(
                request_type="FILE",
                name="file_hash",
                response_time=elapsed,
                response_length=len(result),
                exception=None,
                context={}
            )
            
            # Cleanup
            try:
                os.remove(filepath)
            except OSError:
                pass

#!/usr/bin/env python3
"""STUNIR Batch Processing Example

This example demonstrates batch operations in STUNIR:
- Processing multiple specs in parallel
- Batch manifest generation
- Aggregate receipts
- Progress tracking

Usage:
    python batch_processing.py [--specs-dir <dir>] [--workers <n>]
"""

import json
import hashlib
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import threading

# =============================================================================
# Core Utilities
# =============================================================================

def canonical_json(data: Any) -> str:
    """RFC 8785 compliant canonical JSON."""
    return json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False)

def compute_sha256(data: str) -> str:
    """Compute SHA-256 hash."""
    return hashlib.sha256(data.encode('utf-8')).hexdigest()

# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ProcessingResult:
    """Result of processing a single spec."""
    spec_name: str
    success: bool
    ir_hash: Optional[str]
    receipt_hash: Optional[str]
    duration_ms: float
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class BatchSummary:
    """Summary of batch processing."""
    total: int
    successful: int
    failed: int
    total_duration_ms: float
    results: List[ProcessingResult]
    aggregate_hash: str
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['results'] = [r.to_dict() for r in self.results]
        return data

# =============================================================================
# Sample Specs Generator
# =============================================================================

def generate_sample_specs(count: int = 5) -> List[Dict[str, Any]]:
    """Generate sample specs for demonstration."""
    specs = []
    for i in range(count):
        spec = {
            "name": f"module_{i:03d}",
            "version": "1.0.0",
            "functions": [
                {
                    "name": f"func_{j}",
                    "params": [{"name": "x", "type": "i32"}],
                    "returns": "i32",
                    "body": [{"op": "return", "value": f"x * {j+1}"}]
                }
                for j in range(3)
            ],
            "exports": [f"func_{j}" for j in range(3)]
        }
        specs.append(spec)
    return specs

# =============================================================================
# Single Item Processing
# =============================================================================

def process_spec(spec: Dict[str, Any]) -> ProcessingResult:
    """Process a single spec and return results."""
    start_time = time.time()
    spec_name = spec.get("name", "unnamed")
    
    try:
        # Compute spec hash
        spec_json = canonical_json(spec)
        spec_hash = compute_sha256(spec_json)
        
        # Generate IR
        ir = {
            "ir_version": "1.0.0",
            "ir_epoch": int(datetime.now(timezone.utc).timestamp()),
            "ir_spec_hash": spec_hash,
            "module": {
                "name": spec_name,
                "version": spec.get("version", "0.0.0")
            },
            "functions": spec.get("functions", []),
            "exports": spec.get("exports", [])
        }
        
        ir_json = canonical_json(ir)
        ir_hash = compute_sha256(ir_json)
        
        # Generate receipt
        receipt = {
            "receipt_version": "1.0.0",
            "receipt_epoch": int(datetime.now(timezone.utc).timestamp()),
            "module_name": spec_name,
            "ir_hash": ir_hash,
            "spec_hash": spec_hash
        }
        
        receipt_content = canonical_json(receipt)
        receipt_hash = compute_sha256(receipt_content)
        
        # Simulate some processing time
        time.sleep(0.1)
        
        duration_ms = (time.time() - start_time) * 1000
        
        return ProcessingResult(
            spec_name=spec_name,
            success=True,
            ir_hash=ir_hash,
            receipt_hash=receipt_hash,
            duration_ms=duration_ms
        )
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        return ProcessingResult(
            spec_name=spec_name,
            success=False,
            ir_hash=None,
            receipt_hash=None,
            duration_ms=duration_ms,
            error=str(e)
        )

# =============================================================================
# Batch Processor
# =============================================================================

class BatchProcessor:
    """Processes multiple specs in parallel."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self._lock = threading.Lock()
        self._completed = 0
        self._total = 0
        
    def process(self, specs: List[Dict[str, Any]], 
                progress_callback: Optional[callable] = None) -> BatchSummary:
        """Process all specs in parallel.
        
        Args:
            specs: List of spec dictionaries
            progress_callback: Optional callback(completed, total)
            
        Returns:
            BatchSummary with all results
        """
        self._total = len(specs)
        self._completed = 0
        results: List[ProcessingResult] = []
        
        start_time = time.time()
        
        print(f"🚀 Starting batch processing of {len(specs)} specs...")
        print(f"   Workers: {self.max_workers}")
        print()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(process_spec, spec): spec for spec in specs}
            
            # Collect results as they complete
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                
                with self._lock:
                    self._completed += 1
                    completed = self._completed
                
                # Report progress
                status = "✅" if result.success else "❌"
                print(f"   [{completed}/{self._total}] {status} {result.spec_name} "
                      f"({result.duration_ms:.1f}ms)")
                
                if progress_callback:
                    progress_callback(completed, self._total)
        
        total_duration_ms = (time.time() - start_time) * 1000
        
        # Sort results by spec name for deterministic ordering
        results.sort(key=lambda r: r.spec_name)
        
        # Compute aggregate hash
        aggregate_data = [r.to_dict() for r in results if r.success]
        aggregate_hash = compute_sha256(canonical_json(aggregate_data))
        
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        
        return BatchSummary(
            total=len(results),
            successful=successful,
            failed=failed,
            total_duration_ms=total_duration_ms,
            results=results,
            aggregate_hash=aggregate_hash
        )

# =============================================================================
# Aggregate Manifest Generator
# =============================================================================

class AggregateManifestGenerator:
    """Generates aggregate manifests from batch results."""
    
    def __init__(self, summary: BatchSummary):
        self.summary = summary
        
    def generate(self) -> Dict[str, Any]:
        """Generate an aggregate manifest."""
        entries = []
        
        for result in self.summary.results:
            if result.success:
                entries.append({
                    "name": result.spec_name,
                    "ir_hash": result.ir_hash,
                    "receipt_hash": result.receipt_hash,
                    "processing_time_ms": result.duration_ms
                })
        
        manifest = {
            "schema": "stunir.manifest.aggregate.v1",
            "manifest_epoch": int(datetime.now(timezone.utc).timestamp()),
            "batch_stats": {
                "total": self.summary.total,
                "successful": self.summary.successful,
                "failed": self.summary.failed,
                "total_duration_ms": self.summary.total_duration_ms
            },
            "entries": entries
        }
        
        # Compute manifest hash
        content = {k: v for k, v in manifest.items() if k != "manifest_hash"}
        manifest["manifest_hash"] = compute_sha256(canonical_json(content))
        
        return manifest

# =============================================================================
# Progress Bar (Simple ASCII)
# =============================================================================

class ProgressBar:
    """Simple ASCII progress bar."""
    
    def __init__(self, total: int, width: int = 40):
        self.total = total
        self.width = width
        
    def render(self, completed: int) -> str:
        """Render the progress bar."""
        if self.total == 0:
            return "[" + "=" * self.width + "]"
            
        ratio = completed / self.total
        filled = int(self.width * ratio)
        empty = self.width - filled
        
        bar = "[" + "=" * filled + ">" + " " * (empty - 1) + "]"
        percent = ratio * 100
        
        return f"{bar} {percent:5.1f}% ({completed}/{self.total})"

# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point for batch processing example."""
    parser = argparse.ArgumentParser(description="STUNIR Batch Processing Example")
    parser.add_argument("--count", type=int, default=10, help="Number of specs to process")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    args = parser.parse_args()
    
    print("="*60)
    print("STUNIR Batch Processing Example")
    print("="*60)
    print()
    
    # Generate sample specs
    print(f"📄 Generating {args.count} sample specs...")
    specs = generate_sample_specs(args.count)
    print()
    
    # Process in batch
    processor = BatchProcessor(max_workers=args.workers)
    summary = processor.process(specs)
    print()
    
    # Generate aggregate manifest
    print("📋 Generating aggregate manifest...")
    manifest_gen = AggregateManifestGenerator(summary)
    manifest = manifest_gen.generate()
    print(f"   Manifest hash: {manifest['manifest_hash'][:16]}...")
    print()
    
    # Display summary
    print("="*60)
    print("Batch Processing Summary")
    print("="*60)
    print(f"Total specs:      {summary.total}")
    print(f"Successful:       {summary.successful}")
    print(f"Failed:           {summary.failed}")
    print(f"Total duration:   {summary.total_duration_ms:.1f}ms")
    print(f"Avg per spec:     {summary.total_duration_ms/summary.total:.1f}ms")
    print(f"Aggregate hash:   {summary.aggregate_hash[:16]}...")
    print()
    
    if summary.failed > 0:
        print("Failed specs:")
        for result in summary.results:
            if not result.success:
                print(f"  - {result.spec_name}: {result.error}")
        print()
    
    print("✅ Batch processing example completed!")
    
    return 0 if summary.failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())

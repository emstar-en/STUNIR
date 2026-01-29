#!/usr/bin/env python3
"""STUNIR Advanced Usage Example

This example demonstrates advanced STUNIR features:
- Multi-target code generation
- Manifest verification
- Provenance tracking
- Pipeline orchestration
- Custom configurations

Usage:
    python advanced_usage.py [options]
"""

import json
import hashlib
import argparse
import os
import sys
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

# =============================================================================
# Core Utilities
# =============================================================================

def canonical_json(data: Any) -> str:
    """RFC 8785 compliant canonical JSON."""
    return json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False)

def compute_sha256(data: str) -> str:
    """Compute SHA-256 hash."""
    return hashlib.sha256(data.encode('utf-8')).hexdigest()

def compute_file_hash(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

# =============================================================================
# Target Configuration
# =============================================================================

class TargetLanguage(Enum):
    """Supported target languages."""
    PYTHON = "python"
    RUST = "rust"
    C89 = "c89"
    C99 = "c99"
    X86_64 = "x86_64"
    ARM64 = "arm64"
    WASM = "wasm"

@dataclass
class TargetConfig:
    """Configuration for a code generation target."""
    language: TargetLanguage
    output_dir: str
    optimize: bool = False
    debug_symbols: bool = False
    extra_flags: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['language'] = self.language.value
        return result

# =============================================================================
# Advanced IR Processing
# =============================================================================

@dataclass
class IRFunction:
    """Represents a function in the IR."""
    name: str
    params: List[Dict[str, str]]
    returns: str
    body: List[Dict[str, Any]]
    attributes: Optional[Dict[str, Any]] = None
    
    def signature_hash(self) -> str:
        """Compute hash of function signature."""
        sig = canonical_json({"name": self.name, "params": self.params, "returns": self.returns})
        return compute_sha256(sig)

@dataclass  
class IRModule:
    """Represents a complete IR module."""
    name: str
    version: str
    functions: List[IRFunction]
    exports: List[str]
    imports: Optional[List[Dict[str, str]]] = None
    types: Optional[List[Dict[str, Any]]] = None
    
    def compute_hash(self) -> str:
        """Compute deterministic module hash."""
        data = {
            "name": self.name,
            "version": self.version,
            "functions": [asdict(f) for f in self.functions],
            "exports": self.exports,
            "imports": self.imports or [],
            "types": self.types or []
        }
        return compute_sha256(canonical_json(data))

# =============================================================================
# Multi-Target Code Generation
# =============================================================================

class MultiTargetEmitter:
    """Emits code for multiple target languages."""
    
    TYPE_MAPPINGS = {
        TargetLanguage.PYTHON: {"i32": "int", "f64": "float", "bool": "bool", "str": "str"},
        TargetLanguage.RUST: {"i32": "i32", "f64": "f64", "bool": "bool", "str": "String"},
        TargetLanguage.C89: {"i32": "int", "f64": "double", "bool": "int", "str": "char*"},
        TargetLanguage.C99: {"i32": "int32_t", "f64": "double", "bool": "bool", "str": "char*"},
    }
    
    def __init__(self, module: IRModule):
        self.module = module
        self.outputs: Dict[TargetLanguage, str] = {}
        
    def emit(self, targets: List[TargetConfig]) -> Dict[TargetLanguage, str]:
        """Emit code for all specified targets."""
        print(f"🎯 Emitting code for {len(targets)} targets...")
        
        for config in targets:
            code = self._emit_target(config)
            self.outputs[config.language] = code
            print(f"   ✅ {config.language.value}: {len(code)} bytes")
            
        return self.outputs
    
    def _emit_target(self, config: TargetConfig) -> str:
        """Emit code for a single target."""
        if config.language == TargetLanguage.PYTHON:
            return self._emit_python(config)
        elif config.language == TargetLanguage.RUST:
            return self._emit_rust(config)
        elif config.language in (TargetLanguage.C89, TargetLanguage.C99):
            return self._emit_c(config)
        else:
            return f"// Target {config.language.value} not implemented\n"
    
    def _emit_python(self, config: TargetConfig) -> str:
        """Generate Python code."""
        lines = [
            f'"""Generated Python module: {self.module.name}',
            f'Version: {self.module.version}',
            f'Generated by STUNIR',
            '"""',
            '',
            'from typing import Any',
            ''
        ]
        
        for func in self.module.functions:
            params = ', '.join(f"{p['name']}: {self.TYPE_MAPPINGS[TargetLanguage.PYTHON].get(p['type'], 'Any')}" 
                              for p in func.params)
            ret_type = self.TYPE_MAPPINGS[TargetLanguage.PYTHON].get(func.returns, 'Any')
            
            lines.append(f'def {func.name}({params}) -> {ret_type}:')
            lines.append(f'    """STUNIR generated function: {func.name}"""')
            lines.append('    pass  # Implementation placeholder')
            lines.append('')
        
        return '\n'.join(lines)
    
    def _emit_rust(self, config: TargetConfig) -> str:
        """Generate Rust code."""
        lines = [
            f'//! Generated Rust module: {self.module.name}',
            f'//! Version: {self.module.version}',
            '//! Generated by STUNIR',
            '',
        ]
        
        for func in self.module.functions:
            params = ', '.join(f"{p['name']}: {self.TYPE_MAPPINGS[TargetLanguage.RUST].get(p['type'], 'i32')}" 
                              for p in func.params)
            ret_type = self.TYPE_MAPPINGS[TargetLanguage.RUST].get(func.returns, 'i32')
            
            lines.append(f'pub fn {func.name}({params}) -> {ret_type} {{')
            lines.append('    todo!("Implementation placeholder")')
            lines.append('}')
            lines.append('')
        
        return '\n'.join(lines)
    
    def _emit_c(self, config: TargetConfig) -> str:
        """Generate C code."""
        type_map = self.TYPE_MAPPINGS[config.language]
        dialect = "C89" if config.language == TargetLanguage.C89 else "C99"
        
        lines = [
            f'/* Generated C module: {self.module.name} */',
            f'/* Version: {self.module.version} */',
            f'/* Dialect: {dialect} */',
            '/* Generated by STUNIR */',
            '',
        ]
        
        if config.language == TargetLanguage.C99:
            lines.extend(['#include <stdint.h>', '#include <stdbool.h>', ''])
        
        for func in self.module.functions:
            params = ', '.join(f"{type_map.get(p['type'], 'int')} {p['name']}" 
                              for p in func.params)
            ret_type = type_map.get(func.returns, 'int')
            
            lines.append(f'{ret_type} {func.name}({params}) {{')
            lines.append(f'    /* Implementation placeholder */')
            lines.append(f'    return 0;')
            lines.append('}')
            lines.append('')
        
        return '\n'.join(lines)

# =============================================================================
# Manifest Verification
# =============================================================================

class ManifestVerifier:
    """Verifies manifest integrity and completeness."""
    
    def __init__(self, manifest_path: str):
        self.manifest_path = manifest_path
        self.manifest: Optional[Dict[str, Any]] = None
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
    def load(self) -> bool:
        """Load the manifest file."""
        try:
            with open(self.manifest_path, 'r') as f:
                self.manifest = json.load(f)
            return True
        except FileNotFoundError:
            self.errors.append(f"Manifest not found: {self.manifest_path}")
            return False
        except json.JSONDecodeError as e:
            self.errors.append(f"Invalid JSON in manifest: {e}")
            return False
    
    def verify(self) -> Tuple[bool, List[str], List[str]]:
        """Verify manifest integrity.
        
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        if not self.manifest:
            if not self.load():
                return False, self.errors, self.warnings
        
        print("🔍 Verifying manifest...")
        
        # Check required fields
        required = ['schema', 'manifest_epoch', 'entries', 'manifest_hash']
        for field in required:
            if field not in self.manifest:
                self.errors.append(f"Missing required field: {field}")
        
        if self.errors:
            return False, self.errors, self.warnings
        
        # Verify manifest hash
        content = {k: v for k, v in self.manifest.items() if k != 'manifest_hash'}
        computed_hash = compute_sha256(canonical_json(content))
        if computed_hash != self.manifest['manifest_hash']:
            self.errors.append("Manifest hash mismatch - possible tampering!")
        
        # Verify entry hashes
        for entry in self.manifest.get('entries', []):
            if 'path' in entry and 'hash' in entry:
                if os.path.exists(entry['path']):
                    actual_hash = compute_file_hash(entry['path'])
                    if actual_hash != entry['hash']:
                        self.errors.append(f"Hash mismatch for {entry['path']}")
                else:
                    self.warnings.append(f"File not found: {entry['path']}")
        
        is_valid = len(self.errors) == 0
        status = "✅ Valid" if is_valid else "❌ Invalid"
        print(f"   {status} ({len(self.errors)} errors, {len(self.warnings)} warnings)")
        
        return is_valid, self.errors, self.warnings

# =============================================================================
# Provenance Tracking
# =============================================================================

@dataclass
class Provenance:
    """Tracks build provenance information."""
    build_epoch: int
    spec_hash: str
    ir_hash: str
    tool_version: str
    targets: List[str]
    environment: Dict[str, str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_c_header(self) -> str:
        """Generate C header with provenance macros."""
        return f"""/* STUNIR Provenance Header */
/* Auto-generated - do not edit */

#ifndef STUNIR_PROVENANCE_H
#define STUNIR_PROVENANCE_H

#define STUNIR_PROV_BUILD_EPOCH {self.build_epoch}
#define STUNIR_PROV_SPEC_DIGEST "{self.spec_hash}"
#define STUNIR_PROV_IR_DIGEST "{self.ir_hash}"
#define STUNIR_PROV_TOOL_VERSION "{self.tool_version}"
#define STUNIR_PROV_TARGET_COUNT {len(self.targets)}

#endif /* STUNIR_PROVENANCE_H */
"""

class ProvenanceTracker:
    """Tracks and records build provenance."""
    
    def __init__(self):
        self.events: List[Dict[str, Any]] = []
        
    def record(self, event_type: str, data: Dict[str, Any]):
        """Record a provenance event."""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            "data": data
        }
        self.events.append(event)
        print(f"📋 Recorded: {event_type}")
        
    def generate_provenance(self, spec_hash: str, ir_hash: str, targets: List[str]) -> Provenance:
        """Generate final provenance record."""
        return Provenance(
            build_epoch=int(datetime.now(timezone.utc).timestamp()),
            spec_hash=spec_hash,
            ir_hash=ir_hash,
            tool_version="2.0.0",
            targets=targets,
            environment={
                "python_version": sys.version.split()[0],
                "platform": sys.platform
            }
        )

# =============================================================================
# Pipeline Orchestration
# =============================================================================

class Pipeline:
    """Orchestrates the complete STUNIR build pipeline."""
    
    def __init__(self, spec: Dict[str, Any], targets: List[TargetConfig]):
        self.spec = spec
        self.targets = targets
        self.tracker = ProvenanceTracker()
        self.ir_module: Optional[IRModule] = None
        self.outputs: Dict[TargetLanguage, str] = {}
        
    def run(self) -> Dict[str, Any]:
        """Execute the complete pipeline."""
        print("="*60)
        print("STUNIR Advanced Pipeline")
        print("="*60)
        print()
        
        # Stage 1: Parse spec
        self.tracker.record("spec_parse", {"name": self.spec.get("name")})
        spec_hash = compute_sha256(canonical_json(self.spec))
        
        # Stage 2: Generate IR
        print("📦 Stage 1: Generating IR...")
        self.ir_module = self._spec_to_ir()
        ir_hash = self.ir_module.compute_hash()
        self.tracker.record("ir_generate", {"hash": ir_hash})
        print()
        
        # Stage 3: Emit targets
        print("📦 Stage 2: Emitting targets...")
        emitter = MultiTargetEmitter(self.ir_module)
        self.outputs = emitter.emit(self.targets)
        self.tracker.record("targets_emit", {"count": len(self.outputs)})
        print()
        
        # Stage 4: Generate provenance
        print("📦 Stage 3: Generating provenance...")
        target_names = [t.language.value for t in self.targets]
        provenance = self.tracker.generate_provenance(spec_hash, ir_hash, target_names)
        print()
        
        # Summary
        print("="*60)
        print("Pipeline Summary")
        print("="*60)
        print(f"Module:       {self.ir_module.name}")
        print(f"Functions:    {len(self.ir_module.functions)}")
        print(f"Targets:      {', '.join(target_names)}")
        print(f"Spec Hash:    {spec_hash[:16]}...")
        print(f"IR Hash:      {ir_hash[:16]}...")
        print()
        print("✅ Pipeline completed successfully!")
        
        return {
            "ir_hash": ir_hash,
            "spec_hash": spec_hash,
            "provenance": provenance.to_dict(),
            "outputs": {k.value: len(v) for k, v in self.outputs.items()}
        }
    
    def _spec_to_ir(self) -> IRModule:
        """Convert spec to IR module."""
        functions = []
        for func in self.spec.get("functions", []):
            ir_func = IRFunction(
                name=func["name"],
                params=func.get("params", []),
                returns=func.get("returns", "void"),
                body=func.get("body", [])
            )
            functions.append(ir_func)
            
        return IRModule(
            name=self.spec.get("name", "unnamed"),
            version=self.spec.get("version", "0.0.0"),
            functions=functions,
            exports=self.spec.get("exports", [])
        )

# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point for advanced usage example."""
    parser = argparse.ArgumentParser(description="STUNIR Advanced Usage Example")
    parser.add_argument("--targets", nargs="+", 
                       choices=["python", "rust", "c89", "c99"],
                       default=["python", "rust"],
                       help="Target languages")
    args = parser.parse_args()
    
    # Sample spec
    spec = {
        "name": "advanced_module",
        "version": "2.0.0",
        "functions": [
            {"name": "process", "params": [{"name": "data", "type": "i32"}], "returns": "i32"},
            {"name": "transform", "params": [{"name": "x", "type": "f64"}], "returns": "f64"},
        ],
        "exports": ["process", "transform"]
    }
    
    # Configure targets
    target_map = {
        "python": TargetLanguage.PYTHON,
        "rust": TargetLanguage.RUST,
        "c89": TargetLanguage.C89,
        "c99": TargetLanguage.C99
    }
    targets = [TargetConfig(language=target_map[t], output_dir=f"output/{t}") 
               for t in args.targets]
    
    # Run pipeline
    pipeline = Pipeline(spec, targets)
    results = pipeline.run()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

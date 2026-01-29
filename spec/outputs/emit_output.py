#!/usr/bin/env python3
"""STUNIR Output Emitter

Pipeline Stage: spec -> outputs
Issue: #1049

Emits structured output artifacts from STUNIR pipeline stages.
"""

import json
import hashlib
import time
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional


def canonical_json(data: Any) -> str:
    """Generate RFC 8785 canonical JSON."""
    return json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def compute_sha256(data: str) -> str:
    """Compute SHA-256 hash."""
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def compute_file_hash(path: Path) -> str:
    """Compute SHA-256 hash of file contents."""
    hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


class OutputEmitter:
    """STUNIR Output Emitter.
    
    Generates structured output specifications for pipeline artifacts.
    """
    
    SCHEMA = 'stunir.output.v1'
    VERSION = '1.0.0'
    
    def __init__(self, input_id: str, input_hash: str):
        """Initialize emitter.
        
        Args:
            input_id: Source input spec ID
            input_hash: SHA-256 hash of source input
        """
        self.input_id = input_id
        self.input_hash = input_hash
        self.artifacts: List[Dict[str, Any]] = []
        self.start_time = time.time()
    
    def add_artifact(self, name: str, path: Path, format: str = None) -> Dict[str, Any]:
        """Add an artifact to the output.
        
        Args:
            name: Artifact name
            path: Path to artifact file
            format: Optional format identifier
            
        Returns:
            Artifact entry dictionary
        """
        artifact = {
            'name': name,
            'path': str(path),
            'hash': compute_file_hash(path),
            'size': path.stat().st_size
        }
        
        if format:
            artifact['format'] = format
        elif path.suffix:
            artifact['format'] = path.suffix.lstrip('.')
        
        self.artifacts.append(artifact)
        return artifact
    
    def add_artifact_from_content(self, name: str, path: str, content: str, format: str = None) -> Dict[str, Any]:
        """Add an artifact from content (without file).
        
        Args:
            name: Artifact name
            path: Virtual path
            content: Content string
            format: Optional format identifier
            
        Returns:
            Artifact entry dictionary
        """
        artifact = {
            'name': name,
            'path': path,
            'hash': compute_sha256(content),
            'size': len(content.encode('utf-8'))
        }
        
        if format:
            artifact['format'] = format
        
        self.artifacts.append(artifact)
        return artifact
    
    def emit(self, output_type: str, target: str = None) -> Dict[str, Any]:
        """Emit the output specification.
        
        Args:
            output_type: Type of output (ir, code, binary, receipt, manifest)
            target: Optional target language/platform
            
        Returns:
            Output specification dictionary
        """
        duration_ms = int((time.time() - self.start_time) * 1000)
        
        output = {
            'schema': self.SCHEMA,
            'input_id': self.input_id,
            'input_hash': self.input_hash,
            'output_type': output_type,
            'artifacts': sorted(self.artifacts, key=lambda a: a['name']),
            'metadata': {
                'epoch': int(time.time()),
                'tool_version': self.VERSION,
                'duration_ms': duration_ms
            }
        }
        
        if target:
            output['target'] = target
        
        return output
    
    def emit_canonical(self, output_type: str, target: str = None) -> str:
        """Emit canonical JSON output."""
        return canonical_json(self.emit(output_type, target))
    
    def write(self, output_path: Path, output_type: str, target: str = None) -> str:
        """Write output to file.
        
        Args:
            output_path: Output file path
            output_type: Type of output
            target: Optional target
            
        Returns:
            SHA-256 hash of output
        """
        content = self.emit_canonical(output_type, target)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(content)
            f.write('\n')
        
        return compute_sha256(content)


def main():
    """CLI interface for output emitter."""
    import argparse
    
    parser = argparse.ArgumentParser(description='STUNIR Output Emitter')
    parser.add_argument('--input-id', required=True, help='Source input ID')
    parser.add_argument('--input-hash', required=True, help='Source input hash')
    parser.add_argument('--type', required=True, 
                        choices=['ir', 'code', 'binary', 'receipt', 'manifest'],
                        help='Output type')
    parser.add_argument('--target', help='Target language/platform')
    parser.add_argument('--artifact', action='append', nargs=2, metavar=('NAME', 'PATH'),
                        help='Add artifact (can be repeated)')
    parser.add_argument('-o', '--output', help='Output file path')
    
    args = parser.parse_args()
    
    emitter = OutputEmitter(args.input_id, args.input_hash)
    
    # Add artifacts
    if args.artifact:
        for name, path in args.artifact:
            artifact_path = Path(path)
            if artifact_path.exists():
                emitter.add_artifact(name, artifact_path)
            else:
                print(f"Warning: Artifact not found: {path}", file=sys.stderr)
    
    # Emit output
    if args.output:
        output_hash = emitter.write(Path(args.output), args.type, args.target)
        print(f"Output written to: {args.output}", file=sys.stderr)
        print(f"Hash: {output_hash}", file=sys.stderr)
    else:
        output = emitter.emit(args.type, args.target)
        print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""STUNIR Nested Spec Resolver

Pipeline Stage: spec -> nested
Issue: #1151

Resolves nested spec imports and produces flattened output.
"""

import json
import hashlib
import sys
from pathlib import Path
from typing import Dict, Any, List, Set, Optional, Tuple
from dataclasses import dataclass, field


def canonical_json(data: Any) -> str:
    """Generate RFC 8785 canonical JSON."""
    return json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def compute_sha256(data: str) -> str:
    """Compute SHA-256 hash."""
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


@dataclass
class ImportNode:
    """Import dependency node."""
    module: str
    path: str
    items: List[str]
    resolved: bool = False
    spec: Optional[Dict] = None


@dataclass
class ResolutionResult:
    """Result of nested spec resolution."""
    success: bool
    flattened: Dict[str, Any] = field(default_factory=dict)
    imports_resolved: int = 0
    import_order: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    hash: str = ''


class NestedResolver:
    """STUNIR Nested Spec Resolver.
    
    Resolves import dependencies and produces flattened specs.
    """
    
    def __init__(self, search_paths: Optional[List[Path]] = None):
        """Initialize resolver.
        
        Args:
            search_paths: Directories to search for imports
        """
        self.search_paths = search_paths or [
            Path('.'),
            Path('spec'),
            Path('modules')
        ]
        self._cache: Dict[str, Dict] = {}
        self._resolving: Set[str] = set()  # For circular dependency detection
    
    def add_search_path(self, path: Path):
        """Add a search path for imports."""
        if path not in self.search_paths:
            self.search_paths.insert(0, path)
    
    def find_module(self, module_name: str) -> Optional[Path]:
        """Find a module by name in search paths.
        
        Args:
            module_name: Module name (dot-separated)
            
        Returns:
            Path to module file or None
        """
        # Convert module name to path
        module_path = module_name.replace('.', '/')
        
        # Search for various file patterns
        patterns = [
            f"{module_path}.json",
            f"{module_path}/index.json",
            f"{module_path}.stunir",
            f"{module_path}/spec.json"
        ]
        
        for search_path in self.search_paths:
            for pattern in patterns:
                candidate = search_path / pattern
                if candidate.exists():
                    return candidate
        
        return None
    
    def load_spec(self, path: Path) -> Dict[str, Any]:
        """Load a spec from file (with caching)."""
        key = str(path.resolve())
        if key in self._cache:
            return self._cache[key]
        
        with open(path, 'r') as f:
            spec = json.load(f)
        
        self._cache[key] = spec
        return spec
    
    def extract_imports(self, spec: Dict[str, Any]) -> List[ImportNode]:
        """Extract imports from a spec."""
        imports = []
        
        # Check module imports
        if 'module' in spec:
            module = spec['module']
            if 'imports' in module:
                for imp in module['imports']:
                    if isinstance(imp, str):
                        imports.append(ImportNode(
                            module=imp,
                            path='',
                            items=['*']
                        ))
                    elif isinstance(imp, dict):
                        imports.append(ImportNode(
                            module=imp.get('module', ''),
                            path=imp.get('path', ''),
                            items=imp.get('items', ['*'])
                        ))
        
        # Check top-level dependencies (for including other specs)
        if 'dependencies' in spec:
            for dep in spec.get('dependencies', []):
                if isinstance(dep, dict) and 'spec' in dep:
                    imports.append(ImportNode(
                        module=dep.get('name', ''),
                        path=dep.get('spec', ''),
                        items=['*']
                    ))
        
        return imports
    
    def resolve(self, spec: Dict[str, Any], base_path: Path = None) -> ResolutionResult:
        """Resolve all nested imports in a spec.
        
        Args:
            spec: Root spec to resolve
            base_path: Base directory for relative paths
            
        Returns:
            ResolutionResult with flattened spec
        """
        result = ResolutionResult(success=True)
        
        if base_path:
            self.add_search_path(base_path)
        
        # Get spec identifier for circular detection
        spec_id = spec.get('id', compute_sha256(canonical_json(spec))[:8])
        
        # Check for circular dependency
        if spec_id in self._resolving:
            result.success = False
            result.errors.append(f"Circular dependency detected: {spec_id}")
            return result
        
        self._resolving.add(spec_id)
        
        try:
            # Start with a copy of the original spec
            flattened = dict(spec)
            
            # Extract and resolve imports
            imports = self.extract_imports(spec)
            
            all_types = []
            all_functions = []
            all_constants = []
            
            # Collect existing items from the spec
            if 'module' in flattened:
                module = flattened['module']
                all_types.extend(module.get('types', []))
                all_functions.extend(module.get('functions', []))
                all_constants.extend(module.get('constants', []))
            
            for import_node in imports:
                # Find the module
                module_path = None
                if import_node.path:
                    module_path = Path(import_node.path)
                else:
                    module_path = self.find_module(import_node.module)
                
                if not module_path or not module_path.exists():
                    result.warnings.append(f"Import not found: {import_node.module}")
                    continue
                
                # Load and recursively resolve
                try:
                    imported_spec = self.load_spec(module_path)
                    nested_result = self.resolve(imported_spec, module_path.parent)
                    
                    if not nested_result.success:
                        result.errors.extend(nested_result.errors)
                        continue
                    
                    imported_flat = nested_result.flattened
                    import_node.resolved = True
                    import_node.spec = imported_flat
                    result.imports_resolved += 1
                    result.import_order.append(import_node.module)
                    
                    # Merge items from imported module
                    if 'module' in imported_flat:
                        imp_module = imported_flat['module']
                        
                        # Filter by import items if not '*'
                        items_to_import = import_node.items
                        
                        for typ in imp_module.get('types', []):
                            if '*' in items_to_import or typ.get('name') in items_to_import:
                                # Avoid duplicates
                                if not any(t.get('name') == typ.get('name') for t in all_types):
                                    all_types.append(typ)
                        
                        for func in imp_module.get('functions', []):
                            if '*' in items_to_import or func.get('name') in items_to_import:
                                if not any(f.get('name') == func.get('name') for f in all_functions):
                                    all_functions.append(func)
                        
                        for const in imp_module.get('constants', []):
                            if '*' in items_to_import or const.get('name') in items_to_import:
                                if not any(c.get('name') == const.get('name') for c in all_constants):
                                    all_constants.append(const)
                
                except Exception as e:
                    result.errors.append(f"Error loading {import_node.module}: {e}")
            
            # Build flattened module
            if 'module' not in flattened:
                flattened['module'] = {'name': spec.get('name', 'unnamed')}
            
            # Update with merged items
            if all_types:
                flattened['module']['types'] = sorted(all_types, key=lambda t: t.get('name', ''))
            if all_functions:
                flattened['module']['functions'] = sorted(all_functions, key=lambda f: f.get('name', ''))
            if all_constants:
                flattened['module']['constants'] = sorted(all_constants, key=lambda c: c.get('name', ''))
            
            # Clear imports in flattened output (they're now inlined)
            if 'imports' in flattened.get('module', {}):
                flattened['module']['imports'] = []
            
            # Add resolution metadata
            flattened['_resolved'] = {
                'imports_count': result.imports_resolved,
                'import_order': result.import_order
            }
            
            result.flattened = flattened
            result.hash = compute_sha256(canonical_json(flattened))
            
        finally:
            self._resolving.discard(spec_id)
        
        return result
    
    def resolve_file(self, path: Path) -> ResolutionResult:
        """Resolve a spec file."""
        spec = self.load_spec(path)
        return self.resolve(spec, path.parent)


def main():
    """CLI interface for nested resolver."""
    import argparse
    
    parser = argparse.ArgumentParser(description='STUNIR Nested Spec Resolver')
    parser.add_argument('input', help='Input spec file')
    parser.add_argument('-o', '--output', help='Output file path')
    parser.add_argument('--search-path', '-I', action='append', help='Add search path')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    search_paths = [Path(p) for p in (args.search_path or [])]
    resolver = NestedResolver(search_paths)
    
    # Resolve
    input_path = Path(args.input)
    resolver.add_search_path(input_path.parent)
    
    result = resolver.resolve_file(input_path)
    
    if args.verbose:
        print(f"Imports resolved: {result.imports_resolved}", file=sys.stderr)
        print(f"Import order: {' -> '.join(result.import_order) or 'none'}", file=sys.stderr)
        if result.warnings:
            for warn in result.warnings:
                print(f"WARNING: {warn}", file=sys.stderr)
        if result.errors:
            for err in result.errors:
                print(f"ERROR: {err}", file=sys.stderr)
    
    if not result.success:
        print("Resolution FAILED", file=sys.stderr)
        for err in result.errors:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)
    
    # Output
    if args.json or args.output:
        output = canonical_json(result.flattened)
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
                f.write('\n')
            print(f"Output written to: {args.output}", file=sys.stderr)
            print(f"Hash: {result.hash}", file=sys.stderr)
        else:
            print(output)
    else:
        print(json.dumps(result.flattened, indent=2, sort_keys=True))
        print(f"\nHash: {result.hash}", file=sys.stderr)


if __name__ == '__main__':
    main()

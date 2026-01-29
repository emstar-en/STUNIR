#!/usr/bin/env python3
"""STUNIR Spec Complexity Analyzer

Pipeline Stage: spec -> complexity
Issue: #1150

Analyzes complexity metrics of STUNIR specifications.
"""

import json
import hashlib
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict


def canonical_json(data: Any) -> str:
    """Generate RFC 8785 canonical JSON."""
    return json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def compute_sha256(data: str) -> str:
    """Compute SHA-256 hash."""
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


@dataclass
class ComplexityMetrics:
    """Complexity metrics for a spec."""
    # Basic counts
    function_count: int = 0
    type_count: int = 0
    import_count: int = 0
    export_count: int = 0
    constant_count: int = 0
    
    # Complexity scores
    cyclomatic_complexity: int = 0
    nesting_depth: int = 0
    type_complexity: int = 0
    
    # Size metrics
    total_params: int = 0
    total_fields: int = 0
    total_variants: int = 0
    
    # Derived scores
    overall_complexity: float = 0.0
    maintainability_index: float = 100.0


class ComplexityAnalyzer:
    """STUNIR Spec Complexity Analyzer.
    
    Analyzes various complexity metrics of STUNIR specifications.
    """
    
    # Weights for overall complexity calculation
    WEIGHTS = {
        'function_count': 1.0,
        'type_count': 1.5,
        'cyclomatic_complexity': 2.0,
        'nesting_depth': 1.5,
        'import_count': 0.5
    }
    
    def __init__(self, metrics_path: Optional[Path] = None):
        """Initialize analyzer.
        
        Args:
            metrics_path: Optional path to custom metrics config
        """
        self.metrics_path = metrics_path or Path(__file__).parent / 'metrics.json'
        self._metrics_config = None
    
    @property
    def metrics_config(self) -> Dict[str, Any]:
        """Lazy load metrics configuration."""
        if self._metrics_config is None:
            if self.metrics_path.exists():
                with open(self.metrics_path, 'r') as f:
                    self._metrics_config = json.load(f)
            else:
                self._metrics_config = {'weights': self.WEIGHTS}
        return self._metrics_config
    
    def analyze(self, spec: Dict[str, Any]) -> ComplexityMetrics:
        """Analyze complexity of a spec.
        
        Args:
            spec: Spec dictionary to analyze
            
        Returns:
            ComplexityMetrics dataclass
        """
        metrics = ComplexityMetrics()
        
        # Basic counts from top level
        if 'targets' in spec:
            metrics.export_count = len(spec.get('targets', []))
        
        # Module analysis
        if 'module' in spec:
            module = spec['module']
            
            # Function analysis
            functions = module.get('functions', [])
            metrics.function_count = len(functions)
            
            for func in functions:
                # Count parameters
                params = func.get('params', [])
                metrics.total_params += len(params)
                
                # Estimate cyclomatic complexity from body
                body = func.get('body', [])
                metrics.cyclomatic_complexity += self._estimate_cyclomatic(body)
            
            # Type analysis
            types = module.get('types', [])
            metrics.type_count = len(types)
            
            for typ in types:
                kind = typ.get('kind', '')
                if kind == 'struct':
                    fields = typ.get('fields', [])
                    metrics.total_fields += len(fields)
                    metrics.type_complexity += len(fields)
                elif kind == 'enum':
                    variants = typ.get('variants', [])
                    metrics.total_variants += len(variants)
                    metrics.type_complexity += len(variants) * 0.5
            
            # Import/export analysis
            metrics.import_count = len(module.get('imports', []))
            metrics.export_count = len(module.get('exports', []))
            metrics.constant_count = len(module.get('constants', []))
        
        # Nesting depth analysis
        metrics.nesting_depth = self._calculate_nesting_depth(spec)
        
        # Calculate overall complexity
        metrics.overall_complexity = self._calculate_overall(metrics)
        
        # Calculate maintainability index (simplified)
        metrics.maintainability_index = self._calculate_maintainability(metrics)
        
        return metrics
    
    def _estimate_cyclomatic(self, body: List[Any]) -> int:
        """Estimate cyclomatic complexity from function body."""
        if not body:
            return 1  # Base complexity
        
        complexity = 1
        for stmt in body:
            if isinstance(stmt, dict):
                op = stmt.get('op', '')
                # Decision points increase complexity
                if op in ('if', 'match', 'while', 'for', 'loop'):
                    complexity += 1
                # Nested bodies
                if 'body' in stmt:
                    complexity += self._estimate_cyclomatic(stmt['body'])
                if 'else' in stmt:
                    complexity += self._estimate_cyclomatic(stmt.get('else', []))
        
        return complexity
    
    def _calculate_nesting_depth(self, obj: Any, depth: int = 0) -> int:
        """Calculate maximum nesting depth."""
        max_depth = depth
        
        if isinstance(obj, dict):
            for v in obj.values():
                max_depth = max(max_depth, self._calculate_nesting_depth(v, depth + 1))
        elif isinstance(obj, list):
            for item in obj:
                max_depth = max(max_depth, self._calculate_nesting_depth(item, depth + 1))
        
        return max_depth
    
    def _calculate_overall(self, metrics: ComplexityMetrics) -> float:
        """Calculate overall complexity score."""
        weights = self.metrics_config.get('weights', self.WEIGHTS)
        
        score = 0.0
        score += metrics.function_count * weights.get('function_count', 1.0)
        score += metrics.type_count * weights.get('type_count', 1.5)
        score += metrics.cyclomatic_complexity * weights.get('cyclomatic_complexity', 2.0)
        score += metrics.nesting_depth * weights.get('nesting_depth', 1.5)
        score += metrics.import_count * weights.get('import_count', 0.5)
        
        return round(score, 2)
    
    def _calculate_maintainability(self, metrics: ComplexityMetrics) -> float:
        """Calculate maintainability index (0-100)."""
        # Simplified maintainability formula
        # Higher is better, penalized by complexity
        base = 100.0
        
        # Deductions
        deductions = 0.0
        deductions += metrics.cyclomatic_complexity * 2
        deductions += metrics.nesting_depth * 3
        deductions += (metrics.function_count / 10) * 5 if metrics.function_count > 10 else 0
        deductions += (metrics.type_count / 5) * 5 if metrics.type_count > 5 else 0
        
        return max(0.0, round(base - deductions, 1))
    
    def analyze_file(self, path: Path) -> Dict[str, Any]:
        """Analyze a spec file."""
        with open(path, 'r') as f:
            spec = json.load(f)
        
        metrics = self.analyze(spec)
        
        return {
            'file': str(path),
            'hash': compute_sha256(canonical_json(spec)),
            'metrics': asdict(metrics)
        }
    
    def compare(self, metrics1: ComplexityMetrics, metrics2: ComplexityMetrics) -> Dict[str, Any]:
        """Compare two complexity metrics."""
        m1 = asdict(metrics1)
        m2 = asdict(metrics2)
        
        diff = {}
        for key in m1:
            diff[key] = {
                'before': m1[key],
                'after': m2[key],
                'delta': round(m2[key] - m1[key], 2) if isinstance(m1[key], (int, float)) else None
            }
        
        return diff


def main():
    """CLI interface for complexity analyzer."""
    import argparse
    
    parser = argparse.ArgumentParser(description='STUNIR Spec Complexity Analyzer')
    parser.add_argument('files', nargs='+', help='Spec files to analyze')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    parser.add_argument('--compare', action='store_true', help='Compare two files')
    parser.add_argument('-o', '--output', help='Output file path')
    
    args = parser.parse_args()
    
    analyzer = ComplexityAnalyzer()
    
    if args.compare and len(args.files) == 2:
        # Compare two files
        result1 = analyzer.analyze_file(Path(args.files[0]))
        result2 = analyzer.analyze_file(Path(args.files[1]))
        
        m1 = ComplexityMetrics(**result1['metrics'])
        m2 = ComplexityMetrics(**result2['metrics'])
        
        comparison = {
            'file1': args.files[0],
            'file2': args.files[1],
            'comparison': analyzer.compare(m1, m2)
        }
        
        if args.json:
            print(json.dumps(comparison, indent=2))
        else:
            print(f"Comparing: {args.files[0]} vs {args.files[1]}")
            print("="*50)
            for key, vals in comparison['comparison'].items():
                if vals['delta'] is not None:
                    sign = '+' if vals['delta'] > 0 else ''
                    print(f"{key}: {vals['before']} -> {vals['after']} ({sign}{vals['delta']})")
        
        sys.exit(0)
    
    results = []
    for file_path in args.files:
        try:
            result = analyzer.analyze_file(Path(file_path))
            results.append(result)
            
            if not args.json:
                print(f"\n{file_path}")
                print("="*50)
                metrics = result['metrics']
                print(f"Functions: {metrics['function_count']}")
                print(f"Types: {metrics['type_count']}")
                print(f"Cyclomatic Complexity: {metrics['cyclomatic_complexity']}")
                print(f"Nesting Depth: {metrics['nesting_depth']}")
                print(f"Overall Complexity: {metrics['overall_complexity']}")
                print(f"Maintainability Index: {metrics['maintainability_index']}")
                print(f"Hash: {result['hash'][:16]}...")
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}", file=sys.stderr)
            results.append({'file': file_path, 'error': str(e)})
    
    if args.json:
        output = json.dumps(results, indent=2)
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            print(f"Output written to: {args.output}", file=sys.stderr)
        else:
            print(output)


if __name__ == '__main__':
    main()

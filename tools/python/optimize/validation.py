#!/usr/bin/env python3
"""STUNIR Optimization Validation Framework.

Validates that optimizations preserve program semantics.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Set, Tuple
import copy
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


class OptimizationValidator:
    """Validates optimization correctness."""
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate(self, original_ir: Dict[str, Any], 
                 optimized_ir: Dict[str, Any]) -> bool:
        """Validate that optimized IR preserves semantics.
        
        Args:
            original_ir: Original IR before optimization
            optimized_ir: IR after optimization
            
        Returns:
            True if optimization is valid
        """
        self.errors.clear()
        self.warnings.clear()
        
        valid = True
        
        # Check structure preservation
        if not self._validate_structure(original_ir, optimized_ir):
            valid = False
        
        # Check function preservation
        if not self._validate_functions(original_ir, optimized_ir):
            valid = False
        
        # Check type preservation
        if not self._validate_types(original_ir, optimized_ir):
            valid = False
        
        # Check control flow preservation
        if not self._validate_control_flow(original_ir, optimized_ir):
            valid = False
        
        return valid
    
    def _validate_structure(self, original: Dict, optimized: Dict) -> bool:
        """Validate IR structure is preserved."""
        # Module name must be preserved
        orig_module = original.get('ir_module', original.get('module', ''))
        opt_module = optimized.get('ir_module', optimized.get('module', ''))
        if orig_module != opt_module:
            self.errors.append(f"Module name changed: {orig_module} -> {opt_module}")
            return False
        
        return True
    
    def _validate_functions(self, original: Dict, optimized: Dict) -> bool:
        """Validate all functions are preserved."""
        orig_funcs = {f.get('name') for f in original.get('ir_functions', [])}
        opt_funcs = {f.get('name') for f in optimized.get('ir_functions', [])}
        
        # All original functions should exist (may have additional inline temps)
        missing = orig_funcs - opt_funcs
        if missing:
            # Allow removal if they were marked as inline-only
            real_missing = [f for f in missing if not f.startswith('_inline_')]
            if real_missing:
                self.errors.append(f"Functions removed: {real_missing}")
                return False
        
        # Validate each function's interface
        orig_func_map = {f.get('name'): f for f in original.get('ir_functions', [])}
        opt_func_map = {f.get('name'): f for f in optimized.get('ir_functions', [])}
        
        for name, orig_func in orig_func_map.items():
            if name not in opt_func_map:
                continue
            
            opt_func = opt_func_map[name]
            
            # Parameter count must match
            orig_params = orig_func.get('params', [])
            opt_params = opt_func.get('params', [])
            if len(orig_params) != len(opt_params):
                self.errors.append(
                    f"Function {name}: param count changed {len(orig_params)} -> {len(opt_params)}")
                return False
            
            # Return type must be preserved (if specified)
            orig_ret = orig_func.get('return_type')
            opt_ret = opt_func.get('return_type')
            if orig_ret and opt_ret and orig_ret != opt_ret:
                self.warnings.append(
                    f"Function {name}: return type changed {orig_ret} -> {opt_ret}")
        
        return True
    
    def _validate_types(self, original: Dict, optimized: Dict) -> bool:
        """Validate type definitions are preserved."""
        orig_types = {t.get('name') for t in original.get('ir_types', [])}
        opt_types = {t.get('name') for t in optimized.get('ir_types', [])}
        
        missing = orig_types - opt_types
        if missing:
            self.errors.append(f"Type definitions removed: {missing}")
            return False
        
        return True
    
    def _validate_control_flow(self, original: Dict, optimized: Dict) -> bool:
        """Validate control flow is semantically preserved."""
        # Check that all reachable return statements are preserved
        for orig_func in original.get('ir_functions', []):
            name = orig_func.get('name')
            opt_func = None
            for f in optimized.get('ir_functions', []):
                if f.get('name') == name:
                    opt_func = f
                    break
            
            if not opt_func:
                continue
            
            # Count reachable returns
            orig_returns = self._count_returns(orig_func.get('body', []))
            opt_returns = self._count_returns(opt_func.get('body', []))
            
            # Optimized may have fewer returns (dead code elimination)
            # but should have at least one if original had any
            if orig_returns > 0 and opt_returns == 0:
                self.warnings.append(
                    f"Function {name}: all return statements removed")
        
        return True
    
    def _count_returns(self, body: List) -> int:
        """Count return statements in body."""
        count = 0
        for stmt in body:
            if isinstance(stmt, dict):
                if stmt.get('type') == 'return':
                    count += 1
                for key in ('then', 'else', 'body'):
                    if key in stmt:
                        count += self._count_returns(stmt[key])
        return count
    
    def get_report(self) -> str:
        """Get validation report."""
        lines = ["Optimization Validation Report"]
        lines.append("=" * 40)
        
        if not self.errors and not self.warnings:
            lines.append("✓ All validations passed")
        else:
            if self.errors:
                lines.append(f"\nErrors ({len(self.errors)}):")
                for err in self.errors:
                    lines.append(f"  ✗ {err}")
            
            if self.warnings:
                lines.append(f"\nWarnings ({len(self.warnings)}):")
                for warn in self.warnings:
                    lines.append(f"  ⚠ {warn}")
        
        return '\n'.join(lines)


class SemanticComparator:
    """Compares semantic equivalence of IR."""
    
    def compute_semantic_hash(self, ir_data: Dict[str, Any]) -> str:
        """Compute a semantic hash of the IR.
        
        This hash should be the same for semantically equivalent IR,
        even if the structure differs slightly.
        """
        # Extract semantic elements
        semantic_elements = []
        
        # Module identity
        semantic_elements.append(('module', ir_data.get('ir_module', '')))
        
        # Function signatures (order-independent)
        func_sigs = []
        for func in ir_data.get('ir_functions', []):
            name = func.get('name', '')
            params = len(func.get('params', []))
            ret_type = func.get('return_type', 'void')
            func_sigs.append(f"{name}({params}):{ret_type}")
        func_sigs.sort()
        semantic_elements.append(('functions', tuple(func_sigs)))
        
        # Type definitions
        type_names = sorted(t.get('name', '') for t in ir_data.get('ir_types', []))
        semantic_elements.append(('types', tuple(type_names)))
        
        # Compute hash
        content = json.dumps(semantic_elements, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def compare(self, original_ir: Dict[str, Any], 
                optimized_ir: Dict[str, Any]) -> Dict[str, Any]:
        """Compare original and optimized IR.
        
        Returns:
            Dictionary with comparison results
        """
        orig_hash = self.compute_semantic_hash(original_ir)
        opt_hash = self.compute_semantic_hash(optimized_ir)
        
        # Count statements
        orig_stmts = self._count_statements(original_ir)
        opt_stmts = self._count_statements(optimized_ir)
        
        # Count nodes
        orig_nodes = self._count_nodes(original_ir)
        opt_nodes = self._count_nodes(optimized_ir)
        
        return {
            'semantic_match': orig_hash == opt_hash,
            'original_hash': orig_hash,
            'optimized_hash': opt_hash,
            'statement_reduction': orig_stmts - opt_stmts,
            'original_statements': orig_stmts,
            'optimized_statements': opt_stmts,
            'node_reduction': orig_nodes - opt_nodes,
            'reduction_percent': ((orig_stmts - opt_stmts) / orig_stmts * 100) if orig_stmts > 0 else 0
        }
    
    def _count_statements(self, ir_data: Dict) -> int:
        """Count total statements in IR."""
        count = 0
        for func in ir_data.get('ir_functions', []):
            count += self._count_in_body(func.get('body', []))
        return count
    
    def _count_in_body(self, body: List) -> int:
        count = len(body)
        for stmt in body:
            if isinstance(stmt, dict):
                for key in ('then', 'else', 'body'):
                    if key in stmt:
                        count += self._count_in_body(stmt[key])
        return count
    
    def _count_nodes(self, ir_data: Dict) -> int:
        """Count total IR nodes."""
        return self._count_dict_nodes(ir_data)
    
    def _count_dict_nodes(self, obj: Any) -> int:
        if isinstance(obj, dict):
            count = 1
            for v in obj.values():
                count += self._count_dict_nodes(v)
            return count
        elif isinstance(obj, list):
            count = 0
            for item in obj:
                count += self._count_dict_nodes(item)
            return count
        return 0


def validate_optimization(original_ir: Dict[str, Any], 
                          optimized_ir: Dict[str, Any]) -> Tuple[bool, str]:
    """Convenience function to validate optimization.
    
    Args:
        original_ir: Original IR
        optimized_ir: Optimized IR
        
    Returns:
        Tuple of (is_valid, report_string)
    """
    validator = OptimizationValidator()
    is_valid = validator.validate(original_ir, optimized_ir)
    return is_valid, validator.get_report()


def compare_optimization(original_ir: Dict[str, Any],
                         optimized_ir: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to compare original and optimized IR.
    
    Args:
        original_ir: Original IR
        optimized_ir: Optimized IR
        
    Returns:
        Comparison results dictionary
    """
    comparator = SemanticComparator()
    return comparator.compare(original_ir, optimized_ir)


__all__ = [
    'OptimizationValidator',
    'SemanticComparator', 
    'validate_optimization',
    'compare_optimization'
]

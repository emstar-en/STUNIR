#!/usr/bin/env python3
"""STUNIR Optimization Demo.

Demonstrates the optimization pipeline by generating code at different
optimization levels (O0, O1, O2, O3) and showing the differences.
"""

import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tools.optimize import create_pass_manager, compare_optimization


def create_sample_ir():
    """Create sample IR with optimization opportunities."""
    return {
        'ir_module': 'cjson_demo',
        'ir_types': [
            {'name': 'cJSON', 'kind': 'struct', 'fields': [
                {'name': 'type', 'type': 'i32'},
                {'name': 'valueint', 'type': 'i32'},
                {'name': 'valuedouble', 'type': 'f64'},
                {'name': 'valuestring', 'type': 'ptr_char'},
                {'name': 'string', 'type': 'ptr_char'},
                {'name': 'next', 'type': 'ptr_cJSON'},
                {'name': 'prev', 'type': 'ptr_cJSON'},
                {'name': 'child', 'type': 'ptr_cJSON'}
            ]}
        ],
        'ir_globals': [
            {'name': 'cJSON_Invalid', 'value': {'type': 'literal', 'value': 0}},
            {'name': 'cJSON_False', 'value': {'type': 'literal', 'value': 1}},
            {'name': 'cJSON_True', 'value': {'type': 'literal', 'value': 2}},
            {'name': 'cJSON_NULL', 'value': {'type': 'literal', 'value': 4}},
            {'name': 'cJSON_Number', 'value': {'type': 'literal', 'value': 8}},
            {'name': 'cJSON_String', 'value': {'type': 'literal', 'value': 16}},
            {'name': 'cJSON_Array', 'value': {'type': 'literal', 'value': 32}},
            {'name': 'cJSON_Object', 'value': {'type': 'literal', 'value': 64}},
        ],
        'ir_functions': [
            # Function with dead code after return
            {
                'name': 'cJSON_IsInvalid',
                'params': [{'name': 'item', 'type': 'ptr_cJSON'}],
                'return_type': 'bool',
                'body': [
                    {'type': 'if', 'cond': {
                        'type': 'binary', 'op': '==',
                        'left': {'type': 'var', 'name': 'item'},
                        'right': {'type': 'literal', 'value': 0}
                    }, 'then': [
                        {'type': 'return', 'value': {'type': 'literal', 'value': False}},
                        # Dead code after return
                        {'type': 'var_decl', 'var_name': 'dead', 'init': {'type': 'literal', 'value': 999}}
                    ]},
                    {'type': 'return', 'value': {
                        'type': 'binary', 'op': '==',
                        'left': {
                            'type': 'binary', 'op': '&',
                            'left': {'type': 'field_access', 'object': {'type': 'var', 'name': 'item'}, 'field': 'type'},
                            'right': {'type': 'literal', 'value': 255}
                        },
                        'right': {'type': 'literal', 'value': 0}
                    }}
                ]
            },
            # Function with constant folding opportunities
            {
                'name': 'cJSON_GetArraySize',
                'params': [{'name': 'array', 'type': 'ptr_cJSON'}],
                'return_type': 'i32',
                'body': [
                    # Constant folding opportunity: 10 + 20 = 30
                    {'type': 'var_decl', 'var_name': 'base_offset', 'init': {
                        'type': 'binary', 'op': '+',
                        'left': {'type': 'literal', 'value': 10},
                        'right': {'type': 'literal', 'value': 20}
                    }},
                    # Algebraic simplification: count * 1 = count
                    {'type': 'var_decl', 'var_name': 'count', 'init': {'type': 'literal', 'value': 0}},
                    {'type': 'var_decl', 'var_name': 'result', 'init': {
                        'type': 'binary', 'op': '*',
                        'left': {'type': 'var', 'name': 'count'},
                        'right': {'type': 'literal', 'value': 1}
                    }},
                    {'type': 'var_decl', 'var_name': 'child', 'init': {
                        'type': 'field_access', 
                        'object': {'type': 'var', 'name': 'array'}, 
                        'field': 'child'
                    }},
                    {'type': 'while', 'cond': {
                        'type': 'binary', 'op': '!=',
                        'left': {'type': 'var', 'name': 'child'},
                        'right': {'type': 'literal', 'value': 0}
                    }, 'body': [
                        {'type': 'assign', 'target': 'count', 'value': {
                            'type': 'binary', 'op': '+',
                            'left': {'type': 'var', 'name': 'count'},
                            'right': {'type': 'literal', 'value': 1}
                        }},
                        {'type': 'assign', 'target': 'child', 'value': {
                            'type': 'field_access',
                            'object': {'type': 'var', 'name': 'child'},
                            'field': 'next'
                        }}
                    ]},
                    {'type': 'return', 'value': {'type': 'var', 'name': 'count'}}
                ]
            },
            # Function with strength reduction opportunity
            {
                'name': 'cJSON_GetArrayIndex',
                'params': [{'name': 'index', 'type': 'i32'}],
                'return_type': 'i32',
                'body': [
                    # x * 8 can become x << 3
                    {'type': 'var_decl', 'var_name': 'byte_offset', 'init': {
                        'type': 'binary', 'op': '*',
                        'left': {'type': 'var', 'name': 'index'},
                        'right': {'type': 'literal', 'value': 8}
                    }},
                    # x / 4 can become x >> 2
                    {'type': 'var_decl', 'var_name': 'word_count', 'init': {
                        'type': 'binary', 'op': '/',
                        'left': {'type': 'var', 'name': 'byte_offset'},
                        'right': {'type': 'literal', 'value': 4}
                    }},
                    # x % 16 can become x & 15
                    {'type': 'var_decl', 'var_name': 'remainder', 'init': {
                        'type': 'binary', 'op': '%',
                        'left': {'type': 'var', 'name': 'byte_offset'},
                        'right': {'type': 'literal', 'value': 16}
                    }},
                    {'type': 'return', 'value': {'type': 'var', 'name': 'word_count'}}
                ]
            },
            # Small function for inlining
            {
                'name': 'cJSON_malloc',
                'params': [{'name': 'size', 'type': 'size_t'}],
                'return_type': 'ptr_void',
                'inline': True,
                'body': [
                    {'type': 'return', 'value': {
                        'type': 'call',
                        'func': 'malloc',
                        'args': [{'type': 'var', 'name': 'size'}]
                    }}
                ]
            }
        ]
    }


def count_statements(ir):
    """Count total statements in IR."""
    def count_body(body):
        count = len(body)
        for stmt in body:
            if isinstance(stmt, dict):
                for key in ('then', 'else', 'body'):
                    if key in stmt:
                        count += count_body(stmt[key])
        return count
    
    total = 0
    for func in ir.get('ir_functions', []):
        total += count_body(func.get('body', []))
    return total


def format_ir_brief(ir):
    """Format IR in a brief way for display."""
    lines = []
    for func in ir.get('ir_functions', []):
        name = func.get('name', 'unknown')
        body_len = len(func.get('body', []))
        lines.append(f"  {name}(): {body_len} statements")
    return '\n'.join(lines)


def main():
    print("=" * 60)
    print("STUNIR Optimization Pipeline Demo")
    print("=" * 60)
    
    ir = create_sample_ir()
    original_stmt_count = count_statements(ir)
    
    print(f"\nOriginal IR: {original_stmt_count} statements")
    print(f"  Module: {ir.get('ir_module')}")
    print(f"  Functions: {len(ir.get('ir_functions', []))}")
    print("\nFunction details:")
    print(format_ir_brief(ir))
    
    print("\n" + "-" * 60)
    print("Running optimizations at different levels...")
    print("-" * 60)
    
    results = {}
    for level in ['O0', 'O1', 'O2', 'O3']:
        pm = create_pass_manager(level)
        optimized_ir, stats = pm.optimize(ir)
        
        stmt_count = count_statements(optimized_ir)
        results[level] = {
            'ir': optimized_ir,
            'stats': stats,
            'stmt_count': stmt_count
        }
        
        print(f"\n### {level}: {stmt_count} statements")
        print(f"    Passes run: {len(stats['passes'])}")
        if stats['passes']:
            print(f"    Pass names: {', '.join(stats['passes'])}")
        print(f"    Total changes: {stats['total_changes']}")
        
        # Show comparison
        comparison = compare_optimization(ir, optimized_ir)
        print(f"    Statement reduction: {comparison['statement_reduction']} ({comparison['reduction_percent']:.1f}%)")
    
    print("\n" + "=" * 60)
    print("Summary: Optimization Level Comparison")
    print("=" * 60)
    
    print("\n{:<6} {:<12} {:<8} {:<10} {:<20}".format(
        "Level", "Statements", "Changes", "Passes", "Reduction"))
    print("-" * 60)
    
    for level in ['O0', 'O1', 'O2', 'O3']:
        r = results[level]
        reduction = original_stmt_count - r['stmt_count']
        reduction_pct = (reduction / original_stmt_count * 100) if original_stmt_count > 0 else 0
        print("{:<6} {:<12} {:<8} {:<10} {:>5} ({:>5.1f}%)".format(
            level,
            r['stmt_count'],
            r['stats']['total_changes'],
            len(r['stats']['passes']),
            reduction,
            reduction_pct
        ))
    
    print("\n" + "=" * 60)
    print("Optimization Pass Details by Level")
    print("=" * 60)
    
    print("\n### O1 Passes (Basic):")
    print("  - dead_code_elimination: Removes unreachable code")
    print("  - constant_folding: Evaluates constant expressions (10+20=30)")
    print("  - constant_propagation: Propagates known constants")
    print("  - algebraic_simplification: Simplifies x*1=x, x+0=x")
    
    print("\n### O2 Passes (O1 + Standard):")
    print("  - common_subexpression_elimination: Caches repeated expressions")
    print("  - loop_invariant_code_motion: Hoists invariant code from loops")
    print("  - copy_propagation: Eliminates redundant copies")
    print("  - strength_reduction: x*8 -> x<<3, x/4 -> x>>2, x%16 -> x&15")
    
    print("\n### O3 Passes (O2 + Aggressive):")
    print("  - function_inlining: Inlines small functions")
    print("  - loop_unrolling: Unrolls small loops")
    print("  - aggressive_constant_propagation: Interprocedural propagation")
    print("  - dead_store_elimination: Removes unused assignments")
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

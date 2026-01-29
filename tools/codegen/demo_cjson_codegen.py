#!/usr/bin/env python3
"""Demonstration of STUNIR Advanced Code Generation with cJSON Functions.

This script demonstrates generating code in 8 target languages from STUNIR IR
representing cJSON-style JSON manipulation functions.

Usage:
    python demo_cjson_codegen.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tools.codegen import get_generator, get_supported_targets


def get_cjson_create_object_ir():
    """Get IR for cJSON_CreateObject function."""
    return {
        'name': 'cJSON_CreateObject',
        'params': [],
        'return_type': 'cJSON*',
        'docstring': 'Create a new empty JSON object',
        'body': [
            {
                'type': 'var_decl',
                'var_name': 'item',
                'var_type': 'cJSON*',
                'init': {
                    'type': 'call',
                    'func': 'cJSON_New_Item',
                    'args': []
                }
            },
            {
                'type': 'if',
                'condition': {'type': 'var', 'name': 'item'},
                'then': [
                    {
                        'type': 'assign',
                        'target': 'item->type',
                        'value': 'cJSON_Object'
                    }
                ]
            },
            {
                'type': 'return',
                'value': {'type': 'var', 'name': 'item'}
            }
        ]
    }


def get_cjson_get_array_size_ir():
    """Get IR for cJSON_GetArraySize function."""
    return {
        'name': 'cJSON_GetArraySize',
        'params': [{'name': 'array', 'type': 'cJSON*'}],
        'return_type': 'i32',
        'docstring': 'Get the number of items in a JSON array',
        'body': [
            {
                'type': 'var_decl',
                'var_name': 'child',
                'var_type': 'cJSON*',
                'init': {
                    'type': 'ternary',
                    'condition': {'type': 'var', 'name': 'array'},
                    'then': {'type': 'member', 'base': {'type': 'var', 'name': 'array'}, 'member': 'child'},
                    'else': None
                }
            },
            {
                'type': 'var_decl',
                'var_name': 'size',
                'var_type': 'i32',
                'init': 0
            },
            {
                'type': 'while',
                'condition': {'type': 'var', 'name': 'child'},
                'body': [
                    {
                        'type': 'assign',
                        'target': 'size',
                        'op': '+=',
                        'value': 1
                    },
                    {
                        'type': 'assign',
                        'target': 'child',
                        'value': {'type': 'member', 'base': {'type': 'var', 'name': 'child'}, 'member': 'next'}
                    }
                ]
            },
            {
                'type': 'return',
                'value': {'type': 'var', 'name': 'size'}
            }
        ]
    }


def get_cjson_parse_string_ir():
    """Get IR for a simplified cJSON_Parse function."""
    return {
        'name': 'cJSON_Parse',
        'params': [{'name': 'value', 'type': 'string'}],
        'return_type': 'cJSON*',
        'docstring': 'Parse a JSON string and return a cJSON object tree',
        'body': [
            {
                'type': 'if',
                'condition': {
                    'type': 'binary', 'op': '==',
                    'left': {'type': 'var', 'name': 'value'},
                    'right': None
                },
                'then': [
                    {'type': 'return', 'value': None}
                ]
            },
            {
                'type': 'var_decl',
                'var_name': 'parser',
                'var_type': 'ParseBuffer',
                'init': {
                    'type': 'struct',
                    'struct_type': 'ParseBuffer',
                    'fields': {
                        'content': {'type': 'var', 'name': 'value'},
                        'position': 0,
                        'length': {
                            'type': 'call',
                            'func': 'strlen',
                            'args': [{'type': 'var', 'name': 'value'}]
                        }
                    }
                }
            },
            {
                'type': 'return',
                'value': {
                    'type': 'call',
                    'func': 'parse_value',
                    'args': [
                        {'type': 'ref', 'target': {'type': 'var', 'name': 'parser'}}
                    ]
                }
            }
        ]
    }


def get_cjson_print_value_ir():
    """Get IR for a cJSON_PrintValue-style function with switch."""
    return {
        'name': 'cJSON_PrintValue',
        'params': [{'name': 'item', 'type': 'cJSON*'}],
        'return_type': 'string',
        'docstring': 'Convert a cJSON item to a string representation',
        'body': [
            {
                'type': 'if',
                'condition': {
                    'type': 'binary', 'op': '==',
                    'left': {'type': 'var', 'name': 'item'},
                    'right': None
                },
                'then': [
                    {'type': 'return', 'value': ''}
                ]
            },
            {
                'type': 'switch',
                'value': {'type': 'member', 'base': {'type': 'var', 'name': 'item'}, 'member': 'type'},
                'cases': [
                    {
                        'value': 'cJSON_NULL',
                        'body': [{'type': 'return', 'value': 'null'}]
                    },
                    {
                        'value': 'cJSON_True',
                        'body': [{'type': 'return', 'value': 'true'}]
                    },
                    {
                        'value': 'cJSON_False',
                        'body': [{'type': 'return', 'value': 'false'}]
                    },
                    {
                        'value': 'cJSON_Number',
                        'body': [
                            {
                                'type': 'return',
                                'value': {
                                    'type': 'call',
                                    'func': 'numberToString',
                                    'args': [{'type': 'member', 'base': {'type': 'var', 'name': 'item'}, 'member': 'valuedouble'}]
                                }
                            }
                        ]
                    },
                    {
                        'value': 'cJSON_String',
                        'body': [
                            {
                                'type': 'return',
                                'value': {
                                    'type': 'call',
                                    'func': 'escapeString',
                                    'args': [{'type': 'member', 'base': {'type': 'var', 'name': 'item'}, 'member': 'valuestring'}]
                                }
                            }
                        ]
                    }
                ],
                'default': [
                    {'type': 'return', 'value': ''}
                ]
            }
        ]
    }


def generate_demo():
    """Generate and display code in all target languages."""
    
    # Define the IR for demonstration functions
    functions_ir = [
        get_cjson_create_object_ir(),
        get_cjson_get_array_size_ir(),
        get_cjson_print_value_ir(),
    ]
    
    print("=" * 80)
    print("STUNIR Advanced Code Generation Demo - cJSON Functions")
    print("=" * 80)
    print()
    
    targets = get_supported_targets()
    
    for target in targets:
        print(f"\n{'='*80}")
        print(f"Target Language: {target.upper()}")
        print('='*80)
        
        gen = get_generator(target)
        
        for func_ir in functions_ir:
            print(f"\n--- {func_ir['name']} ---")
            print()
            code = gen.generate_function(func_ir)
            print(code)
            print()
    
    # Also generate a complete module example
    print("\n" + "="*80)
    print("Module Generation Example (Python)")
    print("="*80)
    
    module_ir = {
        'name': 'cjson_helpers',
        'docstring': 'Helper functions for JSON manipulation',
        'imports': [],
        'functions': functions_ir[:2],  # Just first two functions
        'exports': ['cJSON_CreateObject', 'cJSON_GetArraySize']
    }
    
    py_gen = get_generator('python')
    module_code = py_gen.generate_module(module_ir)
    print(module_code)


if __name__ == '__main__':
    generate_demo()

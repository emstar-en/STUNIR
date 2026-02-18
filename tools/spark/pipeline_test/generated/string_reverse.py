#!/usr/bin/env python3
# Auto-generated from spec: string_reverse
# Reverses a string
import sys

def reverse_string(text):
    """Reverses a string"""
    return text[::-1]

def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--describe':
        print('{"tool": "string_reverse", "description": "Reverses a string"}')
        return
    
    if len(sys.argv) < 2:
        print('Usage: string_reverse [--describe] <text>', file=sys.stderr)
        sys.exit(1)
    
    text = sys.argv[1]
    result = reverse_string(text)
    print(result)

if __name__ == '__main__':
    main()

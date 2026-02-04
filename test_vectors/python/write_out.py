#!/usr/bin/env python3
import sys

def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'out.txt'
    with open(out, 'w', encoding='utf-8', newline='\n') as f:
        f.write('hello\n')

if __name__ == '__main__':
    main()

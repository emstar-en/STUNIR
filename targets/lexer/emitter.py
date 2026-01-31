#!/usr/bin/env python3
"""STUNIR Lexer Emitter - Generate lexer code from IR specifications.

This tool is part of the targets → lexer pipeline stage.
It converts STUNIR IR to lexer implementations in multiple languages.

Usage:
    emitter.py <ir.json> --output=<dir> [--target=python|rust|c]
    emitter.py --help
"""

import json
import hashlib
import time
import sys
from pathlib import Path


def canonical_json(data):
    """Generate RFC 8785 / JCS subset canonical JSON."""
    return json.dumps(data, sort_keys=True, separators=(',', ':'))


def compute_sha256(content):
    """Compute SHA256 hash of content."""
    if isinstance(content, str):
        content = content.encode('utf-8')
    return hashlib.sha256(content).hexdigest()


class LexerEmitter:
    """Emitter for lexer code generation."""
    
    def __init__(self, ir_data, out_dir, options=None):
        """Initialize lexer emitter."""
        self.ir_data = ir_data
        self.out_dir = Path(out_dir)
        self.options = options or {}
        self.target = options.get('target', 'python') if options else 'python'
        self.generated_files = []
        self.epoch = int(time.time())
    
    def _write_file(self, path, content):
        """Write content to file."""
        full_path = self.out_dir / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding='utf-8', newline='\n')
        self.generated_files.append({
            'path': str(path),
            'sha256': compute_sha256(content),
            'size': len(content.encode('utf-8'))
        })
        return full_path
    
    def emit_python_lexer(self):
        """Generate Python lexer from IR."""
        name = self.ir_data.get('name', 'Lexer')
        tokens = self.ir_data.get('tokens', [])
        
        # Generate token definitions
        token_patterns = []
        for token in tokens:
            token_name = token.get('name', 'UNKNOWN')
            pattern = token.get('pattern', '')
            skip = token.get('skip', False)
            token_patterns.append({
                'name': token_name,
                'pattern': pattern,
                'skip': skip
            })
        
        # Generate Python lexer code
        code = f'''#!/usr/bin/env python3
"""STUNIR Generated Lexer: {name}

Generated from STUNIR IR specification.
DO NOT EDIT - Generated code.
"""

import re
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Token:
    """Token representation."""
    type: str
    value: str
    line: int
    column: int


class {name}:
    """Lexer implementation for {name}."""
    
    # Token patterns (name, regex, skip)
    TOKEN_PATTERNS = [
'''
        
        for token in token_patterns:
            skip_str = 'True' if token['skip'] else 'False'
            code += f"        ('{token['name']}', r'{token['pattern']}', {skip_str}),\n"
        
        code += '''    ]
    
    def __init__(self, text: str):
        """Initialize lexer with input text."""
        self.text = text
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Token] = []
    
    def tokenize(self) -> List[Token]:
        """Tokenize the input text."""
        while self.pos < len(self.text):
            matched = False
            
            for token_name, pattern, skip in self.TOKEN_PATTERNS:
                regex = re.compile(pattern)
                match = regex.match(self.text, self.pos)
                
                if match:
                    value = match.group(0)
                    
                    if not skip:
                        token = Token(
                            type=token_name,
                            value=value,
                            line=self.line,
                            column=self.column
                        )
                        self.tokens.append(token)
                    
                    # Update position
                    self.pos = match.end()
                    
                    # Update line/column tracking
                    for char in value:
                        if char == '\\n':
                            self.line += 1
                            self.column = 1
                        else:
                            self.column += 1
                    
                    matched = True
                    break
            
            if not matched:
                raise SyntaxError(
                    f"Unexpected character at line {self.line}, "
                    f"column {self.column}: {self.text[self.pos]}"
                )
        
        return self.tokens


def main():
    """Main entry point for generated lexer."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: lexer.py <input_file>")
        sys.exit(1)
    
    with open(sys.argv[1], 'r') as f:
        text = f.read()
    
    lexer = ''' + name + '''(text)
    tokens = lexer.tokenize()
    
    for token in tokens:
        print(f"{token.type:15s} {token.value!r:20s} "
              f"(line {token.line}, col {token.column})")


if __name__ == '__main__':
    main()
'''
        
        return self._write_file('lexer.py', code)
    
    def emit_rust_lexer(self):
        """Generate Rust lexer from IR."""
        name = self.ir_data.get('name', 'Lexer')
        tokens = self.ir_data.get('tokens', [])
        
        # Generate token enum
        token_variants = [token.get('name', 'Unknown') for token in tokens]
        
        code = f'''// STUNIR Generated Lexer: {name}
// Generated from STUNIR IR specification.
// DO NOT EDIT - Generated code.

use std::fmt;

/// Token types for {name}
#[derive(Debug, Clone, PartialEq)]
pub enum TokenType {{
'''
        
        for variant in token_variants:
            code += f'    {variant},\n'
        
        code += '''    Eof,
}

/// Token representation
#[derive(Debug, Clone)]
pub struct Token {
    pub token_type: TokenType,
    pub value: String,
    pub line: usize,
    pub column: usize,
}

impl Token {
    pub fn new(token_type: TokenType, value: String, line: usize, column: usize) -> Self {
        Token {
            token_type,
            value,
            line,
            column,
        }
    }
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?} '{}' (line {}, col {})", 
               self.token_type, self.value, self.line, self.column)
    }
}

/// ''' + name + ''' lexer implementation
pub struct ''' + name + ''' {
    text: String,
    pos: usize,
    line: usize,
    column: usize,
    tokens: Vec<Token>,
}

impl ''' + name + ''' {
    pub fn new(text: String) -> Self {
        ''' + name + ''' {
            text,
            pos: 0,
            line: 1,
            column: 1,
            tokens: Vec::new(),
        }
    }
    
    pub fn tokenize(&mut self) -> Result<Vec<Token>, String> {
        while self.pos < self.text.len() {
            if !self.match_token()? {
                return Err(format!(
                    "Unexpected character at line {}, column {}: {:?}",
                    self.line, self.column,
                    self.text.chars().nth(self.pos).unwrap()
                ));
            }
        }
        
        self.tokens.push(Token::new(
            TokenType::Eof,
            String::new(),
            self.line,
            self.column
        ));
        
        Ok(self.tokens.clone())
    }
    
    fn match_token(&mut self) -> Result<bool, String> {
        // Token matching logic would go here
        // This is a placeholder implementation
        self.pos += 1;
        self.column += 1;
        Ok(true)
    }
}
'''
        
        return self._write_file('lexer.rs', code)
    
    def emit_c_lexer(self):
        """Generate C lexer from IR."""
        name = self.ir_data.get('name', 'Lexer')
        tokens = self.ir_data.get('tokens', [])
        
        # Generate header
        header = f'''/* STUNIR Generated Lexer: {name}
 * Generated from STUNIR IR specification.
 * DO NOT EDIT - Generated code.
 */

#ifndef {name.upper()}_H
#define {name.upper()}_H

#include <stdint.h>
#include <stdbool.h>

/* Token types */
typedef enum {{
'''
        
        for i, token in enumerate(tokens):
            token_name = token.get('name', 'UNKNOWN')
            header += f'    TOKEN_{token_name.upper()} = {i},\n'
        
        header += '''    TOKEN_EOF
} TokenType;

/* Token structure */
typedef struct {
    TokenType type;
    const char *value;
    uint32_t line;
    uint32_t column;
} Token;

/* Lexer structure */
typedef struct {
    const char *text;
    uint32_t pos;
    uint32_t line;
    uint32_t column;
    Token *tokens;
    uint32_t token_count;
    uint32_t token_capacity;
} Lexer;

/* Function declarations */
void lexer_init(Lexer *lexer, const char *text);
int lexer_tokenize(Lexer *lexer);
void lexer_free(Lexer *lexer);
const char *token_type_name(TokenType type);

#endif /* ''' + name.upper() + '''_H */
'''
        
        # Generate source
        source = f'''/* STUNIR Generated Lexer: {name}
 * Generated from STUNIR IR specification.
 * DO NOT EDIT - Generated code.
 */

#include "lexer.h"
#include <stdlib.h>
#include <string.h>

void lexer_init(Lexer *lexer, const char *text) {{
    lexer->text = text;
    lexer->pos = 0;
    lexer->line = 1;
    lexer->column = 1;
    lexer->token_count = 0;
    lexer->token_capacity = 128;
    lexer->tokens = malloc(lexer->token_capacity * sizeof(Token));
}}

int lexer_tokenize(Lexer *lexer) {{
    /* Token matching logic would go here */
    /* This is a placeholder implementation */
    return 0;
}}

void lexer_free(Lexer *lexer) {{
    if (lexer->tokens) {{
        free(lexer->tokens);
        lexer->tokens = NULL;
    }}
}}

const char *token_type_name(TokenType type) {{
    switch (type) {{
'''
        
        for i, token in enumerate(tokens):
            token_name = token.get('name', 'UNKNOWN')
            source += f'        case TOKEN_{token_name.upper()}: return "{token_name}";\n'
        
        source += '''        case TOKEN_EOF: return "EOF";
        default: return "UNKNOWN";
    }
}
'''
        
        self._write_file('lexer.h', header)
        return self._write_file('lexer.c', source)
    
    def emit(self):
        """Emit lexer code based on target."""
        if self.target == 'python':
            self.emit_python_lexer()
        elif self.target == 'rust':
            self.emit_rust_lexer()
        elif self.target == 'c':
            self.emit_c_lexer()
        else:
            raise ValueError(f"Unknown target: {self.target}")
        
        # Generate manifest
        manifest = {
            'timestamp': self.epoch,
            'target': self.target,
            'ir_hash': compute_sha256(canonical_json(self.ir_data)),
            'files': self.generated_files
        }
        
        manifest_path = self.out_dir / 'manifest.json'
        manifest_path.write_text(
            canonical_json(manifest) + '\n',
            encoding='utf-8',
            newline='\n'
        )
        
        return manifest


def main():
    """Main entry point."""
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    
    ir_path = sys.argv[1]
    out_dir = None
    target = 'python'
    
    # Parse arguments
    for arg in sys.argv[2:]:
        if arg.startswith('--output='):
            out_dir = arg.split('=', 1)[1]
        elif arg.startswith('--target='):
            target = arg.split('=', 1)[1]
        elif arg in ('-h', '--help'):
            print(__doc__)
            sys.exit(0)
    
    if not out_dir:
        print("Error: --output=<dir> required")
        sys.exit(1)
    
    # Load IR
    with open(ir_path, 'r') as f:
        ir_data = json.load(f)
    
    # Emit
    emitter = LexerEmitter(ir_data, out_dir, {'target': target})
    manifest = emitter.emit()
    
    print(f"Generated {len(manifest['files'])} files in {out_dir}")
    print(f"Manifest: {out_dir}/manifest.json")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

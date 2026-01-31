#!/usr/bin/env python3
"""STUNIR Parser Emitter - Generate parser code from IR specifications.

This tool is part of the targets → parser pipeline stage.
It converts STUNIR IR to parser implementations in multiple languages.

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


class ParserEmitter:
    """Emitter for parser code generation."""
    
    def __init__(self, ir_data, out_dir, options=None):
        """Initialize parser emitter."""
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
    
    def emit_python_parser(self):
        """Generate Python parser from IR."""
        name = self.ir_data.get('name', 'Parser')
        grammar_rules = self.ir_data.get('rules', [])
        
        code = f'''#!/usr/bin/env python3
"""STUNIR Generated Parser: {name}

Generated from STUNIR IR specification.
DO NOT EDIT - Generated code.
"""

from typing import List, Optional, Any
from dataclasses import dataclass


@dataclass
class ASTNode:
    """Abstract Syntax Tree node."""
    node_type: str
    value: Any
    children: List['ASTNode']
    line: int = 0
    column: int = 0


class ParseError(Exception):
    """Parser error exception."""
    pass


class {name}:
    """Parser implementation for {name}."""
    
    def __init__(self, tokens: List[Any]):
        """Initialize parser with token stream."""
        self.tokens = tokens
        self.pos = 0
        self.current_token = tokens[0] if tokens else None
    
    def advance(self):
        """Move to next token."""
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
            self.current_token = self.tokens[self.pos]
        else:
            self.current_token = None
    
    def expect(self, token_type: str) -> Any:
        """Expect a specific token type."""
        if self.current_token is None:
            raise ParseError(f"Unexpected end of input, expected {{token_type}}")
        
        if hasattr(self.current_token, 'type'):
            actual_type = self.current_token.type
        else:
            actual_type = str(self.current_token)
        
        if actual_type != token_type:
            raise ParseError(
                f"Expected {{token_type}}, got {{actual_type}} "
                f"at line {{getattr(self.current_token, 'line', 0)}}"
            )
        
        token = self.current_token
        self.advance()
        return token
    
    def peek(self, offset: int = 0) -> Optional[Any]:
        """Peek ahead at tokens."""
        pos = self.pos + offset
        if 0 <= pos < len(self.tokens):
            return self.tokens[pos]
        return None
    
    def parse(self) -> ASTNode:
        """Parse token stream into AST."""
        return self.parse_program()
    
    def parse_program(self) -> ASTNode:
        """Parse program (top-level production)."""
        children = []
        
        while self.current_token is not None:
            stmt = self.parse_statement()
            if stmt:
                children.append(stmt)
        
        return ASTNode(
            node_type='Program',
            value=None,
            children=children
        )
    
    def parse_statement(self) -> Optional[ASTNode]:
        """Parse a statement."""
        if self.current_token is None:
            return None
        
        # Placeholder implementation
        # Real parser would have grammar-specific logic
        token = self.current_token
        self.advance()
        
        return ASTNode(
            node_type='Statement',
            value=token,
            children=[],
            line=getattr(token, 'line', 0),
            column=getattr(token, 'column', 0)
        )
    
    def parse_expression(self) -> ASTNode:
        """Parse an expression."""
        return self.parse_primary()
    
    def parse_primary(self) -> ASTNode:
        """Parse a primary expression."""
        token = self.current_token
        self.advance()
        
        return ASTNode(
            node_type='Primary',
            value=token,
            children=[],
            line=getattr(token, 'line', 0),
            column=getattr(token, 'column', 0)
        )


def print_ast(node: ASTNode, indent: int = 0):
    """Pretty-print AST."""
    prefix = "  " * indent
    print(f"{{prefix}}{{node.node_type}}: {{node.value}}")
    for child in node.children:
        print_ast(child, indent + 1)


def main():
    """Main entry point for generated parser."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: parser.py <input_file>")
        sys.exit(1)
    
    # This is a placeholder - real implementation would need lexer
    print("Parser generated successfully.")
    print("To use: import parser module and call {name}(tokens).parse()")


if __name__ == '__main__':
    main()
'''
        
        return self._write_file('parser.py', code)
    
    def emit_rust_parser(self):
        """Generate Rust parser from IR."""
        name = self.ir_data.get('name', 'Parser')
        
        code = f'''// STUNIR Generated Parser: {name}
// Generated from STUNIR IR specification.
// DO NOT EDIT - Generated code.

use std::fmt;

/// AST Node types
#[derive(Debug, Clone, PartialEq)]
pub enum ASTNodeType {{
    Program,
    Statement,
    Expression,
    Primary,
}}

/// Abstract Syntax Tree node
#[derive(Debug, Clone)]
pub struct ASTNode {{
    pub node_type: ASTNodeType,
    pub value: Option<String>,
    pub children: Vec<ASTNode>,
    pub line: usize,
    pub column: usize,
}}

impl ASTNode {{
    pub fn new(node_type: ASTNodeType, value: Option<String>) -> Self {{
        ASTNode {{
            node_type,
            value,
            children: Vec::new(),
            line: 0,
            column: 0,
        }}
    }}
    
    pub fn add_child(&mut self, child: ASTNode) {{
        self.children.push(child);
    }}
}}

/// Parser error type
#[derive(Debug)]
pub struct ParseError {{
    pub message: String,
    pub line: usize,
    pub column: usize,
}}

impl fmt::Display for ParseError {{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {{
        write!(f, "Parse error at line {{}}, column {{}}: {{}}",
               self.line, self.column, self.message)
    }}
}}

impl std::error::Error for ParseError {{}}

/// {name} parser implementation
pub struct {name}<T> {{
    tokens: Vec<T>,
    pos: usize,
}}

impl<T> {name}<T> {{
    pub fn new(tokens: Vec<T>) -> Self {{
        {name} {{
            tokens,
            pos: 0,
        }}
    }}
    
    fn current(&self) -> Option<&T> {{
        if self.pos < self.tokens.len() {{
            Some(&self.tokens[self.pos])
        }} else {{
            None
        }}
    }}
    
    fn advance(&mut self) {{
        if self.pos < self.tokens.len() {{
            self.pos += 1;
        }}
    }}
    
    fn peek(&self, offset: usize) -> Option<&T> {{
        let pos = self.pos + offset;
        if pos < self.tokens.len() {{
            Some(&self.tokens[pos])
        }} else {{
            None
        }}
    }}
    
    pub fn parse(&mut self) -> Result<ASTNode, ParseError> {{
        self.parse_program()
    }}
    
    fn parse_program(&mut self) -> Result<ASTNode, ParseError> {{
        let mut program = ASTNode::new(ASTNodeType::Program, None);
        
        while self.current().is_some() {{
            match self.parse_statement() {{
                Ok(stmt) => program.add_child(stmt),
                Err(e) => return Err(e),
            }}
        }}
        
        Ok(program)
    }}
    
    fn parse_statement(&mut self) -> Result<ASTNode, ParseError> {{
        // Placeholder implementation
        self.advance();
        Ok(ASTNode::new(ASTNodeType::Statement, None))
    }}
}}
'''
        
        return self._write_file('parser.rs', code)
    
    def emit_c_parser(self):
        """Generate C parser from IR."""
        name = self.ir_data.get('name', 'Parser')
        
        # Generate header
        header = f'''/* STUNIR Generated Parser: {name}
 * Generated from STUNIR IR specification.
 * DO NOT EDIT - Generated code.
 */

#ifndef {name.upper()}_PARSER_H
#define {name.upper()}_PARSER_H

#include <stdint.h>
#include <stdbool.h>

/* AST Node types */
typedef enum {{
    NODE_PROGRAM,
    NODE_STATEMENT,
    NODE_EXPRESSION,
    NODE_PRIMARY,
}} ASTNodeType;

/* Forward declaration */
struct ASTNode;

/* AST Node structure */
typedef struct ASTNode {{
    ASTNodeType type;
    const char *value;
    struct ASTNode **children;
    uint32_t child_count;
    uint32_t child_capacity;
    uint32_t line;
    uint32_t column;
}} ASTNode;

/* Parser structure */
typedef struct {{
    void *tokens;  /* Token array */
    uint32_t token_count;
    uint32_t pos;
}} Parser;

/* Function declarations */
void parser_init(Parser *parser, void *tokens, uint32_t token_count);
ASTNode *parser_parse(Parser *parser);
void ast_node_free(ASTNode *node);
ASTNode *ast_node_new(ASTNodeType type, const char *value);
void ast_node_add_child(ASTNode *parent, ASTNode *child);

#endif /* {name.upper()}_PARSER_H */
'''
        
        # Generate source
        source = f'''/* STUNIR Generated Parser: {name}
 * Generated from STUNIR IR specification.
 * DO NOT EDIT - Generated code.
 */

#include "parser.h"
#include <stdlib.h>
#include <string.h>

void parser_init(Parser *parser, void *tokens, uint32_t token_count) {{
    parser->tokens = tokens;
    parser->token_count = token_count;
    parser->pos = 0;
}}

ASTNode *ast_node_new(ASTNodeType type, const char *value) {{
    ASTNode *node = malloc(sizeof(ASTNode));
    if (!node) return NULL;
    
    node->type = type;
    node->value = value ? strdup(value) : NULL;
    node->children = NULL;
    node->child_count = 0;
    node->child_capacity = 0;
    node->line = 0;
    node->column = 0;
    
    return node;
}}

void ast_node_add_child(ASTNode *parent, ASTNode *child) {{
    if (!parent || !child) return;
    
    if (parent->child_count >= parent->child_capacity) {{
        uint32_t new_capacity = parent->child_capacity == 0 ? 4 : parent->child_capacity * 2;
        ASTNode **new_children = realloc(parent->children, new_capacity * sizeof(ASTNode *));
        if (!new_children) return;
        
        parent->children = new_children;
        parent->child_capacity = new_capacity;
    }}
    
    parent->children[parent->child_count++] = child;
}}

void ast_node_free(ASTNode *node) {{
    if (!node) return;
    
    if (node->value) {{
        free((void *)node->value);
    }}
    
    for (uint32_t i = 0; i < node->child_count; i++) {{
        ast_node_free(node->children[i]);
    }}
    
    if (node->children) {{
        free(node->children);
    }}
    
    free(node);
}}

ASTNode *parser_parse(Parser *parser) {{
    /* Placeholder implementation */
    return ast_node_new(NODE_PROGRAM, NULL);
}}
'''
        
        self._write_file('parser.h', header)
        return self._write_file('parser.c', source)
    
    def emit(self):
        """Emit parser code based on target."""
        if self.target == 'python':
            self.emit_python_parser()
        elif self.target == 'rust':
            self.emit_rust_parser()
        elif self.target == 'c':
            self.emit_c_parser()
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
    emitter = ParserEmitter(ir_data, out_dir, {'target': target})
    manifest = emitter.emit()
    
    print(f"Generated {len(manifest['files'])} files in {out_dir}")
    print(f"Manifest: {out_dir}/manifest.json")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

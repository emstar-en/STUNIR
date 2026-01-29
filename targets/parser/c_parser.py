#!/usr/bin/env python3
"""C parser code emitter.

Generates C parser code from parse tables.
"""

from typing import Dict, List, Optional, Any, Union

from targets.parser.base import ParserEmitterBase, ParserEmitterResult

# Import from parser module
try:
    from ir.parser.parse_table import ParseTable, LL1Table, ParserType, Action
    from ir.parser.ast_node import ASTSchema, ASTNodeSpec
    from ir.parser.parser_generator import ParserGeneratorResult
    from ir.grammar.grammar_ir import Grammar
except ImportError:
    ParseTable = Any
    LL1Table = Any
    ParserType = Any
    ASTSchema = Any
    ParserGeneratorResult = Any
    Grammar = Any


class CParserEmitter(ParserEmitterBase):
    """C parser code emitter.
    
    Generates C89/C99 compatible parser code.
    """
    
    LANGUAGE = "c"
    FILE_EXTENSION = ".c"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the C emitter.
        
        Args:
            config: Configuration dictionary
                - c_standard: "c89" or "c99" (default: "c99")
        """
        super().__init__(config)
        self.c_standard = self.config.get('c_standard', 'c99')
    
    def emit(self, result: ParserGeneratorResult, 
             grammar: Grammar) -> ParserEmitterResult:
        """Emit C parser code.
        
        Args:
            result: Parser generator result
            grammar: Source grammar
        
        Returns:
            ParserEmitterResult with C code
        """
        # Generate header file
        header_code = self._emit_header_file(result, grammar)
        
        # Generate implementation file
        impl_parts = [
            self._emit_file_header(grammar),
            self._emit_includes(),
            "",
            self.emit_parse_table(result.parse_table),
            "",
            self._emit_parser_impl(result, grammar),
        ]
        code = "\n".join(impl_parts)
        
        # Generate AST code
        ast_code = ""
        if result.ast_schema:
            ast_code = self.emit_ast_nodes(result.ast_schema)
        
        # Generate Makefile
        makefile = self._emit_makefile(grammar)
        
        # Generate manifest
        manifest = self._generate_manifest(code, ast_code, grammar, {
            "parser.h": header_code,
            "Makefile": makefile
        })
        
        emit_result = ParserEmitterResult(
            code=code,
            ast_code=ast_code,
            manifest=manifest,
            warnings=self._get_warnings()
        )
        emit_result.add_auxiliary_file("parser.h", header_code)
        emit_result.add_auxiliary_file("Makefile", makefile)
        
        return emit_result
    
    def _emit_file_header(self, grammar: Grammar) -> str:
        """Generate file header comment."""
        import datetime
        name = grammar.name if hasattr(grammar, 'name') else 'unknown'
        return f'''/*
 * Parser for: {name}
 * Generated: {datetime.datetime.now().isoformat()}
 * Generator: STUNIR Parser Emitter
 * Standard: {self.c_standard.upper()}
 */
'''
    
    def _emit_includes(self) -> str:
        """Generate include statements."""
        return '''#include "parser.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>'''
    
    def _emit_header_file(self, result: ParserGeneratorResult, grammar: Grammar) -> str:
        """Generate header file."""
        name = grammar.name if hasattr(grammar, 'name') else 'parser'
        guard = f"{name.upper().replace(' ', '_')}_PARSER_H"
        
        lines = [
            f"#ifndef {guard}",
            f"#define {guard}",
            "",
            "#include <stddef.h>",
            "",
        ]
        
        # Token types enum
        lines.append("/* Token types */")
        lines.append("typedef enum {")
        
        terminals = set()
        if isinstance(result.parse_table, ParseTable):
            for (_, sym) in result.parse_table.action.keys():
                if hasattr(sym, 'name'):
                    terminals.add(sym.name)
        
        for i, term in enumerate(sorted(terminals)):
            safe_name = self._c_ident(term)
            lines.append(f"    TOK_{safe_name} = {i},")
        
        lines.append(f"    TOK_EOF = {len(terminals)},")
        lines.append(f"    TOK_ERROR = {len(terminals) + 1}")
        lines.append("} TokenType;")
        lines.append("")
        
        # Token struct
        lines.append("/* Token structure */")
        lines.append("typedef struct {")
        lines.append("    TokenType type;")
        lines.append("    char* value;")
        lines.append("    int line;")
        lines.append("    int column;")
        lines.append("} Token;")
        lines.append("")
        
        # Parser struct
        lines.append("/* Parser structure */")
        lines.append("typedef struct {")
        lines.append("    Token* tokens;")
        lines.append("    size_t token_count;")
        lines.append("    size_t pos;")
        lines.append("    int* stack;")
        lines.append("    size_t stack_size;")
        lines.append("    size_t stack_capacity;")
        lines.append("} Parser;")
        lines.append("")
        
        # Function declarations
        lines.append("/* Parser functions */")
        lines.append("Parser* parser_create(Token* tokens, size_t count);")
        lines.append("void parser_destroy(Parser* parser);")
        lines.append("int parser_parse(Parser* parser);")
        lines.append("")
        
        lines.append(f"#endif /* {guard} */")
        
        return "\n".join(lines)
    
    def _emit_makefile(self, grammar: Grammar) -> str:
        """Generate Makefile."""
        name = grammar.name if hasattr(grammar, 'name') else 'parser'
        target = name.lower().replace(' ', '_')
        
        std_flag = "-ansi" if self.c_standard == "c89" else f"-std={self.c_standard}"
        
        return f'''# Makefile for {name} parser
# Generated by STUNIR

CC = gcc
CFLAGS = -Wall -Wextra {std_flag} -O2
TARGET = {target}_parser

SRCS = parser.c
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
\t$(CC) $(CFLAGS) -o $@ $^

%.o: %.c parser.h
\t$(CC) $(CFLAGS) -c $<

clean:
\trm -f $(OBJS) $(TARGET)

.PHONY: all clean
'''
    
    def _c_ident(self, name: str) -> str:
        """Convert name to valid C identifier."""
        replacements = {
            '+': 'PLUS', '-': 'MINUS', '*': 'STAR', '/': 'SLASH',
            '<': 'LT', '>': 'GT', '=': 'EQ', '!': 'BANG',
            '&': 'AND', '|': 'OR', '^': 'XOR', '%': 'MOD',
            '(': 'LPAREN', ')': 'RPAREN', '[': 'LBRACKET', ']': 'RBRACKET',
            '{': 'LBRACE', '}': 'RBRACE', ',': 'COMMA', '.': 'DOT',
            ';': 'SEMI', ':': 'COLON', '?': 'QUESTION', '@': 'AT',
            '#': 'HASH', '$': 'DOLLAR', '~': 'TILDE',
        }
        
        if name in replacements:
            return replacements[name]
        
        result = []
        for c in name:
            if c.isalnum():
                result.append(c.upper())
            elif c in replacements:
                result.append('_' + replacements[c])
            elif c == '_':
                result.append(c)
            else:
                result.append('_')
        
        ident = ''.join(result)
        
        # Ensure starts with letter or underscore
        if ident and ident[0].isdigit():
            ident = '_' + ident
        
        return ident
    
    def emit_parse_table(self, table: Union[ParseTable, LL1Table]) -> str:
        """Emit parse table as C data structures.
        
        Args:
            table: Parse table to emit
        
        Returns:
            C code defining the parse tables
        """
        lines = ["/* Parse Tables */"]
        
        if isinstance(table, ParseTable):
            lines.extend(self._emit_lr_table(table))
        else:
            lines.extend(self._emit_ll_table(table))
        
        return "\n".join(lines)
    
    def _emit_lr_table(self, table: ParseTable) -> List[str]:
        """Emit LR parse table."""
        lines = []
        
        # Action types
        lines.append("typedef enum { ACT_SHIFT, ACT_REDUCE, ACT_ACCEPT, ACT_ERROR } ActionType;")
        lines.append("")
        lines.append("typedef struct { ActionType type; int value; } Action;")
        lines.append("")
        
        # Production struct
        lines.append("typedef struct { const char* head; int body_len; } Production;")
        lines.append("")
        
        # Productions array
        lines.append(f"static const Production productions[{len(table.productions)}] = {{")
        for prod in table.productions:
            head = prod.head.name if hasattr(prod.head, 'name') else str(prod.head)
            body_len = len(prod.body) if prod.body else 0
            lines.append(f'    {{ "{head}", {body_len} }},')
        lines.append("};")
        lines.append("")
        
        # Determine table dimensions
        num_states = table.state_count()
        terminals = sorted(set(sym for (_, sym) in table.action.keys()), 
                          key=lambda s: s.name if hasattr(s, 'name') else str(s))
        
        # ACTION table as 2D array
        lines.append(f"/* ACTION table: {num_states} states x {len(terminals)} terminals */")
        lines.append(f"static const Action action_table[{num_states}][{len(terminals) + 1}] = {{")
        
        for state in range(num_states):
            row = []
            for sym in terminals:
                action = table.get_action(state, sym)
                if action is None:
                    row.append("{ ACT_ERROR, 0 }")
                elif action.is_shift():
                    row.append(f"{{ ACT_SHIFT, {action.value} }}")
                elif action.is_reduce():
                    row.append(f"{{ ACT_REDUCE, {action.value} }}")
                elif action.is_accept():
                    row.append("{ ACT_ACCEPT, 0 }")
                else:
                    row.append("{ ACT_ERROR, 0 }")
            
            # Add EOF column
            eof_action = table.get_action(state, table.productions[0].body[0] if table.productions else None)
            # Actually get EOF action
            for (s, sym) in table.action.keys():
                if s == state and hasattr(sym, 'name') and sym.name == '$':
                    eof_action = table.get_action(state, sym)
                    break
            
            if eof_action and eof_action.is_accept():
                row.append("{ ACT_ACCEPT, 0 }")
            else:
                row.append("{ ACT_ERROR, 0 }")
            
            lines.append(f"    {{ {', '.join(row)} }},")
        
        lines.append("};")
        lines.append("")
        
        # GOTO table (simplified - would need proper nonterminal indexing)
        nonterminals = sorted(set(sym for (_, sym) in table.goto.keys()),
                             key=lambda s: s.name if hasattr(s, 'name') else str(s))
        
        lines.append(f"/* GOTO table: {num_states} states x {len(nonterminals)} nonterminals */")
        lines.append(f"static const int goto_table[{num_states}][{max(1, len(nonterminals))}] = {{")
        
        for state in range(num_states):
            row = []
            for nt in nonterminals:
                goto_state = table.get_goto(state, nt)
                row.append(str(goto_state if goto_state is not None else -1))
            if not row:
                row = ["-1"]
            lines.append(f"    {{ {', '.join(row)} }},")
        
        lines.append("};")
        lines.append("")
        
        # Terminal index function
        lines.append("static int get_terminal_index(TokenType type) {")
        lines.append("    switch (type) {")
        for i, term in enumerate(terminals):
            safe_name = self._c_ident(term.name if hasattr(term, 'name') else str(term))
            lines.append(f"        case TOK_{safe_name}: return {i};")
        lines.append(f"        case TOK_EOF: return {len(terminals)};")
        lines.append("        default: return -1;")
        lines.append("    }")
        lines.append("}")
        
        return lines
    
    def _emit_ll_table(self, table: LL1Table) -> List[str]:
        """Emit LL(1) parse table."""
        lines = []
        
        # Simplified LL table emission
        lines.append("/* LL(1) Parse Table */")
        lines.append("/* (Simplified - production indices stored) */")
        
        nonterminals = sorted(table.get_nonterminals(), 
                             key=lambda s: s.name if hasattr(s, 'name') else str(s))
        terminals = sorted(table.get_terminals(),
                          key=lambda s: s.name if hasattr(s, 'name') else str(s))
        
        lines.append(f"static const int ll_table[{len(nonterminals)}][{max(1, len(terminals))}] = {{")
        
        for nt in nonterminals:
            row = []
            for t in terminals:
                prod = table.get_production(nt, t)
                if prod:
                    # Would need production indexing
                    row.append("1")
                else:
                    row.append("-1")
            if not row:
                row = ["-1"]
            lines.append(f"    {{ {', '.join(row)} }},  /* {nt.name if hasattr(nt, 'name') else nt} */")
        
        lines.append("};")
        
        return lines
    
    def _emit_parser_impl(self, result: ParserGeneratorResult, grammar: Grammar) -> str:
        """Generate parser implementation."""
        parser_type = result.parser_type
        
        if parser_type in (ParserType.LR0, ParserType.SLR1, ParserType.LALR1, ParserType.LR1):
            return self._emit_lr_parser_impl()
        else:
            return self._emit_ll_parser_impl()
    
    def _emit_lr_parser_impl(self) -> str:
        """Generate LR parser implementation."""
        return '''
/* Stack operations */
static void stack_push(Parser* p, int value) {
    if (p->stack_size >= p->stack_capacity) {
        p->stack_capacity = p->stack_capacity ? p->stack_capacity * 2 : 16;
        p->stack = realloc(p->stack, p->stack_capacity * sizeof(int));
    }
    p->stack[p->stack_size++] = value;
}

static int stack_pop(Parser* p) {
    return p->stack_size > 0 ? p->stack[--p->stack_size] : -1;
}

static int stack_top(Parser* p) {
    return p->stack_size > 0 ? p->stack[p->stack_size - 1] : -1;
}

/* Parser creation */
Parser* parser_create(Token* tokens, size_t count) {
    Parser* p = malloc(sizeof(Parser));
    if (!p) return NULL;
    
    p->tokens = tokens;
    p->token_count = count;
    p->pos = 0;
    p->stack = NULL;
    p->stack_size = 0;
    p->stack_capacity = 0;
    
    stack_push(p, 0);  /* Initial state */
    return p;
}

/* Parser destruction */
void parser_destroy(Parser* parser) {
    if (parser) {
        free(parser->stack);
        free(parser);
    }
}

/* Get current token */
static Token* current_token(Parser* p) {
    if (p->pos < p->token_count) {
        return &p->tokens[p->pos];
    }
    static Token eof = { TOK_EOF, "", 0, 0 };
    return &eof;
}

/* Parse function */
int parser_parse(Parser* parser) {
    while (1) {
        int state = stack_top(parser);
        Token* token = current_token(parser);
        int term_idx = get_terminal_index(token->type);
        
        if (term_idx < 0) {
            fprintf(stderr, "Unknown token type: %d\\n", token->type);
            return -1;
        }
        
        Action action = action_table[state][term_idx];
        
        switch (action.type) {
            case ACT_SHIFT:
                stack_push(parser, action.value);
                parser->pos++;
                break;
                
            case ACT_REDUCE: {
                Production prod = productions[action.value];
                for (int i = 0; i < prod.body_len; i++) {
                    stack_pop(parser);
                }
                /* Simplified: would need nonterminal index lookup */
                int goto_state = goto_table[stack_top(parser)][0];
                if (goto_state < 0) {
                    fprintf(stderr, "No GOTO state\\n");
                    return -1;
                }
                stack_push(parser, goto_state);
                break;
            }
            
            case ACT_ACCEPT:
                return 0;  /* Success */
                
            case ACT_ERROR:
            default:
                fprintf(stderr, "Parse error at line %d, column %d\\n",
                        token->line, token->column);
                return -1;
        }
    }
}
'''
    
    def _emit_ll_parser_impl(self) -> str:
        """Generate LL(1) parser implementation."""
        return '''
/* Parser creation */
Parser* parser_create(Token* tokens, size_t count) {
    Parser* p = malloc(sizeof(Parser));
    if (!p) return NULL;
    
    p->tokens = tokens;
    p->token_count = count;
    p->pos = 0;
    p->stack = NULL;
    p->stack_size = 0;
    p->stack_capacity = 0;
    
    return p;
}

/* Parser destruction */
void parser_destroy(Parser* parser) {
    if (parser) {
        free(parser->stack);
        free(parser);
    }
}

/* Parse function (simplified LL(1)) */
int parser_parse(Parser* parser) {
    /* Would need full LL(1) implementation */
    fprintf(stderr, "LL(1) parser not fully implemented\\n");
    return -1;
}
'''
    
    def emit_ast_nodes(self, schema: ASTSchema) -> str:
        """Emit AST node definitions as C types.
        
        Args:
            schema: AST schema
        
        Returns:
            C code with struct definitions
        """
        lines = [
            "/*",
            " * AST Node Definitions",
            " * Generated by STUNIR Parser Emitter",
            " */",
            "",
            "#ifndef AST_NODES_H",
            "#define AST_NODES_H",
            "",
            "#include <stddef.h>",
            "",
        ]
        
        # Forward declarations
        lines.append("/* Forward declarations */")
        for node in schema.nodes:
            lines.append(f"typedef struct {node.name} {node.name};")
        lines.append("")
        
        # Base node type enum
        lines.append("/* Node types */")
        lines.append("typedef enum {")
        for i, node in enumerate(schema.nodes):
            lines.append(f"    NODE_{node.name.upper()} = {i},")
        lines.append("} NodeType;")
        lines.append("")
        
        # Base node struct
        lines.append("/* Base AST node */")
        lines.append(f"typedef struct {schema.base_node_name} {{")
        lines.append("    NodeType type;")
        lines.append(f"}} {schema.base_node_name};")
        lines.append("")
        
        # Concrete nodes
        for node in schema.nodes:
            lines.extend(self._emit_c_ast_struct(node, schema))
            lines.append("")
        
        lines.append("#endif /* AST_NODES_H */")
        
        return '\n'.join(lines)
    
    def _emit_c_ast_struct(self, node: ASTNodeSpec, schema: ASTSchema) -> List[str]:
        """Emit a single AST node struct."""
        lines = []
        
        lines.append(f"/* {node.name} */")
        lines.append(f"struct {node.name} {{")
        lines.append(f"    {schema.base_node_name} base;")
        
        for fname, ftype in node.fields:
            c_type = self._map_c_type(ftype, schema)
            lines.append(f"    {c_type} {fname};")
        
        if not node.fields:
            lines.append("    /* no additional fields */")
        
        lines.append("};")
        
        return lines
    
    def _map_c_type(self, ir_type: str, schema: ASTSchema) -> str:
        """Map IR type to C type."""
        if ir_type == schema.token_type_name:
            return "Token*"
        elif schema.get_node(ir_type):
            return f"{ir_type}*"
        else:
            return "void*"

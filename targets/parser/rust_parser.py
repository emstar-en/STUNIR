#!/usr/bin/env python3
"""Rust parser code emitter.

Generates Rust parser code from parse tables.
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


class RustParserEmitter(ParserEmitterBase):
    """Rust parser code emitter.
    
    Generates Rust parser code with enum-based AST nodes.
    """
    
    LANGUAGE = "rust"
    FILE_EXTENSION = ".rs"
    
    def _get_comment_style(self) -> Dict[str, str]:
        """Get Rust comment style."""
        return {
            'start': '//!',
            'line': '//!',
            'end': '',
        }
    
    def emit(self, result: ParserGeneratorResult, 
             grammar: Grammar) -> ParserEmitterResult:
        """Emit Rust parser code.
        
        Args:
            result: Parser generator result
            grammar: Source grammar
        
        Returns:
            ParserEmitterResult with Rust code
        """
        # Generate parser code
        code_parts = [
            self._emit_header(grammar),
            self._emit_imports(),
            "",
            self._emit_token_types(result, grammar),
            "",
            self.emit_parse_table(result.parse_table),
            "",
            self._emit_parser_struct(result, grammar),
        ]
        code = "\n".join(code_parts)
        
        # Generate AST code
        ast_code = ""
        if result.ast_schema:
            ast_code = self.emit_ast_nodes(result.ast_schema)
        
        # Generate Cargo.toml
        cargo_toml = self._emit_cargo_toml(grammar)
        
        # Generate manifest
        manifest = self._generate_manifest(code, ast_code, grammar, {"Cargo.toml": cargo_toml})
        
        emit_result = ParserEmitterResult(
            code=code,
            ast_code=ast_code,
            manifest=manifest,
            warnings=self._get_warnings()
        )
        emit_result.add_auxiliary_file("Cargo.toml", cargo_toml)
        
        return emit_result
    
    def _emit_imports(self) -> str:
        """Generate use statements."""
        return '''use std::collections::HashMap;
use std::fmt;'''
    
    def _emit_cargo_toml(self, grammar: Grammar) -> str:
        """Generate Cargo.toml."""
        name = grammar.name if hasattr(grammar, 'name') else 'parser'
        return f'''[package]
name = "{name.lower().replace(' ', '_')}_parser"
version = "0.1.0"
edition = "2021"

[dependencies]
'''
    
    def _emit_token_types(self, result: ParserGeneratorResult, grammar: Grammar) -> str:
        """Generate token type enum."""
        lines = [
            "#[derive(Debug, Clone, PartialEq, Eq, Hash)]",
            "pub enum TokenType {",
        ]
        
        # Add terminals from grammar
        terminals = set()
        if isinstance(result.parse_table, ParseTable):
            for (_, sym) in result.parse_table.action.keys():
                if hasattr(sym, 'name'):
                    terminals.add(sym.name)
        
        for term in sorted(terminals):
            safe_name = self._rust_ident(term)
            lines.append(f"    {safe_name},")
        
        lines.append("    Eof,")
        lines.append("    Error,")
        lines.append("}")
        lines.append("")
        
        lines.append("#[derive(Debug, Clone)]")
        lines.append("pub struct Token {")
        lines.append("    pub token_type: TokenType,")
        lines.append("    pub value: String,")
        lines.append("    pub line: usize,")
        lines.append("    pub column: usize,")
        lines.append("}")
        lines.append("")
        lines.append("impl Token {")
        lines.append("    pub fn new(token_type: TokenType, value: String) -> Self {")
        lines.append("        Token { token_type, value, line: 0, column: 0 }")
        lines.append("    }")
        lines.append("}")
        
        return "\n".join(lines)
    
    def _rust_ident(self, name: str) -> str:
        """Convert name to valid Rust identifier."""
        # Handle special characters
        replacements = {
            '+': 'Plus', '-': 'Minus', '*': 'Star', '/': 'Slash',
            '<': 'Lt', '>': 'Gt', '=': 'Eq', '!': 'Bang',
            '&': 'And', '|': 'Or', '^': 'Caret', '%': 'Percent',
            '(': 'LParen', ')': 'RParen', '[': 'LBracket', ']': 'RBracket',
            '{': 'LBrace', '}': 'RBrace', ',': 'Comma', '.': 'Dot',
            ';': 'Semi', ':': 'Colon', '?': 'Question', '@': 'At',
            '#': 'Hash', '$': 'Dollar', '~': 'Tilde',
        }
        
        if name in replacements:
            return replacements[name]
        
        result = []
        for c in name:
            if c.isalnum():
                result.append(c)
            elif c in replacements:
                result.append(replacements[c])
            elif c == '_':
                result.append(c)
            else:
                result.append('_')
        
        ident = ''.join(result)
        
        # Ensure starts with letter or underscore
        if ident and ident[0].isdigit():
            ident = '_' + ident
        
        # Convert to PascalCase for enum variants
        return ''.join(word.capitalize() for word in ident.split('_') if word)
    
    def emit_parse_table(self, table: Union[ParseTable, LL1Table]) -> str:
        """Emit parse table as Rust data structures.
        
        Args:
            table: Parse table to emit
        
        Returns:
            Rust code defining the parse tables
        """
        lines = ["// Parse Tables"]
        
        if isinstance(table, ParseTable):
            lines.extend(self._emit_lr_table(table))
        else:
            lines.extend(self._emit_ll_table(table))
        
        return "\n".join(lines)
    
    def _emit_lr_table(self, table: ParseTable) -> List[str]:
        """Emit LR parse table."""
        lines = []
        
        # Action enum
        lines.append("#[derive(Debug, Clone, Copy)]")
        lines.append("pub enum Action {")
        lines.append("    Shift(usize),")
        lines.append("    Reduce(usize),")
        lines.append("    Accept,")
        lines.append("    Error,")
        lines.append("}")
        lines.append("")
        
        # Production struct
        lines.append("#[derive(Debug, Clone)]")
        lines.append("pub struct Production {")
        lines.append("    pub head: &'static str,")
        lines.append("    pub body_len: usize,")
        lines.append("}")
        lines.append("")
        
        # Productions array
        lines.append("pub static PRODUCTIONS: &[Production] = &[")
        for prod in table.productions:
            head = prod.head.name if hasattr(prod.head, 'name') else str(prod.head)
            body_len = len(prod.body) if prod.body else 0
            lines.append(f'    Production {{ head: "{head}", body_len: {body_len} }},')
        lines.append("];")
        lines.append("")
        
        # ACTION table as function (more idiomatic than giant static HashMap)
        lines.append("pub fn get_action(state: usize, token: &TokenType) -> Action {")
        lines.append("    match (state, token) {")
        
        for (state, sym), action in sorted(table.action.items()):
            sym_name = sym.name if hasattr(sym, 'name') else str(sym)
            safe_name = self._rust_ident(sym_name)
            
            if action.is_shift():
                rust_action = f"Action::Shift({action.value})"
            elif action.is_reduce():
                rust_action = f"Action::Reduce({action.value})"
            elif action.is_accept():
                rust_action = "Action::Accept"
            else:
                rust_action = "Action::Error"
            
            if sym_name == "$":
                lines.append(f"        ({state}, TokenType::Eof) => {rust_action},")
            else:
                lines.append(f"        ({state}, TokenType::{safe_name}) => {rust_action},")
        
        lines.append("        _ => Action::Error,")
        lines.append("    }")
        lines.append("}")
        lines.append("")
        
        # GOTO table
        lines.append("pub fn get_goto(state: usize, nonterminal: &str) -> Option<usize> {")
        lines.append("    match (state, nonterminal) {")
        
        for (state, sym), target in sorted(table.goto.items()):
            sym_name = sym.name if hasattr(sym, 'name') else str(sym)
            lines.append(f'        ({state}, "{sym_name}") => Some({target}),')
        
        lines.append("        _ => None,")
        lines.append("    }")
        lines.append("}")
        
        return lines
    
    def _emit_ll_table(self, table: LL1Table) -> List[str]:
        """Emit LL(1) parse table."""
        lines = []
        
        lines.append("pub fn get_production(nonterminal: &str, terminal: &TokenType) -> Option<&'static [&'static str]> {")
        lines.append("    match (nonterminal, terminal) {")
        
        for (nt, t), prod in sorted(table.table.items(), key=lambda x: (str(x[0][0]), str(x[0][1]))):
            nt_name = nt.name if hasattr(nt, 'name') else str(nt)
            t_name = t.name if hasattr(t, 'name') else str(t)
            safe_t = self._rust_ident(t_name)
            
            body_syms = [s.name if hasattr(s, 'name') else str(s) for s in (prod.body or [])]
            body_str = ', '.join(f'"{s}"' for s in body_syms)
            
            if t_name == "$":
                lines.append(f'        ("{nt_name}", TokenType::Eof) => Some(&[{body_str}]),')
            else:
                lines.append(f'        ("{nt_name}", TokenType::{safe_t}) => Some(&[{body_str}]),')
        
        lines.append("        _ => None,")
        lines.append("    }")
        lines.append("}")
        
        return lines
    
    def _emit_parser_struct(self, result: ParserGeneratorResult, grammar: Grammar) -> str:
        """Generate Parser struct."""
        parser_type = result.parser_type
        grammar_name = grammar.name if hasattr(grammar, 'name') else 'unknown'
        
        if parser_type in (ParserType.LR0, ParserType.SLR1, ParserType.LALR1, ParserType.LR1):
            return self._emit_lr_parser_struct(grammar_name)
        else:
            return self._emit_ll_parser_struct(grammar_name)
    
    def _emit_lr_parser_struct(self, grammar_name: str) -> str:
        """Generate LR parser struct."""
        return f'''
/// Parse error
#[derive(Debug)]
pub struct ParseError {{
    pub message: String,
    pub line: usize,
    pub column: usize,
}}

impl fmt::Display for ParseError {{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {{
        write!(f, "Parse error at line {{}}, column {{}}: {{}}", 
               self.line, self.column, self.message)
    }}
}}

impl std::error::Error for ParseError {{}}

/// LR Parser for grammar: {grammar_name}
pub struct Parser {{
    tokens: Vec<Token>,
    pos: usize,
    stack: Vec<usize>,
    value_stack: Vec<Box<dyn std::any::Any>>,
}}

impl Parser {{
    pub fn new(tokens: Vec<Token>) -> Self {{
        Parser {{
            tokens,
            pos: 0,
            stack: vec![0],
            value_stack: Vec::new(),
        }}
    }}
    
    fn current_token(&self) -> &Token {{
        if self.pos < self.tokens.len() {{
            &self.tokens[self.pos]
        }} else {{
            // Return a dummy EOF token
            static EOF_TOKEN: Token = Token {{
                token_type: TokenType::Eof,
                value: String::new(),
                line: 0,
                column: 0,
            }};
            &EOF_TOKEN
        }}
    }}
    
    fn advance(&mut self) -> Token {{
        if self.pos < self.tokens.len() {{
            let token = self.tokens[self.pos].clone();
            self.pos += 1;
            token
        }} else {{
            Token::new(TokenType::Eof, String::new())
        }}
    }}
    
    pub fn parse(&mut self) -> Result<(), ParseError> {{
        loop {{
            let state = *self.stack.last().unwrap();
            let token = self.current_token();
            
            let action = get_action(state, &token.token_type);
            
            match action {{
                Action::Shift(next_state) => {{
                    self.stack.push(next_state);
                    self.advance();
                }}
                
                Action::Reduce(prod_index) => {{
                    let production = &PRODUCTIONS[prod_index];
                    
                    // Pop states
                    for _ in 0..production.body_len {{
                        self.stack.pop();
                    }}
                    
                    // Get goto state
                    let state = *self.stack.last().unwrap();
                    match get_goto(state, production.head) {{
                        Some(goto_state) => self.stack.push(goto_state),
                        None => return Err(ParseError {{
                            message: format!("No GOTO for ({{}}, {{}})", state, production.head),
                            line: token.line,
                            column: token.column,
                        }}),
                    }}
                }}
                
                Action::Accept => {{
                    return Ok(());
                }}
                
                Action::Error => {{
                    return Err(ParseError {{
                        message: format!("Unexpected token: {{:?}}", token.token_type),
                        line: token.line,
                        column: token.column,
                    }});
                }}
            }}
        }}
    }}
}}
'''
    
    def _emit_ll_parser_struct(self, grammar_name: str) -> str:
        """Generate LL(1) parser struct."""
        return f'''
/// Parse error
#[derive(Debug)]
pub struct ParseError {{
    pub message: String,
    pub line: usize,
    pub column: usize,
}}

impl fmt::Display for ParseError {{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {{
        write!(f, "Parse error at line {{}}, column {{}}: {{}}", 
               self.line, self.column, self.message)
    }}
}}

impl std::error::Error for ParseError {{}}

/// LL(1) Parser for grammar: {grammar_name}
pub struct Parser {{
    tokens: Vec<Token>,
    pos: usize,
    stack: Vec<String>,
}}

impl Parser {{
    pub fn new(tokens: Vec<Token>) -> Self {{
        Parser {{
            tokens,
            pos: 0,
            stack: Vec::new(),
        }}
    }}
    
    fn current_token(&self) -> &Token {{
        if self.pos < self.tokens.len() {{
            &self.tokens[self.pos]
        }} else {{
            static EOF_TOKEN: Token = Token {{
                token_type: TokenType::Eof,
                value: String::new(),
                line: 0,
                column: 0,
            }};
            &EOF_TOKEN
        }}
    }}
    
    fn advance(&mut self) -> Token {{
        if self.pos < self.tokens.len() {{
            let token = self.tokens[self.pos].clone();
            self.pos += 1;
            token
        }} else {{
            Token::new(TokenType::Eof, String::new())
        }}
    }}
    
    pub fn parse(&mut self, start_symbol: &str) -> Result<(), ParseError> {{
        self.stack.push("$".to_string());
        self.stack.push(start_symbol.to_string());
        
        while let Some(top) = self.stack.pop() {{
            let token = self.current_token();
            
            if top == "$" {{
                if token.token_type == TokenType::Eof {{
                    return Ok(());
                }} else {{
                    return Err(ParseError {{
                        message: "Expected end of input".to_string(),
                        line: token.line,
                        column: token.column,
                    }});
                }}
            }}
            
            // Check if terminal matches
            // (simplified - would need proper terminal matching)
            
            // Consult table for nonterminal
            if let Some(production) = get_production(&top, &token.token_type) {{
                for symbol in production.iter().rev() {{
                    self.stack.push(symbol.to_string());
                }}
            }}
        }}
        
        Ok(())
    }}
}}
'''
    
    def emit_ast_nodes(self, schema: ASTSchema) -> str:
        """Emit AST node definitions as Rust types.
        
        Args:
            schema: AST schema
        
        Returns:
            Rust code with struct/enum definitions
        """
        lines = [
            "//! AST Node Definitions",
            "//!",
            "//! Generated by STUNIR Parser Emitter",
            "",
            "use std::boxed::Box;",
            "",
        ]
        
        # Emit base trait
        lines.append(f"pub trait {schema.base_node_name} {{")
        lines.append("    fn node_type(&self) -> &'static str;")
        lines.append("}")
        lines.append("")
        
        # Group nodes by base class for enum generation
        abstract_nodes = schema.get_abstract_nodes()
        
        for abstract_node in abstract_nodes:
            lines.extend(self._emit_ast_enum(abstract_node, schema))
            lines.append("")
        
        # Emit concrete nodes without base class
        for node in schema.get_concrete_nodes():
            if node.base_class is None:
                lines.extend(self._emit_ast_struct(node, schema))
                lines.append("")
        
        return '\n'.join(lines)
    
    def _emit_ast_enum(self, abstract_node: ASTNodeSpec, schema: ASTSchema) -> List[str]:
        """Emit enum for abstract node with variants."""
        lines = []
        variants = schema.get_nodes_by_base(abstract_node.name)
        
        lines.append("#[derive(Debug, Clone)]")
        lines.append(f"pub enum {abstract_node.name} {{")
        
        for variant in variants:
            # Emit as tuple variant with fields
            if variant.fields:
                field_types = [self._map_rust_type(ft, schema) for _, ft in variant.fields]
                fields_str = ", ".join(field_types)
                lines.append(f"    {self._variant_name(variant.name, abstract_node.name)}({fields_str}),")
            else:
                lines.append(f"    {self._variant_name(variant.name, abstract_node.name)},")
        
        lines.append("}")
        
        # Implement base trait
        lines.append("")
        lines.append(f"impl {schema.base_node_name} for {abstract_node.name} {{")
        lines.append('    fn node_type(&self) -> &\'static str {')
        lines.append("        match self {")
        for variant in variants:
            vname = self._variant_name(variant.name, abstract_node.name)
            lines.append(f'            {abstract_node.name}::{vname}{{..}} => "{variant.name}",')
        lines.append("        }")
        lines.append("    }")
        lines.append("}")
        
        return lines
    
    def _emit_ast_struct(self, node: ASTNodeSpec, schema: ASTSchema) -> List[str]:
        """Emit struct for concrete node."""
        lines = []
        
        lines.append("#[derive(Debug, Clone)]")
        lines.append(f"pub struct {node.name} {{")
        
        for fname, ftype in node.fields:
            rust_type = self._map_rust_type(ftype, schema)
            lines.append(f"    pub {self._rust_field_name(fname)}: {rust_type},")
        
        if not node.fields:
            lines.append("    _marker: std::marker::PhantomData<()>,")
        
        lines.append("}")
        
        # Implement base trait
        lines.append("")
        lines.append(f"impl {schema.base_node_name} for {node.name} {{")
        lines.append('    fn node_type(&self) -> &\'static str {')
        lines.append(f'        "{node.name}"')
        lines.append("    }")
        lines.append("}")
        
        return lines
    
    def _variant_name(self, node_name: str, base_name: str) -> str:
        """Get enum variant name from node name."""
        # Remove base name suffix if present
        if node_name.endswith(base_name):
            return node_name[:-len(base_name)]
        return node_name.replace(base_name, "")
    
    def _rust_field_name(self, name: str) -> str:
        """Convert field name to Rust style."""
        # Convert to snake_case and handle reserved words
        result = name.lower()
        reserved = {'type', 'match', 'fn', 'let', 'mut', 'ref', 'self', 'super', 'mod', 'use'}
        if result in reserved:
            result = f"r#{result}"
        return result
    
    def _map_rust_type(self, ir_type: str, schema: ASTSchema) -> str:
        """Map IR type to Rust type."""
        if ir_type == schema.token_type_name:
            return "Token"
        elif schema.get_node(ir_type):
            node = schema.get_node(ir_type)
            if node and node.is_abstract:
                return f"Box<{ir_type}>"
            return ir_type
        else:
            return f"Box<dyn {schema.base_node_name}>"

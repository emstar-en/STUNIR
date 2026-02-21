"""
STUNIR Bootstrap Compiler.

The bootstrap compiler parses STUNIR source files using the generated
parser and lexer, producing an AST for further processing.

This enables STUNIR to be self-hosting by:
1. Using generated lexer to tokenize source
2. Using generated parser to build parse tree
3. Converting parse tree to AST
4. Validating AST structure
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Iterator

from .stunir_lexer import STUNIR_KEYWORDS, STUNIR_TOKENS


@dataclass
class STUNIRToken:
    """
    Token from STUNIR lexer.
    
    Attributes:
        type: Token type name (e.g., 'KW_MODULE', 'IDENTIFIER')
        value: Lexeme (actual text matched)
        line: Line number (1-based)
        column: Column number (1-based)
    """
    type: str
    value: str
    line: int
    column: int
    
    def __repr__(self):
        return f"Token({self.type}, {self.value!r}, {self.line}:{self.column})"
    
    def is_eof(self) -> bool:
        return self.type == 'EOF'
    
    def is_keyword(self, keyword: str) -> bool:
        return self.type == f'KW_{keyword.upper()}'


@dataclass
class STUNIRASTNode:
    """
    AST node for STUNIR programs.
    
    Attributes:
        kind: Node kind (e.g., 'program', 'function_def', 'binary_op')
        children: Child nodes
        attributes: Node attributes (name, value, etc.)
        location: Source location (line, column)
    """
    kind: str
    children: List['STUNIRASTNode'] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    location: Optional[Tuple[int, int]] = None
    
    def __repr__(self):
        return f"ASTNode({self.kind}, attrs={self.attributes})"
    
    def add_child(self, child: 'STUNIRASTNode') -> None:
        """Add a child node."""
        self.children.append(child)
    
    def get_attr(self, name: str, default: Any = None) -> Any:
        """Get an attribute value."""
        return self.attributes.get(name, default)
    
    def set_attr(self, name: str, value: Any) -> None:
        """Set an attribute value."""
        self.attributes[name] = value
    
    def find_children(self, kind: str) -> List['STUNIRASTNode']:
        """Find all children of a specific kind."""
        return [c for c in self.children if c.kind == kind]


@dataclass
class BootstrapResult:
    """
    Result of bootstrap compilation.
    
    Attributes:
        success: True if compilation succeeded
        ast: Root AST node (if successful)
        tokens: List of tokens from lexer
        errors: List of error messages
        warnings: List of warning messages
    """
    success: bool
    ast: Optional[STUNIRASTNode] = None
    tokens: List[STUNIRToken] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class STUNIRLexerError(Exception):
    """Lexer error with location information."""
    
    def __init__(self, message: str, line: int = 0, column: int = 0):
        super().__init__(message)
        self.line = line
        self.column = column


class STUNIRParseError(Exception):
    """Parse error with location information."""
    
    def __init__(self, message: str, line: int = 0, column: int = 0):
        super().__init__(message)
        self.line = line
        self.column = column


class SimpleLexer:
    """
    Simple lexer for STUNIR bootstrap.
    
    Uses regex-based tokenization with keyword post-processing.
    This is a fallback lexer that doesn't require the generated
    lexer to be available.
    """
    
    def __init__(self, source: str):
        """Initialize lexer with source code."""
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1
        self._build_patterns()
    
    def _build_patterns(self):
        """Build regex patterns from token specifications."""
        patterns = []
        
        # Build patterns from token specs (in priority order)
        token_specs = sorted(STUNIR_TOKENS, key=lambda t: -t.priority)
        
        for spec in token_specs:
            patterns.append((spec.name, spec.pattern, spec.skip))
        
        # Combine into single pattern with named groups
        self._patterns = patterns
        
        # Build combined regex
        parts = []
        for name, pattern, _ in patterns:
            parts.append(f'(?P<{name}>{pattern})')
        
        self._regex = re.compile('|'.join(parts))
    
    def _advance(self, text: str):
        """Advance position by text length, tracking line/column."""
        for ch in text:
            if ch == '\n':
                self.line += 1
                self.column = 1
            else:
                self.column += 1
        self.pos += len(text)
    
    def tokenize(self) -> Iterator[STUNIRToken]:
        """
        Tokenize the source code.
        
        Yields:
            STUNIRToken for each token in source
        """
        while self.pos < len(self.source):
            match = self._regex.match(self.source, self.pos)
            
            if not match:
                # Unexpected character
                ch = self.source[self.pos]
                raise STUNIRLexerError(
                    f"Unexpected character: {ch!r}",
                    self.line, self.column
                )
            
            # Find which pattern matched
            token_type = match.lastgroup
            value = match.group(token_type)
            
            # Check if this is a skip token
            is_skip = False
            for name, _, skip in self._patterns:
                if name == token_type:
                    is_skip = skip
                    break
            
            if not is_skip:
                # Create token
                token = STUNIRToken(
                    type=token_type,
                    value=value,
                    line=self.line,
                    column=self.column
                )
                
                # Handle keywords
                if token.type == 'IDENTIFIER' and token.value in STUNIR_KEYWORDS:
                    token.type = STUNIR_KEYWORDS[token.value]
                
                yield token
            
            self._advance(value)
        
        # Yield EOF token
        yield STUNIRToken('EOF', '', self.line, self.column)


class RecursiveDescentParser:
    """
    Recursive descent parser for STUNIR bootstrap.
    
    Implements a hand-written parser for the STUNIR grammar.
    This is a fallback parser that doesn't require the generated
    parser to be available.
    """
    
    def __init__(self, tokens: List[STUNIRToken]):
        """Initialize parser with tokens."""
        self.tokens = tokens
        self.pos = 0
        self._errors: List[str] = []
    
    def _current(self) -> STUNIRToken:
        """Get current token."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return self.tokens[-1]  # EOF
    
    def _peek(self, offset: int = 0) -> STUNIRToken:
        """Peek at token with offset."""
        pos = self.pos + offset
        if pos < len(self.tokens):
            return self.tokens[pos]
        return self.tokens[-1]
    
    def _advance(self) -> STUNIRToken:
        """Advance to next token and return current."""
        token = self._current()
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
        return token
    
    def _check(self, *types: str) -> bool:
        """Check if current token matches any of the types."""
        return self._current().type in types
    
    def _match(self, *types: str) -> Optional[STUNIRToken]:
        """Match and consume token if it matches."""
        if self._check(*types):
            return self._advance()
        return None
    
    def _expect(self, type_: str, message: str = None) -> STUNIRToken:
        """Expect and consume a specific token type."""
        if self._check(type_):
            return self._advance()
        
        token = self._current()
        msg = message or f"Expected {type_}, got {token.type}"
        raise STUNIRParseError(msg, token.line, token.column)
    
    def _make_node(self, kind: str, **attrs) -> STUNIRASTNode:
        """Create an AST node at current position."""
        token = self._current()
        return STUNIRASTNode(
            kind=kind,
            attributes=attrs,
            location=(token.line, token.column)
        )
    
    def parse(self) -> STUNIRASTNode:
        """
        Parse the token stream.
        
        Returns:
            Root AST node (program)
        """
        return self._parse_program()
    
    def _parse_program(self) -> STUNIRASTNode:
        """Parse: program → module_decl declarations"""
        node = self._make_node('program')
        
        # Parse module declaration
        module = self._parse_module_decl()
        node.add_child(module)
        
        # Parse declarations
        while not self._check('EOF'):
            decl = self._parse_declaration()
            if decl:
                node.add_child(decl)
        
        return node
    
    def _parse_module_decl(self) -> STUNIRASTNode:
        """Parse: module_decl → KW_MODULE IDENTIFIER (SEMICOLON | LBRACE ... RBRACE)"""
        self._expect('KW_MODULE', "Expected 'module'")
        
        name_token = self._expect('IDENTIFIER', "Expected module name")
        node = self._make_node('module_decl', name=name_token.value)
        
        if self._match('SEMICOLON'):
            # Simple module declaration
            return node
        
        if self._match('LBRACE'):
            # Module with body
            while not self._check('RBRACE', 'EOF'):
                item = self._parse_module_item()
                if item:
                    node.add_child(item)
            self._expect('RBRACE', "Expected '}'")
            return node
        
        raise STUNIRParseError("Expected ';' or '{'", 
                              self._current().line, self._current().column)
    
    def _parse_module_item(self) -> Optional[STUNIRASTNode]:
        """Parse module item (import, export, or declaration)."""
        if self._check('KW_IMPORT'):
            return self._parse_import_decl()
        if self._check('KW_FROM'):
            return self._parse_import_from()
        if self._check('KW_EXPORT'):
            return self._parse_export_decl()
        return self._parse_declaration()
    
    def _parse_import_decl(self) -> STUNIRASTNode:
        """Parse: import_decl → KW_IMPORT dotted_name [KW_AS IDENTIFIER] SEMICOLON"""
        self._expect('KW_IMPORT')
        
        name = self._parse_dotted_name()
        node = self._make_node('import_decl', module=name)
        
        if self._match('KW_AS'):
            alias = self._expect('IDENTIFIER')
            node.set_attr('alias', alias.value)
        
        self._expect('SEMICOLON')
        return node
    
    def _parse_import_from(self) -> STUNIRASTNode:
        """Parse: KW_FROM dotted_name KW_IMPORT identifier_list SEMICOLON"""
        self._expect('KW_FROM')
        
        module = self._parse_dotted_name()
        self._expect('KW_IMPORT')
        
        names = self._parse_identifier_list()
        node = self._make_node('import_from', module=module, names=names)
        
        self._expect('SEMICOLON')
        return node
    
    def _parse_export_decl(self) -> STUNIRASTNode:
        """Parse: export_decl → KW_EXPORT identifier_list SEMICOLON"""
        self._expect('KW_EXPORT')
        
        if self._match('STAR'):
            node = self._make_node('export_all')
        else:
            names = self._parse_identifier_list()
            node = self._make_node('export_decl', names=names)
        
        self._expect('SEMICOLON')
        return node
    
    def _parse_dotted_name(self) -> str:
        """Parse: dotted_name → IDENTIFIER (DOT IDENTIFIER)*"""
        parts = [self._expect('IDENTIFIER').value]
        
        while self._match('DOT'):
            parts.append(self._expect('IDENTIFIER').value)
        
        return '.'.join(parts)
    
    def _parse_identifier_list(self) -> List[str]:
        """Parse: identifier_list → IDENTIFIER (COMMA IDENTIFIER)*"""
        names = [self._expect('IDENTIFIER').value]
        
        while self._match('COMMA'):
            names.append(self._expect('IDENTIFIER').value)
        
        return names
    
    def _parse_declaration(self) -> Optional[STUNIRASTNode]:
        """Parse declaration (type, function, ir, target)."""
        if self._check('KW_TYPE'):
            return self._parse_type_def()
        if self._check('KW_FUNCTION'):
            return self._parse_function_def()
        if self._check('KW_IR'):
            return self._parse_ir_def()
        if self._check('KW_TARGET'):
            return self._parse_target_def()
        
        # Skip unknown token
        if not self._check('EOF', 'RBRACE'):
            token = self._advance()
            self._errors.append(
                f"Unexpected token {token.type} at line {token.line}"
            )
        
        return None
    
    def _parse_type_def(self) -> STUNIRASTNode:
        """Parse type definition."""
        self._expect('KW_TYPE')
        
        name = self._expect('IDENTIFIER').value
        node = self._make_node('type_def', name=name)
        
        # Optional type parameters
        if self._match('LT'):
            params = self._parse_identifier_list()
            node.set_attr('type_params', params)
            self._expect('GT')
        
        if self._match('EQUALS'):
            # Type alias
            type_expr = self._parse_type_expr()
            node.set_attr('type_expr', type_expr)
            self._expect('SEMICOLON')
        elif self._match('LBRACE'):
            # Struct/enum type
            while not self._check('RBRACE', 'EOF'):
                member = self._parse_type_member()
                if member:
                    node.add_child(member)
            self._expect('RBRACE')
        
        return node
    
    def _parse_type_member(self) -> Optional[STUNIRASTNode]:
        """Parse type member (field or variant)."""
        if self._match('PIPE'):
            # Variant
            name = self._expect('IDENTIFIER').value
            node = self._make_node('variant', name=name)
            
            if self._match('LPAREN'):
                types = self._parse_type_list()
                node.set_attr('types', types)
                self._expect('RPAREN')
            
            return node
        
        if self._check('IDENTIFIER'):
            # Field
            name = self._advance().value
            self._expect('COLON')
            type_expr = self._parse_type_expr()
            self._expect('SEMICOLON')
            return self._make_node('field', name=name, type=type_expr)
        
        return None
    
    def _parse_type_expr(self) -> Dict[str, Any]:
        """Parse type expression."""
        # Basic types
        basic_types = [
            'KW_I8', 'KW_I16', 'KW_I32', 'KW_I64',
            'KW_U8', 'KW_U16', 'KW_U32', 'KW_U64',
            'KW_F32', 'KW_F64', 'KW_BOOL', 'KW_STRING', 'KW_VOID', 'KW_ANY'
        ]
        
        for bt in basic_types:
            if self._match(bt):
                return {'kind': 'basic', 'type': bt[3:].lower()}
        
        # Named type
        if self._check('IDENTIFIER'):
            name = self._advance().value
            result = {'kind': 'named', 'name': name}
            
            # Generic parameters
            if self._match('LT'):
                result['params'] = self._parse_type_list()
                self._expect('GT')
            
            return result
        
        # Array type
        if self._match('LBRACKET'):
            elem = self._parse_type_expr()
            self._expect('RBRACKET')
            return {'kind': 'array', 'element': elem}
        
        return {'kind': 'unknown'}
    
    def _parse_type_list(self) -> List[Dict[str, Any]]:
        """Parse comma-separated type list."""
        types = [self._parse_type_expr()]
        
        while self._match('COMMA'):
            types.append(self._parse_type_expr())
        
        return types
    
    def _parse_function_def(self) -> STUNIRASTNode:
        """Parse function definition."""
        self._expect('KW_FUNCTION')
        
        name = self._expect('IDENTIFIER').value
        node = self._make_node('function_def', name=name)
        
        # Parameters
        self._expect('LPAREN')
        params = self._parse_param_list()
        node.set_attr('params', params)
        self._expect('RPAREN')
        
        # Return type
        if self._match('COLON'):
            return_type = self._parse_type_expr()
            node.set_attr('return_type', return_type)
        
        # Body
        body = self._parse_block()
        node.add_child(body)
        
        return node
    
    def _parse_param_list(self) -> List[Dict[str, Any]]:
        """Parse parameter list."""
        params = []
        
        if self._check('RPAREN'):
            return params
        
        params.append(self._parse_param())
        
        while self._match('COMMA'):
            params.append(self._parse_param())
        
        return params
    
    def _parse_param(self) -> Dict[str, Any]:
        """Parse a single parameter."""
        name = self._expect('IDENTIFIER').value
        self._expect('COLON')
        type_expr = self._parse_type_expr()
        
        param = {'name': name, 'type': type_expr}
        
        if self._match('EQUALS'):
            param['default'] = self._parse_expression()
        
        return param
    
    def _parse_block(self) -> STUNIRASTNode:
        """Parse block statement."""
        self._expect('LBRACE')
        
        node = self._make_node('block')
        
        while not self._check('RBRACE', 'EOF'):
            stmt = self._parse_statement()
            if stmt:
                node.add_child(stmt)
        
        self._expect('RBRACE')
        return node
    
    def _parse_statement(self) -> Optional[STUNIRASTNode]:
        """Parse a statement."""
        if self._check('KW_LET'):
            return self._parse_var_decl(is_let=True)
        if self._check('KW_VAR'):
            return self._parse_var_decl(is_let=False)
        if self._check('KW_IF'):
            return self._parse_if_stmt()
        if self._check('KW_WHILE'):
            return self._parse_while_stmt()
        if self._check('KW_FOR'):
            return self._parse_for_stmt()
        if self._check('KW_MATCH'):
            return self._parse_match_stmt()
        if self._check('KW_RETURN'):
            return self._parse_return_stmt()
        if self._check('KW_EMIT'):
            return self._parse_emit_stmt()
        
        # Check for assignment: identifier = expression
        if self._check('IDENTIFIER'):
            # Peek ahead to see if this is an assignment
            lookahead_pos = self.pos
            self._advance()  # Consume identifier
            
            # Check for member access or indexing
            while self._check('DOT', 'LBRACKET'):
                if self._match('DOT'):
                    if not self._match('IDENTIFIER'):
                        # Reset and fall through to expression
                        self.pos = lookahead_pos
                        break
                elif self._match('LBRACKET'):
                    # Skip over index expression (simplified)
                    depth = 1
                    while depth > 0 and not self._check('EOF'):
                        if self._check('LBRACKET'):
                            depth += 1
                        elif self._check('RBRACKET'):
                            depth -= 1
                        self._advance()
            
            # Check if followed by assignment operator
            if self._check('EQUALS', 'PLUS_EQ', 'MINUS_EQ', 'STAR_EQ', 'SLASH_EQ'):
                # It's an assignment statement
                # Get the lvalue name(s)
                saved_pos = self.pos
                self.pos = lookahead_pos
                lvalue = self._parse_lvalue()
                
                op_token = self._advance()  # Get assignment operator
                value = self._parse_expression()
                self._expect('SEMICOLON')
                
                node = self._make_node('assignment', op=op_token.value)
                node.add_child(lvalue)
                node.add_child(value)
                return node
            else:
                # Reset and parse as expression
                self.pos = lookahead_pos
        
        # Expression statement
        expr = self._parse_expression()
        self._expect('SEMICOLON')
        
        node = self._make_node('expr_stmt')
        node.add_child(expr)
        return node
    
    def _parse_lvalue(self) -> STUNIRASTNode:
        """Parse an lvalue (target of assignment)."""
        name = self._expect('IDENTIFIER').value
        node = self._make_node('identifier', name=name)
        
        while self._check('DOT', 'LBRACKET'):
            if self._match('DOT'):
                member = self._expect('IDENTIFIER').value
                new_node = self._make_node('member_access', member=member)
                new_node.add_child(node)
                node = new_node
            elif self._match('LBRACKET'):
                index = self._parse_expression()
                self._expect('RBRACKET')
                new_node = self._make_node('index_access')
                new_node.add_child(node)
                new_node.add_child(index)
                node = new_node
        
        return node
    
    def _parse_var_decl(self, is_let: bool) -> STUNIRASTNode:
        """Parse variable declaration."""
        self._advance()  # KW_LET or KW_VAR
        
        name = self._expect('IDENTIFIER').value
        node = self._make_node('var_decl', name=name, is_let=is_let)
        
        # Optional type annotation
        if self._match('COLON'):
            node.set_attr('type', self._parse_type_expr())
        
        # Initializer (required for let, optional for var)
        if self._match('EQUALS'):
            init = self._parse_expression()
            node.add_child(init)
        
        self._expect('SEMICOLON')
        return node
    
    def _parse_if_stmt(self) -> STUNIRASTNode:
        """Parse if statement."""
        self._expect('KW_IF')
        
        node = self._make_node('if_stmt')
        
        # Condition
        cond = self._parse_expression()
        node.add_child(cond)
        
        # Then block
        then_block = self._parse_block()
        node.add_child(then_block)
        
        # Optional else
        if self._match('KW_ELSE'):
            if self._check('KW_IF'):
                # else if
                else_part = self._parse_if_stmt()
            else:
                # else block
                else_part = self._parse_block()
            node.add_child(else_part)
        
        return node
    
    def _parse_while_stmt(self) -> STUNIRASTNode:
        """Parse while statement."""
        self._expect('KW_WHILE')
        
        node = self._make_node('while_stmt')
        
        cond = self._parse_expression()
        node.add_child(cond)
        
        body = self._parse_block()
        node.add_child(body)
        
        return node
    
    def _parse_for_stmt(self) -> STUNIRASTNode:
        """Parse for statement."""
        self._expect('KW_FOR')
        
        var_name = self._expect('IDENTIFIER').value
        self._expect('KW_IN')
        
        node = self._make_node('for_stmt', var=var_name)
        
        iterable = self._parse_expression()
        node.add_child(iterable)
        
        body = self._parse_block()
        node.add_child(body)
        
        return node
    
    def _parse_match_stmt(self) -> STUNIRASTNode:
        """Parse match statement."""
        self._expect('KW_MATCH')
        
        node = self._make_node('match_stmt')
        
        expr = self._parse_expression()
        node.add_child(expr)
        
        self._expect('LBRACE')
        
        while not self._check('RBRACE', 'EOF'):
            arm = self._parse_match_arm()
            if arm:
                node.add_child(arm)
        
        self._expect('RBRACE')
        return node
    
    def _parse_match_arm(self) -> STUNIRASTNode:
        """Parse match arm."""
        pattern = self._parse_pattern()
        self._expect('FAT_ARROW')
        
        node = self._make_node('match_arm')
        node.add_child(pattern)
        
        if self._check('LBRACE'):
            body = self._parse_block()
        else:
            body = self._parse_expression()
            self._match('COMMA')  # Optional trailing comma
        
        node.add_child(body)
        return node
    
    def _parse_pattern(self) -> STUNIRASTNode:
        """Parse pattern."""
        if self._check('INTEGER_LITERAL', 'FLOAT_LITERAL', 'STRING_LITERAL',
                       'KW_TRUE', 'KW_FALSE', 'KW_NULL'):
            token = self._advance()
            return self._make_node('pattern_literal', value=token.value, type=token.type)
        
        if self._check('IDENTIFIER'):
            name = self._advance().value
            
            if self._match('LPAREN'):
                # Constructor pattern
                node = self._make_node('pattern_constructor', name=name)
                
                if not self._check('RPAREN'):
                    while True:
                        sub = self._parse_pattern()
                        node.add_child(sub)
                        if not self._match('COMMA'):
                            break
                
                self._expect('RPAREN')
                return node
            
            return self._make_node('pattern_var', name=name)
        
        if self._match('LBRACKET'):
            node = self._make_node('pattern_array')
            
            if not self._check('RBRACKET'):
                while True:
                    sub = self._parse_pattern()
                    node.add_child(sub)
                    if not self._match('COMMA'):
                        break
            
            self._expect('RBRACKET')
            return node
        
        return self._make_node('pattern_wildcard')
    
    def _parse_return_stmt(self) -> STUNIRASTNode:
        """Parse return statement."""
        self._expect('KW_RETURN')
        
        node = self._make_node('return_stmt')
        
        if not self._check('SEMICOLON'):
            expr = self._parse_expression()
            node.add_child(expr)
        
        self._expect('SEMICOLON')
        return node
    
    def _parse_emit_stmt(self) -> STUNIRASTNode:
        """Parse emit statement."""
        self._expect('KW_EMIT')
        
        node = self._make_node('emit_stmt')
        
        expr = self._parse_expression()
        node.add_child(expr)
        
        self._expect('SEMICOLON')
        return node
    
    def _parse_expression(self) -> STUNIRASTNode:
        """Parse expression (entry point for expression parsing)."""
        return self._parse_ternary()
    
    def _parse_ternary(self) -> STUNIRASTNode:
        """Parse ternary expression."""
        expr = self._parse_or()
        
        if self._match('QUESTION'):
            then_expr = self._parse_expression()
            self._expect('COLON')
            else_expr = self._parse_ternary()
            
            node = self._make_node('ternary')
            node.add_child(expr)
            node.add_child(then_expr)
            node.add_child(else_expr)
            return node
        
        return expr
    
    def _parse_or(self) -> STUNIRASTNode:
        """Parse OR expression."""
        left = self._parse_and()
        
        while self._match('OR'):
            right = self._parse_and()
            node = self._make_node('binary_op', op='||')
            node.add_child(left)
            node.add_child(right)
            left = node
        
        return left
    
    def _parse_and(self) -> STUNIRASTNode:
        """Parse AND expression."""
        left = self._parse_equality()
        
        while self._match('AND'):
            right = self._parse_equality()
            node = self._make_node('binary_op', op='&&')
            node.add_child(left)
            node.add_child(right)
            left = node
        
        return left
    
    def _parse_equality(self) -> STUNIRASTNode:
        """Parse equality expression."""
        left = self._parse_relational()
        
        while True:
            if self._match('EQ'):
                op = '=='
            elif self._match('NE'):
                op = '!='
            else:
                break
            
            right = self._parse_relational()
            node = self._make_node('binary_op', op=op)
            node.add_child(left)
            node.add_child(right)
            left = node
        
        return left
    
    def _parse_relational(self) -> STUNIRASTNode:
        """Parse relational expression."""
        left = self._parse_additive()
        
        while True:
            if self._match('LT'):
                op = '<'
            elif self._match('GT'):
                op = '>'
            elif self._match('LE'):
                op = '<='
            elif self._match('GE'):
                op = '>='
            else:
                break
            
            right = self._parse_additive()
            node = self._make_node('binary_op', op=op)
            node.add_child(left)
            node.add_child(right)
            left = node
        
        return left
    
    def _parse_additive(self) -> STUNIRASTNode:
        """Parse additive expression."""
        left = self._parse_multiplicative()
        
        while True:
            if self._match('PLUS'):
                op = '+'
            elif self._match('MINUS'):
                op = '-'
            else:
                break
            
            right = self._parse_multiplicative()
            node = self._make_node('binary_op', op=op)
            node.add_child(left)
            node.add_child(right)
            left = node
        
        return left
    
    def _parse_multiplicative(self) -> STUNIRASTNode:
        """Parse multiplicative expression."""
        left = self._parse_unary()
        
        while True:
            if self._match('STAR'):
                op = '*'
            elif self._match('SLASH'):
                op = '/'
            elif self._match('PERCENT'):
                op = '%'
            else:
                break
            
            right = self._parse_unary()
            node = self._make_node('binary_op', op=op)
            node.add_child(left)
            node.add_child(right)
            left = node
        
        return left
    
    def _parse_unary(self) -> STUNIRASTNode:
        """Parse unary expression."""
        if self._match('MINUS'):
            expr = self._parse_unary()
            node = self._make_node('unary_op', op='-')
            node.add_child(expr)
            return node
        
        if self._match('NOT'):
            expr = self._parse_unary()
            node = self._make_node('unary_op', op='!')
            node.add_child(expr)
            return node
        
        if self._match('TILDE'):
            expr = self._parse_unary()
            node = self._make_node('unary_op', op='~')
            node.add_child(expr)
            return node
        
        return self._parse_postfix()
    
    def _parse_postfix(self) -> STUNIRASTNode:
        """Parse postfix expression."""
        expr = self._parse_primary()
        
        while True:
            if self._match('DOT'):
                member = self._expect('IDENTIFIER').value
                node = self._make_node('member_access', member=member)
                node.add_child(expr)
                expr = node
            elif self._match('LBRACKET'):
                index = self._parse_expression()
                self._expect('RBRACKET')
                node = self._make_node('index_access')
                node.add_child(expr)
                node.add_child(index)
                expr = node
            elif self._match('LPAREN'):
                # Function call
                args = self._parse_arg_list()
                self._expect('RPAREN')
                node = self._make_node('call')
                node.add_child(expr)
                for arg in args:
                    node.add_child(arg)
                expr = node
            else:
                break
        
        return expr
    
    def _parse_arg_list(self) -> List[STUNIRASTNode]:
        """Parse argument list."""
        args = []
        
        if self._check('RPAREN'):
            return args
        
        args.append(self._parse_expression())
        
        while self._match('COMMA'):
            args.append(self._parse_expression())
        
        return args
    
    def _parse_primary(self) -> STUNIRASTNode:
        """Parse primary expression."""
        if self._match('INTEGER_LITERAL'):
            token = self.tokens[self.pos - 1]
            return self._make_node('literal', value=int(token.value), type='int')
        
        if self._match('FLOAT_LITERAL'):
            token = self.tokens[self.pos - 1]
            return self._make_node('literal', value=float(token.value), type='float')
        
        if self._match('STRING_LITERAL'):
            token = self.tokens[self.pos - 1]
            # Remove quotes
            value = token.value[1:-1]
            return self._make_node('literal', value=value, type='string')
        
        if self._match('KW_TRUE'):
            return self._make_node('literal', value=True, type='bool')
        
        if self._match('KW_FALSE'):
            return self._make_node('literal', value=False, type='bool')
        
        if self._match('KW_NULL'):
            return self._make_node('literal', value=None, type='null')
        
        if self._match('IDENTIFIER'):
            token = self.tokens[self.pos - 1]
            return self._make_node('identifier', name=token.value)
        
        if self._match('LPAREN'):
            expr = self._parse_expression()
            self._expect('RPAREN')
            return expr
        
        if self._match('LBRACKET'):
            # Array literal
            node = self._make_node('array_literal')
            
            if not self._check('RBRACKET'):
                while True:
                    elem = self._parse_expression()
                    node.add_child(elem)
                    if not self._match('COMMA'):
                        break
            
            self._expect('RBRACKET')
            return node
        
        if self._match('LBRACE'):
            # Object literal
            node = self._make_node('object_literal')
            
            if not self._check('RBRACE'):
                while True:
                    key = self._expect('IDENTIFIER').value
                    self._expect('COLON')
                    value = self._parse_expression()
                    
                    field_node = self._make_node('field', name=key)
                    field_node.add_child(value)
                    node.add_child(field_node)
                    
                    if not self._match('COMMA'):
                        break
            
            self._expect('RBRACE')
            return node
        
        # Error
        token = self._current()
        raise STUNIRParseError(
            f"Unexpected token {token.type}",
            token.line, token.column
        )
    
    def _parse_ir_def(self) -> STUNIRASTNode:
        """Parse IR definition."""
        self._expect('KW_IR')
        
        name = self._expect('IDENTIFIER').value
        node = self._make_node('ir_def', name=name)
        
        # Optional type parameters
        if self._match('LT'):
            params = self._parse_identifier_list()
            node.set_attr('type_params', params)
            self._expect('GT')
        
        self._expect('LBRACE')
        
        while not self._check('RBRACE', 'EOF'):
            member = self._parse_ir_member()
            if member:
                node.add_child(member)
        
        self._expect('RBRACE')
        return node
    
    def _parse_ir_member(self) -> Optional[STUNIRASTNode]:
        """Parse IR member (field, child, or op)."""
        if self._match('KW_CHILD'):
            name = self._expect('IDENTIFIER').value
            self._expect('COLON')
            type_expr = self._parse_type_expr()
            self._expect('SEMICOLON')
            return self._make_node('ir_child', name=name, type=type_expr)
        
        # Check for op keyword - but if followed by COLON, it's a field named 'op'
        if self._check('KW_OP'):
            # Peek ahead to see if it's 'op: type' (field) or 'op name(' (operation)
            if self._peek(1).type == 'COLON':
                # It's a field named 'op'
                self._advance()  # consume KW_OP
                self._expect('COLON')
                type_expr = self._parse_type_expr()
                self._expect('SEMICOLON')
                return self._make_node('ir_field', name='op', type=type_expr)
            else:
                # It's an operation definition
                self._advance()  # consume KW_OP
                name = self._expect('IDENTIFIER').value
                self._expect('LPAREN')
                params = self._parse_param_list()
                self._expect('RPAREN')
                
                return_type = None
                if self._match('COLON'):
                    return_type = self._parse_type_expr()
                
                self._expect('SEMICOLON')
                return self._make_node('ir_op', name=name, params=params, return_type=return_type)
        
        if self._check('IDENTIFIER'):
            name = self._advance().value
            self._expect('COLON')
            type_expr = self._parse_type_expr()
            self._expect('SEMICOLON')
            return self._make_node('ir_field', name=name, type=type_expr)
        
        return None
    
    def _parse_target_def(self) -> STUNIRASTNode:
        """Parse target definition."""
        self._expect('KW_TARGET')
        
        name = self._expect('IDENTIFIER').value
        node = self._make_node('target_def', name=name)
        
        self._expect('LBRACE')
        
        while not self._check('RBRACE', 'EOF'):
            member = self._parse_target_member()
            if member:
                node.add_child(member)
        
        self._expect('RBRACE')
        return node
    
    def _parse_target_member(self) -> Optional[STUNIRASTNode]:
        """Parse target member (option or emit rule)."""
        if self._match('KW_EMIT'):
            name = self._expect('IDENTIFIER').value
            self._expect('LPAREN')
            params = self._parse_param_list()
            self._expect('RPAREN')
            body = self._parse_block()
            
            node = self._make_node('emit_rule', name=name, params=params)
            node.add_child(body)
            return node
        
        if self._check('IDENTIFIER'):
            name = self._advance().value
            self._expect('COLON')
            value = self._parse_expression()
            self._expect('SEMICOLON')
            
            node = self._make_node('target_option', name=name)
            node.add_child(value)
            return node
        
        return None


class BootstrapCompiler:
    """
    Bootstrap compiler for STUNIR source files.
    
    Uses the SimpleLexer and RecursiveDescentParser to parse
    STUNIR source files and produce an AST.
    
    Usage:
        compiler = BootstrapCompiler()
        result = compiler.parse(source_code)
        
        if result.success:
            ast = result.ast
            # Process AST
    """
    
    def __init__(self):
        """Initialize the bootstrap compiler."""
        self._errors: List[str] = []
        self._warnings: List[str] = []
    
    def parse(self, source: str, filename: str = "<input>") -> BootstrapResult:
        """
        Parse STUNIR source code.
        
        Args:
            source: STUNIR source code
            filename: Source filename (for error messages)
            
        Returns:
            BootstrapResult with AST or errors
        """
        result = BootstrapResult(success=False)
        
        try:
            # Tokenize
            lexer = SimpleLexer(source)
            tokens = list(lexer.tokenize())
            result.tokens = tokens
            
            # Parse
            parser = RecursiveDescentParser(tokens)
            ast = parser.parse()
            
            result.ast = ast
            result.errors = parser._errors
            result.success = len(parser._errors) == 0
            
        except STUNIRLexerError as e:
            result.errors.append(f"{filename}:{e.line}:{e.column}: Lexer error: {e}")
        except STUNIRParseError as e:
            result.errors.append(f"{filename}:{e.line}:{e.column}: Parse error: {e}")
        except Exception as e:
            result.errors.append(f"{filename}: Internal error: {e}")
        
        return result
    
    def parse_file(self, path: Path) -> BootstrapResult:
        """
        Parse STUNIR source file.
        
        Args:
            path: Path to source file
            
        Returns:
            BootstrapResult with AST or errors
        """
        path = Path(path)
        
        if not path.exists():
            return BootstrapResult(
                success=False,
                errors=[f"File not found: {path}"]
            )
        
        source = path.read_text()
        return self.parse(source, str(path))
    
    def validate_ast(self, ast: STUNIRASTNode) -> List[str]:
        """
        Validate AST structure.
        
        Args:
            ast: Root AST node
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Check root is program
        if ast.kind != 'program':
            errors.append(f"Expected 'program' root, got '{ast.kind}'")
            return errors
        
        # Check has module declaration
        module_decls = ast.find_children('module_decl')
        if len(module_decls) == 0:
            errors.append("Missing module declaration")
        elif len(module_decls) > 1:
            errors.append("Multiple module declarations")
        
        return errors

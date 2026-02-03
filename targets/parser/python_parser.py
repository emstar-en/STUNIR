#!/usr/bin/env python3
"""Python parser code emitter.

Generates Python parser code from parse tables.
"""

from typing import Dict, List, Optional, Any, Union

from targets.parser.base import ParserEmitterBase, ParserEmitterResult

# Import from parser module
try:
    from ir.parser.parse_table import ParseTable, LL1Table, ParserType, Action, ActionType
    from ir.parser.ast_node import ASTSchema, ASTNodeSpec
    from ir.parser.parser_generator import ParserGeneratorResult
    from ir.grammar.grammar_ir import Grammar
    from ir.grammar.symbol import Symbol
except ImportError:
    ParseTable = Any
    LL1Table = Any
    ParserType = Any
    ASTSchema = Any
    ParserGeneratorResult = Any
    Grammar = Any


class PythonParserEmitter(ParserEmitterBase):
    """Python parser code emitter.
    
    Generates Python 3 parser code with dataclass-based AST nodes.
    """
    
    LANGUAGE = "python"
    FILE_EXTENSION = ".py"
    
    def _get_comment_style(self) -> Dict[str, str]:
        """Get Python comment style."""
        return {
            'start': '"""',
            'line': '',
            'end': '"""',
        }
    
    def emit(self, result: ParserGeneratorResult, 
             grammar: Grammar) -> ParserEmitterResult:
        """Emit Python parser code.
        
        Args:
            result: Parser generator result
            grammar: Source grammar
        
        Returns:
            ParserEmitterResult with Python code
        """
        # Generate parser code
        code_parts = [
            self._emit_header(grammar),
            self._emit_imports(),
            "",
            self._emit_token_class(),
            "",
            self.emit_parse_table(result.parse_table),
            "",
            self._emit_parser_class(result, grammar),
        ]
        code = "\n".join(code_parts)
        
        # Generate AST code
        ast_code = ""
        if result.ast_schema:
            ast_code = self.emit_ast_nodes(result.ast_schema)
        
        # Generate manifest
        manifest = self._generate_manifest(code, ast_code, grammar)
        
        return ParserEmitterResult(
            code=code,
            ast_code=ast_code,
            manifest=manifest,
            warnings=self._get_warnings()
        )
    
    def _emit_imports(self) -> str:
        """Generate import statements."""
        return '''from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union, Callable
from enum import Enum, auto'''
    
    def _emit_token_class(self) -> str:
        """Generate Token class."""
        return '''
@dataclass
class Token:
    """A lexical token."""
    type: str
    value: str
    line: int = 0
    column: int = 0
    
    def __str__(self) -> str:
        return f"Token({self.type}, {self.value!r})"


class TokenType(Enum):
    """Token types."""
    EOF = auto()
    ERROR = auto()
'''
    
    def emit_parse_table(self, table: Union[ParseTable, LL1Table]) -> str:
        """Emit parse table as Python data structures.
        
        Args:
            table: Parse table to emit
        
        Returns:
            Python code defining the parse tables
        """
        lines = ["# Parse Tables"]
        
        if isinstance(table, ParseTable):
            lines.extend(self._emit_lr_table(table))
        else:
            lines.extend(self._emit_ll_table(table))
        
        return "\n".join(lines)
    
    def _emit_lr_table(self, table: ParseTable) -> List[str]:
        """Emit LR parse table."""
        lines = []
        
        # Emit productions
        lines.append("PRODUCTIONS = [")
        for i, prod in enumerate(table.productions):
            head = prod.head.name if hasattr(prod.head, 'name') else str(prod.head)
            body_len = len(prod.body) if prod.body else 0
            body_syms = [s.name if hasattr(s, 'name') else str(s) for s in (prod.body or [])]
            lines.append(f"    ({i}, {head!r}, {body_len}, {body_syms}),  # {prod}")
        lines.append("]")
        lines.append("")
        
        # Emit ACTION table
        lines.append("ACTION = {")
        for (state, sym), action in sorted(table.action.items()):
            sym_name = sym.name if hasattr(sym, 'name') else str(sym)
            action_str = str(action)
            lines.append(f"    ({state}, {sym_name!r}): {action_str!r},")
        lines.append("}")
        lines.append("")
        
        # Emit GOTO table
        lines.append("GOTO = {")
        for (state, sym), target in sorted(table.goto.items()):
            sym_name = sym.name if hasattr(sym, 'name') else str(sym)
            lines.append(f"    ({state}, {sym_name!r}): {target},")
        lines.append("}")
        
        return lines
    
    def _emit_ll_table(self, table: LL1Table) -> List[str]:
        """Emit LL(1) parse table."""
        lines = []
        
        lines.append("LL_TABLE = {")
        for (nt, t), prod in sorted(table.table.items(), key=lambda x: (str(x[0][0]), str(x[0][1]))):
            nt_name = nt.name if hasattr(nt, 'name') else str(nt)
            t_name = t.name if hasattr(t, 'name') else str(t)
            body_syms = [s.name if hasattr(s, 'name') else str(s) for s in (prod.body or [])]
            lines.append(f"    ({nt_name!r}, {t_name!r}): {body_syms},")
        lines.append("}")
        
        return lines
    
    def _emit_parser_class(self, result: ParserGeneratorResult, grammar: Grammar) -> str:
        """Generate Parser class."""
        parser_type = result.parser_type
        
        if parser_type in (ParserType.LR0, ParserType.SLR1, ParserType.LALR1, ParserType.LR1):
            return self._emit_lr_parser_class(grammar)
        else:
            return self._emit_ll_parser_class(grammar)
    
    def _emit_lr_parser_class(self, grammar: Grammar) -> str:
        """Generate LR parser class."""
        return '''
class ParseError(Exception):
    """Parser error with location info."""
    def __init__(self, message: str, token: Optional[Token] = None):
        self.token = token
        loc = f" at line {token.line}, column {token.column}" if token else ""
        super().__init__(f"{message}{loc}")


class Parser:
    """LR Parser.
    
    Generated from grammar: ''' + (grammar.name if hasattr(grammar, 'name') else 'unknown') + '''
    """
    
    def __init__(self, tokens: List[Token]):
        """Initialize parser with token list.
        
        Args:
            tokens: List of tokens to parse
        """
        self.tokens = tokens
        self.pos = 0
        self.stack: List[Any] = [0]  # State stack
        self.value_stack: List[Any] = []  # Value/AST stack
    
    def current_token(self) -> Token:
        """Get current token."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return Token("$", "", -1, -1)  # EOF token
    
    def advance(self) -> Token:
        """Advance to next token and return current."""
        token = self.current_token()
        if self.pos < len(self.tokens):
            self.pos += 1
        return token
    
    def parse(self) -> Any:
        """Parse the token stream.
        
        Returns:
            The parsed AST or value
        
        Raises:
            ParseError: If parsing fails
        """
        while True:
            state = self.stack[-1]
            token = self.current_token()
            
            action_key = (state, token.type)
            action = ACTION.get(action_key)
            
            if action is None:
                # Try EOF
                action = ACTION.get((state, "$"))
                if action is None:
                    raise ParseError(f"Unexpected token: {token.type}", token)
            
            if action.startswith("s"):
                # Shift
                next_state = int(action[1:])
                self.stack.append(next_state)
                self.value_stack.append(token)
                self.advance()
            
            elif action.startswith("r"):
                # Reduce
                prod_index = int(action[1:])
                prod_idx, head, body_len, body_syms = PRODUCTIONS[prod_index]
                
                # Pop states and values
                values = []
                for _ in range(body_len):
                    self.stack.pop()
                    if self.value_stack:
                        values.insert(0, self.value_stack.pop())
                
                # Get goto state
                state = self.stack[-1]
                goto_state = GOTO.get((state, head))
                if goto_state is None:
                    raise ParseError(f"No GOTO for ({state}, {head})")
                
                self.stack.append(goto_state)
                
                # Create AST node or use semantic action
                result = self._reduce_action(prod_index, values)
                self.value_stack.append(result)
            
            elif action == "acc":
                # Accept
                if self.value_stack:
                    return self.value_stack[-1]
                return None
            
            else:
                raise ParseError(f"Unknown action: {action}")
    
    def _reduce_action(self, prod_index: int, values: List[Any]) -> Any:
        """Apply semantic action for reduction.
        
        Override this method to provide custom semantic actions.
        
        Args:
            prod_index: Production index
            values: Values from the RHS
        
        Returns:
            Result of the semantic action
        """
        # Default: return values as tuple or single value
        if len(values) == 0:
            return None
        elif len(values) == 1:
            return values[0]
        else:
            return tuple(values)
'''
    
    def _emit_ll_parser_class(self, grammar: Grammar) -> str:
        """Generate LL(1) parser class."""
        return '''
class ParseError(Exception):
    """Parser error with location info."""
    def __init__(self, message: str, token: Optional[Token] = None):
        self.token = token
        loc = f" at line {token.line}, column {token.column}" if token else ""
        super().__init__(f"{message}{loc}")


class Parser:
    """LL(1) Parser.
    
    Generated from grammar: ''' + (grammar.name if hasattr(grammar, 'name') else 'unknown') + '''
    """
    
    def __init__(self, tokens: List[Token]):
        """Initialize parser with token list.
        
        Args:
            tokens: List of tokens to parse
        """
        self.tokens = tokens
        self.pos = 0
        self.stack: List[str] = []  # Symbol stack
    
    def current_token(self) -> Token:
        """Get current token."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return Token("$", "", -1, -1)  # EOF token
    
    def advance(self) -> Token:
        """Advance to next token and return current."""
        token = self.current_token()
        if self.pos < len(self.tokens):
            self.pos += 1
        return token
    
    def match(self, expected: str) -> Token:
        """Match and consume expected token.
        
        Args:
            expected: Expected token type
        
        Returns:
            The matched token
        
        Raises:
            ParseError: If token doesn\'t match
        """
        token = self.current_token()
        if token.type == expected:
            return self.advance()
        raise ParseError(f"Expected {expected}, got {token.type}", token)
    
    def parse(self, start_symbol: str) -> Any:
        """Parse the token stream using predictive parsing.
        
        Args:
            start_symbol: The start symbol
        
        Returns:
            The parsed result
        
        Raises:
            ParseError: If parsing fails
        """
        self.stack = ["$", start_symbol]
        result_stack: List[Any] = []
        
        while self.stack:
            top = self.stack.pop()
            token = self.current_token()
            
            if top == "$":
                if token.type == "$":
                    break
                else:
                    raise ParseError(f"Expected end of input, got {token.type}", token)
            
            if top == token.type:
                # Terminal: match
                result_stack.append(self.advance())
            
            else:
                # Non-terminal: consult table
                production = LL_TABLE.get((top, token.type))
                if production is None:
                    raise ParseError(f"No production for ({top}, {token.type})", token)
                
                # Push RHS in reverse order
                for symbol in reversed(production):
                    self.stack.append(symbol)
        
        return result_stack[0] if result_stack else None
'''
    
    def emit_ast_nodes(self, schema: ASTSchema) -> str:
        """Emit AST node definitions as Python dataclasses.
        
        Args:
            schema: AST schema
        
        Returns:
            Python code with dataclass definitions
        """
        lines = [
            '"""AST Node Definitions',
            '',
            'Generated by STUNIR Parser Emitter',
            '"""',
            '',
            'from dataclasses import dataclass, field',
            'from typing import List, Optional, Any, Union',
            '',
            '',
            '@dataclass',
            f'class {schema.base_node_name}:',
            '    """Base class for all AST nodes."""',
            '    pass',
            '',
        ]
        
        for node in schema.nodes:
            lines.extend(self._emit_ast_node(node, schema))
            lines.append('')
        
        return '\n'.join(lines)
    
    def _emit_ast_node(self, node: ASTNodeSpec, schema: ASTSchema) -> List[str]:
        """Emit a single AST node class."""
        lines = []
        
        # Decorator
        lines.append('@dataclass')
        
        # Class definition
        base = node.base_class if node.base_class else schema.base_node_name
        lines.append(f'class {node.name}({base}):')
        
        # Docstring
        if node.production:
            lines.append(f'    """AST node for: {node.production}"""')
        else:
            lines.append(f'    """AST node: {node.name}"""')
        
        # Fields
        if node.fields:
            for fname, ftype in node.fields:
                # Map types
                py_type = self._map_ast_type(ftype, schema)
                lines.append(f'    {fname}: {py_type}')
        else:
            lines.append('    pass')
        
        return lines
    
    def _map_ast_type(self, ir_type: str, schema: ASTSchema) -> str:
        """Map IR type to Python type."""
        if ir_type == schema.token_type_name:
            return "Token"
        elif schema.get_node(ir_type):
            return ir_type
        else:
            return f"Optional[{ir_type}]"

"""
Lexer Generator Module for STUNIR.

Main interface for lexer generation. Combines:
- Token specifications
- Regex parsing
- NFA construction (Thompson's)
- DFA construction (Subset construction)
- DFA minimization (Hopcroft's algorithm)
- Lexer simulation
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from .token_spec import LexerSpec, TokenSpec, Token, LexerError
from .regex import RegexParser, RegexError
from .nfa import NFABuilder, NFA, combine_nfas
from .dfa import (
    DFA, MinimizedDFA, TransitionTable,
    SubsetConstruction, HopcroftMinimizer
)

if TYPE_CHECKING:
    from targets.lexer.base import BaseLexerEmitter


class LexerGenerator:
    """
    Main lexer generator interface.
    
    Generates a minimized DFA from token specifications that can be
    used for efficient lexical analysis.
    
    Usage:
        spec = LexerSpec("MyLexer", [
            TokenSpec("INT", "[0-9]+"),
            TokenSpec("ID", "[a-z]+"),
            TokenSpec("WS", "[ \\t\\n]+", skip=True)
        ])
        gen = LexerGenerator(spec)
        dfa = gen.generate()
        
        # Use emitter to generate code
        emitter = PythonLexerEmitter()
        code = gen.emit(emitter)
    """
    
    def __init__(self, spec: LexerSpec):
        """
        Initialize lexer generator.
        
        Args:
            spec: Lexer specification
        """
        self.spec = spec
        self.nfa: Optional[NFA] = None
        self.dfa: Optional[DFA] = None
        self.minimized_dfa: Optional[MinimizedDFA] = None
        self.table: Optional[TransitionTable] = None
        self._errors: List[str] = []
    
    def validate(self) -> List[str]:
        """
        Validate lexer specification.
        
        Returns:
            List of error messages (empty if valid)
        """
        errors = self.spec.validate()
        
        # Validate regex patterns
        for token in self.spec.tokens:
            try:
                parser = RegexParser(token.pattern)
                parser.parse()
            except RegexError as e:
                errors.append(f"Invalid pattern for {token.name}: {e}")
        
        return errors
    
    def generate(self) -> MinimizedDFA:
        """
        Generate minimized DFA from token specifications.
        
        Returns:
            MinimizedDFA ready for lexer code generation
            
        Raises:
            ValueError: If specification is invalid
        """
        # Validate
        errors = self.validate()
        if errors:
            raise ValueError(f"Invalid lexer spec: {'; '.join(errors)}")
        
        # Build NFA for each token
        nfas: List[NFA] = []
        for i, token in enumerate(self.spec.tokens):
            try:
                # Parse regex
                parser = RegexParser(token.pattern)
                ast = parser.parse()
                
                # Build NFA
                builder = NFABuilder()
                # Priority: explicit priority * 1000 + definition order (earlier = higher)
                effective_priority = token.priority * 1000 + (len(self.spec.tokens) - i)
                nfa = builder.build(ast, token.name, effective_priority)
                nfas.append(nfa)
            except RegexError as e:
                raise ValueError(f"Failed to parse pattern for {token.name}: {e}")
        
        # Combine all NFAs
        self.nfa = combine_nfas(nfas)
        
        # Convert to DFA
        converter = SubsetConstruction(self.nfa)
        self.dfa = converter.convert()
        
        # Minimize DFA
        minimizer = HopcroftMinimizer(self.dfa)
        self.minimized_dfa = minimizer.minimize()
        
        # Build transition table
        self.table = self.minimized_dfa.to_table()
        
        return self.minimized_dfa
    
    def get_skip_tokens(self) -> Set[str]:
        """Get set of tokens that should be skipped."""
        return self.spec.get_skip_tokens()
    
    def emit(self, emitter: 'BaseLexerEmitter') -> str:
        """
        Emit lexer code using specified emitter.
        
        Args:
            emitter: Lexer code emitter
            
        Returns:
            Generated lexer code
        """
        if self.minimized_dfa is None:
            self.generate()
        
        return emitter.emit(self.spec, self.minimized_dfa, self.table)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the generated lexer.
        
        Returns:
            Dictionary with statistics
        """
        if self.minimized_dfa is None:
            self.generate()
        
        return {
            "lexer_name": self.spec.name,
            "num_tokens": len(self.spec.tokens),
            "num_skip_tokens": len(self.get_skip_tokens()),
            "nfa_states": len(self.nfa.states) if self.nfa else 0,
            "dfa_states": len(self.dfa.states) if self.dfa else 0,
            "minimized_states": self.minimized_dfa.num_states if self.minimized_dfa else 0,
            "alphabet_size": len(self.minimized_dfa.alphabet) if self.minimized_dfa else 0,
            "table_size": len(self.table.table) if self.table else 0
        }


class LexerSimulator:
    """
    Simulates DFA-based lexer with longest match semantics.
    
    Implements the maximal munch algorithm for tokenization.
    """
    
    def __init__(self, table: TransitionTable, skip_tokens: Set[str]):
        """
        Initialize lexer simulator.
        
        Args:
            table: DFA transition table
            skip_tokens: Set of token names to skip
        """
        self.table = table
        self.skip_tokens = skip_tokens
    
    def tokenize(self, input_str: str) -> List[Token]:
        """
        Tokenize input string using longest match rule.
        
        Args:
            input_str: Input to tokenize
            
        Returns:
            List of tokens (excluding skip tokens)
            
        Raises:
            LexerError: If unexpected character encountered
        """
        tokens: List[Token] = []
        pos = 0
        line = 1
        col = 1
        
        while pos < len(input_str):
            # Find longest match
            token, consumed, new_line, new_col = self._next_token(
                input_str, pos, line, col
            )
            
            if token is not None and token.type not in self.skip_tokens:
                tokens.append(token)
            
            # Update position
            pos += consumed
            line = new_line
            col = new_col
        
        return tokens
    
    def _next_token(
        self, 
        input_str: str, 
        start_pos: int, 
        start_line: int, 
        start_col: int
    ) -> tuple[Optional[Token], int, int, int]:
        """
        Get next token using longest match.
        
        Returns:
            (token, chars_consumed, new_line, new_col)
        """
        state = self.table.start_state
        last_accept_pos = -1
        last_accept_info = None
        current_pos = start_pos
        
        # Scan for longest match
        while current_pos < len(input_str):
            char = input_str[current_pos]
            next_state = self.table.next_state(state, char)
            
            if next_state == self.table.error_state:
                break
            
            state = next_state
            current_pos += 1
            
            # Record if this is an accept state
            if self.table.is_accept(state):
                last_accept_pos = current_pos
                last_accept_info = self.table.get_token(state)
        
        if last_accept_pos < 0:
            # No match found
            char = input_str[start_pos] if start_pos < len(input_str) else 'EOF'
            raise LexerError(
                f"Unexpected character '{char}'",
                line=start_line,
                column=start_col
            )
        
        # Extract matched text
        lexeme = input_str[start_pos:last_accept_pos]
        token_name, _ = last_accept_info
        
        # Create token
        token = Token(
            type=token_name,
            value=lexeme,
            line=start_line,
            column=start_col
        )
        
        # Calculate new position
        new_line = start_line
        new_col = start_col
        for c in lexeme:
            if c == '\n':
                new_line += 1
                new_col = 1
            else:
                new_col += 1
        
        return token, len(lexeme), new_line, new_col
    
    def tokenize_with_positions(self, input_str: str) -> List[Token]:
        """
        Tokenize and include all tokens (including skip tokens).
        
        Useful for syntax highlighting.
        """
        tokens: List[Token] = []
        pos = 0
        line = 1
        col = 1
        
        while pos < len(input_str):
            token, consumed, new_line, new_col = self._next_token(
                input_str, pos, line, col
            )
            
            if token is not None:
                tokens.append(token)
            
            pos += consumed
            line = new_line
            col = new_col
        
        return tokens


def create_lexer(spec: LexerSpec) -> LexerSimulator:
    """
    Convenience function to create a lexer from specification.
    
    Args:
        spec: Lexer specification
        
    Returns:
        Ready-to-use lexer simulator
    """
    gen = LexerGenerator(spec)
    gen.generate()
    return LexerSimulator(gen.table, gen.get_skip_tokens())


def tokenize(spec: LexerSpec, input_str: str) -> List[Token]:
    """
    Convenience function to tokenize input with a specification.
    
    Args:
        spec: Lexer specification
        input_str: Input to tokenize
        
    Returns:
        List of tokens
    """
    lexer = create_lexer(spec)
    return lexer.tokenize(input_str)

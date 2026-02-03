"""
Regular Expression Parser for STUNIR Lexer Generator.

Parses regular expressions into an AST representation that can be
converted to NFA using Thompson's construction.

Supported syntax:
- Literal characters
- . (any character except newline)
- Character classes: [abc], [a-z], [^abc]
- Escapes: \\d, \\w, \\s, \\\\, \\., etc.
- Quantifiers: *, +, ?
- Grouping: ()
- Alternation: |
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .nfa import NFABuilder, NFAState


# Character sets for escape sequences
DIGITS = frozenset('0123456789')
WORD_CHARS = frozenset(
    'abcdefghijklmnopqrstuvwxyz'
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    '0123456789_'
)
WHITESPACE = frozenset(' \t\n\r\f\v')

# All printable ASCII characters (for negated classes)
ALL_PRINTABLE = frozenset(chr(i) for i in range(32, 127))


class RegexNode(ABC):
    """Base class for regex AST nodes."""
    
    @abstractmethod
    def to_nfa(self, builder: 'NFABuilder') -> Tuple['NFAState', 'NFAState']:
        """
        Convert regex node to NFA fragment.
        
        Returns:
            Tuple of (start_state, end_state)
        """
        pass
    
    @abstractmethod
    def __repr__(self) -> str:
        pass


@dataclass
class EpsilonNode(RegexNode):
    """Matches empty string (epsilon)."""
    
    def to_nfa(self, builder: 'NFABuilder') -> Tuple['NFAState', 'NFAState']:
        start = builder.new_state()
        end = builder.new_state()
        start.add_epsilon(end)
        return start, end
    
    def __repr__(self) -> str:
        return "Epsilon()"


@dataclass
class CharNode(RegexNode):
    """Matches a single character."""
    char: str
    
    def to_nfa(self, builder: 'NFABuilder') -> Tuple['NFAState', 'NFAState']:
        start = builder.new_state()
        end = builder.new_state()
        start.add_transition(self.char, end)
        builder.add_to_alphabet(self.char)
        return start, end
    
    def __repr__(self) -> str:
        return f"Char({self.char!r})"


@dataclass
class AnyCharNode(RegexNode):
    """Matches any character (.)."""
    exclude_newline: bool = True
    
    def to_nfa(self, builder: 'NFABuilder') -> Tuple['NFAState', 'NFAState']:
        start = builder.new_state()
        end = builder.new_state()
        chars = ALL_PRINTABLE
        if self.exclude_newline:
            chars = chars - {'\n'}
        for c in chars:
            start.add_transition(c, end)
            builder.add_to_alphabet(c)
        return start, end
    
    def __repr__(self) -> str:
        return "AnyChar()"


@dataclass
class CharClassNode(RegexNode):
    """
    Matches a character class [abc] or [a-z].
    
    Attributes:
        chars: Set of characters in the class
        negated: If True, matches any character NOT in the class
    """
    chars: FrozenSet[str]
    negated: bool = False
    
    def to_nfa(self, builder: 'NFABuilder') -> Tuple['NFAState', 'NFAState']:
        start = builder.new_state()
        end = builder.new_state()
        
        if self.negated:
            match_chars = ALL_PRINTABLE - self.chars
        else:
            match_chars = self.chars
        
        for c in match_chars:
            start.add_transition(c, end)
            builder.add_to_alphabet(c)
        
        return start, end
    
    def __repr__(self) -> str:
        prefix = "^" if self.negated else ""
        return f"CharClass({prefix}{set(self.chars)})"


@dataclass
class ConcatNode(RegexNode):
    """Concatenation: AB (A followed by B)."""
    left: RegexNode
    right: RegexNode
    
    def to_nfa(self, builder: 'NFABuilder') -> Tuple['NFAState', 'NFAState']:
        s1, e1 = self.left.to_nfa(builder)
        s2, e2 = self.right.to_nfa(builder)
        e1.add_epsilon(s2)
        return s1, e2
    
    def __repr__(self) -> str:
        return f"Concat({self.left}, {self.right})"


@dataclass
class UnionNode(RegexNode):
    """Alternation: A|B (A or B)."""
    left: RegexNode
    right: RegexNode
    
    def to_nfa(self, builder: 'NFABuilder') -> Tuple['NFAState', 'NFAState']:
        start = builder.new_state()
        end = builder.new_state()
        s1, e1 = self.left.to_nfa(builder)
        s2, e2 = self.right.to_nfa(builder)
        start.add_epsilon(s1)
        start.add_epsilon(s2)
        e1.add_epsilon(end)
        e2.add_epsilon(end)
        return start, end
    
    def __repr__(self) -> str:
        return f"Union({self.left}, {self.right})"


@dataclass
class StarNode(RegexNode):
    """Kleene star: A* (zero or more A)."""
    child: RegexNode
    
    def to_nfa(self, builder: 'NFABuilder') -> Tuple['NFAState', 'NFAState']:
        start = builder.new_state()
        end = builder.new_state()
        s, e = self.child.to_nfa(builder)
        start.add_epsilon(s)
        start.add_epsilon(end)
        e.add_epsilon(s)
        e.add_epsilon(end)
        return start, end
    
    def __repr__(self) -> str:
        return f"Star({self.child})"


@dataclass
class PlusNode(RegexNode):
    """One or more: A+ (one or more A)."""
    child: RegexNode
    
    def to_nfa(self, builder: 'NFABuilder') -> Tuple['NFAState', 'NFAState']:
        start = builder.new_state()
        end = builder.new_state()
        s, e = self.child.to_nfa(builder)
        start.add_epsilon(s)
        e.add_epsilon(s)
        e.add_epsilon(end)
        return start, end
    
    def __repr__(self) -> str:
        return f"Plus({self.child})"


@dataclass
class OptionalNode(RegexNode):
    """Optional: A? (zero or one A)."""
    child: RegexNode
    
    def to_nfa(self, builder: 'NFABuilder') -> Tuple['NFAState', 'NFAState']:
        start = builder.new_state()
        end = builder.new_state()
        s, e = self.child.to_nfa(builder)
        start.add_epsilon(s)
        start.add_epsilon(end)
        e.add_epsilon(end)
        return start, end
    
    def __repr__(self) -> str:
        return f"Optional({self.child})"


class RegexError(Exception):
    """Exception raised for regex parsing errors."""
    pass


class RegexParser:
    """
    Parse regex strings into AST.
    
    Supports:
    - Literal characters
    - . (any character)
    - Character classes: [abc], [a-z], [^abc]
    - Escapes: \\d, \\w, \\s, \\\\, \\., etc.
    - Quantifiers: *, +, ?
    - Grouping: ()
    - Alternation: |
    """
    
    def __init__(self, pattern: str):
        self.pattern = pattern
        self.pos = 0
        self.length = len(pattern)
    
    def parse(self) -> RegexNode:
        """Parse the regex pattern into an AST."""
        if not self.pattern:
            return EpsilonNode()
        
        result = self._parse_alternation()
        
        if self.pos < self.length:
            raise RegexError(f"Unexpected character at position {self.pos}: {self.pattern[self.pos]!r}")
        
        return result
    
    def _peek(self) -> Optional[str]:
        """Look at current character without consuming."""
        if self.pos < self.length:
            return self.pattern[self.pos]
        return None
    
    def _next(self) -> str:
        """Consume and return current character."""
        if self.pos >= self.length:
            raise RegexError("Unexpected end of pattern")
        char = self.pattern[self.pos]
        self.pos += 1
        return char
    
    def _match(self, char: str) -> bool:
        """Consume character if it matches."""
        if self._peek() == char:
            self.pos += 1
            return True
        return False
    
    def _expect(self, char: str) -> None:
        """Consume character or raise error."""
        if not self._match(char):
            actual = self._peek() or 'end of pattern'
            raise RegexError(f"Expected {char!r}, got {actual!r} at position {self.pos}")
    
    def _parse_alternation(self) -> RegexNode:
        """Parse alternation: A|B."""
        left = self._parse_concatenation()
        
        while self._match('|'):
            right = self._parse_concatenation()
            left = UnionNode(left, right)
        
        return left
    
    def _parse_concatenation(self) -> RegexNode:
        """Parse concatenation: AB."""
        nodes: List[RegexNode] = []
        
        while self.pos < self.length and self._peek() not in '|)':
            nodes.append(self._parse_quantified())
        
        if not nodes:
            return EpsilonNode()
        
        result = nodes[0]
        for node in nodes[1:]:
            result = ConcatNode(result, node)
        
        return result
    
    def _parse_quantified(self) -> RegexNode:
        """Parse quantified expression: A*, A+, A?."""
        node = self._parse_atom()
        
        if self._match('*'):
            return StarNode(node)
        elif self._match('+'):
            return PlusNode(node)
        elif self._match('?'):
            return OptionalNode(node)
        
        return node
    
    def _parse_atom(self) -> RegexNode:
        """Parse atomic expression."""
        char = self._peek()
        
        if char == '(':
            self._next()
            node = self._parse_alternation()
            self._expect(')')
            return node
        elif char == '[':
            return self._parse_char_class()
        elif char == '\\':
            self._next()
            return self._parse_escape()
        elif char == '.':
            self._next()
            return AnyCharNode()
        elif char in '*+?|)':
            raise RegexError(f"Unexpected metacharacter {char!r} at position {self.pos}")
        else:
            self._next()
            return CharNode(char)
    
    def _parse_char_class(self) -> RegexNode:
        """Parse character class [abc] or [a-z] or [^abc]."""
        self._expect('[')
        
        negated = self._match('^')
        chars: Set[str] = set()
        
        # Handle ] as first character (literal)
        if self._peek() == ']':
            chars.add(self._next())
        
        while self._peek() != ']':
            if self._peek() is None:
                raise RegexError("Unclosed character class")
            
            char = self._next()
            
            if char == '\\':
                # Handle escape in char class
                chars.update(self._parse_char_class_escape())
            elif self._peek() == '-' and self.pos + 1 < self.length and self.pattern[self.pos + 1] != ']':
                # Range: a-z
                self._next()  # consume '-'
                end_char = self._next()
                if end_char == '\\':
                    end_chars = self._parse_char_class_escape()
                    if len(end_chars) != 1:
                        raise RegexError("Invalid range endpoint")
                    end_char = next(iter(end_chars))
                
                if ord(char) > ord(end_char):
                    raise RegexError(f"Invalid range: {char}-{end_char}")
                
                for c in range(ord(char), ord(end_char) + 1):
                    chars.add(chr(c))
            else:
                chars.add(char)
        
        self._expect(']')
        
        return CharClassNode(frozenset(chars), negated)
    
    def _parse_char_class_escape(self) -> Set[str]:
        """Parse escape sequence inside character class."""
        if self.pos >= self.length:
            raise RegexError("Incomplete escape sequence")
        
        char = self._next()
        
        # Character class shortcuts
        if char == 'd':
            return set(DIGITS)
        elif char == 'D':
            return set(ALL_PRINTABLE - DIGITS)
        elif char == 'w':
            return set(WORD_CHARS)
        elif char == 'W':
            return set(ALL_PRINTABLE - WORD_CHARS)
        elif char == 's':
            return set(WHITESPACE)
        elif char == 'S':
            return set(ALL_PRINTABLE - WHITESPACE)
        elif char == 'n':
            return {'\n'}
        elif char == 'r':
            return {'\r'}
        elif char == 't':
            return {'\t'}
        elif char == 'f':
            return {'\f'}
        elif char == 'v':
            return {'\v'}
        else:
            # Literal escape
            return {char}
    
    def _parse_escape(self) -> RegexNode:
        """Parse escape sequence."""
        if self.pos >= self.length:
            raise RegexError("Incomplete escape sequence")
        
        char = self._next()
        
        # Character class shortcuts
        if char == 'd':
            return CharClassNode(DIGITS)
        elif char == 'D':
            return CharClassNode(DIGITS, negated=True)
        elif char == 'w':
            return CharClassNode(WORD_CHARS)
        elif char == 'W':
            return CharClassNode(WORD_CHARS, negated=True)
        elif char == 's':
            return CharClassNode(WHITESPACE)
        elif char == 'S':
            return CharClassNode(WHITESPACE, negated=True)
        elif char == 'n':
            return CharNode('\n')
        elif char == 'r':
            return CharNode('\r')
        elif char == 't':
            return CharNode('\t')
        elif char == 'f':
            return CharNode('\f')
        elif char == 'v':
            return CharNode('\v')
        elif char == '0':
            return CharNode('\0')
        else:
            # Literal escape (e.g., \\, \., \*, etc.)
            return CharNode(char)


def parse_regex(pattern: str) -> RegexNode:
    """
    Parse a regex pattern into an AST.
    
    Args:
        pattern: Regular expression pattern
        
    Returns:
        RegexNode AST
        
    Raises:
        RegexError: If pattern is invalid
    """
    return RegexParser(pattern).parse()

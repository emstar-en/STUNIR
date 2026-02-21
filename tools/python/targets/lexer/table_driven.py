"""
Table-Driven Lexer Emitter for STUNIR.

Generates portable JSON representation of lexer DFA tables
that can be loaded by any runtime.
"""

import json
from datetime import datetime
from typing import Any, Dict, List

from ir.lexer.token_spec import LexerSpec, TokenSpec
from ir.lexer.dfa import MinimizedDFA, TransitionTable
from .base import BaseLexerEmitter, canonical_json, compute_sha256


class TableDrivenEmitter(BaseLexerEmitter):
    """
    Emit portable table-driven lexer representation.
    
    Generates a JSON file containing:
    - Token specifications
    - DFA structure (states, transitions, accept states)
    - Transition table in flat format
    - Metadata for verification
    
    This format can be loaded by runtime libraries in any language.
    """
    
    def emit(self, spec: LexerSpec, dfa: MinimizedDFA, table: TransitionTable) -> str:
        """Emit JSON representation of lexer tables."""
        data = self._build_data(spec, dfa, table)
        return canonical_json(data)
    
    def _build_data(self, spec: LexerSpec, dfa: MinimizedDFA, table: TransitionTable) -> Dict[str, Any]:
        """Build complete data structure."""
        # Build token list
        tokens = []
        for i, t in enumerate(spec.tokens):
            tokens.append({
                "index": i,
                "name": t.name,
                "pattern": t.pattern,
                "priority": t.priority,
                "skip": t.skip,
                "token_type": t.token_type.name if hasattr(t.token_type, 'name') else str(t.token_type)
            })
        
        # Build DFA structure
        dfa_data = {
            "start_state": dfa.start_state,
            "num_states": dfa.num_states,
            "alphabet": dfa.alphabet,
            "transitions": {
                str(state): trans 
                for state, trans in dfa.transitions.items()
            },
            "accept_states": {
                str(state): {
                    "token": token_name,
                    "priority": priority
                }
                for state, (token_name, priority) in dfa.accept_states.items()
            }
        }
        
        # Build flat transition table
        table_data = {
            "num_states": table.num_states,
            "num_symbols": table.num_symbols,
            "start_state": table.start_state,
            "error_state": table.error_state,
            "symbol_to_index": table.symbol_to_index,
            "index_to_symbol": table.index_to_symbol,
            "transitions": table.table,
            "accept_table": [
                {"token": a[0], "priority": a[1]} if a else None
                for a in table.accept_table
            ]
        }
        
        # Build skip tokens list
        skip_tokens = [t.name for t in spec.tokens if t.skip]
        
        # Complete structure
        data = {
            "schema": "stunir.lexer.table.v1",
            "version": "1.0.0",
            "generated": datetime.utcnow().isoformat() + "Z",
            "generator": "STUNIR Lexer Generator",
            "lexer_name": spec.name,
            "case_sensitive": spec.case_sensitive,
            "tokens": tokens,
            "skip_tokens": skip_tokens,
            "keywords": spec.keywords,
            "dfa": dfa_data,
            "table": table_data,
            "statistics": {
                "num_tokens": len(spec.tokens),
                "num_skip_tokens": len(skip_tokens),
                "dfa_states": dfa.num_states,
                "alphabet_size": len(dfa.alphabet),
                "table_size": len(table.table)
            }
        }
        
        # Add content hash for verification
        content_for_hash = canonical_json({
            "tokens": tokens,
            "dfa": dfa_data,
            "table": table_data
        })
        data["content_hash"] = compute_sha256(content_for_hash)
        
        return data
    
    def emit_transition_table(self, table: TransitionTable) -> str:
        """Emit transition table as JSON."""
        return canonical_json(table.to_dict())
    
    def emit_token_class(self, spec: LexerSpec) -> str:
        """Emit token specifications as JSON."""
        tokens = []
        for i, t in enumerate(spec.tokens):
            tokens.append({
                "index": i,
                "name": t.name,
                "pattern": t.pattern,
                "priority": t.priority,
                "skip": t.skip
            })
        return canonical_json({"tokens": tokens})
    
    def emit_lexer_class(self, spec: LexerSpec, dfa: MinimizedDFA) -> str:
        """Emit DFA specification as JSON."""
        return canonical_json({
            "start_state": dfa.start_state,
            "num_states": dfa.num_states,
            "alphabet": dfa.alphabet,
            "transitions": dfa.transitions,
            "accept_states": {
                str(k): {"token": v[0], "priority": v[1]}
                for k, v in dfa.accept_states.items()
            }
        })
    
    def emit_pretty(self, spec: LexerSpec, dfa: MinimizedDFA, table: TransitionTable) -> str:
        """Emit pretty-printed JSON (for human readability)."""
        data = self._build_data(spec, dfa, table)
        return json.dumps(data, indent=2, sort_keys=True)


class CompactTableEmitter(BaseLexerEmitter):
    """
    Emit compact binary-friendly table representation.
    
    Produces a minimal JSON structure optimized for size,
    suitable for embedding in binaries or network transfer.
    """
    
    def emit(self, spec: LexerSpec, dfa: MinimizedDFA, table: TransitionTable) -> str:
        """Emit compact representation."""
        # Token name to index mapping
        token_names = [t.name for t in spec.tokens]
        skip_indices = [i for i, t in enumerate(spec.tokens) if t.skip]
        
        # Accept table with token indices instead of names
        accept_compact = []
        for a in table.accept_table:
            if a:
                token_name, priority = a
                token_idx = token_names.index(token_name)
                accept_compact.append([token_idx, priority])
            else:
                accept_compact.append(None)
        
        data = {
            "v": 1,  # Version
            "n": spec.name,  # Lexer name
            "t": token_names,  # Token names
            "s": skip_indices,  # Skip token indices
            "d": {  # DFA
                "ss": table.start_state,  # Start state
                "ns": table.num_states,  # Num states
                "a": table.index_to_symbol,  # Alphabet
                "tr": table.table,  # Transitions (flat)
                "ac": accept_compact  # Accept table
            }
        }
        
        return canonical_json(data)
    
    def emit_transition_table(self, table: TransitionTable) -> str:
        """Emit minimal transition table."""
        return canonical_json({
            "tr": table.table,
            "ss": table.start_state,
            "ns": table.num_states
        })
    
    def emit_token_class(self, spec: LexerSpec) -> str:
        """Emit token names only."""
        return canonical_json([t.name for t in spec.tokens])
    
    def emit_lexer_class(self, spec: LexerSpec, dfa: MinimizedDFA) -> str:
        """Emit DFA in compact form."""
        return canonical_json({
            "s": dfa.start_state,
            "n": dfa.num_states,
            "a": dfa.alphabet,
            "t": dfa.transitions
        })

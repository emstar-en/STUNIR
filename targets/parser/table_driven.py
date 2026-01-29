#!/usr/bin/env python3
"""Table-driven parser emitter.

Generates portable parser tables in JSON format that can be used
by table-driven parser interpreters in any language.
"""

import json
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


class TableDrivenEmitter(ParserEmitterBase):
    """Table-driven parser emitter.
    
    Generates parse tables in a portable JSON format that can be
    interpreted by a generic table-driven parser runtime.
    """
    
    LANGUAGE = "table_driven"
    FILE_EXTENSION = ".json"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the emitter.
        
        Args:
            config: Configuration dictionary
                - compact: bool, whether to use compact JSON (default: False)
                - include_debug: bool, include debug info (default: True)
        """
        super().__init__(config)
        self.compact = self.config.get('compact', False)
        self.include_debug = self.config.get('include_debug', True)
    
    def emit(self, result: ParserGeneratorResult, 
             grammar: Grammar) -> ParserEmitterResult:
        """Emit table-driven parser data.
        
        Args:
            result: Parser generator result
            grammar: Source grammar
        
        Returns:
            ParserEmitterResult with JSON data
        """
        # Build complete parser data
        parser_data = self._build_parser_data(result, grammar)
        
        # Serialize to JSON
        if self.compact:
            code = json.dumps(parser_data, separators=(',', ':'), sort_keys=True)
        else:
            code = json.dumps(parser_data, indent=2, sort_keys=True)
        
        # Generate AST schema as JSON
        ast_code = ""
        if result.ast_schema:
            ast_data = self._build_ast_schema_data(result.ast_schema)
            if self.compact:
                ast_code = json.dumps(ast_data, separators=(',', ':'), sort_keys=True)
            else:
                ast_code = json.dumps(ast_data, indent=2, sort_keys=True)
        
        # Generate generic runtime code (Python example)
        runtime_code = self._emit_runtime_code()
        
        # Generate manifest
        manifest = self._generate_manifest(code, ast_code, grammar, {
            "parser_runtime.py": runtime_code
        })
        
        emit_result = ParserEmitterResult(
            code=code,
            ast_code=ast_code,
            manifest=manifest,
            warnings=self._get_warnings()
        )
        emit_result.add_auxiliary_file("parser_runtime.py", runtime_code)
        
        return emit_result
    
    def _build_parser_data(self, result: ParserGeneratorResult, grammar: Grammar) -> Dict[str, Any]:
        """Build complete parser data structure."""
        data: Dict[str, Any] = {
            "schema": "stunir.parser.table_driven.v1",
            "grammar_name": grammar.name if hasattr(grammar, 'name') else "unknown",
            "parser_type": result.parser_type.name,
        }
        
        if isinstance(result.parse_table, ParseTable):
            data["table_type"] = "LR"
            data.update(self._build_lr_data(result.parse_table))
        else:
            data["table_type"] = "LL"
            data.update(self._build_ll_data(result.parse_table))
        
        if self.include_debug:
            data["debug"] = {
                "state_count": result.info.get("state_count", 0),
                "conflict_count": result.info.get("conflict_count", 0),
                "warnings": result.warnings,
            }
        
        return data
    
    def _build_lr_data(self, table: ParseTable) -> Dict[str, Any]:
        """Build LR table data."""
        # Build terminal and nonterminal lists
        terminals: List[str] = []
        terminal_index: Dict[str, int] = {}
        for (_, sym) in table.action.keys():
            name = sym.name if hasattr(sym, 'name') else str(sym)
            if name not in terminal_index:
                terminal_index[name] = len(terminals)
                terminals.append(name)
        
        nonterminals: List[str] = []
        nonterminal_index: Dict[str, int] = {}
        for (_, sym) in table.goto.keys():
            name = sym.name if hasattr(sym, 'name') else str(sym)
            if name not in nonterminal_index:
                nonterminal_index[name] = len(nonterminals)
                nonterminals.append(name)
        
        # Build productions
        productions = []
        for prod in table.productions:
            head = prod.head.name if hasattr(prod.head, 'name') else str(prod.head)
            body = [s.name if hasattr(s, 'name') else str(s) for s in (prod.body or [])]
            productions.append({
                "head": head,
                "body": body,
                "label": prod.label,
            })
        
        # Build action table
        action_table: Dict[str, Dict[str, List]] = {}
        for (state, sym), action in table.action.items():
            state_key = str(state)
            sym_name = sym.name if hasattr(sym, 'name') else str(sym)
            
            if state_key not in action_table:
                action_table[state_key] = {}
            
            if action.is_shift():
                action_table[state_key][sym_name] = ["shift", action.value]
            elif action.is_reduce():
                action_table[state_key][sym_name] = ["reduce", action.value]
            elif action.is_accept():
                action_table[state_key][sym_name] = ["accept"]
            else:
                action_table[state_key][sym_name] = ["error"]
        
        # Build goto table
        goto_table: Dict[str, Dict[str, int]] = {}
        for (state, sym), target in table.goto.items():
            state_key = str(state)
            sym_name = sym.name if hasattr(sym, 'name') else str(sym)
            
            if state_key not in goto_table:
                goto_table[state_key] = {}
            
            goto_table[state_key][sym_name] = target
        
        return {
            "terminals": terminals,
            "nonterminals": nonterminals,
            "productions": productions,
            "action": action_table,
            "goto": goto_table,
            "start_state": 0,
        }
    
    def _build_ll_data(self, table: LL1Table) -> Dict[str, Any]:
        """Build LL(1) table data."""
        # Build terminal and nonterminal lists
        terminals = sorted(set(
            t.name if hasattr(t, 'name') else str(t) 
            for (_, t) in table.table.keys()
        ))
        
        nonterminals = sorted(set(
            nt.name if hasattr(nt, 'name') else str(nt)
            for (nt, _) in table.table.keys()
        ))
        
        # Build productions from table
        productions: List[Dict[str, Any]] = []
        prod_index: Dict[str, int] = {}
        
        for (nt, t), prod in table.table.items():
            prod_str = str(prod)
            if prod_str not in prod_index:
                head = prod.head.name if hasattr(prod.head, 'name') else str(prod.head)
                body = [s.name if hasattr(s, 'name') else str(s) for s in (prod.body or [])]
                prod_index[prod_str] = len(productions)
                productions.append({
                    "head": head,
                    "body": body,
                    "label": prod.label,
                })
        
        # Build LL table
        ll_table: Dict[str, Dict[str, List[str]]] = {}
        for (nt, t), prod in table.table.items():
            nt_name = nt.name if hasattr(nt, 'name') else str(nt)
            t_name = t.name if hasattr(t, 'name') else str(t)
            
            if nt_name not in ll_table:
                ll_table[nt_name] = {}
            
            body = [s.name if hasattr(s, 'name') else str(s) for s in (prod.body or [])]
            ll_table[nt_name][t_name] = body
        
        return {
            "terminals": terminals,
            "nonterminals": nonterminals,
            "productions": productions,
            "table": ll_table,
            "start_symbol": nonterminals[0] if nonterminals else None,
        }
    
    def _build_ast_schema_data(self, schema: ASTSchema) -> Dict[str, Any]:
        """Build AST schema as JSON data."""
        nodes = []
        for node in schema.nodes:
            node_data = {
                "name": node.name,
                "fields": [{"name": n, "type": t} for n, t in node.fields],
                "is_abstract": node.is_abstract,
            }
            if node.base_class:
                node_data["base_class"] = node.base_class
            nodes.append(node_data)
        
        return {
            "schema": "stunir.ast_schema.v1",
            "base_node_name": schema.base_node_name,
            "token_type_name": schema.token_type_name,
            "nodes": nodes,
        }
    
    def emit_parse_table(self, table: Union[ParseTable, LL1Table]) -> str:
        """Emit parse table as JSON.
        
        Args:
            table: Parse table to emit
        
        Returns:
            JSON string
        """
        if isinstance(table, ParseTable):
            data = self._build_lr_data(table)
        else:
            data = self._build_ll_data(table)
        
        if self.compact:
            return json.dumps(data, separators=(',', ':'), sort_keys=True)
        else:
            return json.dumps(data, indent=2, sort_keys=True)
    
    def emit_ast_nodes(self, schema: ASTSchema) -> str:
        """Emit AST schema as JSON.
        
        Args:
            schema: AST schema
        
        Returns:
            JSON string
        """
        data = self._build_ast_schema_data(schema)
        
        if self.compact:
            return json.dumps(data, separators=(',', ':'), sort_keys=True)
        else:
            return json.dumps(data, indent=2, sort_keys=True)
    
    def _emit_runtime_code(self) -> str:
        """Generate a generic Python runtime for table-driven parsing."""
        return '''#!/usr/bin/env python3
"""Generic Table-Driven Parser Runtime.

This module provides a runtime interpreter for table-driven parsers
generated by STUNIR.

Usage:
    from parser_runtime import TableDrivenParser
    
    # Load parser tables
    with open("parser_tables.json") as f:
        tables = json.load(f)
    
    # Create parser
    parser = TableDrivenParser(tables)
    
    # Parse tokens
    result = parser.parse(tokens)
"""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Token:
    """A lexical token."""
    type: str
    value: str
    line: int = 0
    column: int = 0


class ParseError(Exception):
    """Parse error with location information."""
    def __init__(self, message: str, token: Optional[Token] = None):
        self.token = token
        loc = f" at line {token.line}, column {token.column}" if token else ""
        super().__init__(f"{message}{loc}")


class TableDrivenParser:
    """Generic table-driven parser runtime."""
    
    def __init__(self, tables: Dict[str, Any]):
        """Initialize parser with table data.
        
        Args:
            tables: Parser tables loaded from JSON
        """
        self.tables = tables
        self.table_type = tables.get("table_type", "LR")
        self.grammar_name = tables.get("grammar_name", "unknown")
    
    def parse(self, tokens: List[Token]) -> Any:
        """Parse a list of tokens.
        
        Args:
            tokens: List of Token objects
        
        Returns:
            Parse result (AST or value stack)
        
        Raises:
            ParseError: If parsing fails
        """
        if self.table_type == "LR":
            return self._parse_lr(tokens)
        else:
            return self._parse_ll(tokens)
    
    def _parse_lr(self, tokens: List[Token]) -> Any:
        """LR parsing algorithm."""
        action_table = self.tables["action"]
        goto_table = self.tables["goto"]
        productions = self.tables["productions"]
        
        pos = 0
        stack = [self.tables.get("start_state", 0)]
        value_stack: List[Any] = []
        
        while True:
            state = stack[-1]
            token = tokens[pos] if pos < len(tokens) else Token("$", "")
            
            state_actions = action_table.get(str(state), {})
            action = state_actions.get(token.type)
            
            if action is None:
                raise ParseError(f"Unexpected token: {token.type}", token)
            
            if action[0] == "shift":
                next_state = action[1]
                stack.append(next_state)
                value_stack.append(token)
                pos += 1
            
            elif action[0] == "reduce":
                prod_index = action[1]
                prod = productions[prod_index]
                body_len = len(prod["body"])
                
                # Pop states and values
                values = []
                for _ in range(body_len):
                    stack.pop()
                    if value_stack:
                        values.insert(0, value_stack.pop())
                
                # Get goto state
                state = stack[-1]
                state_gotos = goto_table.get(str(state), {})
                goto_state = state_gotos.get(prod["head"])
                
                if goto_state is None:
                    raise ParseError(f"No GOTO for ({state}, {prod['head']})")
                
                stack.append(goto_state)
                value_stack.append((prod["head"], values))
            
            elif action[0] == "accept":
                return value_stack[-1] if value_stack else None
            
            else:
                raise ParseError(f"Parse error", token)
    
    def _parse_ll(self, tokens: List[Token]) -> Any:
        """LL(1) parsing algorithm."""
        ll_table = self.tables["table"]
        start_symbol = self.tables.get("start_symbol")
        
        pos = 0
        stack = ["$", start_symbol]
        result_stack: List[Any] = []
        
        while stack:
            top = stack.pop()
            token = tokens[pos] if pos < len(tokens) else Token("$", "")
            
            if top == "$":
                if token.type == "$":
                    break
                raise ParseError("Expected end of input", token)
            
            if top == token.type:
                # Terminal match
                result_stack.append(token)
                pos += 1
            else:
                # Nonterminal: consult table
                nt_entries = ll_table.get(top, {})
                production = nt_entries.get(token.type)
                
                if production is None:
                    raise ParseError(f"No production for ({top}, {token.type})", token)
                
                # Push RHS in reverse order
                for symbol in reversed(production):
                    stack.append(symbol)
        
        return result_stack


def main():
    """Example usage."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: parser_runtime.py <tables.json>")
        sys.exit(1)
    
    with open(sys.argv[1]) as f:
        tables = json.load(f)
    
    parser = TableDrivenParser(tables)
    print(f"Loaded {tables.get('grammar_name', 'unknown')} parser")
    print(f"Type: {tables.get('table_type', 'unknown')}")


if __name__ == "__main__":
    main()
'''

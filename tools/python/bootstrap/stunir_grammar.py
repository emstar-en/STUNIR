"""
STUNIR Grammar Specification.

Defines the complete grammar for the STUNIR language using the Grammar IR
classes from Phase 6A. This enables STUNIR to be self-hosting.

The grammar supports:
- Module declarations with imports/exports
- Type definitions (structs, variants, aliases)
- Function definitions with full expression support
- IR node definitions
- Target specifications for code generation
"""

from typing import List, Dict, Optional, Tuple

from ir.grammar.symbol import Symbol, SymbolType, EPSILON
from ir.grammar.production import ProductionRule, BodyElement
from ir.grammar.grammar_ir import Grammar, GrammarType


class STUNIRGrammarBuilder:
    """
    Builder for STUNIR grammar specification.
    
    Creates a complete Grammar object representing the STUNIR language
    using the Grammar IR framework from Phase 6A.
    
    The grammar is organized into sections:
    1. Program structure (module, imports, exports)
    2. Declarations (type, function, ir, target, const)
    3. Types (primitive types, compound types)
    4. Statements (var, if, while, for, match, return, emit)
    5. Expressions (precedence-climbing grammar)
    6. Patterns (for match statements)
    
    Usage:
        builder = STUNIRGrammarBuilder()
        grammar = builder.build()
    """
    
    def __init__(self):
        """Initialize the grammar builder."""
        self._terminals: Dict[str, Symbol] = {}
        self._nonterminals: Dict[str, Symbol] = {}
        self._productions: List[ProductionRule] = []
        
        # Build terminal and non-terminal tables
        self._build_terminals()
        self._build_nonterminals()
    
    def _t(self, name: str) -> Symbol:
        """Get or create a terminal symbol."""
        if name not in self._terminals:
            self._terminals[name] = Symbol(name, SymbolType.TERMINAL)
        return self._terminals[name]
    
    def _nt(self, name: str) -> Symbol:
        """Get or create a non-terminal symbol."""
        if name not in self._nonterminals:
            self._nonterminals[name] = Symbol(name, SymbolType.NONTERMINAL)
        return self._nonterminals[name]
    
    def _add_rule(self, head: str, *body: str, label: Optional[str] = None):
        """Add a production rule."""
        head_symbol = self._nt(head)
        body_symbols = []
        for s in body:
            if s.startswith('$'):  # Non-terminal marker
                body_symbols.append(self._nt(s[1:]))
            elif s == 'EPSILON':
                body_symbols.append(EPSILON)
            else:  # Terminal
                body_symbols.append(self._t(s))
        self._productions.append(ProductionRule(
            head=head_symbol,
            body=tuple(body_symbols),
            label=label
        ))
    
    def _build_terminals(self):
        """Build all terminal symbols."""
        # Keywords
        keywords = [
            'KW_MODULE', 'KW_IMPORT', 'KW_FROM', 'KW_EXPORT', 'KW_AS',
            'KW_TYPE', 'KW_FUNCTION', 'KW_IR', 'KW_TARGET', 'KW_CONST',
            'KW_I8', 'KW_I16', 'KW_I32', 'KW_I64',
            'KW_U8', 'KW_U16', 'KW_U32', 'KW_U64',
            'KW_F32', 'KW_F64', 'KW_BOOL', 'KW_STRING', 'KW_VOID', 'KW_ANY',
            'KW_LET', 'KW_VAR', 'KW_IF', 'KW_ELSE', 'KW_WHILE', 'KW_FOR', 'KW_IN',
            'KW_MATCH', 'KW_RETURN', 'KW_EMIT',
            'KW_CHILD', 'KW_OP',
            'KW_TRUE', 'KW_FALSE', 'KW_NULL',
        ]
        
        # Operators
        operators = [
            'PLUS', 'MINUS', 'STAR', 'SLASH', 'PERCENT',
            'EQ', 'NE', 'LT', 'GT', 'LE', 'GE',
            'AND', 'OR', 'NOT',
            'AMPERSAND', 'PIPE', 'CARET', 'TILDE', 'LSHIFT', 'RSHIFT',
            'EQUALS', 'PLUS_EQ', 'MINUS_EQ', 'STAR_EQ', 'SLASH_EQ', 'PERCENT_EQ',
            'ARROW', 'FAT_ARROW', 'QUESTION',
        ]
        
        # Punctuation
        punctuation = [
            'LPAREN', 'RPAREN', 'LBRACE', 'RBRACE', 'LBRACKET', 'RBRACKET',
            'COMMA', 'SEMICOLON', 'DOT', 'COLON',
        ]
        
        # Literals
        literals = [
            'IDENTIFIER', 'INTEGER_LITERAL', 'FLOAT_LITERAL', 'STRING_LITERAL',
        ]
        
        # Create all terminal symbols
        for name in keywords + operators + punctuation + literals:
            self._t(name)
    
    def _build_nonterminals(self):
        """Build all non-terminal symbols."""
        nonterminals = [
            # Program structure
            'program', 'module_decl', 'module_body', 'module_item',
            'import_decl', 'import_alias', 'import_list',
            'export_decl', 'declarations', 'declaration',
            
            # Types
            'type_def', 'type_params', 'type_params_opt', 'type_body',
            'type_member', 'field_def', 'variant_def',
            'type_expr', 'type_expr_tail', 'basic_type', 'type_list',
            
            # Functions
            'function_def', 'param_list', 'param_list_items', 'param',
            'return_type_opt', 'block', 'statements', 'statement',
            
            # Statements
            'var_decl', 'type_annot_opt', 'init_opt',
            'if_stmt', 'else_part',
            'while_stmt', 'for_stmt',
            'match_stmt', 'match_arms', 'match_arm',
            'return_stmt', 'emit_stmt', 'expr_stmt',
            
            # Expressions (precedence climbing)
            'expression', 'ternary_expr', 'ternary_tail',
            'or_expr', 'or_tail',
            'and_expr', 'and_tail',
            'equality_expr', 'equality_tail',
            'relational_expr', 'relational_tail',
            'additive_expr', 'additive_tail',
            'multiplicative_expr', 'multiplicative_tail',
            'unary_expr', 'postfix_expr', 'postfix_tail',
            'primary_expr', 'array_literal', 'object_literal',
            'field_init', 'field_init_list', 'expression_list',
            'arg_list', 'arg_list_items',
            
            # IR definitions
            'ir_def', 'ir_body', 'ir_member', 'ir_field', 'ir_child', 'ir_op',
            
            # Target definitions
            'target_def', 'target_body', 'target_member', 'target_option', 'emit_rule',
            
            # Patterns
            'pattern', 'pattern_list', 'pattern_list_items',
            
            # Utilities
            'dotted_name', 'dotted_name_tail',
            'identifier_list', 'identifier_list_tail',
            'expression_opt',
        ]
        
        for name in nonterminals:
            self._nt(name)
    
    def _build_program_rules(self):
        """Build program structure rules."""
        # program → module_decl declarations
        self._add_rule('program', '$module_decl', '$declarations', label='program')
        
        # declarations → declaration declarations | ε
        self._add_rule('declarations', '$declaration', '$declarations', label='declarations_list')
        self._add_rule('declarations', 'EPSILON', label='declarations_empty')
        
        # module_decl → KW_MODULE IDENTIFIER SEMICOLON
        #             | KW_MODULE IDENTIFIER LBRACE module_body RBRACE
        self._add_rule('module_decl', 'KW_MODULE', 'IDENTIFIER', 'SEMICOLON', label='module_simple')
        self._add_rule('module_decl', 'KW_MODULE', 'IDENTIFIER', 'LBRACE', '$module_body', 'RBRACE', label='module_block')
        
        # module_body → module_item module_body | ε
        self._add_rule('module_body', '$module_item', '$module_body', label='module_body_list')
        self._add_rule('module_body', 'EPSILON', label='module_body_empty')
        
        # module_item → import_decl | export_decl | declaration
        self._add_rule('module_item', '$import_decl', label='module_item_import')
        self._add_rule('module_item', '$export_decl', label='module_item_export')
        self._add_rule('module_item', '$declaration', label='module_item_decl')
    
    def _build_import_export_rules(self):
        """Build import/export rules."""
        # import_decl → KW_IMPORT dotted_name import_alias SEMICOLON
        #             | KW_FROM dotted_name KW_IMPORT import_list SEMICOLON
        self._add_rule('import_decl', 'KW_IMPORT', '$dotted_name', '$import_alias', 'SEMICOLON', label='import_simple')
        self._add_rule('import_decl', 'KW_FROM', '$dotted_name', 'KW_IMPORT', '$import_list', 'SEMICOLON', label='import_from')
        
        # import_alias → KW_AS IDENTIFIER | ε
        self._add_rule('import_alias', 'KW_AS', 'IDENTIFIER', label='import_alias_as')
        self._add_rule('import_alias', 'EPSILON', label='import_alias_none')
        
        # import_list → identifier_list
        self._add_rule('import_list', '$identifier_list', label='import_list')
        
        # export_decl → KW_EXPORT identifier_list SEMICOLON
        #             | KW_EXPORT STAR SEMICOLON
        self._add_rule('export_decl', 'KW_EXPORT', '$identifier_list', 'SEMICOLON', label='export_list')
        self._add_rule('export_decl', 'KW_EXPORT', 'STAR', 'SEMICOLON', label='export_all')
        
        # dotted_name → IDENTIFIER dotted_name_tail
        self._add_rule('dotted_name', 'IDENTIFIER', '$dotted_name_tail', label='dotted_name')
        
        # dotted_name_tail → DOT IDENTIFIER dotted_name_tail | ε
        self._add_rule('dotted_name_tail', 'DOT', 'IDENTIFIER', '$dotted_name_tail', label='dotted_name_tail_more')
        self._add_rule('dotted_name_tail', 'EPSILON', label='dotted_name_tail_empty')
        
        # identifier_list → IDENTIFIER identifier_list_tail
        self._add_rule('identifier_list', 'IDENTIFIER', '$identifier_list_tail', label='identifier_list')
        
        # identifier_list_tail → COMMA IDENTIFIER identifier_list_tail | ε
        self._add_rule('identifier_list_tail', 'COMMA', 'IDENTIFIER', '$identifier_list_tail', label='identifier_list_tail_more')
        self._add_rule('identifier_list_tail', 'EPSILON', label='identifier_list_tail_empty')
    
    def _build_declaration_rules(self):
        """Build declaration rules."""
        # declaration → type_def | function_def | ir_def | target_def | const_def
        self._add_rule('declaration', '$type_def', label='decl_type')
        self._add_rule('declaration', '$function_def', label='decl_function')
        self._add_rule('declaration', '$ir_def', label='decl_ir')
        self._add_rule('declaration', '$target_def', label='decl_target')
    
    def _build_type_rules(self):
        """Build type definition rules."""
        # type_def → KW_TYPE IDENTIFIER type_params_opt EQUALS type_expr SEMICOLON
        #          | KW_TYPE IDENTIFIER type_params_opt LBRACE type_body RBRACE
        self._add_rule('type_def', 'KW_TYPE', 'IDENTIFIER', '$type_params_opt', 'EQUALS', '$type_expr', 'SEMICOLON', label='type_alias')
        self._add_rule('type_def', 'KW_TYPE', 'IDENTIFIER', '$type_params_opt', 'LBRACE', '$type_body', 'RBRACE', label='type_struct')
        
        # type_params_opt → LT identifier_list GT | ε
        self._add_rule('type_params_opt', 'LT', '$identifier_list', 'GT', label='type_params')
        self._add_rule('type_params_opt', 'EPSILON', label='type_params_none')
        
        # type_body → type_member type_body | ε
        self._add_rule('type_body', '$type_member', '$type_body', label='type_body_list')
        self._add_rule('type_body', 'EPSILON', label='type_body_empty')
        
        # type_member → field_def | variant_def
        self._add_rule('type_member', '$field_def', label='type_member_field')
        self._add_rule('type_member', '$variant_def', label='type_member_variant')
        
        # field_def → IDENTIFIER COLON type_expr SEMICOLON
        self._add_rule('field_def', 'IDENTIFIER', 'COLON', '$type_expr', 'SEMICOLON', label='field_def')
        
        # variant_def → PIPE IDENTIFIER
        #             | PIPE IDENTIFIER LPAREN type_list RPAREN
        self._add_rule('variant_def', 'PIPE', 'IDENTIFIER', label='variant_simple')
        self._add_rule('variant_def', 'PIPE', 'IDENTIFIER', 'LPAREN', '$type_list', 'RPAREN', label='variant_tuple')
        
        # type_expr → basic_type type_expr_tail
        #           | IDENTIFIER type_expr_tail
        #           | LBRACKET type_expr RBRACKET type_expr_tail
        #           | LPAREN type_list RPAREN ARROW type_expr
        self._add_rule('type_expr', '$basic_type', '$type_expr_tail', label='type_basic')
        self._add_rule('type_expr', 'IDENTIFIER', '$type_expr_tail', label='type_named')
        self._add_rule('type_expr', 'LBRACKET', '$type_expr', 'RBRACKET', '$type_expr_tail', label='type_array')
        self._add_rule('type_expr', 'LPAREN', '$type_list', 'RPAREN', 'ARROW', '$type_expr', label='type_function')
        
        # type_expr_tail → LT type_list GT | QUESTION | PIPE type_expr | ε
        self._add_rule('type_expr_tail', 'LT', '$type_list', 'GT', label='type_generic')
        self._add_rule('type_expr_tail', 'QUESTION', label='type_optional')
        self._add_rule('type_expr_tail', 'PIPE', '$type_expr', label='type_union')
        self._add_rule('type_expr_tail', 'EPSILON', label='type_expr_tail_empty')
        
        # basic_type → primitive types
        for kw in ['KW_I8', 'KW_I16', 'KW_I32', 'KW_I64',
                   'KW_U8', 'KW_U16', 'KW_U32', 'KW_U64',
                   'KW_F32', 'KW_F64', 'KW_BOOL', 'KW_STRING', 'KW_VOID', 'KW_ANY']:
            self._add_rule('basic_type', kw, label=f'basic_{kw.lower()[3:]}')
        
        # type_list → type_expr type_list_tail
        self._add_rule('type_list', '$type_expr', '$identifier_list_tail', label='type_list')  # Reuse identifier_list_tail pattern
    
    def _build_function_rules(self):
        """Build function definition rules."""
        # function_def → KW_FUNCTION IDENTIFIER LPAREN param_list RPAREN return_type_opt block
        self._add_rule('function_def', 'KW_FUNCTION', 'IDENTIFIER', 'LPAREN', '$param_list', 'RPAREN', '$return_type_opt', '$block', label='function_def')
        
        # param_list → param_list_items | ε
        self._add_rule('param_list', '$param_list_items', label='param_list')
        self._add_rule('param_list', 'EPSILON', label='param_list_empty')
        
        # param_list_items → param COMMA param_list_items | param
        self._add_rule('param_list_items', '$param', 'COMMA', '$param_list_items', label='param_list_items_more')
        self._add_rule('param_list_items', '$param', label='param_list_items_one')
        
        # param → IDENTIFIER COLON type_expr
        #       | IDENTIFIER COLON type_expr EQUALS expression
        self._add_rule('param', 'IDENTIFIER', 'COLON', '$type_expr', label='param_simple')
        self._add_rule('param', 'IDENTIFIER', 'COLON', '$type_expr', 'EQUALS', '$expression', label='param_default')
        
        # return_type_opt → COLON type_expr | ε
        self._add_rule('return_type_opt', 'COLON', '$type_expr', label='return_type')
        self._add_rule('return_type_opt', 'EPSILON', label='return_type_none')
        
        # block → LBRACE statements RBRACE
        self._add_rule('block', 'LBRACE', '$statements', 'RBRACE', label='block')
        
        # statements → statement statements | ε
        self._add_rule('statements', '$statement', '$statements', label='statements_list')
        self._add_rule('statements', 'EPSILON', label='statements_empty')
    
    def _build_statement_rules(self):
        """Build statement rules."""
        # statement → var_decl | if_stmt | while_stmt | for_stmt | match_stmt
        #           | return_stmt | emit_stmt | expr_stmt
        self._add_rule('statement', '$var_decl', label='stmt_var')
        self._add_rule('statement', '$if_stmt', label='stmt_if')
        self._add_rule('statement', '$while_stmt', label='stmt_while')
        self._add_rule('statement', '$for_stmt', label='stmt_for')
        self._add_rule('statement', '$match_stmt', label='stmt_match')
        self._add_rule('statement', '$return_stmt', label='stmt_return')
        self._add_rule('statement', '$emit_stmt', label='stmt_emit')
        self._add_rule('statement', '$expr_stmt', label='stmt_expr')
        
        # var_decl → KW_LET IDENTIFIER type_annot_opt EQUALS expression SEMICOLON
        #          | KW_VAR IDENTIFIER type_annot_opt init_opt SEMICOLON
        self._add_rule('var_decl', 'KW_LET', 'IDENTIFIER', '$type_annot_opt', 'EQUALS', '$expression', 'SEMICOLON', label='var_let')
        self._add_rule('var_decl', 'KW_VAR', 'IDENTIFIER', '$type_annot_opt', '$init_opt', 'SEMICOLON', label='var_var')
        
        # type_annot_opt → COLON type_expr | ε
        self._add_rule('type_annot_opt', 'COLON', '$type_expr', label='type_annot')
        self._add_rule('type_annot_opt', 'EPSILON', label='type_annot_none')
        
        # init_opt → EQUALS expression | ε
        self._add_rule('init_opt', 'EQUALS', '$expression', label='init')
        self._add_rule('init_opt', 'EPSILON', label='init_none')
        
        # if_stmt → KW_IF expression block else_part
        self._add_rule('if_stmt', 'KW_IF', '$expression', '$block', '$else_part', label='if_stmt')
        
        # else_part → KW_ELSE block | KW_ELSE if_stmt | ε
        self._add_rule('else_part', 'KW_ELSE', '$block', label='else_block')
        self._add_rule('else_part', 'KW_ELSE', '$if_stmt', label='else_if')
        self._add_rule('else_part', 'EPSILON', label='else_none')
        
        # while_stmt → KW_WHILE expression block
        self._add_rule('while_stmt', 'KW_WHILE', '$expression', '$block', label='while_stmt')
        
        # for_stmt → KW_FOR IDENTIFIER KW_IN expression block
        self._add_rule('for_stmt', 'KW_FOR', 'IDENTIFIER', 'KW_IN', '$expression', '$block', label='for_stmt')
        
        # match_stmt → KW_MATCH expression LBRACE match_arms RBRACE
        self._add_rule('match_stmt', 'KW_MATCH', '$expression', 'LBRACE', '$match_arms', 'RBRACE', label='match_stmt')
        
        # match_arms → match_arm match_arms | ε
        self._add_rule('match_arms', '$match_arm', '$match_arms', label='match_arms_list')
        self._add_rule('match_arms', 'EPSILON', label='match_arms_empty')
        
        # match_arm → pattern FAT_ARROW expression COMMA
        #           | pattern FAT_ARROW block
        self._add_rule('match_arm', '$pattern', 'FAT_ARROW', '$expression', 'COMMA', label='match_arm_expr')
        self._add_rule('match_arm', '$pattern', 'FAT_ARROW', '$block', label='match_arm_block')
        
        # return_stmt → KW_RETURN expression_opt SEMICOLON
        self._add_rule('return_stmt', 'KW_RETURN', '$expression_opt', 'SEMICOLON', label='return_stmt')
        
        # expression_opt → expression | ε
        self._add_rule('expression_opt', '$expression', label='expr_opt')
        self._add_rule('expression_opt', 'EPSILON', label='expr_opt_none')
        
        # emit_stmt → KW_EMIT expression SEMICOLON
        self._add_rule('emit_stmt', 'KW_EMIT', '$expression', 'SEMICOLON', label='emit_stmt')
        
        # expr_stmt → expression SEMICOLON
        self._add_rule('expr_stmt', '$expression', 'SEMICOLON', label='expr_stmt')
    
    def _build_expression_rules(self):
        """Build expression rules with precedence climbing."""
        # expression → ternary_expr
        self._add_rule('expression', '$ternary_expr', label='expression')
        
        # ternary_expr → or_expr ternary_tail
        self._add_rule('ternary_expr', '$or_expr', '$ternary_tail', label='ternary_expr')
        
        # ternary_tail → QUESTION expression COLON ternary_expr | ε
        self._add_rule('ternary_tail', 'QUESTION', '$expression', 'COLON', '$ternary_expr', label='ternary_op')
        self._add_rule('ternary_tail', 'EPSILON', label='ternary_none')
        
        # or_expr → and_expr or_tail
        self._add_rule('or_expr', '$and_expr', '$or_tail', label='or_expr')
        
        # or_tail → OR and_expr or_tail | ε
        self._add_rule('or_tail', 'OR', '$and_expr', '$or_tail', label='or_op')
        self._add_rule('or_tail', 'EPSILON', label='or_none')
        
        # and_expr → equality_expr and_tail
        self._add_rule('and_expr', '$equality_expr', '$and_tail', label='and_expr')
        
        # and_tail → AND equality_expr and_tail | ε
        self._add_rule('and_tail', 'AND', '$equality_expr', '$and_tail', label='and_op')
        self._add_rule('and_tail', 'EPSILON', label='and_none')
        
        # equality_expr → relational_expr equality_tail
        self._add_rule('equality_expr', '$relational_expr', '$equality_tail', label='equality_expr')
        
        # equality_tail → EQ relational_expr equality_tail
        #               | NE relational_expr equality_tail | ε
        self._add_rule('equality_tail', 'EQ', '$relational_expr', '$equality_tail', label='eq_op')
        self._add_rule('equality_tail', 'NE', '$relational_expr', '$equality_tail', label='ne_op')
        self._add_rule('equality_tail', 'EPSILON', label='equality_none')
        
        # relational_expr → additive_expr relational_tail
        self._add_rule('relational_expr', '$additive_expr', '$relational_tail', label='relational_expr')
        
        # relational_tail → LT additive_expr relational_tail
        #                 | GT additive_expr relational_tail
        #                 | LE additive_expr relational_tail
        #                 | GE additive_expr relational_tail | ε
        self._add_rule('relational_tail', 'LT', '$additive_expr', '$relational_tail', label='lt_op')
        self._add_rule('relational_tail', 'GT', '$additive_expr', '$relational_tail', label='gt_op')
        self._add_rule('relational_tail', 'LE', '$additive_expr', '$relational_tail', label='le_op')
        self._add_rule('relational_tail', 'GE', '$additive_expr', '$relational_tail', label='ge_op')
        self._add_rule('relational_tail', 'EPSILON', label='relational_none')
        
        # additive_expr → multiplicative_expr additive_tail
        self._add_rule('additive_expr', '$multiplicative_expr', '$additive_tail', label='additive_expr')
        
        # additive_tail → PLUS multiplicative_expr additive_tail
        #               | MINUS multiplicative_expr additive_tail | ε
        self._add_rule('additive_tail', 'PLUS', '$multiplicative_expr', '$additive_tail', label='add_op')
        self._add_rule('additive_tail', 'MINUS', '$multiplicative_expr', '$additive_tail', label='sub_op')
        self._add_rule('additive_tail', 'EPSILON', label='additive_none')
        
        # multiplicative_expr → unary_expr multiplicative_tail
        self._add_rule('multiplicative_expr', '$unary_expr', '$multiplicative_tail', label='multiplicative_expr')
        
        # multiplicative_tail → STAR unary_expr multiplicative_tail
        #                     | SLASH unary_expr multiplicative_tail
        #                     | PERCENT unary_expr multiplicative_tail | ε
        self._add_rule('multiplicative_tail', 'STAR', '$unary_expr', '$multiplicative_tail', label='mul_op')
        self._add_rule('multiplicative_tail', 'SLASH', '$unary_expr', '$multiplicative_tail', label='div_op')
        self._add_rule('multiplicative_tail', 'PERCENT', '$unary_expr', '$multiplicative_tail', label='mod_op')
        self._add_rule('multiplicative_tail', 'EPSILON', label='multiplicative_none')
        
        # unary_expr → MINUS unary_expr | NOT unary_expr | TILDE unary_expr | postfix_expr
        self._add_rule('unary_expr', 'MINUS', '$unary_expr', label='unary_neg')
        self._add_rule('unary_expr', 'NOT', '$unary_expr', label='unary_not')
        self._add_rule('unary_expr', 'TILDE', '$unary_expr', label='unary_bitnot')
        self._add_rule('unary_expr', '$postfix_expr', label='unary_postfix')
        
        # postfix_expr → primary_expr postfix_tail
        self._add_rule('postfix_expr', '$primary_expr', '$postfix_tail', label='postfix_expr')
        
        # postfix_tail → DOT IDENTIFIER postfix_tail
        #              | LBRACKET expression RBRACKET postfix_tail
        #              | LPAREN arg_list RPAREN postfix_tail | ε
        self._add_rule('postfix_tail', 'DOT', 'IDENTIFIER', '$postfix_tail', label='postfix_member')
        self._add_rule('postfix_tail', 'LBRACKET', '$expression', 'RBRACKET', '$postfix_tail', label='postfix_index')
        self._add_rule('postfix_tail', 'LPAREN', '$arg_list', 'RPAREN', '$postfix_tail', label='postfix_call')
        self._add_rule('postfix_tail', 'EPSILON', label='postfix_none')
        
        # primary_expr → INTEGER_LITERAL | FLOAT_LITERAL | STRING_LITERAL
        #              | KW_TRUE | KW_FALSE | KW_NULL
        #              | IDENTIFIER | LPAREN expression RPAREN
        #              | array_literal | object_literal
        self._add_rule('primary_expr', 'INTEGER_LITERAL', label='primary_int')
        self._add_rule('primary_expr', 'FLOAT_LITERAL', label='primary_float')
        self._add_rule('primary_expr', 'STRING_LITERAL', label='primary_string')
        self._add_rule('primary_expr', 'KW_TRUE', label='primary_true')
        self._add_rule('primary_expr', 'KW_FALSE', label='primary_false')
        self._add_rule('primary_expr', 'KW_NULL', label='primary_null')
        self._add_rule('primary_expr', 'IDENTIFIER', label='primary_ident')
        self._add_rule('primary_expr', 'LPAREN', '$expression', 'RPAREN', label='primary_paren')
        self._add_rule('primary_expr', '$array_literal', label='primary_array')
        self._add_rule('primary_expr', '$object_literal', label='primary_object')
        
        # array_literal → LBRACKET expression_list RBRACKET
        #               | LBRACKET RBRACKET
        self._add_rule('array_literal', 'LBRACKET', '$expression_list', 'RBRACKET', label='array_lit')
        self._add_rule('array_literal', 'LBRACKET', 'RBRACKET', label='array_empty')
        
        # expression_list → expression COMMA expression_list | expression
        self._add_rule('expression_list', '$expression', 'COMMA', '$expression_list', label='expr_list_more')
        self._add_rule('expression_list', '$expression', label='expr_list_one')
        
        # object_literal → LBRACE field_init_list RBRACE
        #                | LBRACE RBRACE
        self._add_rule('object_literal', 'LBRACE', '$field_init_list', 'RBRACE', label='object_lit')
        self._add_rule('object_literal', 'LBRACE', 'RBRACE', label='object_empty')
        
        # field_init_list → field_init COMMA field_init_list | field_init
        self._add_rule('field_init_list', '$field_init', 'COMMA', '$field_init_list', label='field_init_list_more')
        self._add_rule('field_init_list', '$field_init', label='field_init_list_one')
        
        # field_init → IDENTIFIER COLON expression
        self._add_rule('field_init', 'IDENTIFIER', 'COLON', '$expression', label='field_init')
        
        # arg_list → arg_list_items | ε
        self._add_rule('arg_list', '$arg_list_items', label='arg_list')
        self._add_rule('arg_list', 'EPSILON', label='arg_list_empty')
        
        # arg_list_items → expression COMMA arg_list_items | expression
        self._add_rule('arg_list_items', '$expression', 'COMMA', '$arg_list_items', label='arg_list_more')
        self._add_rule('arg_list_items', '$expression', label='arg_list_one')
    
    def _build_ir_rules(self):
        """Build IR definition rules."""
        # ir_def → KW_IR IDENTIFIER type_params_opt LBRACE ir_body RBRACE
        self._add_rule('ir_def', 'KW_IR', 'IDENTIFIER', '$type_params_opt', 'LBRACE', '$ir_body', 'RBRACE', label='ir_def')
        
        # ir_body → ir_member ir_body | ε
        self._add_rule('ir_body', '$ir_member', '$ir_body', label='ir_body_list')
        self._add_rule('ir_body', 'EPSILON', label='ir_body_empty')
        
        # ir_member → ir_field | ir_child | ir_op
        self._add_rule('ir_member', '$ir_field', label='ir_member_field')
        self._add_rule('ir_member', '$ir_child', label='ir_member_child')
        self._add_rule('ir_member', '$ir_op', label='ir_member_op')
        
        # ir_field → IDENTIFIER COLON type_expr SEMICOLON
        self._add_rule('ir_field', 'IDENTIFIER', 'COLON', '$type_expr', 'SEMICOLON', label='ir_field')
        
        # ir_child → KW_CHILD IDENTIFIER COLON type_expr SEMICOLON
        self._add_rule('ir_child', 'KW_CHILD', 'IDENTIFIER', 'COLON', '$type_expr', 'SEMICOLON', label='ir_child')
        
        # ir_op → KW_OP IDENTIFIER LPAREN param_list RPAREN return_type_opt SEMICOLON
        self._add_rule('ir_op', 'KW_OP', 'IDENTIFIER', 'LPAREN', '$param_list', 'RPAREN', '$return_type_opt', 'SEMICOLON', label='ir_op')
    
    def _build_target_rules(self):
        """Build target definition rules."""
        # target_def → KW_TARGET IDENTIFIER LBRACE target_body RBRACE
        self._add_rule('target_def', 'KW_TARGET', 'IDENTIFIER', 'LBRACE', '$target_body', 'RBRACE', label='target_def')
        
        # target_body → target_member target_body | ε
        self._add_rule('target_body', '$target_member', '$target_body', label='target_body_list')
        self._add_rule('target_body', 'EPSILON', label='target_body_empty')
        
        # target_member → target_option | emit_rule
        self._add_rule('target_member', '$target_option', label='target_member_option')
        self._add_rule('target_member', '$emit_rule', label='target_member_emit')
        
        # target_option → IDENTIFIER COLON expression SEMICOLON
        self._add_rule('target_option', 'IDENTIFIER', 'COLON', '$expression', 'SEMICOLON', label='target_option')
        
        # emit_rule → KW_EMIT IDENTIFIER LPAREN param_list RPAREN block
        self._add_rule('emit_rule', 'KW_EMIT', 'IDENTIFIER', 'LPAREN', '$param_list', 'RPAREN', '$block', label='emit_rule')
    
    def _build_pattern_rules(self):
        """Build pattern rules for match statements."""
        # pattern → IDENTIFIER | literal | IDENTIFIER LPAREN pattern_list RPAREN
        #         | LBRACKET pattern_list RBRACKET | _
        self._add_rule('pattern', 'IDENTIFIER', label='pattern_ident')
        self._add_rule('pattern', 'INTEGER_LITERAL', label='pattern_int')
        self._add_rule('pattern', 'FLOAT_LITERAL', label='pattern_float')
        self._add_rule('pattern', 'STRING_LITERAL', label='pattern_string')
        self._add_rule('pattern', 'KW_TRUE', label='pattern_true')
        self._add_rule('pattern', 'KW_FALSE', label='pattern_false')
        self._add_rule('pattern', 'KW_NULL', label='pattern_null')
        self._add_rule('pattern', 'IDENTIFIER', 'LPAREN', '$pattern_list', 'RPAREN', label='pattern_construct')
        self._add_rule('pattern', 'LBRACKET', '$pattern_list', 'RBRACKET', label='pattern_array')
        
        # pattern_list → pattern_list_items | ε
        self._add_rule('pattern_list', '$pattern_list_items', label='pattern_list')
        self._add_rule('pattern_list', 'EPSILON', label='pattern_list_empty')
        
        # pattern_list_items → pattern COMMA pattern_list_items | pattern
        self._add_rule('pattern_list_items', '$pattern', 'COMMA', '$pattern_list_items', label='pattern_list_more')
        self._add_rule('pattern_list_items', '$pattern', label='pattern_list_one')
    
    def build(self) -> Grammar:
        """
        Build complete STUNIR grammar.
        
        Returns:
            Grammar object representing STUNIR language
        """
        # Build all production rules
        self._build_program_rules()
        self._build_import_export_rules()
        self._build_declaration_rules()
        self._build_type_rules()
        self._build_function_rules()
        self._build_statement_rules()
        self._build_expression_rules()
        self._build_ir_rules()
        self._build_target_rules()
        self._build_pattern_rules()
        
        # Create grammar object
        grammar = Grammar(
            name='STUNIR',
            grammar_type=GrammarType.EBNF,
            start_symbol=self._nt('program')
        )
        
        # Add all productions
        for prod in self._productions:
            grammar.add_production(prod)
        
        return grammar
    
    def get_terminals(self) -> Dict[str, Symbol]:
        """Get all terminal symbols."""
        return self._terminals.copy()
    
    def get_nonterminals(self) -> Dict[str, Symbol]:
        """Get all non-terminal symbols."""
        return self._nonterminals.copy()
    
    def get_productions(self) -> List[ProductionRule]:
        """Get all production rules."""
        return self._productions.copy()


def create_stunir_grammar() -> Grammar:
    """
    Create STUNIR grammar.
    
    Convenience function for creating the STUNIR grammar.
    
    Returns:
        Grammar for STUNIR language
    """
    return STUNIRGrammarBuilder().build()

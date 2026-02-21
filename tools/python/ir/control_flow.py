#!/usr/bin/env python3
"""STUNIR Control Flow Analysis Module.

Provides control flow graph construction, dominance analysis, loop detection,
and control flow structure translation for all target languages.

This module is part of the STUNIR code generation enhancement suite.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Any, Tuple


class BlockType(Enum):
    """Types of basic blocks in CFG."""
    ENTRY = auto()
    EXIT = auto()
    NORMAL = auto()
    CONDITIONAL = auto()
    LOOP_HEADER = auto()
    LOOP_BODY = auto()
    LOOP_EXIT = auto()
    SWITCH = auto()
    CASE = auto()
    TRY = auto()
    CATCH = auto()
    FINALLY = auto()


class ControlFlowType(Enum):
    """Types of control flow structures."""
    IF = auto()
    IF_ELSE = auto()
    IF_ELIF_ELSE = auto()
    WHILE = auto()
    FOR = auto()
    DO_WHILE = auto()
    LOOP = auto()  # Infinite loop (Rust style)
    SWITCH = auto()
    MATCH = auto()  # Pattern matching (Rust, Haskell)
    TRY_CATCH = auto()
    TRY_CATCH_FINALLY = auto()
    GOTO = auto()
    BREAK = auto()
    CONTINUE = auto()
    RETURN = auto()
    RECURSION = auto()


@dataclass
class BasicBlock:
    """Represents a basic block in the CFG."""
    id: int
    block_type: BlockType = BlockType.NORMAL
    statements: List[Any] = field(default_factory=list)
    successors: List[int] = field(default_factory=list)
    predecessors: List[int] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    dominators: Set[int] = field(default_factory=set)
    immediate_dominator: Optional[int] = None
    loop_depth: int = 0
    is_loop_header: bool = False
    
    def add_statement(self, stmt: Any) -> None:
        """Add a statement to this block."""
        self.statements.append(stmt)
    
    def add_successor(self, block_id: int) -> None:
        """Add a successor block."""
        if block_id not in self.successors:
            self.successors.append(block_id)
    
    def add_predecessor(self, block_id: int) -> None:
        """Add a predecessor block."""
        if block_id not in self.predecessors:
            self.predecessors.append(block_id)


@dataclass
class LoopInfo:
    """Information about a detected loop."""
    header_id: int
    body_ids: Set[int]
    exit_ids: Set[int]
    loop_type: ControlFlowType
    back_edge_source: int
    depth: int = 0
    is_natural: bool = True  # Single entry point
    has_break: bool = False
    has_continue: bool = False


@dataclass
class BranchInfo:
    """Information about a branch/conditional."""
    condition: Any
    true_target: int
    false_target: int
    merge_point: Optional[int] = None
    is_multiway: bool = False
    case_targets: Dict[Any, int] = field(default_factory=dict)


class ControlFlowGraph:
    """Control Flow Graph with analysis capabilities."""
    
    def __init__(self):
        self.blocks: Dict[int, BasicBlock] = {}
        self.entry_id: Optional[int] = None
        self.exit_id: Optional[int] = None
        self._next_block_id = 0
        self.loops: List[LoopInfo] = []
        self.branches: List[BranchInfo] = []
        
    def create_block(self, block_type: BlockType = BlockType.NORMAL) -> BasicBlock:
        """Create a new basic block."""
        block = BasicBlock(id=self._next_block_id, block_type=block_type)
        self.blocks[block.id] = block
        self._next_block_id += 1
        return block
    
    def create_entry(self) -> BasicBlock:
        """Create the entry block."""
        block = self.create_block(BlockType.ENTRY)
        self.entry_id = block.id
        return block
    
    def create_exit(self) -> BasicBlock:
        """Create the exit block."""
        block = self.create_block(BlockType.EXIT)
        self.exit_id = block.id
        return block
    
    def add_edge(self, from_id: int, to_id: int) -> None:
        """Add an edge between blocks."""
        if from_id in self.blocks and to_id in self.blocks:
            self.blocks[from_id].add_successor(to_id)
            self.blocks[to_id].add_predecessor(from_id)
    
    def compute_dominators(self) -> None:
        """Compute dominators for all blocks using iterative algorithm."""
        if self.entry_id is None:
            return
            
        # Initialize: entry dominates only itself, others are dominated by all
        all_blocks = set(self.blocks.keys())
        for block_id in self.blocks:
            if block_id == self.entry_id:
                self.blocks[block_id].dominators = {block_id}
            else:
                self.blocks[block_id].dominators = all_blocks.copy()
        
        # Iterate until fixed point
        changed = True
        while changed:
            changed = False
            for block_id in self.blocks:
                if block_id == self.entry_id:
                    continue
                    
                block = self.blocks[block_id]
                if not block.predecessors:
                    continue
                
                # Dom(n) = {n} ∪ (∩ Dom(p) for p in pred(n))
                new_doms = all_blocks.copy()
                for pred_id in block.predecessors:
                    new_doms &= self.blocks[pred_id].dominators
                new_doms.add(block_id)
                
                if new_doms != block.dominators:
                    block.dominators = new_doms
                    changed = True
        
        # Compute immediate dominators
        self._compute_immediate_dominators()
    
    def _compute_immediate_dominators(self) -> None:
        """Compute immediate dominator for each block."""
        for block_id, block in self.blocks.items():
            if block_id == self.entry_id:
                continue
            
            strict_doms = block.dominators - {block_id}
            if not strict_doms:
                continue
            
            # idom is the unique dominator that doesn't dominate any other strict dominator
            for candidate in strict_doms:
                is_idom = True
                for other in strict_doms:
                    if other != candidate and candidate in self.blocks[other].dominators:
                        is_idom = False
                        break
                if is_idom:
                    block.immediate_dominator = candidate
                    break
    
    def detect_loops(self) -> List[LoopInfo]:
        """Detect natural loops in the CFG."""
        self.loops = []
        self.compute_dominators()
        
        # Find back edges (n -> d where d dominates n)
        for block_id, block in self.blocks.items():
            for succ_id in block.successors:
                succ = self.blocks[succ_id]
                if succ_id in block.dominators:
                    # Found a back edge: block_id -> succ_id
                    loop = self._identify_loop(succ_id, block_id)
                    self.loops.append(loop)
        
        # Compute loop depths
        self._compute_loop_depths()
        return self.loops
    
    def _identify_loop(self, header_id: int, back_edge_source: int) -> LoopInfo:
        """Identify all blocks in a natural loop."""
        body = {header_id}
        stack = [back_edge_source]
        
        while stack:
            block_id = stack.pop()
            if block_id not in body:
                body.add(block_id)
                for pred_id in self.blocks[block_id].predecessors:
                    if pred_id not in body:
                        stack.append(pred_id)
        
        # Find exit nodes (blocks in loop with successors outside)
        exits = set()
        for block_id in body:
            for succ_id in self.blocks[block_id].successors:
                if succ_id not in body:
                    exits.add(block_id)
        
        # Mark header
        self.blocks[header_id].is_loop_header = True
        self.blocks[header_id].block_type = BlockType.LOOP_HEADER
        
        # Determine loop type
        loop_type = self._classify_loop(header_id, body)
        
        return LoopInfo(
            header_id=header_id,
            body_ids=body,
            exit_ids=exits,
            loop_type=loop_type,
            back_edge_source=back_edge_source
        )
    
    def _classify_loop(self, header_id: int, body: Set[int]) -> ControlFlowType:
        """Classify the type of loop based on structure."""
        header = self.blocks[header_id]
        
        # Check for for-loop pattern (init; cond; update)
        # Heuristic: header has 2 successors and init block before it
        if len(header.successors) == 2 and header.predecessors:
            has_condition = any(
                isinstance(stmt, dict) and stmt.get('type') == 'branch'
                for stmt in header.statements
            )
            if has_condition:
                return ControlFlowType.FOR
        
        # Check for do-while (entry into body, condition at end)
        # Heuristic: back edge comes from a conditional block
        back_sources = [
            bid for bid in body 
            if header_id in self.blocks[bid].successors and bid != header_id
        ]
        if back_sources:
            back_block = self.blocks[back_sources[0]]
            if back_block.block_type == BlockType.CONDITIONAL:
                return ControlFlowType.DO_WHILE
        
        # Default to while loop
        if len(header.successors) == 2:
            return ControlFlowType.WHILE
        
        return ControlFlowType.LOOP
    
    def _compute_loop_depths(self) -> None:
        """Compute loop nesting depth for all blocks."""
        for block_id in self.blocks:
            depth = sum(1 for loop in self.loops if block_id in loop.body_ids)
            self.blocks[block_id].loop_depth = depth
            
        for i, loop in enumerate(self.loops):
            loop.depth = sum(
                1 for other in self.loops 
                if other != loop and loop.header_id in other.body_ids
            )
    
    def detect_branches(self) -> List[BranchInfo]:
        """Detect conditional branches and their merge points."""
        self.branches = []
        
        for block_id, block in self.blocks.items():
            if block.block_type == BlockType.CONDITIONAL or len(block.successors) > 1:
                # Extract condition from last statement
                condition = None
                if block.statements:
                    last = block.statements[-1]
                    if isinstance(last, dict) and last.get('type') in ('branch', 'cond'):
                        condition = last.get('condition')
                
                if len(block.successors) == 2:
                    true_target, false_target = block.successors[0], block.successors[1]
                    merge = self._find_merge_point(true_target, false_target)
                    branch = BranchInfo(
                        condition=condition,
                        true_target=true_target,
                        false_target=false_target,
                        merge_point=merge
                    )
                    self.branches.append(branch)
                elif len(block.successors) > 2:
                    # Switch/match statement
                    branch = BranchInfo(
                        condition=condition,
                        true_target=block.successors[0],
                        false_target=block.successors[-1],
                        is_multiway=True,
                        case_targets={i: s for i, s in enumerate(block.successors)}
                    )
                    self.branches.append(branch)
        
        return self.branches
    
    def _find_merge_point(self, block_a: int, block_b: int) -> Optional[int]:
        """Find the merge point (common dominator) of two blocks."""
        # Get blocks reachable from both
        reachable_a = self._get_reachable(block_a)
        reachable_b = self._get_reachable(block_b)
        common = reachable_a & reachable_b
        
        if not common:
            return None
        
        # Return the first common block (post-order)
        for block_id in sorted(common):
            block = self.blocks[block_id]
            # Check if all paths to exit go through this block
            is_merge = all(
                block_id in self.blocks[a].dominators
                for a in common if a != block_id
            )
            if is_merge:
                return block_id
        
        return min(common) if common else None
    
    def _get_reachable(self, start_id: int) -> Set[int]:
        """Get all blocks reachable from start."""
        visited = set()
        stack = [start_id]
        while stack:
            block_id = stack.pop()
            if block_id not in visited:
                visited.add(block_id)
                stack.extend(self.blocks[block_id].successors)
        return visited
    
    def to_dict(self) -> Dict:
        """Convert CFG to dictionary for serialization."""
        return {
            'blocks': {
                str(bid): {
                    'id': block.id,
                    'type': block.block_type.name,
                    'successors': block.successors,
                    'predecessors': block.predecessors,
                    'loop_depth': block.loop_depth,
                    'is_loop_header': block.is_loop_header,
                    'statement_count': len(block.statements)
                }
                for bid, block in self.blocks.items()
            },
            'entry_id': self.entry_id,
            'exit_id': self.exit_id,
            'loop_count': len(self.loops),
            'branch_count': len(self.branches)
        }


class ControlFlowAnalyzer:
    """Analyzes control flow patterns in IR and builds CFG."""
    
    def __init__(self, ir_data: Dict):
        self.ir_data = ir_data
        self.cfg = ControlFlowGraph()
        self.label_to_block: Dict[str, int] = {}
        self.pending_gotos: List[Tuple[int, str]] = []
    
    def analyze(self) -> ControlFlowGraph:
        """Analyze IR and build CFG."""
        functions = self.ir_data.get('ir_functions', [])
        
        for func in functions:
            self._analyze_function(func)
        
        # Resolve pending gotos
        self._resolve_gotos()
        
        # Run analyses
        self.cfg.compute_dominators()
        self.cfg.detect_loops()
        self.cfg.detect_branches()
        
        return self.cfg
    
    def _analyze_function(self, func: Dict) -> None:
        """Analyze a single function's control flow."""
        entry = self.cfg.create_entry()
        current_block = entry
        
        body = func.get('body', [])
        for stmt in body:
            current_block = self._analyze_statement(stmt, current_block)
        
        # Create exit if needed
        if self.cfg.exit_id is None:
            exit_block = self.cfg.create_exit()
            self.cfg.add_edge(current_block.id, exit_block.id)
    
    def _analyze_statement(self, stmt: Any, current: BasicBlock) -> BasicBlock:
        """Analyze a statement and update CFG."""
        if not isinstance(stmt, dict):
            current.add_statement(stmt)
            return current
        
        stmt_type = stmt.get('type', '')
        
        if stmt_type == 'if':
            return self._analyze_if(stmt, current)
        elif stmt_type == 'while':
            return self._analyze_while(stmt, current)
        elif stmt_type == 'for':
            return self._analyze_for(stmt, current)
        elif stmt_type == 'do_while':
            return self._analyze_do_while(stmt, current)
        elif stmt_type == 'switch':
            return self._analyze_switch(stmt, current)
        elif stmt_type == 'try':
            return self._analyze_try(stmt, current)
        elif stmt_type == 'loop':
            return self._analyze_loop(stmt, current)
        elif stmt_type == 'goto':
            return self._analyze_goto(stmt, current)
        elif stmt_type == 'label':
            return self._analyze_label(stmt, current)
        elif stmt_type == 'return':
            return self._analyze_return(stmt, current)
        elif stmt_type == 'break':
            current.add_statement(stmt)
            return current
        elif stmt_type == 'continue':
            current.add_statement(stmt)
            return current
        else:
            current.add_statement(stmt)
            return current
    
    def _analyze_if(self, stmt: Dict, current: BasicBlock) -> BasicBlock:
        """Analyze if/else statement."""
        current.block_type = BlockType.CONDITIONAL
        current.add_statement({'type': 'branch', 'condition': stmt.get('cond')})
        
        # Create blocks
        then_block = self.cfg.create_block()
        merge_block = self.cfg.create_block()
        
        # Process 'then' branch
        self.cfg.add_edge(current.id, then_block.id)
        then_end = then_block
        for s in stmt.get('then', []):
            then_end = self._analyze_statement(s, then_end)
        self.cfg.add_edge(then_end.id, merge_block.id)
        
        # Process 'else' branch
        if 'else' in stmt:
            else_block = self.cfg.create_block()
            self.cfg.add_edge(current.id, else_block.id)
            else_end = else_block
            for s in stmt.get('else', []):
                else_end = self._analyze_statement(s, else_end)
            self.cfg.add_edge(else_end.id, merge_block.id)
        else:
            self.cfg.add_edge(current.id, merge_block.id)
        
        # Process 'elif' branches
        for elif_clause in stmt.get('elif', []):
            elif_block = self.cfg.create_block(BlockType.CONDITIONAL)
            elif_block.add_statement({'type': 'branch', 'condition': elif_clause.get('cond')})
            elif_then = self.cfg.create_block()
            self.cfg.add_edge(elif_block.id, elif_then.id)
            elif_end = elif_then
            for s in elif_clause.get('then', []):
                elif_end = self._analyze_statement(s, elif_end)
            self.cfg.add_edge(elif_end.id, merge_block.id)
        
        return merge_block
    
    def _analyze_while(self, stmt: Dict, current: BasicBlock) -> BasicBlock:
        """Analyze while loop."""
        # Create header block
        header = self.cfg.create_block(BlockType.LOOP_HEADER)
        header.is_loop_header = True
        header.add_statement({'type': 'branch', 'condition': stmt.get('cond')})
        self.cfg.add_edge(current.id, header.id)
        
        # Create body and exit blocks
        body_block = self.cfg.create_block(BlockType.LOOP_BODY)
        exit_block = self.cfg.create_block(BlockType.LOOP_EXIT)
        
        self.cfg.add_edge(header.id, body_block.id)  # condition true
        self.cfg.add_edge(header.id, exit_block.id)  # condition false
        
        # Process body
        body_end = body_block
        for s in stmt.get('body', []):
            body_end = self._analyze_statement(s, body_end)
        
        # Back edge
        self.cfg.add_edge(body_end.id, header.id)
        
        return exit_block
    
    def _analyze_for(self, stmt: Dict, current: BasicBlock) -> BasicBlock:
        """Analyze for loop."""
        # Init block
        init_block = self.cfg.create_block()
        if 'init' in stmt:
            init_block.add_statement(stmt['init'])
        self.cfg.add_edge(current.id, init_block.id)
        
        # Header with condition
        header = self.cfg.create_block(BlockType.LOOP_HEADER)
        header.is_loop_header = True
        header.add_statement({'type': 'branch', 'condition': stmt.get('cond')})
        self.cfg.add_edge(init_block.id, header.id)
        
        # Body and exit
        body_block = self.cfg.create_block(BlockType.LOOP_BODY)
        exit_block = self.cfg.create_block(BlockType.LOOP_EXIT)
        
        self.cfg.add_edge(header.id, body_block.id)
        self.cfg.add_edge(header.id, exit_block.id)
        
        # Process body
        body_end = body_block
        for s in stmt.get('body', []):
            body_end = self._analyze_statement(s, body_end)
        
        # Update block
        update_block = self.cfg.create_block()
        if 'update' in stmt:
            update_block.add_statement(stmt['update'])
        self.cfg.add_edge(body_end.id, update_block.id)
        self.cfg.add_edge(update_block.id, header.id)
        
        return exit_block
    
    def _analyze_do_while(self, stmt: Dict, current: BasicBlock) -> BasicBlock:
        """Analyze do-while loop."""
        body_block = self.cfg.create_block(BlockType.LOOP_BODY)
        self.cfg.add_edge(current.id, body_block.id)
        
        # Process body
        body_end = body_block
        for s in stmt.get('body', []):
            body_end = self._analyze_statement(s, body_end)
        
        # Condition block (at end)
        cond_block = self.cfg.create_block(BlockType.CONDITIONAL)
        cond_block.add_statement({'type': 'branch', 'condition': stmt.get('cond')})
        self.cfg.add_edge(body_end.id, cond_block.id)
        
        # Back edge and exit
        exit_block = self.cfg.create_block(BlockType.LOOP_EXIT)
        self.cfg.add_edge(cond_block.id, body_block.id)  # true - loop back
        self.cfg.add_edge(cond_block.id, exit_block.id)  # false - exit
        
        # Mark header
        body_block.is_loop_header = True
        body_block.block_type = BlockType.LOOP_HEADER
        
        return exit_block
    
    def _analyze_switch(self, stmt: Dict, current: BasicBlock) -> BasicBlock:
        """Analyze switch/case statement."""
        current.block_type = BlockType.SWITCH
        current.add_statement({'type': 'switch', 'value': stmt.get('value')})
        
        merge_block = self.cfg.create_block()
        cases = stmt.get('cases', [])
        default_case = stmt.get('default', [])
        
        prev_case = None
        for case in cases:
            case_block = self.cfg.create_block(BlockType.CASE)
            case_block.add_statement({'type': 'case', 'value': case.get('value')})
            self.cfg.add_edge(current.id, case_block.id)
            
            # Handle fallthrough
            if prev_case and case.get('fallthrough', False):
                self.cfg.add_edge(prev_case.id, case_block.id)
            
            case_end = case_block
            for s in case.get('body', []):
                case_end = self._analyze_statement(s, case_end)
            
            if not case.get('fallthrough', False):
                self.cfg.add_edge(case_end.id, merge_block.id)
            
            prev_case = case_end
        
        # Default case
        if default_case:
            default_block = self.cfg.create_block(BlockType.CASE)
            self.cfg.add_edge(current.id, default_block.id)
            default_end = default_block
            for s in default_case:
                default_end = self._analyze_statement(s, default_end)
            self.cfg.add_edge(default_end.id, merge_block.id)
        else:
            self.cfg.add_edge(current.id, merge_block.id)
        
        return merge_block
    
    def _analyze_try(self, stmt: Dict, current: BasicBlock) -> BasicBlock:
        """Analyze try/catch/finally statement."""
        try_block = self.cfg.create_block(BlockType.TRY)
        self.cfg.add_edge(current.id, try_block.id)
        
        # Process try body
        try_end = try_block
        for s in stmt.get('try', []):
            try_end = self._analyze_statement(s, try_end)
        
        merge_block = self.cfg.create_block()
        
        # Catch blocks
        for catch in stmt.get('catch', []):
            catch_block = self.cfg.create_block(BlockType.CATCH)
            catch_block.add_statement({'type': 'catch', 'exception': catch.get('exception')})
            self.cfg.add_edge(try_block.id, catch_block.id)  # Exception edge
            
            catch_end = catch_block
            for s in catch.get('body', []):
                catch_end = self._analyze_statement(s, catch_end)
            self.cfg.add_edge(catch_end.id, merge_block.id)
        
        # Finally block
        if 'finally' in stmt:
            finally_block = self.cfg.create_block(BlockType.FINALLY)
            self.cfg.add_edge(try_end.id, finally_block.id)
            
            finally_end = finally_block
            for s in stmt.get('finally', []):
                finally_end = self._analyze_statement(s, finally_end)
            self.cfg.add_edge(finally_end.id, merge_block.id)
        else:
            self.cfg.add_edge(try_end.id, merge_block.id)
        
        return merge_block
    
    def _analyze_loop(self, stmt: Dict, current: BasicBlock) -> BasicBlock:
        """Analyze infinite loop (Rust style)."""
        header = self.cfg.create_block(BlockType.LOOP_HEADER)
        header.is_loop_header = True
        self.cfg.add_edge(current.id, header.id)
        
        body_end = header
        for s in stmt.get('body', []):
            body_end = self._analyze_statement(s, body_end)
        
        # Back edge (always loops unless break)
        self.cfg.add_edge(body_end.id, header.id)
        
        exit_block = self.cfg.create_block(BlockType.LOOP_EXIT)
        self.cfg.add_edge(header.id, exit_block.id)  # break target
        
        return exit_block
    
    def _analyze_goto(self, stmt: Dict, current: BasicBlock) -> BasicBlock:
        """Analyze goto statement."""
        label = stmt.get('label', '')
        current.add_statement(stmt)
        
        # Track pending goto for later resolution
        self.pending_gotos.append((current.id, label))
        
        # Create new block after goto
        next_block = self.cfg.create_block()
        return next_block
    
    def _analyze_label(self, stmt: Dict, current: BasicBlock) -> BasicBlock:
        """Analyze label statement."""
        label = stmt.get('name', '')
        
        # Start new block at label
        label_block = self.cfg.create_block()
        label_block.labels.append(label)
        self.label_to_block[label] = label_block.id
        
        self.cfg.add_edge(current.id, label_block.id)
        return label_block
    
    def _analyze_return(self, stmt: Dict, current: BasicBlock) -> BasicBlock:
        """Analyze return statement."""
        current.add_statement(stmt)
        
        # Connect to exit
        if self.cfg.exit_id is None:
            self.cfg.create_exit()
        self.cfg.add_edge(current.id, self.cfg.exit_id)
        
        # Create new block (unreachable after return)
        return self.cfg.create_block()
    
    def _resolve_gotos(self) -> None:
        """Resolve pending goto targets."""
        for block_id, label in self.pending_gotos:
            if label in self.label_to_block:
                target_id = self.label_to_block[label]
                self.cfg.add_edge(block_id, target_id)


class ControlFlowTranslator:
    """Translates control flow structures to target language code."""
    
    def __init__(self, target: str):
        self.target = target
        self.indent_level = 0
        self.indent_str = '    '
    
    def _indent(self) -> str:
        return self.indent_str * self.indent_level
    
    def translate_if(self, condition: str, then_body: List[str], 
                     else_body: Optional[List[str]] = None,
                     elif_clauses: Optional[List[Tuple[str, List[str]]]] = None) -> str:
        """Translate if/else to target language."""
        lines = []
        
        if self.target == 'python':
            lines.append(f'{self._indent()}if {condition}:')
            self.indent_level += 1
            lines.extend([f'{self._indent()}{s}' for s in then_body])
            self.indent_level -= 1
            
            for elif_cond, elif_body in (elif_clauses or []):
                lines.append(f'{self._indent()}elif {elif_cond}:')
                self.indent_level += 1
                lines.extend([f'{self._indent()}{s}' for s in elif_body])
                self.indent_level -= 1
            
            if else_body:
                lines.append(f'{self._indent()}else:')
                self.indent_level += 1
                lines.extend([f'{self._indent()}{s}' for s in else_body])
                self.indent_level -= 1
                
        elif self.target == 'rust':
            lines.append(f'{self._indent()}if {condition} {{')
            self.indent_level += 1
            lines.extend([f'{self._indent()}{s}' for s in then_body])
            self.indent_level -= 1
            
            for elif_cond, elif_body in (elif_clauses or []):
                lines.append(f'{self._indent()}}} else if {elif_cond} {{')
                self.indent_level += 1
                lines.extend([f'{self._indent()}{s}' for s in elif_body])
                self.indent_level -= 1
            
            if else_body:
                lines.append(f'{self._indent()}}} else {{')
                self.indent_level += 1
                lines.extend([f'{self._indent()}{s}' for s in else_body])
                self.indent_level -= 1
            lines.append(f'{self._indent()}}}')
            
        elif self.target == 'haskell':
            lines.append(f'{self._indent()}if {condition}')
            self.indent_level += 1
            lines.append(f'{self._indent()}then {" ".join(then_body) if then_body else "()"}')
            if else_body:
                lines.append(f'{self._indent()}else {" ".join(else_body)}')
            else:
                lines.append(f'{self._indent()}else ()')
            self.indent_level -= 1
            
        elif self.target in ('c89', 'c99', 'c'):
            lines.append(f'{self._indent()}if ({condition}) {{')
            self.indent_level += 1
            lines.extend([f'{self._indent()}{s}' for s in then_body])
            self.indent_level -= 1
            
            for elif_cond, elif_body in (elif_clauses or []):
                lines.append(f'{self._indent()}}} else if ({elif_cond}) {{')
                self.indent_level += 1
                lines.extend([f'{self._indent()}{s}' for s in elif_body])
                self.indent_level -= 1
            
            if else_body:
                lines.append(f'{self._indent()}}} else {{')
                self.indent_level += 1
                lines.extend([f'{self._indent()}{s}' for s in else_body])
                self.indent_level -= 1
            lines.append(f'{self._indent()}}}')
            
        elif self.target in ('asm', 'x86', 'arm'):
            # Assembly uses conditional jumps
            label_else = f'_else_{id(condition) % 10000}'
            label_end = f'_endif_{id(condition) % 10000}'
            lines.append(f'{self._indent()}; if {condition}')
            lines.append(f'{self._indent()}cmp eax, 0')
            lines.append(f'{self._indent()}je {label_else}')
            lines.extend([f'{self._indent()}{s}' for s in then_body])
            lines.append(f'{self._indent()}jmp {label_end}')
            lines.append(f'{label_else}:')
            if else_body:
                lines.extend([f'{self._indent()}{s}' for s in else_body])
            lines.append(f'{label_end}:')
        
        return '\n'.join(lines)
    
    def translate_while(self, condition: str, body: List[str]) -> str:
        """Translate while loop to target language."""
        lines = []
        
        if self.target == 'python':
            lines.append(f'{self._indent()}while {condition}:')
            self.indent_level += 1
            lines.extend([f'{self._indent()}{s}' for s in body])
            self.indent_level -= 1
            
        elif self.target == 'rust':
            lines.append(f'{self._indent()}while {condition} {{')
            self.indent_level += 1
            lines.extend([f'{self._indent()}{s}' for s in body])
            self.indent_level -= 1
            lines.append(f'{self._indent()}}}')
            
        elif self.target == 'haskell':
            # Haskell uses recursion instead of while
            loop_name = f'loop_{id(condition) % 10000}'
            lines.append(f'{self._indent()}let {loop_name} = if {condition}')
            self.indent_level += 1
            lines.append(f'{self._indent()}then do')
            self.indent_level += 1
            lines.extend([f'{self._indent()}{s}' for s in body])
            lines.append(f'{self._indent()}{loop_name}')
            self.indent_level -= 1
            lines.append(f'{self._indent()}else return ()')
            self.indent_level -= 1
            lines.append(f'{self._indent()}in {loop_name}')
            
        elif self.target in ('c89', 'c99', 'c'):
            lines.append(f'{self._indent()}while ({condition}) {{')
            self.indent_level += 1
            lines.extend([f'{self._indent()}{s}' for s in body])
            self.indent_level -= 1
            lines.append(f'{self._indent()}}}')
            
        elif self.target in ('asm', 'x86', 'arm'):
            label_start = f'_while_{id(condition) % 10000}'
            label_end = f'_endwhile_{id(condition) % 10000}'
            lines.append(f'{label_start}:')
            lines.append(f'{self._indent()}; while {condition}')
            lines.append(f'{self._indent()}cmp eax, 0')
            lines.append(f'{self._indent()}je {label_end}')
            lines.extend([f'{self._indent()}{s}' for s in body])
            lines.append(f'{self._indent()}jmp {label_start}')
            lines.append(f'{label_end}:')
        
        return '\n'.join(lines)
    
    def translate_for(self, init: str, condition: str, update: str, body: List[str]) -> str:
        """Translate for loop to target language."""
        lines = []
        
        if self.target == 'python':
            # Python uses for-in, convert C-style to range-based
            lines.append(f'{self._indent()}# for {init}; {condition}; {update}')
            lines.append(f'{self._indent()}for _ in range(...):'  )  # Placeholder
            self.indent_level += 1
            lines.extend([f'{self._indent()}{s}' for s in body])
            self.indent_level -= 1
            
        elif self.target == 'rust':
            # Rust prefers for-in, but supports while-based for
            lines.append(f'{self._indent()}// for {init}; {condition}; {update}')
            lines.append(f'{self._indent()}{init};')
            lines.append(f'{self._indent()}while {condition} {{')
            self.indent_level += 1
            lines.extend([f'{self._indent()}{s}' for s in body])
            lines.append(f'{self._indent()}{update};')
            self.indent_level -= 1
            lines.append(f'{self._indent()}}}')
            
        elif self.target == 'haskell':
            # Use recursive helper with state
            lines.append(f'{self._indent()}-- for {init}; {condition}; {update}')
            lines.append(f'{self._indent()}forM_ [..] $ \\_ -> do')
            self.indent_level += 1
            lines.extend([f'{self._indent()}{s}' for s in body])
            self.indent_level -= 1
            
        elif self.target in ('c89', 'c99', 'c'):
            lines.append(f'{self._indent()}for ({init}; {condition}; {update}) {{')
            self.indent_level += 1
            lines.extend([f'{self._indent()}{s}' for s in body])
            self.indent_level -= 1
            lines.append(f'{self._indent()}}}')
            
        elif self.target in ('asm', 'x86', 'arm'):
            label_start = f'_for_{id(condition) % 10000}'
            label_end = f'_endfor_{id(condition) % 10000}'
            lines.append(f'{self._indent()}; {init}')
            lines.append(f'{label_start}:')
            lines.append(f'{self._indent()}; check {condition}')
            lines.append(f'{self._indent()}cmp eax, 0')
            lines.append(f'{self._indent()}je {label_end}')
            lines.extend([f'{self._indent()}{s}' for s in body])
            lines.append(f'{self._indent()}; {update}')
            lines.append(f'{self._indent()}jmp {label_start}')
            lines.append(f'{label_end}:')
        
        return '\n'.join(lines)
    
    def translate_switch(self, value: str, cases: List[Tuple[str, List[str]]], 
                        default: Optional[List[str]] = None) -> str:
        """Translate switch/case to target language."""
        lines = []
        
        if self.target == 'python':
            # Python 3.10+ match
            lines.append(f'{self._indent()}match {value}:')
            self.indent_level += 1
            for case_val, case_body in cases:
                lines.append(f'{self._indent()}case {case_val}:')
                self.indent_level += 1
                lines.extend([f'{self._indent()}{s}' for s in case_body])
                self.indent_level -= 1
            if default:
                lines.append(f'{self._indent()}case _:')
                self.indent_level += 1
                lines.extend([f'{self._indent()}{s}' for s in default])
                self.indent_level -= 1
            self.indent_level -= 1
            
        elif self.target == 'rust':
            lines.append(f'{self._indent()}match {value} {{')
            self.indent_level += 1
            for case_val, case_body in cases:
                body_str = ' '.join(case_body) if case_body else '()'
                lines.append(f'{self._indent()}{case_val} => {{ {body_str} }}')
            if default:
                body_str = ' '.join(default)
                lines.append(f'{self._indent()}_ => {{ {body_str} }}')
            self.indent_level -= 1
            lines.append(f'{self._indent()}}}')
            
        elif self.target == 'haskell':
            lines.append(f'{self._indent()}case {value} of')
            self.indent_level += 1
            for case_val, case_body in cases:
                body_str = ' '.join(case_body) if case_body else '()'
                lines.append(f'{self._indent()}{case_val} -> {body_str}')
            if default:
                body_str = ' '.join(default)
                lines.append(f'{self._indent()}_ -> {body_str}')
            self.indent_level -= 1
            
        elif self.target in ('c89', 'c99', 'c'):
            lines.append(f'{self._indent()}switch ({value}) {{')
            self.indent_level += 1
            for case_val, case_body in cases:
                lines.append(f'{self._indent()}case {case_val}:')
                self.indent_level += 1
                lines.extend([f'{self._indent()}{s}' for s in case_body])
                lines.append(f'{self._indent()}break;')
                self.indent_level -= 1
            if default:
                lines.append(f'{self._indent()}default:')
                self.indent_level += 1
                lines.extend([f'{self._indent()}{s}' for s in default])
                self.indent_level -= 1
            self.indent_level -= 1
            lines.append(f'{self._indent()}}}')
            
        elif self.target in ('asm', 'x86', 'arm'):
            # Jump table or chained comparisons
            lines.append(f'{self._indent()}; switch {value}')
            for i, (case_val, case_body) in enumerate(cases):
                label = f'_case_{i}_{id(value) % 10000}'
                lines.append(f'{self._indent()}cmp eax, {case_val}')
                lines.append(f'{self._indent()}je {label}')
            lines.append(f'{self._indent()}jmp _default_{id(value) % 10000}')
            for i, (case_val, case_body) in enumerate(cases):
                label = f'_case_{i}_{id(value) % 10000}'
                lines.append(f'{label}:')
                lines.extend([f'{self._indent()}{s}' for s in case_body])
                lines.append(f'{self._indent()}jmp _endswitch_{id(value) % 10000}')
            lines.append(f'_default_{id(value) % 10000}:')
            if default:
                lines.extend([f'{self._indent()}{s}' for s in default])
            lines.append(f'_endswitch_{id(value) % 10000}:')
        
        return '\n'.join(lines)
    
    def translate_try_catch(self, try_body: List[str], 
                           catches: List[Tuple[str, List[str]]],
                           finally_body: Optional[List[str]] = None) -> str:
        """Translate try/catch/finally to target language."""
        lines = []
        
        if self.target == 'python':
            lines.append(f'{self._indent()}try:')
            self.indent_level += 1
            lines.extend([f'{self._indent()}{s}' for s in try_body])
            self.indent_level -= 1
            for exc_type, catch_body in catches:
                lines.append(f'{self._indent()}except {exc_type}:')
                self.indent_level += 1
                lines.extend([f'{self._indent()}{s}' for s in catch_body])
                self.indent_level -= 1
            if finally_body:
                lines.append(f'{self._indent()}finally:')
                self.indent_level += 1
                lines.extend([f'{self._indent()}{s}' for s in finally_body])
                self.indent_level -= 1
                
        elif self.target == 'rust':
            # Rust uses Result and ? operator instead of exceptions
            lines.append(f'{self._indent()}// Rust uses Result<T, E> instead of try/catch')
            lines.append(f'{self._indent()}let result = (|| -> Result<(), Box<dyn std::error::Error>> {{')
            self.indent_level += 1
            lines.extend([f'{self._indent()}{s}' for s in try_body])
            lines.append(f'{self._indent()}Ok(())')
            self.indent_level -= 1
            lines.append(f'{self._indent()}}})();')
            lines.append(f'{self._indent()}match result {{')
            self.indent_level += 1
            lines.append(f'{self._indent()}Ok(_) => {{}},')
            for exc_type, catch_body in catches:
                lines.append(f'{self._indent()}Err(e) if e.is::<{exc_type}>() => {{')
                self.indent_level += 1
                lines.extend([f'{self._indent()}{s}' for s in catch_body])
                self.indent_level -= 1
                lines.append(f'{self._indent()}}}')
            lines.append(f'{self._indent()}Err(_) => {{}}')
            self.indent_level -= 1
            lines.append(f'{self._indent()}}}')
            if finally_body:
                lines.append(f'{self._indent()}// finally')
                lines.extend([f'{self._indent()}{s}' for s in finally_body])
                
        elif self.target == 'haskell':
            # Haskell uses Control.Exception
            lines.append(f'{self._indent()}catch')
            self.indent_level += 1
            lines.append(f'{self._indent()}( do')
            self.indent_level += 1
            lines.extend([f'{self._indent()}{s}' for s in try_body])
            self.indent_level -= 1
            lines.append(f'{self._indent()})')
            for exc_type, catch_body in catches:
                lines.append(f'{self._indent()}(\\({exc_type} _) -> do')
                self.indent_level += 1
                lines.extend([f'{self._indent()}{s}' for s in catch_body])
                self.indent_level -= 1
                lines.append(f'{self._indent()})')
            self.indent_level -= 1
            if finally_body:
                lines.append(f'{self._indent()}finally_')
                self.indent_level += 1
                lines.extend([f'{self._indent()}{s}' for s in finally_body])
                self.indent_level -= 1
                
        elif self.target in ('c89', 'c99', 'c'):
            # C doesn't have try/catch, use setjmp/longjmp pattern
            lines.append(f'{self._indent()}/* C: Using setjmp/longjmp for exception handling */')
            lines.append(f'{self._indent()}if (setjmp(exc_env) == 0) {{')
            self.indent_level += 1
            lines.extend([f'{self._indent()}{s}' for s in try_body])
            self.indent_level -= 1
            lines.append(f'{self._indent()}}} else {{')
            self.indent_level += 1
            for exc_type, catch_body in catches:
                lines.append(f'{self._indent()}/* catch {exc_type} */')
                lines.extend([f'{self._indent()}{s}' for s in catch_body])
            self.indent_level -= 1
            lines.append(f'{self._indent()}}}')
            if finally_body:
                lines.append(f'{self._indent()}/* finally */')
                lines.extend([f'{self._indent()}{s}' for s in finally_body])
        
        return '\n'.join(lines)
    
    def translate_break(self) -> str:
        """Translate break statement."""
        if self.target == 'python':
            return f'{self._indent()}break'
        elif self.target == 'rust':
            return f'{self._indent()}break;'
        elif self.target == 'haskell':
            return f'{self._indent()}-- break (use recursion control)'
        elif self.target in ('c89', 'c99', 'c'):
            return f'{self._indent()}break;'
        elif self.target in ('asm', 'x86', 'arm'):
            return f'{self._indent()}jmp _loop_exit'
        return ''
    
    def translate_continue(self) -> str:
        """Translate continue statement."""
        if self.target == 'python':
            return f'{self._indent()}continue'
        elif self.target == 'rust':
            return f'{self._indent()}continue;'
        elif self.target == 'haskell':
            return f'{self._indent()}-- continue (use recursion)'
        elif self.target in ('c89', 'c99', 'c'):
            return f'{self._indent()}continue;'
        elif self.target in ('asm', 'x86', 'arm'):
            return f'{self._indent()}jmp _loop_start'
        return ''
    
    def translate_goto(self, label: str) -> str:
        """Translate goto statement."""
        if self.target in ('c89', 'c99', 'c'):
            return f'{self._indent()}goto {label};'
        elif self.target in ('asm', 'x86', 'arm'):
            return f'{self._indent()}jmp {label}'
        elif self.target == 'rust':
            return f'{self._indent()}// goto not supported in Rust'
        elif self.target == 'python':
            return f'{self._indent()}# goto not supported in Python'
        elif self.target == 'haskell':
            return f'{self._indent()}-- goto not supported in Haskell'
        return ''
    
    def translate_label(self, name: str) -> str:
        """Translate label declaration."""
        if self.target in ('c89', 'c99', 'c'):
            return f'{name}:'
        elif self.target in ('asm', 'x86', 'arm'):
            return f'{name}:'
        elif self.target == 'rust':
            return f"'{name}: loop {{"  # Rust uses labeled loops
        elif self.target == 'python':
            return f'{self._indent()}# label: {name}'
        elif self.target == 'haskell':
            return f'{self._indent()}-- label: {name}'
        return ''


# Export public API
__all__ = [
    'BasicBlock', 'BlockType', 'ControlFlowType',
    'LoopInfo', 'BranchInfo',
    'ControlFlowGraph', 'ControlFlowAnalyzer', 'ControlFlowTranslator'
]

"""Category-specific parsers for Semantic IR.

This package provides specialized parsers for all 24 target categories.
"""

from .base import CategoryParser
from .embedded import EmbeddedParser
from .assembly import AssemblyParser
from .polyglot import PolyglotParser
from .gpu import GPUParser
from .wasm import WASMParser
from .lisp import LispParser
from .prolog import PrologParser
from .business import BusinessParser
from .bytecode import BytecodeParser
from .constraints import ConstraintsParser
from .expert_systems import ExpertSystemsParser
from .fpga import FPGAParser
from .functional import FunctionalParser
from .grammar import GrammarParser
from .lexer import LexerParser
from .parser import ParserParser
from .mobile import MobileParser
from .oop import OOPParser
from .planning import PlanningParser
from .scientific import ScientificParser
from .systems import SystemsParser
from .asm_ir import ASMIRParser
from .beam import BEAMParser
from .asp import ASPParser

# Category parser registry
CATEGORY_PARSERS = {
    "embedded": EmbeddedParser,
    "assembly": AssemblyParser,
    "polyglot": PolyglotParser,
    "gpu": GPUParser,
    "wasm": WASMParser,
    "lisp": LispParser,
    "prolog": PrologParser,
    "business": BusinessParser,
    "bytecode": BytecodeParser,
    "constraints": ConstraintsParser,
    "expert_systems": ExpertSystemsParser,
    "fpga": FPGAParser,
    "functional": FunctionalParser,
    "grammar": GrammarParser,
    "lexer": LexerParser,
    "parser": ParserParser,
    "mobile": MobileParser,
    "oop": OOPParser,
    "planning": PlanningParser,
    "scientific": ScientificParser,
    "systems": SystemsParser,
    "asm_ir": ASMIRParser,
    "beam": BEAMParser,
    "asp": ASPParser,
}

__all__ = [
    "CategoryParser",
    "CATEGORY_PARSERS",
    "EmbeddedParser",
    "AssemblyParser",
    "PolyglotParser",
    "GPUParser",
    "WASMParser",
    "LispParser",
    "PrologParser",
    "BusinessParser",
    "BytecodeParser",
    "ConstraintsParser",
    "ExpertSystemsParser",
    "FPGAParser",
    "FunctionalParser",
    "GrammarParser",
    "LexerParser",
    "ParserParser",
    "MobileParser",
    "OOPParser",
    "PlanningParser",
    "ScientificParser",
    "SystemsParser",
    "ASMIRParser",
    "BEAMParser",
    "ASPParser",
]

"""Test JSON serialization and deserialization.

Tests:
- Module serialization
- Round-trip conversion
- Pydantic model validation
- Schema compliance
"""

import json
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tools"))

from semantic_ir.ir_types import (
    IRNodeKind, IRPrimitiveType, BinaryOperator,
    TargetCategory, SafetyLevel
)
from semantic_ir.nodes import PrimitiveTypeRef, TypeKind
from semantic_ir.expressions import IntegerLiteral, BinaryExpr
from semantic_ir.modules import IRModule, ModuleMetadata


class TestNodeSerialization:
    """Test node serialization."""

    def test_integer_literal_serialization(self):
        """Test integer literal JSON serialization."""
        lit = IntegerLiteral(
            node_id="n_lit_42",
            kind=IRNodeKind.INTEGER_LITERAL,
            type=PrimitiveTypeRef(
                kind=TypeKind.PRIMITIVE,
                primitive=IRPrimitiveType.I32
            ),
            value=42
        )
        
        # Serialize to dict
        lit_dict = lit.model_dump(exclude_none=True)
        
        assert lit_dict["node_id"] == "n_lit_42"
        assert lit_dict["kind"] == "integer_literal"
        assert lit_dict["value"] == 42
        
        # Serialize to JSON
        lit_json = lit.model_dump_json(exclude_none=True)
        parsed = json.loads(lit_json)
        
        assert parsed["node_id"] == "n_lit_42"
        assert parsed["value"] == 42


class TestModuleSerialization:
    """Test module serialization."""

    def test_module_serialization(self):
        """Test module JSON serialization."""
        module = IRModule(
            node_id="n_mod_test",
            kind=IRNodeKind.MODULE,
            name="test_module",
            metadata=ModuleMetadata(
                target_categories=[TargetCategory.EMBEDDED],
                safety_level=SafetyLevel.DO178C_A
            )
        )
        
        # Serialize to dict
        mod_dict = module.model_dump(exclude_none=True)
        
        assert mod_dict["node_id"] == "n_mod_test"
        assert mod_dict["name"] == "test_module"
        assert mod_dict["kind"] == "module"
        
        # Check metadata
        assert "metadata" in mod_dict
        assert mod_dict["metadata"]["safety_level"] == "DO-178C_Level_A"
    
    def test_module_round_trip(self):
        """Test module round-trip serialization."""
        original = IRModule(
            node_id="n_mod_test",
            kind=IRNodeKind.MODULE,
            name="test_module"
        )
        
        # Serialize
        json_str = original.model_dump_json(exclude_none=True)
        
        # Deserialize
        json_dict = json.loads(json_str)
        reconstructed = IRModule(**json_dict)
        
        # Compare
        assert reconstructed.node_id == original.node_id
        assert reconstructed.name == original.name
        assert reconstructed.kind == original.kind

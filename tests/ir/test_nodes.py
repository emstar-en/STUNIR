"""Test IR node creation and validation.

Tests:
- Node base structure
- Node ID validation
- Hash validation
- Type references
"""

import pytest
from pathlib import Path
import sys

# Add tools/semantic_ir to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tools"))

from semantic_ir.ir_types import IRNodeKind, IRPrimitiveType
from semantic_ir.nodes import IRNodeBase, PrimitiveTypeRef, TypeRef, TypeKind
from semantic_ir.validation import validate_node_id, validate_hash, ValidationStatus


class TestNodeBase:
    """Test base node functionality."""

    def test_create_node_base(self):
        """Test creating a basic node."""
        node = IRNodeBase(
            node_id="n_test_123",
            kind=IRNodeKind.INTEGER_LITERAL
        )
        
        assert node.node_id == "n_test_123"
        assert node.kind == IRNodeKind.INTEGER_LITERAL
        assert node.location is None
        assert node.attributes == {}
    
    def test_node_id_validation(self):
        """Test node ID validation."""
        # Valid node IDs
        assert validate_node_id("n_test").status == ValidationStatus.VALID
        assert validate_node_id("n_123").status == ValidationStatus.VALID
        assert validate_node_id("n_foo_bar_baz").status == ValidationStatus.VALID
        
        # Invalid node IDs
        assert validate_node_id("test").status == ValidationStatus.INVALID
        assert validate_node_id("n_").status == ValidationStatus.INVALID
        assert validate_node_id("").status == ValidationStatus.INVALID
    
    def test_hash_validation(self):
        """Test hash validation."""
        # Valid hash
        valid_hash = "sha256:" + "a" * 64
        assert validate_hash(valid_hash).status == ValidationStatus.VALID
        
        # Invalid hashes
        assert validate_hash("invalid").status == ValidationStatus.INVALID
        assert validate_hash("sha256:abc").status == ValidationStatus.INVALID
        assert validate_hash("").status == ValidationStatus.INVALID


class TestTypeReferences:
    """Test type reference structures."""

    def test_primitive_type_ref(self):
        """Test primitive type reference."""
        type_ref = PrimitiveTypeRef(
            kind=TypeKind.PRIMITIVE,
            primitive=IRPrimitiveType.I32
        )
        
        assert type_ref.kind == TypeKind.PRIMITIVE
        assert type_ref.primitive == IRPrimitiveType.I32
    
    def test_type_ref(self):
        """Test named type reference."""
        type_ref = TypeRef(
            kind=TypeKind.REF,
            name="Point3D",
            binding="n_type_decl_point3d"
        )
        
        assert type_ref.kind == TypeKind.REF
        assert type_ref.name == "Point3D"
        assert type_ref.binding == "n_type_decl_point3d"

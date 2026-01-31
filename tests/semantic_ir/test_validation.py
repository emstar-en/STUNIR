"""Test validation logic.

Tests:
- Module validation
- Declaration validation
- Statement validation
- Expression validation
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tools"))

from semantic_ir.ir_types import IRNodeKind, IRPrimitiveType, VisibilityKind
from semantic_ir.nodes import IRNodeBase, PrimitiveTypeRef, TypeKind
from semantic_ir.modules import IRModule, ModuleMetadata
from semantic_ir.declarations import FunctionDecl
from semantic_ir.validation import validate_module, validate_node_id, ValidationStatus


class TestModuleValidation:
    """Test module validation."""

    def test_valid_module(self):
        """Test validation of a valid module."""
        module = IRModule(
            node_id="n_mod_test",
            kind=IRNodeKind.MODULE,
            name="test_module",
            hash="sha256:" + "a" * 64
        )
        
        result = validate_module(module)
        assert result.status == ValidationStatus.VALID
    
    def test_module_with_invalid_node_id(self):
        """Test that module with invalid node ID fails validation."""
        # Pydantic will raise ValidationError during construction
        with pytest.raises(Exception):  # ValidationError from pydantic
            module = IRModule(
                node_id="invalid_id",  # Missing 'n_' prefix
                kind=IRNodeKind.MODULE,
                name="test_module"
            )
    
    def test_module_with_empty_name(self):
        """Test that module with empty name fails validation."""
        # Pydantic will raise ValidationError during construction
        with pytest.raises(Exception):  # ValidationError from pydantic
            module = IRModule(
                node_id="n_mod_test",
                kind=IRNodeKind.MODULE,
                name=""  # Empty name
            )


class TestDeclarationValidation:
    """Test declaration validation."""

    def test_valid_function_declaration(self):
        """Test valid function declaration structure."""
        func_decl = FunctionDecl(
            node_id="n_func_add",
            kind=IRNodeKind.FUNCTION_DECL,
            name="add",
            visibility=VisibilityKind.PUBLIC,
            return_type=PrimitiveTypeRef(
                kind=TypeKind.PRIMITIVE,
                primitive=IRPrimitiveType.I32
            ),
            parameters=[]
        )
        
        # Basic structure validation
        assert func_decl.node_id == "n_func_add"
        assert func_decl.name == "add"
        assert validate_node_id(func_decl.node_id).status == ValidationStatus.VALID

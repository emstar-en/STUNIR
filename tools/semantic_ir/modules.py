"""STUNIR Semantic IR Module Structures.

DO-178C Level A Compliant
Module and import definitions.
"""

from typing import List, Optional, Union
from pydantic import BaseModel, Field, conint

from .ir_types import (
    NodeID, IRName, IRNodeKind,
    TargetCategory, SafetyLevel
)
from .nodes import IRNodeBase


class ImportStmt(BaseModel):
    """Import statement."""
    module: IRName
    symbols: Union[List[IRName], str] = Field(default_factory=list)  # "*" for all
    alias: Optional[IRName] = None

    class Config:
        frozen = True


class ModuleMetadata(BaseModel):
    """Module metadata."""
    target_categories: List[TargetCategory] = Field(default_factory=list)
    safety_level: SafetyLevel = SafetyLevel.NONE
    optimization_level: conint(ge=0, le=3) = 0  # O0..O3
    language_standard: Optional[str] = None
    custom_attributes: dict = Field(default_factory=dict)

    class Config:
        frozen = True


class IRModule(IRNodeBase):
    """IR Module structure."""
    kind: IRNodeKind = IRNodeKind.MODULE
    name: IRName
    imports: List[ImportStmt] = Field(default_factory=list)
    exports: List[IRName] = Field(default_factory=list)
    declarations: List[Union[NodeID, "DeclarationNode"]] = Field(default_factory=list)
    metadata: ModuleMetadata = Field(default_factory=ModuleMetadata)

    def add_import(self, import_stmt: ImportStmt) -> bool:
        """Add an import statement."""
        self.imports.append(import_stmt)
        return True

    def add_export(self, name: IRName) -> bool:
        """Add an exported symbol."""
        self.exports.append(name)
        return True

    def add_declaration(self, decl: Union[NodeID, "DeclarationNode"]) -> bool:
        """Add a declaration."""
        self.declarations.append(decl)
        return True


# Enable forward references
from .declarations import DeclarationNode  # noqa: E402

IRModule.model_rebuild()

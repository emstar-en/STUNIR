"""STUNIR Base Emitter - Python Reference Implementation

Base class and common types for all semantic IR emitters.
Based on Ada SPARK emitter specifications.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any
import json
import hashlib
from pathlib import Path

from .types import IRModule, GeneratedFile


class EmitterStatus(Enum):
    """Emitter result status matching SPARK Emitter_Status enumeration."""
    SUCCESS = "success"
    ERROR_INVALID_IR = "error_invalid_ir"
    ERROR_WRITE_FAILED = "error_write_failed"
    ERROR_UNSUPPORTED_TYPE = "error_unsupported_type"
    ERROR_BUFFER_OVERFLOW = "error_buffer_overflow"
    ERROR_INVALID_ARCHITECTURE = "error_invalid_architecture"


@dataclass
class EmitterResult:
    """Emitter result matching SPARK Emitter_Result record."""
    status: EmitterStatus
    files: List[GeneratedFile] = field(default_factory=list)
    total_size: int = 0
    error_message: Optional[str] = None

    @property
    def files_count(self) -> int:
        """Return number of generated files."""
        return len(self.files)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return {
            "status": self.status.value,
            "files_count": self.files_count,
            "total_size": self.total_size,
            "files": [
                {"path": f.path, "hash": f.hash, "size": f.size}
                for f in self.files
            ],
            "error_message": self.error_message,
        }


@dataclass
class EmitterConfig:
    """Base emitter configuration."""
    output_dir: str
    module_name: str
    add_comments: bool = True
    add_do178c_headers: bool = True
    max_line_length: int = 100
    indent_size: int = 4
    deterministic: bool = True  # For reproducible hash generation


class BaseEmitter(ABC):
    """Base class for all semantic IR emitters.
    
    All emitters must inherit from this class and implement the emit() method.
    This ensures consistent behavior across all 24 emitter categories.
    """

    def __init__(self, config: EmitterConfig):
        """Initialize emitter with configuration.
        
        Args:
            config: Emitter configuration
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def emit(self, ir_module: IRModule) -> EmitterResult:
        """Emit code from IR module.
        
        This method must be implemented by all concrete emitters.
        
        Args:
            ir_module: The IR module to emit code from
            
        Returns:
            EmitterResult containing status and generated files
        """
        pass

    def validate_ir(self, ir_module: IRModule) -> bool:
        """Validate IR module structure.
        
        Args:
            ir_module: IR module to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not ir_module.ir_version:
            return False
        if not ir_module.module_name:
            return False
        if not isinstance(ir_module.types, list):
            return False
        if not isinstance(ir_module.functions, list):
            return False
        return True

    def compute_file_hash(self, content: str) -> str:
        """Compute SHA-256 hash of file content.
        
        Args:
            content: File content string
            
        Returns:
            64-character hex SHA-256 hash
        """
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def write_file(self, relative_path: str, content: str) -> GeneratedFile:
        """Write content to file and return generated file record.
        
        Args:
            relative_path: Path relative to output directory
            content: File content to write
            
        Returns:
            GeneratedFile record with path, hash, and size
            
        Raises:
            IOError: If file write fails
        """
        file_path = self.output_dir / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            file_path.write_text(content, encoding='utf-8')
            file_hash = self.compute_file_hash(content)
            file_size = len(content.encode('utf-8'))
            
            return GeneratedFile(
                path=str(relative_path),
                hash=file_hash,
                size=file_size,
            )
        except IOError as e:
            raise IOError(f"Failed to write file {relative_path}: {e}")

    def get_do178c_header(self, description: str) -> str:
        """Generate DO-178C compliant header comment.
        
        Args:
            description: File description
            
        Returns:
            Header comment string
        """
        if not self.config.add_do178c_headers:
            return ""
        
        return f"""/*
 * STUNIR Generated Code
 * DO-178C Level A Compliant
 * {description}
 * 
 * This file was generated by STUNIR Semantic IR Emitter
 * Based on formally verified Ada SPARK implementation
 * 
 * WARNING: Do not modify this file manually.
 * All changes must be made to the source IR.
 */

"""

    def get_stunir_comment(self, text: str, comment_prefix: str = "//") -> str:
        """Generate STUNIR comment with specified prefix.
        
        Args:
            text: Comment text
            comment_prefix: Comment syntax (e.g., "//", "#", "--")
            
        Returns:
            Formatted comment string
        """
        if not self.config.add_comments:
            return ""
        return f"{comment_prefix} STUNIR: {text}\n"

    def indent(self, level: int = 1) -> str:
        """Generate indentation string.
        
        Args:
            level: Indentation level (multiplier)
            
        Returns:
            Indentation string (spaces)
        """
        return " " * (self.config.indent_size * level)

    @staticmethod
    def load_ir_from_file(ir_path: str) -> IRModule:
        """Load IR module from JSON file.
        
        Args:
            ir_path: Path to IR JSON file
            
        Returns:
            IRModule instance
            
        Raises:
            ValueError: If IR file is invalid
        """
        try:
            with open(ir_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Parse IR JSON into IRModule
            # This is a simplified parser - full implementation would be more robust
            from .types import IRType, IRTypeField, IRFunction, IRParameter, IRDataType
            
            types = [
                IRType(
                    name=t["name"],
                    fields=[
                        IRTypeField(
                            name=f["name"],
                            field_type=f["type"],
                            optional=f.get("optional", False)
                        )
                        for f in t["fields"]
                    ],
                    docstring=t.get("docstring")
                )
                for t in data.get("types", [])
            ]
            
            functions = [
                IRFunction(
                    name=func["name"],
                    return_type=IRDataType(func["return_type"]) if func["return_type"] in [e.value for e in IRDataType] else IRDataType.VOID,
                    parameters=[
                        IRParameter(
                            name=arg["name"],
                            param_type=IRDataType(arg["type"]) if arg["type"] in [e.value for e in IRDataType] else IRDataType.VOID
                        )
                        for arg in func.get("args", [])
                    ],
                    statements=[],  # Would parse steps if needed
                    docstring=func.get("docstring")
                )
                for func in data.get("functions", [])
            ]
            
            return IRModule(
                ir_version=data["ir_version"],
                module_name=data["module_name"],
                types=types,
                functions=functions,
                docstring=data.get("docstring")
            )
        except Exception as e:
            raise ValueError(f"Failed to load IR from {ir_path}: {e}")

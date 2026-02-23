"""Test base emitter functionality."""

import pytest
import tempfile
from pathlib import Path

from tools.semantic_ir.emitters.base_emitter import (
    BaseEmitter,
    EmitterConfig,
    EmitterResult,
    EmitterStatus,
)
from tools.semantic_ir.emitters.types import (
    IRModule,
    IRFunction,
    IRType,
    IRDataType,
    IRParameter,
    IRTypeField,
)


class MockEmitter(BaseEmitter):
    """Mock emitter for testing."""

    def emit(self, ir_module: IRModule) -> EmitterResult:
        if not self.validate_ir(ir_module):
            return EmitterResult(
                status=EmitterStatus.ERROR_INVALID_IR,
                error_message="Invalid IR"
            )
        
        content = f"// Module: {ir_module.module_name}\n"
        gen_file = self.write_file("output.txt", content)
        
        return EmitterResult(
            status=EmitterStatus.SUCCESS,
            files=[gen_file],
            total_size=gen_file.size
        )


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_ir():
    """Create sample IR module."""
    return IRModule(
        ir_version="v1",
        module_name="test_module",
        types=[
            IRType(
                name="TestType",
                fields=[
                    IRTypeField(name="field1", field_type="i32"),
                    IRTypeField(name="field2", field_type="string", optional=True),
                ],
                docstring="A test type"
            )
        ],
        functions=[
            IRFunction(
                name="test_func",
                return_type=IRDataType.I32,
                parameters=[
                    IRParameter(name="arg1", param_type=IRDataType.I32),
                    IRParameter(name="arg2", param_type=IRDataType.STRING),
                ],
                statements=[],
                docstring="A test function"
            )
        ],
        docstring="Test module"
    )


class TestBaseEmitter:
    """Test base emitter functionality."""

    def test_emitter_initialization(self, temp_dir):
        """Test emitter initialization."""
        config = EmitterConfig(
            output_dir=temp_dir,
            module_name="test"
        )
        emitter = MockEmitter(config)
        
        assert emitter.config == config
        assert emitter.output_dir.exists()

    def test_validate_ir_success(self, sample_ir, temp_dir):
        """Test IR validation with valid IR."""
        config = EmitterConfig(output_dir=temp_dir, module_name="test")
        emitter = MockEmitter(config)
        
        assert emitter.validate_ir(sample_ir)

    def test_validate_ir_failure(self, temp_dir):
        """Test IR validation with invalid IR."""
        config = EmitterConfig(output_dir=temp_dir, module_name="test")
        emitter = MockEmitter(config)
        
        # Missing required fields
        invalid_ir = IRModule(
            ir_version="",
            module_name="",
            types=[],
            functions=[]
        )
        
        assert not emitter.validate_ir(invalid_ir)

    def test_compute_file_hash(self, temp_dir):
        """Test file hash computation."""
        config = EmitterConfig(output_dir=temp_dir, module_name="test")
        emitter = MockEmitter(config)
        
        content = "test content"
        hash1 = emitter.compute_file_hash(content)
        hash2 = emitter.compute_file_hash(content)
        
        # Same content should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex length
        
        # Different content should produce different hash
        hash3 = emitter.compute_file_hash("different content")
        assert hash1 != hash3

    def test_write_file(self, temp_dir):
        """Test file writing."""
        config = EmitterConfig(output_dir=temp_dir, module_name="test")
        emitter = MockEmitter(config)
        
        content = "test file content\n"
        gen_file = emitter.write_file("test.txt", content)
        
        assert gen_file.path == "test.txt"
        assert gen_file.size == len(content.encode('utf-8'))
        assert len(gen_file.hash) == 64
        
        # Verify file was written
        file_path = Path(temp_dir) / "test.txt"
        assert file_path.exists()
        assert file_path.read_text() == content

    def test_emit_success(self, sample_ir, temp_dir):
        """Test successful emission."""
        config = EmitterConfig(output_dir=temp_dir, module_name="test")
        emitter = MockEmitter(config)
        
        result = emitter.emit(sample_ir)
        
        assert result.status == EmitterStatus.SUCCESS
        assert result.files_count == 1
        assert result.total_size > 0
        assert result.error_message is None

    def test_emit_invalid_ir(self, temp_dir):
        """Test emission with invalid IR."""
        config = EmitterConfig(output_dir=temp_dir, module_name="test")
        emitter = MockEmitter(config)
        
        invalid_ir = IRModule(
            ir_version="",
            module_name="",
            types=[],
            functions=[]
        )
        
        result = emitter.emit(invalid_ir)
        
        assert result.status == EmitterStatus.ERROR_INVALID_IR
        assert result.files_count == 0
        assert result.error_message is not None

    def test_do178c_header(self, temp_dir):
        """Test DO-178C header generation."""
        config = EmitterConfig(
            output_dir=temp_dir,
            module_name="test",
            add_do178c_headers=True
        )
        emitter = MockEmitter(config)
        
        header = emitter.get_do178c_header("Test description")
        
        assert "STUNIR Generated Code" in header
        assert "DO-178C Level A" in header
        assert "Test description" in header

    def test_indent(self, temp_dir):
        """Test indentation generation."""
        config = EmitterConfig(
            output_dir=temp_dir,
            module_name="test",
            indent_size=4
        )
        emitter = MockEmitter(config)
        
        assert emitter.indent(0) == ""
        assert emitter.indent(1) == "    "
        assert emitter.indent(2) == "        "
        assert emitter.indent(3) == "            "


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

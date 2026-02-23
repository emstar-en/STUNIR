"""Test all 24 emitters for basic functionality and confluence."""

import pytest
import tempfile
import json
from pathlib import Path

from tools.semantic_ir.emitters.types import (
    IRModule,
    IRFunction,
    IRType,
    IRDataType,
    IRParameter,
    IRTypeField,
)

# Import all emitters
from tools.semantic_ir.emitters.core import (
    EmbeddedEmitter,
    GPUEmitter,
    WebAssemblyEmitter,
    AssemblyEmitter,
    PolyglotEmitter,
)
from tools.semantic_ir.emitters.language_families import (
    LispEmitter,
    PrologEmitter,
)
from tools.semantic_ir.emitters.specialized import (
    BusinessEmitter,
    FPGAEmitter,
    GrammarEmitter,
    LexerEmitter,
    ParserEmitter,
    ExpertSystemEmitter,
    ConstraintEmitter,
    FunctionalEmitter,
    OOPEmitter,
    MobileEmitter,
    ScientificEmitter,
    BytecodeEmitter,
    SystemsEmitter,
    PlanningEmitter,
    AssemblyIREmitter,
    BEAMEmitter,
    ASPEmitter,
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_ir():
    """Create sample IR module for testing."""
    return IRModule(
        ir_version="v1",
        module_name="test_module",
        types=[
            IRType(
                name="TestType",
                fields=[
                    IRTypeField(name="id", field_type="i32"),
                    IRTypeField(name="name", field_type="string"),
                ],
                docstring="Test type definition"
            )
        ],
        functions=[
            IRFunction(
                name="test_function",
                return_type=IRDataType.I32,
                parameters=[
                    IRParameter(name="x", param_type=IRDataType.I32),
                    IRParameter(name="y", param_type=IRDataType.I32),
                ],
                statements=[],
                docstring="Test function"
            )
        ],
        docstring="Test module for emitter validation"
    )


# All 24 emitters to test
ALL_EMITTERS = [
    # Core (5)
    ("embedded", EmbeddedEmitter, "EmbeddedEmitterConfig"),
    ("gpu", GPUEmitter, "GPUEmitterConfig"),
    ("wasm", WebAssemblyEmitter, "WebAssemblyEmitterConfig"),
    ("assembly", AssemblyEmitter, "AssemblyEmitterConfig"),
    ("polyglot", PolyglotEmitter, "PolyglotEmitterConfig"),
    # Language Families (2)
    ("lisp", LispEmitter, "LispEmitterConfig"),
    ("prolog", PrologEmitter, "PrologEmitterConfig"),
    # Specialized (17)
    ("business", BusinessEmitter, "BusinessEmitterConfig"),
    ("fpga", FPGAEmitter, "FPGAEmitterConfig"),
    ("grammar", GrammarEmitter, "GrammarEmitterConfig"),
    ("lexer", LexerEmitter, "LexerEmitterConfig"),
    ("parser", ParserEmitter, "ParserEmitterConfig"),
    ("expert", ExpertSystemEmitter, "ExpertSystemEmitterConfig"),
    ("constraints", ConstraintEmitter, "ConstraintEmitterConfig"),
    ("functional", FunctionalEmitter, "FunctionalEmitterConfig"),
    ("oop", OOPEmitter, "OOPEmitterConfig"),
    ("mobile", MobileEmitter, "MobileEmitterConfig"),
    ("scientific", ScientificEmitter, "ScientificEmitterConfig"),
    ("bytecode", BytecodeEmitter, "BytecodeEmitterConfig"),
    ("systems", SystemsEmitter, "SystemsEmitterConfig"),
    ("planning", PlanningEmitter, "PlanningEmitterConfig"),
    ("asm_ir", AssemblyIREmitter, "AssemblyIREmitterConfig"),
    ("beam", BEAMEmitter, "BEAMEmitterConfig"),
    ("asp", ASPEmitter, "ASPEmitterConfig"),
]


class TestAllEmitters:
    """Test all 24 emitters."""

    @pytest.mark.parametrize("name,emitter_class,config_name", ALL_EMITTERS)
    def test_emitter_initialization(self, name, emitter_class, config_name, temp_dir):
        """Test that each emitter can be initialized."""
        # Get config class
        config_class = getattr(
            __import__(f"tools.semantic_ir.emitters.{name}", fromlist=[config_name]),
            config_name
        )
        
        config = config_class(
            output_dir=temp_dir,
            module_name="test"
        )
        emitter = emitter_class(config)
        
        assert emitter is not None
        assert emitter.config is not None

    @pytest.mark.parametrize("name,emitter_class,config_name", ALL_EMITTERS)
    def test_emitter_emit(self, name, emitter_class, config_name, temp_dir, sample_ir):
        """Test that each emitter can emit code."""
        # Get config class
        config_class = getattr(
            __import__(f"tools.semantic_ir.emitters.{name}", fromlist=[config_name]),
            config_name
        )
        
        config = config_class(
            output_dir=temp_dir,
            module_name=sample_ir.module_name
        )
        emitter = emitter_class(config)
        result = emitter.emit(sample_ir)
        
        # All emitters should succeed
        assert result.status.value == "success"
        assert result.files_count > 0
        assert result.total_size > 0
        assert result.error_message is None

    @pytest.mark.parametrize("name,emitter_class,config_name", ALL_EMITTERS)
    def test_emitter_deterministic(self, name, emitter_class, config_name, temp_dir, sample_ir):
        """Test that each emitter produces deterministic output."""
        config_class = getattr(
            __import__(f"tools.semantic_ir.emitters.{name}", fromlist=[config_name]),
            config_name
        )
        
        # Emit twice
        config1 = config_class(
            output_dir=temp_dir + "/run1",
            module_name=sample_ir.module_name
        )
        emitter1 = emitter_class(config1)
        result1 = emitter1.emit(sample_ir)
        
        config2 = config_class(
            output_dir=temp_dir + "/run2",
            module_name=sample_ir.module_name
        )
        emitter2 = emitter_class(config2)
        result2 = emitter2.emit(sample_ir)
        
        # Results should be identical
        assert result1.status == result2.status
        assert result1.files_count == result2.files_count
        assert result1.total_size == result2.total_size
        
        # File hashes should match
        for f1, f2 in zip(result1.files, result2.files):
            assert f1.hash == f2.hash


class TestEmitterConfluence:
    """Test confluence between Python and SPARK implementations."""

    def test_confluence_directory_structure(self):
        """Test that Python emitters match SPARK structure."""
        repo_root = Path(__file__).parent.parent.parent.parent
        python_dir = repo_root / "tools" / "semantic_ir" / "emitters"
        spark_dir = repo_root / "targets" / "spark"
        
        # Both should exist
        assert python_dir.exists()
        assert spark_dir.exists()
        
        # Check key emitter categories exist
        for category in ["core", "language_families", "specialized"]:
            assert (python_dir / category).exists()

    @pytest.mark.parametrize("name,emitter_class,config_name", ALL_EMITTERS)
    def test_output_format(self, name, emitter_class, config_name, temp_dir, sample_ir):
        """Test that output format is valid."""
        config_class = getattr(
            __import__(f"tools.semantic_ir.emitters.{name}", fromlist=[config_name]),
            config_name
        )
        
        config = config_class(
            output_dir=temp_dir,
            module_name=sample_ir.module_name
        )
        emitter = emitter_class(config)
        result = emitter.emit(sample_ir)
        
        # Check that files were created
        for gen_file in result.files:
            file_path = Path(temp_dir) / gen_file.path
            assert file_path.exists()
            
            # Check content is non-empty
            content = file_path.read_text()
            assert len(content) > 0
            
            # Check hash matches
            import hashlib
            expected_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
            assert gen_file.hash == expected_hash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

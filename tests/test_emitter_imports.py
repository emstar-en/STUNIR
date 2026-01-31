#!/usr/bin/env python3
"""Import tests for AsmIrEmitter."""

import pytest


class TestAsmIrEmitterImport:
    """Test AsmIrEmitter can be imported."""
    
    def test_module_import(self):
        """Test module can be imported."""
        import targets.asm_ir.emitter
        assert targets.asm_ir.emitter is not None
    
    def test_class_import(self):
        """Test class can be imported."""
        from targets.asm_ir.emitter import AsmIrEmitter
        assert AsmIrEmitter is not None
    
    def test_class_is_callable(self):
        """Test class is callable."""
        from targets.asm_ir.emitter import AsmIrEmitter
        assert callable(AsmIrEmitter)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


#!/usr/bin/env python3
"""Import tests for ClingoEmitter."""

import pytest


class TestClingoEmitterImport:
    """Test ClingoEmitter can be imported."""
    
    def test_module_import(self):
        """Test module can be imported."""
        import targets.asp.clingo_emitter
        assert targets.asp.clingo_emitter is not None
    
    def test_class_import(self):
        """Test class can be imported."""
        from targets.asp.clingo_emitter import ClingoEmitter
        assert ClingoEmitter is not None
    
    def test_class_is_callable(self):
        """Test class is callable."""
        from targets.asp.clingo_emitter import ClingoEmitter
        assert callable(ClingoEmitter)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


#!/usr/bin/env python3
"""Import tests for DLVEmitter."""

import pytest


class TestDLVEmitterImport:
    """Test DLVEmitter can be imported."""
    
    def test_module_import(self):
        """Test module can be imported."""
        import targets.asp.dlv_emitter
        assert targets.asp.dlv_emitter is not None
    
    def test_class_import(self):
        """Test class can be imported."""
        from targets.asp.dlv_emitter import DLVEmitter
        assert DLVEmitter is not None
    
    def test_class_is_callable(self):
        """Test class is callable."""
        from targets.asp.dlv_emitter import DLVEmitter
        assert callable(DLVEmitter)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


#!/usr/bin/env python3
"""Import tests for ElixirEmitter."""

import pytest


class TestElixirEmitterImport:
    """Test ElixirEmitter can be imported."""
    
    def test_module_import(self):
        """Test module can be imported."""
        import targets.beam.elixir_emitter
        assert targets.beam.elixir_emitter is not None
    
    def test_class_import(self):
        """Test class can be imported."""
        from targets.beam.elixir_emitter import ElixirEmitter
        assert ElixirEmitter is not None
    
    def test_class_is_callable(self):
        """Test class is callable."""
        from targets.beam.elixir_emitter import ElixirEmitter
        assert callable(ElixirEmitter)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


#!/usr/bin/env python3
"""Import tests for ErlangEmitter."""

import pytest


class TestErlangEmitterImport:
    """Test ErlangEmitter can be imported."""
    
    def test_module_import(self):
        """Test module can be imported."""
        import targets.beam.erlang_emitter
        assert targets.beam.erlang_emitter is not None
    
    def test_class_import(self):
        """Test class can be imported."""
        from targets.beam.erlang_emitter import ErlangEmitter
        assert ErlangEmitter is not None
    
    def test_class_is_callable(self):
        """Test class is callable."""
        from targets.beam.erlang_emitter import ErlangEmitter
        assert callable(ErlangEmitter)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


#!/usr/bin/env python3
"""Import tests for BASICEmitter."""

import pytest


class TestBASICEmitterImport:
    """Test BASICEmitter can be imported."""
    
    def test_module_import(self):
        """Test module can be imported."""
        import targets.business.basic_emitter
        assert targets.business.basic_emitter is not None
    
    def test_class_import(self):
        """Test class can be imported."""
        from targets.business.basic_emitter import BASICEmitter
        assert BASICEmitter is not None
    
    def test_class_is_callable(self):
        """Test class is callable."""
        from targets.business.basic_emitter import BASICEmitter
        assert callable(BASICEmitter)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


#!/usr/bin/env python3
"""Import tests for COBOLEmitter."""

import pytest


class TestCOBOLEmitterImport:
    """Test COBOLEmitter can be imported."""
    
    def test_module_import(self):
        """Test module can be imported."""
        import targets.business.cobol_emitter
        assert targets.business.cobol_emitter is not None
    
    def test_class_import(self):
        """Test class can be imported."""
        from targets.business.cobol_emitter import COBOLEmitter
        assert COBOLEmitter is not None
    
    def test_class_is_callable(self):
        """Test class is callable."""
        from targets.business.cobol_emitter import COBOLEmitter
        assert callable(COBOLEmitter)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


#!/usr/bin/env python3
"""Import tests for MiniZincEmitter."""

import pytest


class TestMiniZincEmitterImport:
    """Test MiniZincEmitter can be imported."""
    
    def test_module_import(self):
        """Test module can be imported."""
        import targets.constraints.minizinc_emitter
        assert targets.constraints.minizinc_emitter is not None
    
    def test_class_import(self):
        """Test class can be imported."""
        from targets.constraints.minizinc_emitter import MiniZincEmitter
        assert MiniZincEmitter is not None
    
    def test_class_is_callable(self):
        """Test class is callable."""
        from targets.constraints.minizinc_emitter import MiniZincEmitter
        assert callable(MiniZincEmitter)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


#!/usr/bin/env python3
"""Import tests for CHREmitter."""

import pytest


class TestCHREmitterImport:
    """Test CHREmitter can be imported."""
    
    def test_module_import(self):
        """Test module can be imported."""
        import targets.constraints.chr_emitter
        assert targets.constraints.chr_emitter is not None
    
    def test_class_import(self):
        """Test class can be imported."""
        from targets.constraints.chr_emitter import CHREmitter
        assert CHREmitter is not None
    
    def test_class_is_callable(self):
        """Test class is callable."""
        from targets.constraints.chr_emitter import CHREmitter
        assert callable(CHREmitter)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


#!/usr/bin/env python3
"""Import tests for CLIPSEmitter."""

import pytest


class TestCLIPSEmitterImport:
    """Test CLIPSEmitter can be imported."""
    
    def test_module_import(self):
        """Test module can be imported."""
        import targets.expert_systems.clips_emitter
        assert targets.expert_systems.clips_emitter is not None
    
    def test_class_import(self):
        """Test class can be imported."""
        from targets.expert_systems.clips_emitter import CLIPSEmitter
        assert CLIPSEmitter is not None
    
    def test_class_is_callable(self):
        """Test class is callable."""
        from targets.expert_systems.clips_emitter import CLIPSEmitter
        assert callable(CLIPSEmitter)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


#!/usr/bin/env python3
"""Import tests for JessEmitter."""

import pytest


class TestJessEmitterImport:
    """Test JessEmitter can be imported."""
    
    def test_module_import(self):
        """Test module can be imported."""
        import targets.expert_systems.jess_emitter
        assert targets.expert_systems.jess_emitter is not None
    
    def test_class_import(self):
        """Test class can be imported."""
        from targets.expert_systems.jess_emitter import JessEmitter
        assert JessEmitter is not None
    
    def test_class_is_callable(self):
        """Test class is callable."""
        from targets.expert_systems.jess_emitter import JessEmitter
        assert callable(JessEmitter)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


#!/usr/bin/env python3
"""Import tests for HaskellEmitter."""

import pytest


class TestHaskellEmitterImport:
    """Test HaskellEmitter can be imported."""
    
    def test_module_import(self):
        """Test module can be imported."""
        import targets.functional.haskell_emitter
        assert targets.functional.haskell_emitter is not None
    
    def test_class_import(self):
        """Test class can be imported."""
        from targets.functional.haskell_emitter import HaskellEmitter
        assert HaskellEmitter is not None
    
    def test_class_is_callable(self):
        """Test class is callable."""
        from targets.functional.haskell_emitter import HaskellEmitter
        assert callable(HaskellEmitter)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


#!/usr/bin/env python3
"""Import tests for FSharpEmitter."""

import pytest


class TestFSharpEmitterImport:
    """Test FSharpEmitter can be imported."""
    
    def test_module_import(self):
        """Test module can be imported."""
        import targets.functional.fsharp_emitter
        assert targets.functional.fsharp_emitter is not None
    
    def test_class_import(self):
        """Test class can be imported."""
        from targets.functional.fsharp_emitter import FSharpEmitter
        assert FSharpEmitter is not None
    
    def test_class_is_callable(self):
        """Test class is callable."""
        from targets.functional.fsharp_emitter import FSharpEmitter
        assert callable(FSharpEmitter)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


#!/usr/bin/env python3
"""Import tests for OCamlEmitter."""

import pytest


class TestOCamlEmitterImport:
    """Test OCamlEmitter can be imported."""
    
    def test_module_import(self):
        """Test module can be imported."""
        import targets.functional.ocaml_emitter
        assert targets.functional.ocaml_emitter is not None
    
    def test_class_import(self):
        """Test class can be imported."""
        from targets.functional.ocaml_emitter import OCamlEmitter
        assert OCamlEmitter is not None
    
    def test_class_is_callable(self):
        """Test class is callable."""
        from targets.functional.ocaml_emitter import OCamlEmitter
        assert callable(OCamlEmitter)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


#!/usr/bin/env python3
"""Import tests for ANTLREmitter."""

import pytest


class TestANTLREmitterImport:
    """Test ANTLREmitter can be imported."""
    
    def test_module_import(self):
        """Test module can be imported."""
        import targets.grammar.antlr_emitter
        assert targets.grammar.antlr_emitter is not None
    
    def test_class_import(self):
        """Test class can be imported."""
        from targets.grammar.antlr_emitter import ANTLREmitter
        assert ANTLREmitter is not None
    
    def test_class_is_callable(self):
        """Test class is callable."""
        from targets.grammar.antlr_emitter import ANTLREmitter
        assert callable(ANTLREmitter)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


#!/usr/bin/env python3
"""Import tests for BNFEmitter."""

import pytest


class TestBNFEmitterImport:
    """Test BNFEmitter can be imported."""
    
    def test_module_import(self):
        """Test module can be imported."""
        import targets.grammar.bnf_emitter
        assert targets.grammar.bnf_emitter is not None
    
    def test_class_import(self):
        """Test class can be imported."""
        from targets.grammar.bnf_emitter import BNFEmitter
        assert BNFEmitter is not None
    
    def test_class_is_callable(self):
        """Test class is callable."""
        from targets.grammar.bnf_emitter import BNFEmitter
        assert callable(BNFEmitter)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


#!/usr/bin/env python3
"""Import tests for EBNFEmitter."""

import pytest


class TestEBNFEmitterImport:
    """Test EBNFEmitter can be imported."""
    
    def test_module_import(self):
        """Test module can be imported."""
        import targets.grammar.ebnf_emitter
        assert targets.grammar.ebnf_emitter is not None
    
    def test_class_import(self):
        """Test class can be imported."""
        from targets.grammar.ebnf_emitter import EBNFEmitter
        assert EBNFEmitter is not None
    
    def test_class_is_callable(self):
        """Test class is callable."""
        from targets.grammar.ebnf_emitter import EBNFEmitter
        assert callable(EBNFEmitter)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


#!/usr/bin/env python3
"""Import tests for PEGEmitter."""

import pytest


class TestPEGEmitterImport:
    """Test PEGEmitter can be imported."""
    
    def test_module_import(self):
        """Test module can be imported."""
        import targets.grammar.peg_emitter
        assert targets.grammar.peg_emitter is not None
    
    def test_class_import(self):
        """Test class can be imported."""
        from targets.grammar.peg_emitter import PEGEmitter
        assert PEGEmitter is not None
    
    def test_class_is_callable(self):
        """Test class is callable."""
        from targets.grammar.peg_emitter import PEGEmitter
        assert callable(PEGEmitter)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


#!/usr/bin/env python3
"""Import tests for YaccEmitter."""

import pytest


class TestYaccEmitterImport:
    """Test YaccEmitter can be imported."""
    
    def test_module_import(self):
        """Test module can be imported."""
        import targets.grammar.yacc_emitter
        assert targets.grammar.yacc_emitter is not None
    
    def test_class_import(self):
        """Test class can be imported."""
        from targets.grammar.yacc_emitter import YaccEmitter
        assert YaccEmitter is not None
    
    def test_class_is_callable(self):
        """Test class is callable."""
        from targets.grammar.yacc_emitter import YaccEmitter
        assert callable(YaccEmitter)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


#!/usr/bin/env python3
"""Import tests for ALGOLEmitter."""

import pytest


class TestALGOLEmitterImport:
    """Test ALGOLEmitter can be imported."""
    
    def test_module_import(self):
        """Test module can be imported."""
        import targets.oop.algol_emitter
        assert targets.oop.algol_emitter is not None
    
    def test_class_import(self):
        """Test class can be imported."""
        from targets.oop.algol_emitter import ALGOLEmitter
        assert ALGOLEmitter is not None
    
    def test_class_is_callable(self):
        """Test class is callable."""
        from targets.oop.algol_emitter import ALGOLEmitter
        assert callable(ALGOLEmitter)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


#!/usr/bin/env python3
"""Import tests for PDDLEmitter."""

import pytest


class TestPDDLEmitterImport:
    """Test PDDLEmitter can be imported."""
    
    def test_module_import(self):
        """Test module can be imported."""
        import targets.planning.pddl_emitter
        assert targets.planning.pddl_emitter is not None
    
    def test_class_import(self):
        """Test class can be imported."""
        from targets.planning.pddl_emitter import PDDLEmitter
        assert PDDLEmitter is not None
    
    def test_class_is_callable(self):
        """Test class is callable."""
        from targets.planning.pddl_emitter import PDDLEmitter
        assert callable(PDDLEmitter)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


#!/usr/bin/env python3
"""Import tests for AdaEmitter."""

import pytest


class TestAdaEmitterImport:
    """Test AdaEmitter can be imported."""
    
    def test_module_import(self):
        """Test module can be imported."""
        import targets.systems.ada_emitter
        assert targets.systems.ada_emitter is not None
    
    def test_class_import(self):
        """Test class can be imported."""
        from targets.systems.ada_emitter import AdaEmitter
        assert AdaEmitter is not None
    
    def test_class_is_callable(self):
        """Test class is callable."""
        from targets.systems.ada_emitter import AdaEmitter
        assert callable(AdaEmitter)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


#!/usr/bin/env python3
"""Import tests for DEmitter."""

import pytest


class TestDEmitterImport:
    """Test DEmitter can be imported."""
    
    def test_module_import(self):
        """Test module can be imported."""
        import targets.systems.d_emitter
        assert targets.systems.d_emitter is not None
    
    def test_class_import(self):
        """Test class can be imported."""
        from targets.systems.d_emitter import DEmitter
        assert DEmitter is not None
    
    def test_class_is_callable(self):
        """Test class is callable."""
        from targets.systems.d_emitter import DEmitter
        assert callable(DEmitter)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

{-# LANGUAGE OverloadedStrings #-}

module STUNIR.SemanticIR.Emitters.SpecializedSpec (spec) where

import Test.Hspec
import STUNIR.SemanticIR.Emitters.Types
import STUNIR.SemanticIR.Emitters.Specialized.Business
import STUNIR.SemanticIR.Emitters.Specialized.FPGA
import STUNIR.SemanticIR.Emitters.Specialized.Grammar
import STUNIR.SemanticIR.Emitters.Specialized.Lexer
import STUNIR.SemanticIR.Emitters.Specialized.Parser
import STUNIR.SemanticIR.Emitters.Specialized.Expert
import STUNIR.SemanticIR.Emitters.Specialized.Constraints
import STUNIR.SemanticIR.Emitters.Specialized.Functional
import STUNIR.SemanticIR.Emitters.Specialized.OOP
import STUNIR.SemanticIR.Emitters.Specialized.Mobile
import STUNIR.SemanticIR.Emitters.Specialized.Scientific
import STUNIR.SemanticIR.Emitters.Specialized.Bytecode
import STUNIR.SemanticIR.Emitters.Specialized.Systems
import STUNIR.SemanticIR.Emitters.Specialized.Planning
import STUNIR.SemanticIR.Emitters.Specialized.AsmIR
import STUNIR.SemanticIR.Emitters.Specialized.BEAM
import STUNIR.SemanticIR.Emitters.Specialized.ASP

spec :: Spec
spec = do
  describe "Specialized Category Emitters (17 emitters)" $ do
    
    let testModule = IRModule "1.0" "TestModule" [] [] Nothing
    
    describe "Business Emitter" $ do
      it "emits COBOL code" $ do
        let result = emitBusiness testModule "/tmp" LangCOBOL
        result `shouldSatisfy` isRight
    
    describe "FPGA Emitter" $ do
      it "emits VHDL code" $ do
        let result = emitFPGA testModule "/tmp" VHDL
        result `shouldSatisfy` isRight
    
    describe "Grammar Emitter" $ do
      it "emits ANTLR grammar" $ do
        let result = emitGrammar testModule "/tmp" ANTLR
        result `shouldSatisfy` isRight
    
    describe "Lexer Emitter" $ do
      it "emits Flex lexer" $ do
        let result = emitLexer testModule "/tmp" Flex
        result `shouldSatisfy` isRight
    
    describe "Parser Emitter" $ do
      it "emits Yacc parser" $ do
        let result = emitParser testModule "/tmp" YaccGen
        result `shouldSatisfy` isRight
    
    describe "Expert Emitter" $ do
      it "emits expert system code" $ do
        let result = emitExpert testModule "/tmp"
        result `shouldSatisfy` isRight
    
    describe "Constraints Emitter" $ do
      it "emits constraint programming code" $ do
        let result = emitConstraints testModule "/tmp"
        result `shouldSatisfy` isRight
    
    describe "Functional Emitter" $ do
      it "emits functional programming code" $ do
        let result = emitFunctional testModule "/tmp"
        result `shouldSatisfy` isRight
    
    describe "OOP Emitter" $ do
      it "emits object-oriented code" $ do
        let result = emitOOP testModule "/tmp"
        result `shouldSatisfy` isRight
    
    describe "Mobile Emitter" $ do
      it "emits mobile code" $ do
        let result = emitMobile testModule "/tmp"
        result `shouldSatisfy` isRight
    
    describe "Scientific Emitter" $ do
      it "emits scientific computing code" $ do
        let result = emitScientific testModule "/tmp"
        result `shouldSatisfy` isRight
    
    describe "Bytecode Emitter" $ do
      it "emits bytecode" $ do
        let result = emitBytecode testModule "/tmp"
        result `shouldSatisfy` isRight
    
    describe "Systems Emitter" $ do
      it "emits systems programming code" $ do
        let result = emitSystems testModule "/tmp"
        result `shouldSatisfy` isRight
    
    describe "Planning Emitter" $ do
      it "emits planning language code" $ do
        let result = emitPlanning testModule "/tmp"
        result `shouldSatisfy` isRight
    
    describe "AsmIR Emitter" $ do
      it "emits assembly IR code" $ do
        let result = emitAsmIR testModule "/tmp"
        result `shouldSatisfy` isRight
    
    describe "BEAM Emitter" $ do
      it "emits BEAM VM code" $ do
        let result = emitBEAM testModule "/tmp"
        result `shouldSatisfy` isRight
    
    describe "ASP Emitter" $ do
      it "emits Answer Set Programming code" $ do
        let result = emitASP testModule "/tmp"
        result `shouldSatisfy` isRight

isRight :: Either a b -> Bool
isRight (Right _) = True
isRight _ = False

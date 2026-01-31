{-# LANGUAGE OverloadedStrings #-}

module STUNIR.SemanticIR.Emitters.CoreSpec (spec) where

import Test.Hspec
import Data.Text (Text)
import qualified Data.Text as T
import STUNIR.SemanticIR.Emitters.Types
import STUNIR.SemanticIR.Emitters.Core.Embedded
import STUNIR.SemanticIR.Emitters.Core.GPU
import STUNIR.SemanticIR.Emitters.Core.WASM
import STUNIR.SemanticIR.Emitters.Core.Assembly
import STUNIR.SemanticIR.Emitters.Core.Polyglot

spec :: Spec
spec = do
  describe "Core Category Emitters (5 emitters)" $ do
    
    let testModule = IRModule "1.0" "TestModule" [] [] Nothing
    
    describe "Embedded Emitter" $ do
      it "emits ARM code" $ do
        let result = emitEmbedded testModule "/tmp" TargetARM
        result `shouldSatisfy` isRight
      
      it "emits ARM64 code" $ do
        let result = emitEmbedded testModule "/tmp" TargetARM64
        result `shouldSatisfy` isRight
      
      it "emits RISC-V code" $ do
        let result = emitEmbedded testModule "/tmp" TargetRISCV
        result `shouldSatisfy` isRight
    
    describe "GPU Emitter" $ do
      it "emits CUDA code" $ do
        let result = emitGPU testModule "/tmp" BackendCUDA
        result `shouldSatisfy` isRight
      
      it "emits OpenCL code" $ do
        let result = emitGPU testModule "/tmp" BackendOpenCL
        result `shouldSatisfy` isRight
    
    describe "WASM Emitter" $ do
      it "emits basic WASM code" $ do
        let result = emitWASM testModule "/tmp" []
        result `shouldSatisfy` isRight
      
      it "emits WASM with WASI" $ do
        let result = emitWASM testModule "/tmp" [FeatureWASI]
        result `shouldSatisfy` isRight
    
    describe "Assembly Emitter" $ do
      it "emits x86 assembly" $ do
        let result = emitAssembly testModule "/tmp" AsmX86 SyntaxIntel
        result `shouldSatisfy` isRight
      
      it "emits x86_64 assembly" $ do
        let result = emitAssembly testModule "/tmp" AsmX86_64 SyntaxATT
        result `shouldSatisfy` isRight
    
    describe "Polyglot Emitter" $ do
      it "emits C89 code" $ do
        let result = emitPolyglot testModule "/tmp" LangC89
        result `shouldSatisfy` isRight
      
      it "emits C99 code" $ do
        let result = emitPolyglot testModule "/tmp" LangC99
        result `shouldSatisfy` isRight
      
      it "emits Rust code" $ do
        let result = emitPolyglot testModule "/tmp" LangRust
        result `shouldSatisfy` isRight

isRight :: Either a b -> Bool
isRight (Right _) = True
isRight _ = False

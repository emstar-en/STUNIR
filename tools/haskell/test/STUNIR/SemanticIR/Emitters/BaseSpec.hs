{-# LANGUAGE OverloadedStrings #-}

module STUNIR.SemanticIR.Emitters.BaseSpec (spec) where

import Test.Hspec
import Test.QuickCheck
import Data.Text (Text)
import qualified Data.Text as T
import STUNIR.SemanticIR.Emitters.Base
import STUNIR.SemanticIR.Emitters.Types

spec :: Spec
spec = do
  describe "Base Emitter Infrastructure" $ do
    
    describe "validateIR" $ do
      it "validates a valid IR module" $ do
        let validModule = IRModule "1.0" "TestModule" [] [] Nothing
        validateIR validModule `shouldBe` True
      
      it "rejects IR module with empty version" $ do
        let invalidModule = IRModule "" "TestModule" [] [] Nothing
        validateIR invalidModule `shouldBe` False
      
      it "rejects IR module with empty name" $ do
        let invalidModule = IRModule "1.0" "" [] [] Nothing
        validateIR invalidModule `shouldBe` False
    
    describe "computeFileHash" $ do
      it "computes SHA-256 hash correctly" $ do
        let content = "test content"
        let hash = computeFileHash content
        T.length hash `shouldBe` 64
      
      it "produces consistent hashes for same content" $ do
        let content = "test content"
        let hash1 = computeFileHash content
        let hash2 = computeFileHash content
        hash1 `shouldBe` hash2
      
      it "produces different hashes for different content" $ property $
        \s1 s2 -> s1 /= s2 ==> computeFileHash (T.pack s1) /= computeFileHash (T.pack s2)
    
    describe "getDO178CHeader" $ do
      it "generates header when enabled" $ do
        let header = getDO178CHeader True "Test Description"
        T.length header `shouldSatisfy` (> 0)
      
      it "generates empty string when disabled" $ do
        let header = getDO178CHeader False "Test Description"
        header `shouldBe` ""
    
    describe "indentString" $ do
      it "generates correct indentation" $ do
        indentString 4 2 `shouldBe` "        "
      
      it "handles zero level" $ do
        indentString 4 0 `shouldBe` ""

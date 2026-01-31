{-# LANGUAGE OverloadedStrings #-}

module STUNIR.SemanticIR.Emitters.LanguageFamiliesSpec (spec) where

import Test.Hspec
import STUNIR.SemanticIR.Emitters.Types
import STUNIR.SemanticIR.Emitters.LanguageFamilies.Lisp
import STUNIR.SemanticIR.Emitters.LanguageFamilies.Prolog

spec :: Spec
spec = do
  describe "Language Family Emitters (2 emitters)" $ do
    
    let testModule = IRModule "1.0" "TestModule" [] [] Nothing
    
    describe "Lisp Emitter" $ do
      it "emits Common Lisp code" $ do
        let result = emitLisp testModule "/tmp" DialectCommonLisp
        result `shouldSatisfy` isRight
      
      it "emits Scheme code" $ do
        let result = emitLisp testModule "/tmp" DialectScheme
        result `shouldSatisfy` isRight
      
      it "emits Clojure code" $ do
        let result = emitLisp testModule "/tmp" DialectClojure
        result `shouldSatisfy` isRight
      
      it "emits Racket code" $ do
        let result = emitLisp testModule "/tmp" DialectRacket
        result `shouldSatisfy` isRight
      
      it "emits Emacs Lisp code" $ do
        let result = emitLisp testModule "/tmp" DialectEmacsLisp
        result `shouldSatisfy` isRight
    
    describe "Prolog Emitter" $ do
      it "emits SWI-Prolog code" $ do
        let result = emitProlog testModule "/tmp" SystemSWIProlog
        result `shouldSatisfy` isRight
      
      it "emits GNU Prolog code" $ do
        let result = emitProlog testModule "/tmp" SystemGNUProlog
        result `shouldSatisfy` isRight
      
      it "emits SICStus Prolog code" $ do
        let result = emitProlog testModule "/tmp" SystemSICStus
        result `shouldSatisfy` isRight

isRight :: Either a b -> Bool
isRight (Right _) = True
isRight _ = False

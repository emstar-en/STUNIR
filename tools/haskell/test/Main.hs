{-|
Module      : Main
Description : Test suite entry point
Copyright   : (c) STUNIR Team, 2026
License     : MIT
-}

module Main (main) where

import Test.Hspec
import qualified STUNIR.SemanticIR.Emitters.BaseSpec as BaseSpec
import qualified STUNIR.SemanticIR.Emitters.CoreSpec as CoreSpec
import qualified STUNIR.SemanticIR.Emitters.LanguageFamiliesSpec as LangFamSpec
import qualified STUNIR.SemanticIR.Emitters.SpecializedSpec as SpecSpec

main :: IO ()
main = hspec $ do
  describe "STUNIR Haskell Emitters Test Suite" $ do
    describe "Base Infrastructure" BaseSpec.spec
    describe "Core Emitters" CoreSpec.spec
    describe "Language Family Emitters" LangFamSpec.spec
    describe "Specialized Emitters" SpecSpec.spec

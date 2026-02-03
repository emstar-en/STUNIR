{-# LANGUAGE OverloadedStrings #-}
module STUNIR.SemanticIR.Emitters.Specialized.Functional
  ( FunctionalEmitter, FunctionalConfig(..)
  , defaultFunctionalConfig, emitFunctional
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import STUNIR.SemanticIR.Emitters.Base
import STUNIR.SemanticIR.Emitters.Types

data FunctionalConfig = FunctionalConfig
  { functionalBaseConfig :: !EmitterConfig
  } deriving (Show)

defaultFunctionalConfig :: FilePath -> Text -> FunctionalConfig
defaultFunctionalConfig outputDir moduleName = FunctionalConfig
  { functionalBaseConfig = defaultEmitterConfig outputDir moduleName
  }

data FunctionalEmitter = FunctionalEmitter FunctionalConfig

instance Emitter FunctionalEmitter where
  emit (FunctionalEmitter config) irModule
    | not (validateIR irModule) = Left "Invalid IR module"
    | otherwise = Right $ EmitterResult Success [mainFile] (gfSize mainFile) Nothing
    where mainFile = GeneratedFile (T.unpack $ imModuleName irModule <> ".txt")
                                    (computeFileHash content) (T.length content)
          content = T.unlines ["-- STUNIR Generated Functional Code", "-- Module: " <> imModuleName irModule]

emitFunctional :: IRModule -> FilePath -> Either Text EmitterResult
emitFunctional irModule outputDir =
  emit (FunctionalEmitter $ defaultFunctionalConfig outputDir (imModuleName irModule)) irModule

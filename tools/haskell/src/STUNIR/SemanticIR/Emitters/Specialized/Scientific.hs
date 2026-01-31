{-# LANGUAGE OverloadedStrings #-}
module STUNIR.SemanticIR.Emitters.Specialized.Scientific
  ( ScientificEmitter, ScientificConfig(..)
  , defaultScientificConfig, emitScientific
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import STUNIR.SemanticIR.Emitters.Base
import STUNIR.SemanticIR.Emitters.Types

data ScientificConfig = ScientificConfig
  { scientificBaseConfig :: !EmitterConfig
  } deriving (Show)

defaultScientificConfig :: FilePath -> Text -> ScientificConfig
defaultScientificConfig outputDir moduleName = ScientificConfig
  { scientificBaseConfig = defaultEmitterConfig outputDir moduleName
  }

data ScientificEmitter = ScientificEmitter ScientificConfig

instance Emitter ScientificEmitter where
  emit (ScientificEmitter config) irModule
    | not (validateIR irModule) = Left "Invalid IR module"
    | otherwise = Right $ EmitterResult Success [mainFile] (gfSize mainFile) Nothing
    where mainFile = GeneratedFile (T.unpack $ imModuleName irModule <> ".txt")
                                    (computeFileHash content) (T.length content)
          content = T.unlines ["-- STUNIR Generated Scientific Code", "-- Module: " <> imModuleName irModule]

emitScientific :: IRModule -> FilePath -> Either Text EmitterResult
emitScientific irModule outputDir =
  emit (ScientificEmitter $ defaultScientificConfig outputDir (imModuleName irModule)) irModule

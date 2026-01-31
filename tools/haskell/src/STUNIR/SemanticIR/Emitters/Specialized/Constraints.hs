{-# LANGUAGE OverloadedStrings #-}
module STUNIR.SemanticIR.Emitters.Specialized.Constraints
  ( ConstraintsEmitter, ConstraintsConfig(..)
  , defaultConstraintsConfig, emitConstraints
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import STUNIR.SemanticIR.Emitters.Base
import STUNIR.SemanticIR.Emitters.Types

data ConstraintsConfig = ConstraintsConfig
  { constraintsBaseConfig :: !EmitterConfig
  } deriving (Show)

defaultConstraintsConfig :: FilePath -> Text -> ConstraintsConfig
defaultConstraintsConfig outputDir moduleName = ConstraintsConfig
  { constraintsBaseConfig = defaultEmitterConfig outputDir moduleName
  }

data ConstraintsEmitter = ConstraintsEmitter ConstraintsConfig

instance Emitter ConstraintsEmitter where
  emit (ConstraintsEmitter config) irModule
    | not (validateIR irModule) = Left "Invalid IR module"
    | otherwise = Right $ EmitterResult Success [mainFile] (gfSize mainFile) Nothing
    where mainFile = GeneratedFile (T.unpack $ imModuleName irModule <> ".txt")
                                    (computeFileHash content) (T.length content)
          content = T.unlines ["-- STUNIR Generated Constraints Code", "-- Module: " <> imModuleName irModule]

emitConstraints :: IRModule -> FilePath -> Either Text EmitterResult
emitConstraints irModule outputDir =
  emit (ConstraintsEmitter $ defaultConstraintsConfig outputDir (imModuleName irModule)) irModule

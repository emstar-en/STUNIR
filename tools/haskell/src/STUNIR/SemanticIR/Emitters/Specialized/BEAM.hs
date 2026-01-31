{-# LANGUAGE OverloadedStrings #-}
module STUNIR.SemanticIR.Emitters.Specialized.BEAM
  ( BEAMEmitter, BEAMConfig(..)
  , defaultBEAMConfig, emitBEAM
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import STUNIR.SemanticIR.Emitters.Base
import STUNIR.SemanticIR.Emitters.Types

data BEAMConfig = BEAMConfig
  { beamBaseConfig :: !EmitterConfig
  } deriving (Show)

defaultBEAMConfig :: FilePath -> Text -> BEAMConfig
defaultBEAMConfig outputDir moduleName = BEAMConfig
  { beamBaseConfig = defaultEmitterConfig outputDir moduleName
  }

data BEAMEmitter = BEAMEmitter BEAMConfig

instance Emitter BEAMEmitter where
  emit (BEAMEmitter config) irModule
    | not (validateIR irModule) = Left "Invalid IR module"
    | otherwise = Right $ EmitterResult Success [mainFile] (gfSize mainFile) Nothing
    where mainFile = GeneratedFile (T.unpack $ imModuleName irModule <> ".txt")
                                    (computeFileHash content) (T.length content)
          content = T.unlines ["-- STUNIR Generated BEAM Code", "-- Module: " <> imModuleName irModule]

emitBEAM :: IRModule -> FilePath -> Either Text EmitterResult
emitBEAM irModule outputDir =
  emit (BEAMEmitter $ defaultBEAMConfig outputDir (imModuleName irModule)) irModule

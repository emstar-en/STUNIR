{-# LANGUAGE OverloadedStrings #-}
module STUNIR.SemanticIR.Emitters.Specialized.Expert
  ( ExpertEmitter, ExpertConfig(..)
  , defaultExpertConfig, emitExpert
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import STUNIR.SemanticIR.Emitters.Base
import STUNIR.SemanticIR.Emitters.Types

data ExpertConfig = ExpertConfig
  { expertBaseConfig :: !EmitterConfig
  } deriving (Show)

defaultExpertConfig :: FilePath -> Text -> ExpertConfig
defaultExpertConfig outputDir moduleName = ExpertConfig
  { expertBaseConfig = defaultEmitterConfig outputDir moduleName
  }

data ExpertEmitter = ExpertEmitter ExpertConfig

instance Emitter ExpertEmitter where
  emit (ExpertEmitter config) irModule
    | not (validateIR irModule) = Left "Invalid IR module"
    | otherwise = Right $ EmitterResult Success [mainFile] (gfSize mainFile) Nothing
    where mainFile = GeneratedFile (T.unpack $ imModuleName irModule <> ".txt")
                                    (computeFileHash content) (T.length content)
          content = T.unlines ["-- STUNIR Generated Expert Code", "-- Module: " <> imModuleName irModule]

emitExpert :: IRModule -> FilePath -> Either Text EmitterResult
emitExpert irModule outputDir =
  emit (ExpertEmitter $ defaultExpertConfig outputDir (imModuleName irModule)) irModule

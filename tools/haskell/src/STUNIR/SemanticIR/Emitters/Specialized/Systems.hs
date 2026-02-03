{-# LANGUAGE OverloadedStrings #-}
module STUNIR.SemanticIR.Emitters.Specialized.Systems
  ( SystemsEmitter, SystemsConfig(..)
  , defaultSystemsConfig, emitSystems
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import STUNIR.SemanticIR.Emitters.Base
import STUNIR.SemanticIR.Emitters.Types

data SystemsConfig = SystemsConfig
  { systemsBaseConfig :: !EmitterConfig
  } deriving (Show)

defaultSystemsConfig :: FilePath -> Text -> SystemsConfig
defaultSystemsConfig outputDir moduleName = SystemsConfig
  { systemsBaseConfig = defaultEmitterConfig outputDir moduleName
  }

data SystemsEmitter = SystemsEmitter SystemsConfig

instance Emitter SystemsEmitter where
  emit (SystemsEmitter config) irModule
    | not (validateIR irModule) = Left "Invalid IR module"
    | otherwise = Right $ EmitterResult Success [mainFile] (gfSize mainFile) Nothing
    where mainFile = GeneratedFile (T.unpack $ imModuleName irModule <> ".txt")
                                    (computeFileHash content) (T.length content)
          content = T.unlines ["-- STUNIR Generated Systems Code", "-- Module: " <> imModuleName irModule]

emitSystems :: IRModule -> FilePath -> Either Text EmitterResult
emitSystems irModule outputDir =
  emit (SystemsEmitter $ defaultSystemsConfig outputDir (imModuleName irModule)) irModule

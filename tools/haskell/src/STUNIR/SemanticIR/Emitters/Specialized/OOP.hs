{-# LANGUAGE OverloadedStrings #-}
module STUNIR.SemanticIR.Emitters.Specialized.OOP
  ( OOPEmitter, OOPConfig(..)
  , defaultOOPConfig, emitOOP
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import STUNIR.SemanticIR.Emitters.Base
import STUNIR.SemanticIR.Emitters.Types

data OOPConfig = OOPConfig
  { oopBaseConfig :: !EmitterConfig
  } deriving (Show)

defaultOOPConfig :: FilePath -> Text -> OOPConfig
defaultOOPConfig outputDir moduleName = OOPConfig
  { oopBaseConfig = defaultEmitterConfig outputDir moduleName
  }

data OOPEmitter = OOPEmitter OOPConfig

instance Emitter OOPEmitter where
  emit (OOPEmitter config) irModule
    | not (validateIR irModule) = Left "Invalid IR module"
    | otherwise = Right $ EmitterResult Success [mainFile] (gfSize mainFile) Nothing
    where mainFile = GeneratedFile (T.unpack $ imModuleName irModule <> ".txt")
                                    (computeFileHash content) (T.length content)
          content = T.unlines ["-- STUNIR Generated OOP Code", "-- Module: " <> imModuleName irModule]

emitOOP :: IRModule -> FilePath -> Either Text EmitterResult
emitOOP irModule outputDir =
  emit (OOPEmitter $ defaultOOPConfig outputDir (imModuleName irModule)) irModule

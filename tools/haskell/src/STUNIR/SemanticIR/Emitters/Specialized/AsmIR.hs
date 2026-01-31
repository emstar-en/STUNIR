{-# LANGUAGE OverloadedStrings #-}
module STUNIR.SemanticIR.Emitters.Specialized.AsmIR
  ( AsmIREmitter, AsmIRConfig(..)
  , defaultAsmIRConfig, emitAsmIR
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import STUNIR.SemanticIR.Emitters.Base
import STUNIR.SemanticIR.Emitters.Types

data AsmIRConfig = AsmIRConfig
  { asmirBaseConfig :: !EmitterConfig
  } deriving (Show)

defaultAsmIRConfig :: FilePath -> Text -> AsmIRConfig
defaultAsmIRConfig outputDir moduleName = AsmIRConfig
  { asmirBaseConfig = defaultEmitterConfig outputDir moduleName
  }

data AsmIREmitter = AsmIREmitter AsmIRConfig

instance Emitter AsmIREmitter where
  emit (AsmIREmitter config) irModule
    | not (validateIR irModule) = Left "Invalid IR module"
    | otherwise = Right $ EmitterResult Success [mainFile] (gfSize mainFile) Nothing
    where mainFile = GeneratedFile (T.unpack $ imModuleName irModule <> ".txt")
                                    (computeFileHash content) (T.length content)
          content = T.unlines ["-- STUNIR Generated AsmIR Code", "-- Module: " <> imModuleName irModule]

emitAsmIR :: IRModule -> FilePath -> Either Text EmitterResult
emitAsmIR irModule outputDir =
  emit (AsmIREmitter $ defaultAsmIRConfig outputDir (imModuleName irModule)) irModule

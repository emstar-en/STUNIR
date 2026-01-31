{-# LANGUAGE OverloadedStrings #-}
module STUNIR.SemanticIR.Emitters.Specialized.Business
  ( BusinessEmitter, BusinessConfig(..), BusinessLanguage(..)
  , defaultBusinessConfig, emitBusiness
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import STUNIR.SemanticIR.Emitters.Base
import STUNIR.SemanticIR.Emitters.Types
import STUNIR.SemanticIR.Emitters.CodeGen

data BusinessLanguage = LangCOBOL | LangBASIC | LangVisualBasic deriving (Eq, Show)
data BusinessConfig = BusinessConfig
  { busBaseConfig :: !EmitterConfig
  , busLanguage   :: !BusinessLanguage
  } deriving (Show)

defaultBusinessConfig :: FilePath -> Text -> BusinessLanguage -> BusinessConfig
defaultBusinessConfig outputDir moduleName lang = BusinessConfig
  { busBaseConfig = defaultEmitterConfig outputDir moduleName
  , busLanguage = lang
  }

data BusinessEmitter = BusinessEmitter BusinessConfig

instance Emitter BusinessEmitter where
  emit (BusinessEmitter config) irModule
    | not (validateIR irModule) = Left "Invalid IR module"
    | otherwise = Right $ EmitterResult Success [mainFile] (gfSize mainFile) Nothing
    where mainFile = GeneratedFile (T.unpack $ imModuleName irModule <> ".cob")
                                    (computeFileHash content) (T.length content)
          content = T.unlines ["       IDENTIFICATION DIVISION.",
                               "       PROGRAM-ID. " <> imModuleName irModule <> "."]

emitBusiness :: IRModule -> FilePath -> BusinessLanguage -> Either Text EmitterResult
emitBusiness irModule outputDir lang =
  emit (BusinessEmitter $ defaultBusinessConfig outputDir (imModuleName irModule) lang) irModule

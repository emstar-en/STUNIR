{-# LANGUAGE OverloadedStrings #-}
module STUNIR.SemanticIR.Emitters.Specialized.ASP
  ( ASPEmitter, ASPConfig(..)
  , defaultASPConfig, emitASP
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import STUNIR.SemanticIR.Emitters.Base
import STUNIR.SemanticIR.Emitters.Types

data ASPConfig = ASPConfig
  { aspBaseConfig :: !EmitterConfig
  } deriving (Show)

defaultASPConfig :: FilePath -> Text -> ASPConfig
defaultASPConfig outputDir moduleName = ASPConfig
  { aspBaseConfig = defaultEmitterConfig outputDir moduleName
  }

data ASPEmitter = ASPEmitter ASPConfig

instance Emitter ASPEmitter where
  emit (ASPEmitter config) irModule
    | not (validateIR irModule) = Left "Invalid IR module"
    | otherwise = Right $ EmitterResult Success [mainFile] (gfSize mainFile) Nothing
    where mainFile = GeneratedFile (T.unpack $ imModuleName irModule <> ".txt")
                                    (computeFileHash content) (T.length content)
          content = T.unlines ["-- STUNIR Generated ASP Code", "-- Module: " <> imModuleName irModule]

emitASP :: IRModule -> FilePath -> Either Text EmitterResult
emitASP irModule outputDir =
  emit (ASPEmitter $ defaultASPConfig outputDir (imModuleName irModule)) irModule

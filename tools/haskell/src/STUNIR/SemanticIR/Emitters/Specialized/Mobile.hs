{-# LANGUAGE OverloadedStrings #-}
module STUNIR.SemanticIR.Emitters.Specialized.Mobile
  ( MobileEmitter, MobileConfig(..)
  , defaultMobileConfig, emitMobile
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import STUNIR.SemanticIR.Emitters.Base
import STUNIR.SemanticIR.Emitters.Types

data MobileConfig = MobileConfig
  { mobileBaseConfig :: !EmitterConfig
  } deriving (Show)

defaultMobileConfig :: FilePath -> Text -> MobileConfig
defaultMobileConfig outputDir moduleName = MobileConfig
  { mobileBaseConfig = defaultEmitterConfig outputDir moduleName
  }

data MobileEmitter = MobileEmitter MobileConfig

instance Emitter MobileEmitter where
  emit (MobileEmitter config) irModule
    | not (validateIR irModule) = Left "Invalid IR module"
    | otherwise = Right $ EmitterResult Success [mainFile] (gfSize mainFile) Nothing
    where mainFile = GeneratedFile (T.unpack $ imModuleName irModule <> ".txt")
                                    (computeFileHash content) (T.length content)
          content = T.unlines ["-- STUNIR Generated Mobile Code", "-- Module: " <> imModuleName irModule]

emitMobile :: IRModule -> FilePath -> Either Text EmitterResult
emitMobile irModule outputDir =
  emit (MobileEmitter $ defaultMobileConfig outputDir (imModuleName irModule)) irModule

{-# LANGUAGE OverloadedStrings #-}
module STUNIR.SemanticIR.Emitters.Specialized.Planning
  ( PlanningEmitter, PlanningConfig(..)
  , defaultPlanningConfig, emitPlanning
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import STUNIR.SemanticIR.Emitters.Base
import STUNIR.SemanticIR.Emitters.Types

data PlanningConfig = PlanningConfig
  { planningBaseConfig :: !EmitterConfig
  } deriving (Show)

defaultPlanningConfig :: FilePath -> Text -> PlanningConfig
defaultPlanningConfig outputDir moduleName = PlanningConfig
  { planningBaseConfig = defaultEmitterConfig outputDir moduleName
  }

data PlanningEmitter = PlanningEmitter PlanningConfig

instance Emitter PlanningEmitter where
  emit (PlanningEmitter config) irModule
    | not (validateIR irModule) = Left "Invalid IR module"
    | otherwise = Right $ EmitterResult Success [mainFile] (gfSize mainFile) Nothing
    where mainFile = GeneratedFile (T.unpack $ imModuleName irModule <> ".txt")
                                    (computeFileHash content) (T.length content)
          content = T.unlines ["-- STUNIR Generated Planning Code", "-- Module: " <> imModuleName irModule]

emitPlanning :: IRModule -> FilePath -> Either Text EmitterResult
emitPlanning irModule outputDir =
  emit (PlanningEmitter $ defaultPlanningConfig outputDir (imModuleName irModule)) irModule

{-# LANGUAGE OverloadedStrings #-}
module STUNIR.SemanticIR.Emitters.Specialized.Parser
  ( ParserEmitter, ParserConfig(..), ParserGen(..)
  , defaultParserConfig, emitParser
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import STUNIR.SemanticIR.Emitters.Base
import STUNIR.SemanticIR.Emitters.Types

data ParserGen = YaccGen | BisonGen | ANTLRParser | JavaCC | CUP deriving (Eq, Show)
data ParserConfig = ParserConfig
  { parBaseConfig :: !EmitterConfig
  , parGen        :: !ParserGen
  } deriving (Show)

defaultParserConfig :: FilePath -> Text -> ParserGen -> ParserConfig
defaultParserConfig outputDir moduleName gen = ParserConfig
  { parBaseConfig = defaultEmitterConfig outputDir moduleName
  , parGen = gen
  }

data ParserEmitter = ParserEmitter ParserConfig

instance Emitter ParserEmitter where
  emit (ParserEmitter config) irModule
    | not (validateIR irModule) = Left "Invalid IR module"
    | otherwise = Right $ EmitterResult Success [mainFile] (gfSize mainFile) Nothing
    where mainFile = GeneratedFile (T.unpack $ imModuleName irModule <> ".y")
                                    (computeFileHash content) (T.length content)
          content = T.unlines ["%{", "/* STUNIR Generated Parser */", "%}"]

emitParser :: IRModule -> FilePath -> ParserGen -> Either Text EmitterResult
emitParser irModule outputDir gen =
  emit (ParserEmitter $ defaultParserConfig outputDir (imModuleName irModule) gen) irModule

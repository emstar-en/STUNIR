{-# LANGUAGE OverloadedStrings #-}
module STUNIR.SemanticIR.Emitters.Specialized.Lexer
  ( LexerEmitter, LexerConfig(..), LexerGen(..)
  , defaultLexerConfig, emitLexer
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import STUNIR.SemanticIR.Emitters.Base
import STUNIR.SemanticIR.Emitters.Types

data LexerGen = Flex | Lex | JFlex | ANTLRLexer | RE2C | Ragel deriving (Eq, Show)
data LexerConfig = LexerConfig
  { lexBaseConfig :: !EmitterConfig
  , lexGen        :: !LexerGen
  } deriving (Show)

defaultLexerConfig :: FilePath -> Text -> LexerGen -> LexerConfig
defaultLexerConfig outputDir moduleName gen = LexerConfig
  { lexBaseConfig = defaultEmitterConfig outputDir moduleName
  , lexGen = gen
  }

data LexerEmitter = LexerEmitter LexerConfig

instance Emitter LexerEmitter where
  emit (LexerEmitter config) irModule
    | not (validateIR irModule) = Left "Invalid IR module"
    | otherwise = Right $ EmitterResult Success [mainFile] (gfSize mainFile) Nothing
    where mainFile = GeneratedFile (T.unpack $ imModuleName irModule <> ".l")
                                    (computeFileHash content) (T.length content)
          content = T.unlines ["%{", "/* STUNIR Generated Lexer */", "%}"]

emitLexer :: IRModule -> FilePath -> LexerGen -> Either Text EmitterResult
emitLexer irModule outputDir gen =
  emit (LexerEmitter $ defaultLexerConfig outputDir (imModuleName irModule) gen) irModule

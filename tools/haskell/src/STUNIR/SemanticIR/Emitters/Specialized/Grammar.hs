{-# LANGUAGE OverloadedStrings #-}
module STUNIR.SemanticIR.Emitters.Specialized.Grammar
  ( GrammarEmitter, GrammarConfig(..), GrammarSpec(..)
  , defaultGrammarConfig, emitGrammar
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import STUNIR.SemanticIR.Emitters.Base
import STUNIR.SemanticIR.Emitters.Types

data GrammarSpec = ANTLR | PEG | BNF | EBNF | Yacc | Bison deriving (Eq, Show)
data GrammarConfig = GrammarConfig
  { gramBaseConfig :: !EmitterConfig
  , gramSpec       :: !GrammarSpec
  } deriving (Show)

defaultGrammarConfig :: FilePath -> Text -> GrammarSpec -> GrammarConfig
defaultGrammarConfig outputDir moduleName spec = GrammarConfig
  { gramBaseConfig = defaultEmitterConfig outputDir moduleName
  , gramSpec = spec
  }

data GrammarEmitter = GrammarEmitter GrammarConfig

instance Emitter GrammarEmitter where
  emit (GrammarEmitter config) irModule
    | not (validateIR irModule) = Left "Invalid IR module"
    | otherwise = Right $ EmitterResult Success [mainFile] (gfSize mainFile) Nothing
    where mainFile = GeneratedFile (T.unpack $ imModuleName irModule <> ".g4")
                                    (computeFileHash content) (T.length content)
          content = T.unlines ["grammar " <> imModuleName irModule <> ";"]

emitGrammar :: IRModule -> FilePath -> GrammarSpec -> Either Text EmitterResult
emitGrammar irModule outputDir spec =
  emit (GrammarEmitter $ defaultGrammarConfig outputDir (imModuleName irModule) spec) irModule

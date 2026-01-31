{-# LANGUAGE OverloadedStrings #-}

{-|
Module      : STUNIR.SemanticIR.Emitters.Core.Polyglot
Description : Multi-language emitter (C89, C99, Rust)
Copyright   : (c) STUNIR Team, 2026
License     : MIT

Emits code in multiple high-level languages.
Supports C89, C99, and Rust with appropriate standards compliance.
Based on Ada SPARK polyglot emitters.
-}

module STUNIR.SemanticIR.Emitters.Core.Polyglot
  ( PolyglotEmitter
  , PolyglotConfig(..)
  , PolyglotLanguage(..)
  , defaultPolyglotConfig
  , emitPolyglot
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import STUNIR.SemanticIR.Emitters.Base
import STUNIR.SemanticIR.Emitters.Types
import STUNIR.SemanticIR.Emitters.CodeGen

-- | Polyglot language targets
data PolyglotLanguage
  = LangC89
  | LangC99
  | LangRust
  deriving (Eq, Show)

-- | Polyglot emitter configuration
data PolyglotConfig = PolyglotConfig
  { polyBaseConfig :: !EmitterConfig
  , polyLanguage   :: !PolyglotLanguage
  } deriving (Show)

-- | Default polyglot configuration
defaultPolyglotConfig :: FilePath -> Text -> PolyglotLanguage -> PolyglotConfig
defaultPolyglotConfig outputDir moduleName language = PolyglotConfig
  { polyBaseConfig = defaultEmitterConfig outputDir moduleName
  , polyLanguage = language
  }

-- | Polyglot emitter
data PolyglotEmitter = PolyglotEmitter PolyglotConfig

instance Emitter PolyglotEmitter where
  emit (PolyglotEmitter config) irModule
    | not (validateIR irModule) = Left "Invalid IR module structure"
    | otherwise = Right $ EmitterResult
        { erStatus = Success
        , erFiles = [mainFile]
        , erTotalSize = gfSize mainFile
        , erErrorMessage = Nothing
        }
    where
      mainFile = generatePolyglotFile config irModule

-- | Generate polyglot file
generatePolyglotFile :: PolyglotConfig -> IRModule -> GeneratedFile
generatePolyglotFile config irModule =
  let content = generatePolyglotCode config irModule
      extension = getLanguageExtension (polyLanguage config)
      fileName = imModuleName irModule <> extension
  in GeneratedFile
       { gfPath = T.unpack fileName
       , gfHash = computeFileHash content
       , gfSize = T.length content
       }

-- | Generate polyglot code
generatePolyglotCode :: PolyglotConfig -> IRModule -> Text
generatePolyglotCode config irModule =
  case polyLanguage config of
    LangC89 -> generateC89Code config irModule
    LangC99 -> generateC99Code config irModule
    LangRust -> generateRustCode config irModule

-- | Generate C89 code
generateC89Code :: PolyglotConfig -> IRModule -> Text
generateC89Code config irModule = T.unlines $
  [ getDO178CHeader (ecAddDO178CHeaders baseConfig) "C89 Code"
  , "/* ANSI C89 compliant code */"
  , ""
  , "/* Note: Using long for 32-bit, long long for 64-bit (pre-stdint.h) */"
  , "typedef signed char int8_t;"
  , "typedef unsigned char uint8_t;"
  , "typedef short int16_t;"
  , "typedef unsigned short uint16_t;"
  , "typedef long int32_t;"
  , "typedef unsigned long uint32_t;"
  , ""
  ] ++
  concatMap (generateCFunction baseConfig "c89") (imFunctions irModule)
  where
    baseConfig = polyBaseConfig config

-- | Generate C99 code
generateC99Code :: PolyglotConfig -> IRModule -> Text
generateC99Code config irModule = T.unlines $
  [ getDO178CHeader (ecAddDO178CHeaders baseConfig) "C99 Code"
  , "#include <stdint.h>"
  , "#include <stdbool.h>"
  , "#include <stddef.h>"
  , ""
  ] ++
  concatMap (generateCFunction baseConfig "c99") (imFunctions irModule)
  where
    baseConfig = polyBaseConfig config

-- | Generate Rust code
generateRustCode :: PolyglotConfig -> IRModule -> Text
generateRustCode config irModule = T.unlines $
  [ "//! STUNIR Generated Rust Code"
  , "//! Module: " <> imModuleName irModule
  , ""
  , "#![allow(unused_variables)]"
  , ""
  ] ++
  concatMap (generateRustFunction baseConfig) (imFunctions irModule)
  where
    baseConfig = polyBaseConfig config

-- | Generate C function
generateCFunction :: EmitterConfig -> Text -> IRFunction -> [Text]
generateCFunction config lang func =
  let signature = generateFunctionSignature
                    (ifName func)
                    [(ipName p, mapIRTypeToC (ipType p)) | p <- ifParameters func]
                    (mapIRTypeToC (ifReturnType func))
                    (T.unpack lang)
      indent = indentString (ecIndentSize config) 1
  in [ signature
     , "{"
     , indent <> "/* Function body */"
     ] ++
     (if ifReturnType func /= TypeVoid
      then [indent <> "return 0;"]
      else []) ++
     ["}", ""]

-- | Generate Rust function
generateRustFunction :: EmitterConfig -> IRFunction -> [Text]
generateRustFunction config func =
  let params = T.intercalate ", " [ipName p <> ": " <> mapIRTypeToRust (ipType p)
                                   | p <- ifParameters func]
      returnType = mapIRTypeToRust (ifReturnType func)
      indent = indentString (ecIndentSize config) 1
  in [ "pub fn " <> ifName func <> "(" <> params <> ") -> " <> returnType <> " {"
     , indent <> "// Function body"
     ] ++
     (if ifReturnType func /= TypeVoid
      then [indent <> "unimplemented!()"]
      else []) ++
     ["}", ""]

-- | Get file extension for language
getLanguageExtension :: PolyglotLanguage -> Text
getLanguageExtension LangC89 = ".c"
getLanguageExtension LangC99 = ".c"
getLanguageExtension LangRust = ".rs"

-- | Convenience function for direct usage
emitPolyglot
  :: IRModule
  -> FilePath
  -> PolyglotLanguage
  -> Either Text EmitterResult
emitPolyglot irModule outputDir language =
  let config = defaultPolyglotConfig outputDir (imModuleName irModule) language
      emitter = PolyglotEmitter config
  in emit emitter irModule

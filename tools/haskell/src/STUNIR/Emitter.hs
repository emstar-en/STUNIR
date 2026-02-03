{-# LANGUAGE OverloadedStrings #-}

{-|
Module      : STUNIR.Emitter
Description : Code emitters for various targets
Copyright   : (c) STUNIR Team, 2026
License     : MIT
-}

module STUNIR.Emitter
  ( emitCode
  , emitC99
  , emitRust
  , emitPython
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import STUNIR.Types

-- | Emit code for specified target
emitCode :: IRModule -> Text -> Either String Text
emitCode module_ target
  | target == "c" || target == "c99" = Right $ emitC99 module_
  | target == "rust"                 = Right $ emitRust module_
  | target == "python"               = Right $ emitPython module_
  | otherwise                        = Left $ "Unsupported target: " <> T.unpack target

-- | Emit C99 code
emitC99 :: IRModule -> Text
emitC99 (IRModule name _ functions) = T.unlines $
  [ "/*"
  , " * STUNIR Generated Code"
  , " * Language: C99"
  , " * Module: " <> name
  , " * Generator: Haskell Pipeline"
  , " */"
  , ""
  , "#include <stdint.h>"
  , "#include <stdbool.h>"
  , ""
  ] ++ concatMap emitC99Function functions

emitC99Function :: IRFunction -> [Text]
emitC99Function (IRFunction name retType params _) =
  [ toCType retType
  , name <> "(" <> T.intercalate ", " (map emitC99Param params) <> ")"
  , "{"
  , "    /* Function body */"
  , "}"
  , ""
  ]
  where
    emitC99Param (IRParameter n t) = toCType t <> " " <> n

-- | Emit Rust code
emitRust :: IRModule -> Text
emitRust (IRModule name _ functions) = T.unlines $
  [ "//! STUNIR Generated Code"
  , "//! Language: Rust"
  , "//! Module: " <> name
  , "//! Generator: Haskell Pipeline"
  , ""
  ] ++ concatMap emitRustFunction functions

emitRustFunction :: IRFunction -> [Text]
emitRustFunction (IRFunction name retType params _) =
  [ "pub fn " <> name <> "(" <> T.intercalate ", " (map emitRustParam params) <> ") -> " <> toRustType retType <> " {"
  , "    // Function body"
  , "    unimplemented!()"
  , "}"
  , ""
  ]
  where
    emitRustParam (IRParameter n t) = n <> ": " <> toRustType t

-- | Emit Python code
emitPython :: IRModule -> Text
emitPython (IRModule name _ functions) = T.unlines $
  [ "\"\"\""
  , "STUNIR Generated Code"
  , "Language: Python"
  , "Module: " <> name
  , "Generator: Haskell Pipeline"
  , "\"\"\""
  , ""
  ] ++ concatMap emitPythonFunction functions

emitPythonFunction :: IRFunction -> [Text]
emitPythonFunction (IRFunction name _ params _) =
  [ "def " <> name <> "(" <> T.intercalate ", " (map paramName params) <> "):"
  , "    \"\"\"Function body\"\"\""
  , "    pass"
  , ""
  ]

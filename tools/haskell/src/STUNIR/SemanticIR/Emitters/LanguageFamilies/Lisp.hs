{-# LANGUAGE OverloadedStrings #-}

{-|
Module      : STUNIR.SemanticIR.Emitters.LanguageFamilies.Lisp
Description : Lisp family emitter (Common Lisp, Scheme, Clojure, Racket, Emacs Lisp, Guile, Hy, Janet)
Copyright   : (c) STUNIR Team, 2026
License     : MIT

Emits code for various Lisp dialects.
Supports Common Lisp, Scheme, Clojure, Racket, Emacs Lisp, Guile, Hy, and Janet.
Based on Ada SPARK lisp emitters.
-}

module STUNIR.SemanticIR.Emitters.LanguageFamilies.Lisp
  ( LispEmitter
  , LispConfig(..)
  , LispDialect(..)
  , defaultLispConfig
  , emitLisp
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import STUNIR.SemanticIR.Emitters.Base
import STUNIR.SemanticIR.Emitters.Types
import STUNIR.SemanticIR.Emitters.CodeGen

-- | Lisp dialect types
data LispDialect
  = DialectCommonLisp
  | DialectScheme
  | DialectClojure
  | DialectRacket
  | DialectEmacsLisp
  | DialectGuile
  | DialectHy
  | DialectJanet
  deriving (Eq, Show)

-- | Lisp emitter configuration
data LispConfig = LispConfig
  { lispBaseConfig :: !EmitterConfig
  , lispDialect    :: !LispDialect
  } deriving (Show)

-- | Default Lisp configuration
defaultLispConfig :: FilePath -> Text -> LispDialect -> LispConfig
defaultLispConfig outputDir moduleName dialect = LispConfig
  { lispBaseConfig = defaultEmitterConfig outputDir moduleName
  , lispDialect = dialect
  }

-- | Lisp emitter
data LispEmitter = LispEmitter LispConfig

instance Emitter LispEmitter where
  emit (LispEmitter config) irModule
    | not (validateIR irModule) = Left "Invalid IR module structure"
    | otherwise = Right $ EmitterResult
        { erStatus = Success
        , erFiles = [mainFile]
        , erTotalSize = gfSize mainFile
        , erErrorMessage = Nothing
        }
    where
      mainFile = generateLispFile config irModule

-- | Generate Lisp file
generateLispFile :: LispConfig -> IRModule -> GeneratedFile
generateLispFile config irModule =
  let content = generateLispCode config irModule
      extension = getDialectExtension (lispDialect config)
      fileName = imModuleName irModule <> extension
  in GeneratedFile
       { gfPath = T.unpack fileName
       , gfHash = computeFileHash content
       , gfSize = T.length content
       }

-- | Generate Lisp code
generateLispCode :: LispConfig -> IRModule -> Text
generateLispCode config irModule =
  case lispDialect config of
    DialectCommonLisp -> generateCommonLispCode config irModule
    DialectScheme -> generateSchemeCode config irModule
    DialectClojure -> generateClojureCode config irModule
    DialectRacket -> generateRacketCode config irModule
    DialectEmacsLisp -> generateEmacsLispCode config irModule
    DialectGuile -> generateGuileCode config irModule
    DialectHy -> generateHyCode config irModule
    DialectJanet -> generateJanetCode config irModule

-- | Generate Common Lisp code
generateCommonLispCode :: LispConfig -> IRModule -> Text
generateCommonLispCode config irModule = T.unlines $
  [ ";;; STUNIR Generated Common Lisp Code"
  , ";;; Module: " <> imModuleName irModule
  , ""
  , "(defpackage :" <> T.toLower (imModuleName irModule)
  , "  (:use :cl)"
  , "  (:export" ++ concatMap ((":" <>) . ifName) (imFunctions irModule) ++ ")"
  , ")"
  , ""
  , "(in-package :" <> T.toLower (imModuleName irModule) <> ")"
  , ""
  ] ++
  concatMap (generateCommonLispFunction config) (imFunctions irModule)

-- | Generate Scheme code
generateSchemeCode :: LispConfig -> IRModule -> Text
generateSchemeCode config irModule = T.unlines $
  [ ";; STUNIR Generated Scheme Code"
  , ";; Module: " <> imModuleName irModule
  , ""
  ] ++
  concatMap (generateSchemeFunction config) (imFunctions irModule)

-- | Generate Clojure code
generateClojureCode :: LispConfig -> IRModule -> Text
generateClojureCode config irModule = T.unlines $
  [ ";; STUNIR Generated Clojure Code"
  , ";; Module: " <> imModuleName irModule
  , ""
  , "(ns " <> T.toLower (imModuleName irModule) <> ")"
  , ""
  ] ++
  concatMap (generateClojureFunction config) (imFunctions irModule)

-- | Generate Racket code
generateRacketCode :: LispConfig -> IRModule -> Text
generateRacketCode config irModule = T.unlines $
  [ "#lang racket"
  , ";; STUNIR Generated Racket Code"
  , ";; Module: " <> imModuleName irModule
  , ""
  ] ++
  concatMap (generateRacketFunction config) (imFunctions irModule)

-- | Generate Emacs Lisp code
generateEmacsLispCode :: LispConfig -> IRModule -> Text
generateEmacsLispCode config irModule = T.unlines $
  [ ";;; STUNIR Generated Emacs Lisp Code"
  , ";;; Module: " <> imModuleName irModule
  , ""
  ] ++
  concatMap (generateEmacsLispFunction config) (imFunctions irModule) ++
  [ "(provide '" <> T.toLower (imModuleName irModule) <> ")" ]

-- | Generate Guile code
generateGuileCode :: LispConfig -> IRModule -> Text
generateGuileCode config irModule = T.unlines $
  [ ";; STUNIR Generated Guile Scheme Code"
  , ";; Module: " <> imModuleName irModule
  , ""
  , "(define-module (" <> T.toLower (imModuleName irModule) <> ")"
  , "  #:export (" <> T.intercalate " " (map ifName (imFunctions irModule)) <> "))"
  , ""
  ] ++
  concatMap (generateGuileFunction config) (imFunctions irModule)

-- | Generate Hy code
generateHyCode :: LispConfig -> IRModule -> Text
generateHyCode config irModule = T.unlines $
  [ "; STUNIR Generated Hy Code"
  , "; Module: " <> imModuleName irModule
  , ""
  ] ++
  concatMap (generateHyFunction config) (imFunctions irModule)

-- | Generate Janet code
generateJanetCode :: LispConfig -> IRModule -> Text
generateJanetCode config irModule = T.unlines $
  [ "# STUNIR Generated Janet Code"
  , "# Module: " <> imModuleName irModule
  , ""
  ] ++
  concatMap (generateJanetFunction config) (imFunctions irModule)

-- | Generate Common Lisp function
generateCommonLispFunction :: LispConfig -> IRFunction -> [Text]
generateCommonLispFunction config func =
  let params = map ipName (ifParameters func)
  in [ "(defun " <> ifName func <> " (" <> T.intercalate " " params <> ")"
     , "  \"Function generated from STUNIR IR.\""
     , "  ;; Function body"
     , "  nil)"
     , ""
     ]

-- | Generate Scheme function
generateSchemeFunction :: LispConfig -> IRFunction -> [Text]
generateSchemeFunction config func =
  let params = map ipName (ifParameters func)
  in [ "(define (" <> ifName func <> " " <> T.intercalate " " params <> ")"
     , "  ;; Function body"
     , "  #f)"
     , ""
     ]

-- | Generate Clojure function
generateClojureFunction :: LispConfig -> IRFunction -> [Text]
generateClojureFunction config func =
  let params = map ipName (ifParameters func)
  in [ "(defn " <> ifName func
     , "  [" <> T.intercalate " " params <> "]"
     , "  ;; Function body"
     , "  nil)"
     , ""
     ]

-- | Generate Racket function
generateRacketFunction :: LispConfig -> IRFunction -> [Text]
generateRacketFunction config func =
  let params = map ipName (ifParameters func)
  in [ "(define (" <> ifName func <> " " <> T.intercalate " " params <> ")"
     , "  ;; Function body"
     , "  #f)"
     , ""
     ]

-- | Generate Emacs Lisp function
generateEmacsLispFunction :: LispConfig -> IRFunction -> [Text]
generateEmacsLispFunction config func =
  let params = map ipName (ifParameters func)
  in [ "(defun " <> ifName func <> " (" <> T.intercalate " " params <> ")"
     , "  \"Function generated from STUNIR IR.\""
     , "  ;; Function body"
     , "  nil)"
     , ""
     ]

-- | Generate Guile function
generateGuileFunction :: LispConfig -> IRFunction -> [Text]
generateGuileFunction config func =
  let params = map ipName (ifParameters func)
  in [ "(define (" <> ifName func <> " " <> T.intercalate " " params <> ")"
     , "  ;; Function body"
     , "  #f)"
     , ""
     ]

-- | Generate Hy function
generateHyFunction :: LispConfig -> IRFunction -> [Text]
generateHyFunction config func =
  let params = map ipName (ifParameters func)
  in [ "(defn " <> ifName func <> " [" <> T.intercalate " " params <> "]"
     , "  ; Function body"
     , "  None)"
     , ""
     ]

-- | Generate Janet function
generateJanetFunction :: LispConfig -> IRFunction -> [Text]
generateJanetFunction config func =
  let params = map ipName (ifParameters func)
  in [ "(defn " <> ifName func
     , "  [" <> T.intercalate " " params <> "]"
     , "  # Function body"
     , "  nil)"
     , ""
     ]

-- | Get file extension for dialect
getDialectExtension :: LispDialect -> Text
getDialectExtension DialectCommonLisp = ".lisp"
getDialectExtension DialectScheme = ".scm"
getDialectExtension DialectClojure = ".clj"
getDialectExtension DialectRacket = ".rkt"
getDialectExtension DialectEmacsLisp = ".el"
getDialectExtension DialectGuile = ".scm"
getDialectExtension DialectHy = ".hy"
getDialectExtension DialectJanet = ".janet"

-- | Convenience function for direct usage
emitLisp
  :: IRModule
  -> FilePath
  -> LispDialect
  -> Either Text EmitterResult
emitLisp irModule outputDir dialect =
  let config = defaultLispConfig outputDir (imModuleName irModule) dialect
      emitter = LispEmitter config
  in emit emitter irModule

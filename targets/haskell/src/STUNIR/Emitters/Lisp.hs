{-# LANGUAGE OverloadedStrings #-}

-- | Lisp family code emitters
module STUNIR.Emitters.Lisp
  ( emitCommonLisp
  , emitScheme
  , emitClojure
  , LispDialect(..)
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import STUNIR.Emitters.Types

-- | Lisp dialect
data LispDialect
  = CommonLisp
  | Scheme
  | Clojure
  | Racket
  | EmacsLisp
  deriving (Show, Eq)

-- | Emit Common Lisp code
emitCommonLisp :: Text -> EmitterResult Text
emitCommonLisp moduleName = Right $ T.unlines
  [ ";; STUNIR Generated Common Lisp"
  , ";; Module: " <> moduleName
  , ";; Generator: Haskell Pipeline"
  , ""
  , "(defpackage :" <> T.toLower moduleName
  , "  (:use :cl))"
  , ""
  , "(in-package :" <> T.toLower moduleName <> ")"
  , ""
  , "(defun factorial (n)"
  , "  (if (<= n 1)"
  , "      1"
  , "      (* n (factorial (- n 1)))))"
  ]

-- | Emit Scheme code
emitScheme :: Text -> EmitterResult Text
emitScheme moduleName = Right $ T.unlines
  [ ";; STUNIR Generated Scheme"
  , ";; Module: " <> moduleName
  , ";; Generator: Haskell Pipeline"
  , ""
  , "(define (factorial n)"
  , "  (if (<= n 1)"
  , "      1"
  , "      (* n (factorial (- n 1)))))"
  , ""
  , "(define (map-double lst)"
  , "  (map (lambda (x) (* x 2)) lst))"
  ]

-- | Emit Clojure code
emitClojure :: Text -> EmitterResult Text
emitClojure moduleName = Right $ T.unlines
  [ ";; STUNIR Generated Clojure"
  , ";; Module: " <> moduleName
  , ";; Generator: Haskell Pipeline"
  , ""
  , "(ns " <> T.toLower moduleName <> ")"
  , ""
  , "(defn factorial [n]"
  , "  (if (<= n 1)"
  , "    1"
  , "    (* n (factorial (dec n)))))"
  , ""
  , "(defn map-double [coll]"
  , "  (map #(* % 2) coll))"
  ]

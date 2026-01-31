{-# LANGUAGE OverloadedStrings #-}

{-|
Module      : STUNIR.IR
Description : IR processing utilities
Copyright   : (c) STUNIR Team, 2026
License     : MIT
-}

module STUNIR.IR
  ( parseIR
  , irToJSON
  ) where

import Data.Aeson (Value, Result(..), fromJSON, toJSON)
import STUNIR.Types (IRModule)

-- | Parse IR from JSON
parseIR :: Value -> Either String IRModule
parseIR v = case fromJSON v of
  Success m -> Right m
  Error err -> Left err

-- | Convert IR to JSON
irToJSON :: IRModule -> Value
irToJSON = toJSON

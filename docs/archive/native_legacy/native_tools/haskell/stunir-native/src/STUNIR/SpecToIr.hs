{-# LANGUAGE OverloadedStrings #-}
-- |
-- Module      : STUNIR.SpecToIr
-- Description : Specification to IR Conversion
-- Copyright   : (c) STUNIR Team
-- License     : MIT
--
-- Maintainer  : stunir@example.com
-- Stability   : experimental
-- Portability : portable
--
-- Converts STUNIR specification files into Intermediate Representation (IR).
-- This is a core transformation step in the STUNIR pipeline.
--
-- = Conversion Process
--
-- 1. Parse the specification JSON file
-- 2. Extract modules and their metadata
-- 3. Generate IR structure from specification
-- 4. Serialize IR to JSON format
--
-- = Error Handling
--
-- JSON parse failures result in program termination with an error message.
-- All other errors are propagated as IO exceptions.
--
-- = Safety
--
-- The conversion is deterministic - the same specification always
-- produces the same IR output. This is critical for reproducible builds.
module STUNIR.SpecToIr (run) where

import qualified Data.ByteString.Lazy as B
import Data.Aeson
import qualified STUNIR.Spec as S
import qualified STUNIR.IR.V1 as IR
import System.Exit (die)

-- | Convert a specification file to IR.
--
-- Reads a specification JSON file, parses it, and generates
-- corresponding IR output.
--
-- === Arguments
--
-- * @inJson@ - Path to the input specification JSON file
-- * @outIr@ - Path for the output IR JSON file
--
-- === Returns
--
-- Returns @IO ()@, exiting with an error message if parsing fails.
--
-- === Example
--
-- @
-- run "spec.json" "output.ir.json"
-- @
--
-- === Safety
--
-- This function handles IO errors gracefully and provides
-- informative error messages for parse failures.
run :: FilePath -> FilePath -> IO ()
run inJson outIr = do
  input <- B.readFile inJson
  case decode input of
    Nothing -> die "Failed to parse Spec JSON"
    Just spec -> do
      let meta = IR.IrMetadata (S.kind spec) (S.modules spec)
      let ir = IR.IrV1 "ir" "stunir-native-haskell" "v1" "main" [] [] meta
      B.writeFile outIr (encode ir)
      putStrLn $ "Generated IR at " ++ outIr

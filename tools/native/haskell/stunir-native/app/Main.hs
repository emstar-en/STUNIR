{-# LANGUAGE OverloadedStrings #-}
-- |
-- Module      : Main
-- Description : STUNIR Native Core - Main Entry Point (Haskell)
-- Copyright   : (c) STUNIR Team
-- License     : MIT
--
-- Maintainer  : stunir@example.com
-- Stability   : experimental
-- Portability : portable
--
-- This is the main entry point for the STUNIR Native Core CLI tool.
-- STUNIR (Structured Typed Universal Notation for Intermediate Representation)
-- is a deterministic IR generation and verification toolkit for critical systems.
--
-- = Commands
--
-- * @spec-to-ir@: Convert specification JSON to IR format
-- * @gen-provenance@: Generate provenance information for IR files
-- * @check-toolchain@: Verify toolchain lock file integrity
--
-- = Safety
--
-- This tool is designed for use in critical systems. All operations are
-- deterministic and produce reproducible outputs given the same inputs.
module Main (main) where

import Options.Applicative
import qualified STUNIR.SpecToIr as SpecToIr
import qualified STUNIR.GenProvenance as GenProvenance
import qualified STUNIR.CheckToolchain as CheckToolchain

-- | Available CLI commands.
data Command
  = CmdSpecToIr FilePath FilePath      -- ^ Convert spec to IR
  | CmdGenProvenance FilePath FilePath FilePath  -- ^ Generate provenance
  | CmdCheckToolchain FilePath         -- ^ Check toolchain lock

-- | Application entry point.
--
-- Parses command-line arguments and dispatches to the appropriate
-- subcommand handler.
main :: IO ()
main = do
  cmd <- execParser opts
  case cmd of
    CmdSpecToIr inJson outIr -> SpecToIr.run inJson outIr
    CmdGenProvenance inIr epochJson outProv -> GenProvenance.run inIr epochJson outProv
    CmdCheckToolchain lockfile -> CheckToolchain.run lockfile
  where
    opts = info (parser <**> helper) (fullDesc <> progDesc "STUNIR Native Core (Haskell)")

-- | CLI argument parser.
--
-- Defines the available subcommands and their argument parsers.
parser :: Parser Command
parser = hsubparser
  ( command "spec-to-ir" (info pSpecToIr (progDesc "Convert Spec to IR"))
 <> command "gen-provenance" (info pGenProvenance (progDesc "Generate Provenance"))
 <> command "check-toolchain" (info pCheckToolchain (progDesc "Check Toolchain Lock"))
  )

-- | Parser for the spec-to-ir subcommand.
pSpecToIr :: Parser Command
pSpecToIr = CmdSpecToIr <$> strOption (long "in-json") <*> strOption (long "out-ir")

-- | Parser for the gen-provenance subcommand.
pGenProvenance :: Parser Command
pGenProvenance = CmdGenProvenance <$> strOption (long "in-ir") <*> strOption (long "epoch-json") <*> strOption (long "out-prov")

-- | Parser for the check-toolchain subcommand.
pCheckToolchain :: Parser Command
pCheckToolchain = CmdCheckToolchain <$> strOption (long "lockfile")

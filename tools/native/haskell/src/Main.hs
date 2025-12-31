{-# LANGUAGE OverloadedStrings #-}

module Main where

import Options.Applicative
import Data.Semigroup ((<>))
import qualified Commands.SpecToIr
import qualified Commands.GenProvenance
import qualified Commands.CheckToolchain

data Command
  = SpecToIr { inJson :: String, outIr :: String }
  | GenProvenance { inIr :: String, epochJson :: String, outProv :: String }
  | CheckToolchain { lockfile :: String }

specToIrOpts :: Parser Command
specToIrOpts = SpecToIr
  <$> strOption (long "in-json" <> metavar "FILE" <> help "Input JSON spec")
  <*> strOption (long "out-ir" <> metavar "FILE" <> help "Output IR file")

genProvOpts :: Parser Command
genProvOpts = GenProvenance
  <$> strOption (long "in-ir" <> metavar "FILE" <> help "Input IR file")
  <*> strOption (long "epoch-json" <> metavar "FILE" <> help "Epoch JSON file")
  <*> strOption (long "out-prov" <> metavar "FILE" <> help "Output Provenance file")

checkToolchainOpts :: Parser Command
checkToolchainOpts = CheckToolchain
  <$> strOption (long "lockfile" <> metavar "FILE" <> help "Toolchain lockfile")

opts :: Parser Command
opts = subparser
  ( command "spec-to-ir" (info specToIrOpts (progDesc "Convert Spec to IR"))
 <> command "gen-provenance" (info genProvOpts (progDesc "Generate Provenance"))
 <> command "check-toolchain" (info checkToolchainOpts (progDesc "Check Toolchain"))
  )

main :: IO ()
main = do
  cmd <- execParser opts'
  case cmd of
    SpecToIr i o -> Commands.SpecToIr.run i o
    GenProvenance i e o -> Commands.GenProvenance.run i e o
    CheckToolchain l -> Commands.CheckToolchain.run l
  where
    opts' = info (opts <**> helper)
      ( fullDesc
     <> progDesc "STUNIR Native Toolchain"
     <> header "stunir-native - Deterministic Build Tool" )

{-# LANGUAGE OverloadedStrings #-}
module Main (main) where
import Options.Applicative
import qualified STUNIR.SpecToIr as SpecToIr
import qualified STUNIR.GenProvenance as GenProvenance
import qualified STUNIR.CheckToolchain as CheckToolchain

data Command
  = CmdSpecToIr FilePath FilePath
  | CmdGenProvenance FilePath FilePath FilePath
  | CmdCheckToolchain FilePath

main :: IO ()
main = do
  cmd <- execParser opts
  case cmd of
    CmdSpecToIr inJson outIr -> SpecToIr.run inJson outIr
    CmdGenProvenance inIr epochJson outProv -> GenProvenance.run inIr epochJson outProv
    CmdCheckToolchain lockfile -> CheckToolchain.run lockfile
  where
    opts = info (parser <**> helper) (fullDesc <> progDesc "STUNIR Native Core (Haskell)")

parser :: Parser Command
parser = hsubparser
  ( command "spec-to-ir" (info pSpecToIr (progDesc "Convert Spec to IR"))
 <> command "gen-provenance" (info pGenProvenance (progDesc "Generate Provenance"))
 <> command "check-toolchain" (info pCheckToolchain (progDesc "Check Toolchain Lock"))
  )

pSpecToIr = CmdSpecToIr <$> strOption (long "in-json") <*> strOption (long "out-ir")
pGenProvenance = CmdGenProvenance <$> strOption (long "in-ir") <*> strOption (long "epoch-json") <*> strOption (long "out-prov")
pCheckToolchain = CmdCheckToolchain <$> strOption (long "lockfile")

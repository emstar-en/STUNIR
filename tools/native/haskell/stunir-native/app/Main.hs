{-# LANGUAGE OverloadedStrings #-}

module Main (main) where

import Options.Applicative
import qualified STUNIR.Validate as Validate
import qualified STUNIR.Verify.Pack as VerifyPack
import qualified STUNIR.Verify.Emit as VerifyEmit
import qualified STUNIR.SpecToIr as SpecToIr
import qualified STUNIR.GenProvenance as GenProvenance
import qualified STUNIR.CheckToolchain as CheckToolchain

data Command
  = CmdValidate FilePath Bool
  | CmdVerifyPack FilePath FilePath FilePath FilePath Bool (Maybe String)
  | CmdVerifyEmit FilePath FilePath
  | CmdSpecToIr FilePath FilePath
  | CmdGenProvenance FilePath FilePath FilePath
  | CmdCheckToolchain FilePath

main :: IO ()
main = do
  cmd <- execParser opts
  case cmd of
    CmdValidate ir allowTrailingLf -> Validate.run ir allowTrailingLf
    CmdVerifyPack root objectsDir packManifest rootAttestation checkCompleteness pubkeyB64 ->
      VerifyPack.run root objectsDir packManifest rootAttestation checkCompleteness pubkeyB64
    CmdVerifyEmit receipt root -> VerifyEmit.run receipt root
    CmdSpecToIr inJson outIr -> SpecToIr.run inJson outIr
    CmdGenProvenance inIr epochJson outProv -> GenProvenance.run inIr epochJson outProv
    CmdCheckToolchain lockfile -> CheckToolchain.run lockfile
  where
    opts = info (parser <**> helper)
      ( fullDesc <> progDesc "STUNIR Native Core (Haskell)" )

parser :: Parser Command
parser = hsubparser
  ( command "validate" (info pValidate (progDesc "Validate STUNIR IR"))
 <> command "verify" (info pVerify (progDesc "Verify pack/receipts"))
 <> command "spec-to-ir" (info pSpecToIr (progDesc "Convert Spec to IR"))
 <> command "gen-provenance" (info pGenProvenance (progDesc "Generate Provenance"))
 <> command "check-toolchain" (info pCheckToolchain (progDesc "Check Toolchain Lock"))
  )

pValidate :: Parser Command
pValidate = CmdValidate <$> argument str (metavar "IR_JSON") <*> switch (long "allow-trailing-lf")

pVerify :: Parser Command
pVerify = hsubparser
  ( command "pack" (info pVerifyPack (progDesc "Verify Pack"))
 <> command "emit" (info pVerifyEmit (progDesc "Verify Emit"))
  )

pVerifyPack :: Parser Command
pVerifyPack = CmdVerifyPack
  <$> strOption (long "root" <> value ".")
  <*> strOption (long "objects-dir" <> value "objects/sha256")
  <*> strOption (long "pack-manifest" <> value "pack_manifest.tsv")
  <*> strOption (long "root-attestation" <> value "root_attestation.txt")
  <*> switch (long "check-completeness")
  <*> optional (strOption (long "ed25519-pubkey-b64"))

pVerifyEmit :: Parser Command
pVerifyEmit = CmdVerifyEmit <$> argument str (metavar "RECEIPT_JSON") <*> strOption (long "root" <> value ".")

pSpecToIr :: Parser Command
pSpecToIr = CmdSpecToIr
  <$> strOption (long "in-json")
  <*> strOption (long "out-ir")

pGenProvenance :: Parser Command
pGenProvenance = CmdGenProvenance
  <$> strOption (long "in-ir")
  <*> strOption (long "epoch-json")
  <*> strOption (long "out-prov")

pCheckToolchain :: Parser Command
pCheckToolchain = CmdCheckToolchain <$> strOption (long "lockfile")

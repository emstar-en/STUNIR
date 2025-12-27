{-# LANGUAGE OverloadedStrings #-}
module Main (main) where
import Options.Applicative
<<<<<<< HEAD
import qualified STUNIR.Validate as Validate
import qualified STUNIR.Verify.Pack as VerifyPack
import qualified STUNIR.Verify.Emit as VerifyEmit
=======
>>>>>>> origin/rescue/main-pre-force
import qualified STUNIR.SpecToIr as SpecToIr
import qualified STUNIR.GenProvenance as GenProvenance
import qualified STUNIR.CheckToolchain as CheckToolchain

data Command
<<<<<<< HEAD
  = CmdValidate FilePath Bool
  | CmdVerifyPack FilePath FilePath FilePath FilePath Bool (Maybe String)
  | CmdVerifyEmit FilePath FilePath
  | CmdSpecToIr FilePath FilePath
=======
  = CmdSpecToIr FilePath FilePath
>>>>>>> origin/rescue/main-pre-force
  | CmdGenProvenance FilePath FilePath FilePath
  | CmdCheckToolchain FilePath

main :: IO ()
main = do
  cmd <- execParser opts
  case cmd of
<<<<<<< HEAD
    CmdValidate ir allowTrailingLf -> Validate.run ir allowTrailingLf
    CmdVerifyPack root objectsDir packManifest rootAttestation checkCompleteness pubkeyB64 ->
      VerifyPack.run root objectsDir packManifest rootAttestation checkCompleteness pubkeyB64
    CmdVerifyEmit receipt root -> VerifyEmit.run receipt root
=======
>>>>>>> origin/rescue/main-pre-force
    CmdSpecToIr inJson outIr -> SpecToIr.run inJson outIr
    CmdGenProvenance inIr epochJson outProv -> GenProvenance.run inIr epochJson outProv
    CmdCheckToolchain lockfile -> CheckToolchain.run lockfile
  where
<<<<<<< HEAD
    opts = info (parser <**> helper)
      ( fullDesc <> progDesc "STUNIR Native Core (Haskell)" )

parser :: Parser Command
parser = hsubparser
  ( command "validate" (info pValidate (progDesc "Validate STUNIR IR"))
 <> command "verify" (info pVerify (progDesc "Verify pack/receipts"))
 <> command "spec-to-ir" (info pSpecToIr (progDesc "Convert Spec to IR"))
=======
    opts = info (parser <**> helper) (fullDesc <> progDesc "STUNIR Native Core (Haskell)")

parser :: Parser Command
parser = hsubparser
  ( command "spec-to-ir" (info pSpecToIr (progDesc "Convert Spec to IR"))
>>>>>>> origin/rescue/main-pre-force
 <> command "gen-provenance" (info pGenProvenance (progDesc "Generate Provenance"))
 <> command "check-toolchain" (info pCheckToolchain (progDesc "Check Toolchain Lock"))
  )

<<<<<<< HEAD
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
=======
pSpecToIr = CmdSpecToIr <$> strOption (long "in-json") <*> strOption (long "out-ir")
pGenProvenance = CmdGenProvenance <$> strOption (long "in-ir") <*> strOption (long "epoch-json") <*> strOption (long "out-prov")
>>>>>>> origin/rescue/main-pre-force
pCheckToolchain = CmdCheckToolchain <$> strOption (long "lockfile")

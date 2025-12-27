{-# LANGUAGE OverloadedStrings #-}

module Main (main) where

import Options.Applicative

import qualified STUNIR.Validate as Validate
import qualified STUNIR.Verify.Pack as VerifyPack
import qualified STUNIR.Verify.Emit as VerifyEmit

data Command
  = CmdValidate FilePath Bool
  | CmdVerifyPack FilePath FilePath FilePath FilePath Bool (Maybe String)
  | CmdVerifyEmit FilePath FilePath

main :: IO ()
main = do
  cmd <- execParser opts
  case cmd of
    CmdValidate ir allowTrailingLf -> Validate.run ir allowTrailingLf
    CmdVerifyPack root objectsDir packManifest rootAttestation checkCompleteness pubkeyB64 ->
      VerifyPack.run root objectsDir packManifest rootAttestation checkCompleteness pubkeyB64
    CmdVerifyEmit receipt root -> VerifyEmit.run receipt root
  where
    opts = info (parser <**> helper)
      ( fullDesc
     <> progDesc "STUNIR native stages: validate + verify (pack/receipts)"
     <> header "stunir-native" )

parser :: Parser Command
parser = hsubparser
  ( command "validate" (info pValidate (progDesc "Validate STUNIR IR v1 + RFC8785 canonical JSON"))
 <> command "verify" (info pVerify (progDesc "Verify pack integrity / receipts"))
  )

pValidate :: Parser Command
pValidate = CmdValidate
  <$> argument str (metavar "IR_JSON")
  <*> switch (long "allow-trailing-lf" <> help "Allow a single trailing LF after canonical JSON")

pVerify :: Parser Command
pVerify = hsubparser
  ( command "pack" (info pVerifyPack (progDesc "Verify Profile-3-style pack"))
 <> command "emit" (info pVerifyEmit (progDesc "Verify emission receipt JSON"))
  )

pVerifyPack :: Parser Command
pVerifyPack = CmdVerifyPack
  <$> strOption (long "root" <> value "." <> showDefault <> help "Root directory")
  <*> strOption (long "objects-dir" <> value "objects/sha256" <> showDefault <> help "Objects dir relative to root")
  <*> strOption (long "pack-manifest" <> value "pack_manifest.tsv" <> showDefault <> help "Pack manifest path relative to root")
  <*> strOption (long "root-attestation" <> value "root_attestation.txt" <> showDefault <> help "Root attestation path relative to root")
  <*> switch (long "check-completeness" <> help "Strictly require manifest matches actual file tree")
  <*> optional (strOption (long "ed25519-pubkey-b64" <> help "Base64 Ed25519 public key (32 bytes)"))

pVerifyEmit :: Parser Command
pVerifyEmit = CmdVerifyEmit
  <$> argument str (metavar "RECEIPT_JSON")
  <*> strOption (long "root" <> value "." <> showDefault <> help "Root directory for resolving output relpaths")

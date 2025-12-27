module STUNIR.Validate
  ( run
  ) where

import qualified Data.Aeson as A
import qualified Data.ByteString as BS
import qualified Data.Text.Encoding as TE
import System.Exit (exitFailure, exitSuccess)

import STUNIR.Hash (sha256Hex)
import STUNIR.JCS (canonicalize)
import STUNIR.IR.V1 (validateIrV1)

run :: FilePath -> Bool -> IO ()
run fp allowTrailingLf = do
  bytes <- BS.readFile fp

  -- IR validation is LF-only.
  if BS.any (== 13) bytes then die "bad_digest" else pure ()

  let bytes' =
        if allowTrailingLf && not (BS.null bytes) && BS.last bytes == 10
          then BS.init bytes
          else bytes

  case TE.decodeUtf8' bytes' of
    Left _ -> die "bad_digest"
    Right _ -> pure ()

  v <- case A.eitherDecodeStrict' bytes' of
    Left _ -> die "bad_digest"
    Right x -> pure x

  canon <- case canonicalize v of
    Left e -> die e
    Right c -> pure c

  if canon /= bytes' then die "bad_digest" else pure ()

  case validateIrV1 v of
    Left e -> die e
    Right () -> pure ()

  putStrLn ("OK ir_sha256_jcs=" <> sha256Hex canon)
  exitSuccess

die :: String -> IO a
die tag = do
  putStrLn tag
  exitFailure

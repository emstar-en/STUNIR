{-# LANGUAGE OverloadedStrings #-}
module Main where

import qualified Data.ByteString.Lazy as LBS
import qualified Data.Aeson as Aeson
import qualified Data.CBOR.Write as CBORWrite
import qualified Crypto.Hash.SHA256 as SHA256
import System.Environment (getArgs)
import System.Exit (exitFailure, exitSuccess)
import System.IO (IOMode(WriteMode), hPutStr, withFile)

specToDCBOR :: FilePath -> FilePath -> IO ()
specToDCBOR specPath irPath = do
  spec <- Aeson.eitherDecodeFileStrict specPath
  case spec of
    Left err -> putStrLn ("JSON invalid: " ++ err) >> exitFailure
    Right obj -> do
      let dcbor = CBORWrite.toLazyByteString $ CBORWrite.encodeCanonical obj
      LBS.writeFile irPath dcbor
      putStrLn $ "IR: " ++ show (SHA256.hashlazy dcbor)

main :: IO ()
main = do
  args <- getArgs
  case args of
    [spec, ir] -> specToDCBOR spec ir >> exitSuccess
    _ -> putStrLn "stunir-ir <spec.json> <ir.dcbor>" >> exitFailure

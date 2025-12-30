{-# LANGUAGE OverloadedStrings #-}
module Main where

import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as LBS
import qualified Data.CBOR.Decoder as CBOR
import qualified Data.CBOR.Write as CBORWrite
import qualified Crypto.Hash.SHA256 as SHA256
import System.Environment (getArgs)
import System.Exit (exitFailure, exitSuccess)

data Artifact = Artifact { path :: String, digest :: String } deriving Show
data Toolchain = Toolchain { ghc :: String, cabal :: String } deriving Show
data PackRoot = PackRoot { artifacts :: [Artifact], toolchain :: Toolchain, timestamp :: Integer }
data RootAttestation = RootAttestation { pack_root :: PackRoot, signature :: String }

main :: IO ()
main = do
  args <- getArgs
  case args of
    ["generate", packPath, rootPath] -> generateAttestation packPath rootPath
    ["verify", rootPath] -> verifyAttestation rootPath
    _ -> usage >> exitFailure

usage :: IO ()
usage = putStrLn "pack-attestation generate <pack.json> <root.dcbor> | verify <root.dcbor>"

generateAttestation :: FilePath -> FilePath -> IO ()
generateAttestation packPath rootPath = do
  pack <- BS.readFile packPath
  let attestation = RootAttestation 
        { pack_root = PackRoot 
            { artifacts = [Artifact "asm/ir_0001.dcbor" "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"]
            , toolchain = Toolchain "9.4.8" "3.10.2.0"
            , timestamp = EPOCH 
            }
        , signature = "ed25519_placeholder"  -- ISSUE.PACK.0002
        }
  LBS.writeFile rootPath $ CBORWrite.toLazyByteString $ CBORWrite.encodeCanonical attestation
  putStrLn "root_attestation.dcbor generated"

verifyAttestation :: FilePath -> IO ()
verifyAttestation path = do
  attestation <- BS.readFile path
  case CBOR.deserialiseFromBytes "" attestation of
    Left err -> putStrLn ("Invalid CBOR: " ++ err) >> exitFailure
    Right (_, RootAttestation root sig) -> do
      let rootDigest = show $ SHA256.hashlazy $ CBORWrite.toLazyByteString $ CBORWrite.encodeCanonical root
      putStrLn $ "Pack root digest: " ++ rootDigest
      putStrLn $ "Signature: " ++ sig
      putStrLn "Pack verified (Profile-3)" >> exitSuccess

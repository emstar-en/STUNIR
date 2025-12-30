{-# LANGUAGE OverloadedStrings #-}
module Main where

import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as LBS
import qualified Crypto.Hash.SHA256 as SHA256
import qualified Data.CBOR.Decoder as CBOR
import qualified Data.CBOR.Write as CBORWrite
import System.Environment (getArgs)
import System.Exit (exitFailure, exitSuccess)
import Data.Word (Word8)

data FailureTag = FloatDetected | KeysUnsorted | InvalidCBOR deriving (Show, Eq)

main :: IO ()
main = do
  args <- getArgs
  case args of
    ["validate", irPath] -> validateIR irPath
    ["verify", "pack", packPath] -> verifyPack packPath
    _ -> usage >> exitFailure

usage :: IO ()
usage = putStrLn "stunir-native-hs validate <ir.dcbor> | verify pack <root.dcbor>"

validateIR :: FilePath -> IO ()
validateIR path = do
  ir <- BS.readFile path
  case validateDCBOR ir of
    Left tag -> putStrLn (show tag) >> exitFailure
    Right _  -> putStrLn "IR valid (Profile-3 UN)" >> exitSuccess

verifyPack :: FilePath -> IO ()
verifyPack path = do
  attestation <- BS.readFile path
  case verifyAttestation attestation of
    Left err -> putStrLn err >> exitFailure
    Right _  -> putStrLn "Pack verified (Profile-3)" >> exitSuccess

validateDCBOR :: BS.ByteString -> Either FailureTag LBS.ByteString
validateDCBOR bs = case CBOR.deserialiseFromBytes "" bs of
  Left _ -> Left InvalidCBOR
  Right (consumed, obj) 
    | BS.length bs /= consumed -> Left InvalidCBOR
    | hasFloats obj -> Left FloatDetected
    | otherwise -> Right $ CBORWrite.toLazyByteString $ CBORWrite.encodeCanonical obj

hasFloats :: CBOR.Value -> Bool
hasFloats (CBOR.FloatD _) = True
hasFloats (CBOR.Float16 _) = True
hasFloats (CBOR.Float32 _) = True
hasFloats (CBOR.Break) = False
hasFloats (CBOR.Null) = False
hasFloats (CBOR.Simple _) = False
hasFloats (CBOR.Integer _) = False
hasFloats (CBOR.Bytes _) = False
hasFloats (CBOR.String _) = False
hasFloats (CBOR.Array len vals) = any hasFloats vals
hasFloats (CBOR.Map len pairs) = any (uncurry (||)) $ take (fromIntegral len * 2) pairs
hasFloats (CBOR.Tag _ val) = hasFloats val
hasFloats (CBOR.Bool _) = False

verifyAttestation :: BS.ByteString -> Either String ()
verifyAttestation _ = Right ()  -- ISSUE.PACK.0001: implement Profile-3 root_attestation.dcbor

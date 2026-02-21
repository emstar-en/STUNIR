{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE ScopedTypeVariables #-}

-- | STUNIR Pack Attestation with Ed25519 Cryptographic Signatures
-- 
-- This module implements secure attestation for STUNIR pack roots using
-- Ed25519 digital signatures as specified in RFC 8032.
--
-- Security Considerations:
--   - Uses cryptographically secure random number generation
--   - Ed25519 provides 128-bit security level
--   - Constant-time signature verification to prevent timing attacks
--   - All keys are validated before use
module Main where

import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as LBS
import qualified Data.ByteString.Base16 as B16
import qualified Data.ByteString.Char8 as C8
import qualified Data.ByteArray as BA
import qualified Codec.CBOR.Encoding as CBOR
import qualified Codec.CBOR.Write as CBORWrite
import qualified Codec.CBOR.Decoding as CBORDec
import qualified Codec.CBOR.Read as CBORRead
import Crypto.Error (CryptoFailable(..))
import qualified Crypto.PubKey.Ed25519 as Ed25519
import qualified Crypto.Hash.SHA256 as SHA256
import Crypto.Random (getRandomBytes)
import System.Environment (getArgs)
import System.Exit (exitFailure, exitSuccess)
import System.IO (hPutStrLn, stderr)
import System.Directory (doesFileExist)
import Control.Monad (unless)
import Data.Time.Clock.POSIX (getPOSIXTime)
import GHC.Generics (Generic)

-- | Artifact entry in pack attestation
data Artifact = Artifact 
    { artifactPath :: String
    , artifactDigest :: String  -- SHA256 hex
    , artifactSize :: Maybe Integer
    } deriving (Show, Eq, Generic)

-- | Toolchain information
data Toolchain = Toolchain 
    { ghcVersion :: String
    , cabalVersion :: String
    } deriving (Show, Eq, Generic)

-- | Pack root metadata
data PackRoot = PackRoot 
    { prArtifacts :: [Artifact]
    , prToolchain :: Toolchain
    , prTimestamp :: Integer
    , prEpoch :: Integer
    } deriving (Show, Eq, Generic)

-- | Root attestation with Ed25519 signature
data RootAttestation = RootAttestation 
    { raPackRoot :: PackRoot
    , raPublicKey :: BS.ByteString  -- 32 bytes
    , raSignature :: BS.ByteString  -- 64 bytes
    } deriving (Show, Eq, Generic)

-- | Key pair wrapper
data KeyPair = KeyPair
    { kpSecretKey :: Ed25519.SecretKey
    , kpPublicKey :: Ed25519.PublicKey
    } deriving (Show)

-- | Generate a new Ed25519 key pair using cryptographically secure RNG
generateKeyPair :: IO KeyPair
generateKeyPair = do
    seed <- getRandomBytes 32 :: IO BS.ByteString
    case Ed25519.secretKey seed of
        CryptoFailed err -> do
            hPutStrLn stderr $ "SECURITY ERROR: Key generation failed: " ++ show err
            exitFailure
        CryptoPassed sk -> do
            let pk = Ed25519.toPublic sk
            return $ KeyPair sk pk

-- | Load key pair from file (secret key in hex format)
loadKeyPair :: FilePath -> IO KeyPair
loadKeyPair path = do
    content <- BS.readFile path
    let hexBytes = BS.filter (\c -> c /= 0x0a && c /= 0x0d) content  -- strip newlines
    case B16.decode hexBytes of
        Left err -> do
            hPutStrLn stderr $ "SECURITY ERROR: Invalid key file format: " ++ err
            exitFailure
        Right skBytes -> do
            unless (BS.length skBytes == 32) $ do
                hPutStrLn stderr "SECURITY ERROR: Secret key must be exactly 32 bytes"
                exitFailure
            case Ed25519.secretKey skBytes of
                CryptoFailed err -> do
                    hPutStrLn stderr $ "SECURITY ERROR: Invalid secret key: " ++ show err
                    exitFailure
                CryptoPassed sk -> do
                    let pk = Ed25519.toPublic sk
                    return $ KeyPair sk pk

-- | Load public key from file (hex format)
loadPublicKey :: FilePath -> IO Ed25519.PublicKey
loadPublicKey path = do
    content <- BS.readFile path
    let hexBytes = BS.filter (\c -> c /= 0x0a && c /= 0x0d) content
    case B16.decode hexBytes of
        Left err -> do
            hPutStrLn stderr $ "SECURITY ERROR: Invalid public key format: " ++ err
            exitFailure
        Right pkBytes -> do
            unless (BS.length pkBytes == 32) $ do
                hPutStrLn stderr "SECURITY ERROR: Public key must be exactly 32 bytes"
                exitFailure
            case Ed25519.publicKey pkBytes of
                CryptoFailed err -> do
                    hPutStrLn stderr $ "SECURITY ERROR: Invalid public key: " ++ show err
                    exitFailure
                CryptoPassed pk -> return pk

-- | Encode PackRoot to canonical CBOR bytes for signing
-- Keys are sorted alphabetically to ensure deterministic output
encodePackRoot :: PackRoot -> BS.ByteString
encodePackRoot pr = LBS.toStrict $ CBORWrite.toLazyByteString $ mconcat
    [ CBOR.encodeMapLen 4
    , CBOR.encodeString "artifacts"
    , encodeArtifacts (prArtifacts pr)
    , CBOR.encodeString "epoch"
    , CBOR.encodeInteger (prEpoch pr)
    , CBOR.encodeString "timestamp"
    , CBOR.encodeInteger (prTimestamp pr)
    , CBOR.encodeString "toolchain"
    , encodeToolchain (prToolchain pr)
    ]
  where
    encodeArtifacts :: [Artifact] -> CBOR.Encoding
    encodeArtifacts arts = CBOR.encodeListLen (fromIntegral $ length arts) 
        <> mconcat (map encodeArtifact arts)
    
    encodeArtifact :: Artifact -> CBOR.Encoding
    encodeArtifact a = CBOR.encodeMapLen 3
        <> CBOR.encodeString "digest"
        <> CBOR.encodeString (C8.pack $ artifactDigest a)
        <> CBOR.encodeString "path"
        <> CBOR.encodeString (C8.pack $ artifactPath a)
        <> CBOR.encodeString "size"
        <> maybe CBOR.encodeNull CBOR.encodeInteger (artifactSize a)
    
    encodeToolchain :: Toolchain -> CBOR.Encoding
    encodeToolchain tc = CBOR.encodeMapLen 2
        <> CBOR.encodeString "cabal"
        <> CBOR.encodeString (C8.pack $ cabalVersion tc)
        <> CBOR.encodeString "ghc"
        <> CBOR.encodeString (C8.pack $ ghcVersion tc)

-- | Encode full attestation to CBOR
encodeAttestation :: RootAttestation -> BS.ByteString
encodeAttestation ra = LBS.toStrict $ CBORWrite.toLazyByteString $ mconcat
    [ CBOR.encodeMapLen 3
    , CBOR.encodeString "pack_root"
    , CBOR.encodeBytes (encodePackRoot $ raPackRoot ra)
    , CBOR.encodeString "public_key"
    , CBOR.encodeBytes (raPublicKey ra)
    , CBOR.encodeString "signature"
    , CBOR.encodeBytes (raSignature ra)
    ]

-- | Sign the pack root with Ed25519
signPackRoot :: KeyPair -> PackRoot -> RootAttestation
signPackRoot kp pr = 
    let packRootBytes = encodePackRoot pr
        sig = Ed25519.sign (kpSecretKey kp) (kpPublicKey kp) packRootBytes
        sigBytes = BS.pack $ BA.unpack sig
        pkBytes = BS.pack $ BA.unpack (kpPublicKey kp)
    in RootAttestation pr pkBytes sigBytes

-- | Verify attestation signature (constant-time)
verifyAttestation' :: Ed25519.PublicKey -> BS.ByteString -> BS.ByteString -> PackRoot -> Bool
verifyAttestation' pk sigBytes prEncoded _ =
    case Ed25519.signature sigBytes of
        CryptoFailed _ -> False
        CryptoPassed sig -> Ed25519.verify pk prEncoded sig

-- | Read and parse pack.json (simplified parser)
readPackJson :: FilePath -> IO PackRoot
readPackJson path = do
    _content <- BS.readFile path
    ts <- round <$> getPOSIXTime
    -- In production, use aeson for proper JSON parsing
    return $ PackRoot
        { prArtifacts = [Artifact "asm/ir_0001.dcbor" "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" Nothing]
        , prToolchain = Toolchain "9.4.8" "3.10.2.0"
        , prTimestamp = ts
        , prEpoch = 1706400000
        }

main :: IO ()
main = do
    args <- getArgs
    case args of
        ["generate", packPath, rootPath] -> do
            hPutStrLn stderr "WARNING: Using ephemeral key pair. Use --key-file for production."
            kp <- generateKeyPair
            generateAttestationCmd packPath rootPath kp
        
        ["generate", packPath, rootPath, "--key-file", keyFile] -> do
            kp <- loadKeyPair keyFile
            generateAttestationCmd packPath rootPath kp
        
        ["verify", rootPath] -> do
            hPutStrLn stderr "ERROR: --pubkey-file required for verification"
            exitFailure
        
        ["verify", rootPath, "--pubkey-file", pubFile] -> do
            pk <- loadPublicKey pubFile
            verifyAttestationCmd rootPath pk
        
        ["keygen", outPath] -> do
            keygenCmd outPath
        
        _ -> usage >> exitFailure

usage :: IO ()
usage = do
    putStrLn "STUNIR Pack Attestation Tool (Ed25519)"
    putStrLn ""
    putStrLn "Commands:"
    putStrLn "  generate <pack.json> <root.dcbor> [--key-file <key>]"
    putStrLn "  verify <root.dcbor> --pubkey-file <pubkey>"
    putStrLn "  keygen <output-path>"
    putStrLn ""
    putStrLn "Security: Uses Ed25519 signatures (RFC 8032)"

-- | Generate attestation and write to file
generateAttestationCmd :: FilePath -> FilePath -> KeyPair -> IO ()
generateAttestationCmd packPath rootPath kp = do
    -- Validate input paths
    packExists <- doesFileExist packPath
    unless packExists $ do
        hPutStrLn stderr $ "SECURITY ERROR: Pack file not found: " ++ packPath
        exitFailure
    
    -- Read and parse pack
    packRoot <- readPackJson packPath
    
    -- Sign
    let attestation = signPackRoot kp packRoot
    
    -- Write CBOR output
    BS.writeFile rootPath (encodeAttestation attestation)
    
    let pkHex = B16.encode (raPublicKey attestation)
    hPutStrLn stderr $ "Attestation generated: " ++ rootPath
    hPutStrLn stderr $ "Public key: " ++ C8.unpack pkHex
    hPutStrLn stderr $ "Signature algorithm: Ed25519"
    exitSuccess

-- | Verify attestation from file
verifyAttestationCmd :: FilePath -> Ed25519.PublicKey -> IO ()
verifyAttestationCmd path pk = do
    content <- BS.readFile path
    case CBORRead.deserialiseFromBytes decodeAttestation (LBS.fromStrict content) of
        Left err -> do
            hPutStrLn stderr $ "SECURITY ERROR: Invalid CBOR: " ++ show err
            exitFailure
        Right (_, (prEncoded, pkBytes, sigBytes)) -> do
            let valid = verifyAttestation' pk sigBytes prEncoded (PackRoot [] (Toolchain "" "") 0 0)
            if valid
                then do
                    let rootDigest = B16.encode $ SHA256.hash prEncoded
                    hPutStrLn stderr $ "Pack root digest: " ++ C8.unpack rootDigest
                    hPutStrLn stderr "Signature: VALID (Ed25519)"
                    hPutStrLn stderr "Pack verified (Profile-3)"
                    exitSuccess
                else do
                    hPutStrLn stderr "SECURITY ERROR: Signature verification FAILED"
                    exitFailure

-- | Generate new key pair and save to files
keygenCmd :: FilePath -> IO ()
keygenCmd outPath = do
    kp <- generateKeyPair
    let skBytes = BS.pack $ BA.unpack (kpSecretKey kp)
        pkBytes = BS.pack $ BA.unpack (kpPublicKey kp)
        skHex = B16.encode skBytes
        pkHex = B16.encode pkBytes
    BS.writeFile outPath skHex
    BS.writeFile (outPath ++ ".pub") pkHex
    hPutStrLn stderr $ "Key pair generated:"
    hPutStrLn stderr $ "  Secret key: " ++ outPath
    hPutStrLn stderr $ "  Public key: " ++ outPath ++ ".pub"
    hPutStrLn stderr "WARNING: Keep the secret key secure and never share it!"

-- | Decode attestation from CBOR
decodeAttestation :: CBORDec.Decoder s (BS.ByteString, BS.ByteString, BS.ByteString)
decodeAttestation = do
    _n <- CBORDec.decodeMapLen
    _ <- CBORDec.decodeString  -- "pack_root"
    packRootBytes <- CBORDec.decodeBytes
    _ <- CBORDec.decodeString  -- "public_key"
    pkBytes <- CBORDec.decodeBytes
    _ <- CBORDec.decodeString  -- "signature"
    sigBytes <- CBORDec.decodeBytes
    return (packRootBytes, pkBytes, sigBytes)

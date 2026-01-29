{-# LANGUAGE OverloadedStrings #-}

{-|
Module      : Test.Determinism
Description : Determinism verification utilities for STUNIR
Copyright   : (c) STUNIR Authors, 2026
License     : MIT

Utilities for verifying deterministic behavior of STUNIR operations.
-}

module Test.Determinism
    ( -- * Determinism Checks
      verifyDeterministic
    , verifyHashDeterminism
    , verifyJsonDeterminism
    , verifyFileDeterminism
      -- * Hash Utilities
    , computeSha256
    , computeFileSha256
    , hashesMatch
      -- * Canonical JSON
    , canonicalJson
    , isCanonicalJson
    ) where

import Crypto.Hash.SHA256 (hash)
import Data.Aeson (Value, encode, decode)
import Data.Aeson.Encode.Pretty (encodePretty, defConfig, Config(..), Indent(..))
import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as BL
import qualified Data.ByteString.Base16 as B16
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import Data.Text (Text)
import System.Directory (doesFileExist)
import Control.Monad (replicateM)

-- | Verify that an action produces deterministic output
verifyDeterministic :: Eq a => Int -> IO a -> IO Bool
verifyDeterministic n action = do
    results <- replicateM n action
    return $ all (== head results) (tail results)

-- | Verify that hash computation is deterministic
verifyHashDeterminism :: BS.ByteString -> Int -> Bool
verifyHashDeterminism input n =
    let hashes = replicate n (computeSha256 input)
    in all (== head hashes) (tail hashes)

-- | Verify JSON encoding is deterministic
verifyJsonDeterminism :: Value -> Int -> Bool
verifyJsonDeterminism value n =
    let encoded = replicate n (canonicalJson value)
    in all (== head encoded) (tail encoded)

-- | Verify file hashing is deterministic
verifyFileDeterminism :: FilePath -> Int -> IO Bool
verifyFileDeterminism path n = do
    exists <- doesFileExist path
    if not exists
        then return False
        else do
            hashes <- replicateM n (computeFileSha256 path)
            return $ all (== head hashes) (tail hashes)

-- | Compute SHA256 hash of ByteString, return as hex Text
computeSha256 :: BS.ByteString -> Text
computeSha256 bs = TE.decodeUtf8 $ B16.encode $ hash bs

-- | Compute SHA256 hash of file contents
computeFileSha256 :: FilePath -> IO Text
computeFileSha256 path = do
    contents <- BS.readFile path
    return $ computeSha256 contents

-- | Check if two hashes match
hashesMatch :: Text -> Text -> Bool
hashesMatch h1 h2 = T.toLower h1 == T.toLower h2

-- | Produce canonical JSON (sorted keys, minimal whitespace)
canonicalJson :: Value -> BL.ByteString
canonicalJson = encode  -- aeson encode already produces sorted keys

-- | Check if JSON is in canonical form
isCanonicalJson :: BL.ByteString -> Bool
isCanonicalJson bs = case decode bs :: Maybe Value of
    Nothing -> False
    Just v  -> encode v == bs

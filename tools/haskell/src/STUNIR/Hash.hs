{-# LANGUAGE OverloadedStrings #-}

{-|
Module      : STUNIR.Hash
Description : Hashing utilities for STUNIR
Copyright   : (c) STUNIR Team, 2026
License     : MIT

Deterministic hashing for confluence verification.
-}

module STUNIR.Hash
  ( sha256Bytes
  , sha256JSON
  , canonicalizeJSON
  ) where

import Crypto.Hash (Digest, SHA256, hash)
import Data.Aeson (Value, encode)
import Data.ByteString (ByteString)
import qualified Data.ByteString.Char8 as BS
import qualified Data.ByteString.Lazy as BL
import Data.Text (Text)
import qualified Data.Text as T

-- | Compute SHA-256 hash of bytes
sha256Bytes :: ByteString -> Text
sha256Bytes bs =
  let digest :: Digest SHA256
      digest = hash bs
  in T.pack (show digest)

-- | Canonicalize JSON to deterministic format
canonicalizeJSON :: Value -> ByteString
canonicalizeJSON v = BL.toStrict (encode v) <> "\n"

-- | Compute SHA-256 of JSON value
sha256JSON :: Value -> Text
sha256JSON = sha256Bytes . canonicalizeJSON

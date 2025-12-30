{-# LANGUAGE OverloadedStrings #-}
module Stunir.Canonical (
    canonicalEncode,
    hashCanonical
) where

import Data.Aeson (Value(..), ToJSON(..), encode)
import qualified Data.Aeson.KeyMap as KM
import qualified Data.Aeson.Key as K
import qualified Data.ByteString.Lazy as BL
import qualified Data.ByteString as BS
import qualified Data.Map.Strict as Map
import qualified Data.Vector as V
import Crypto.Hash.SHA256 (hash)

-- | Canonical JSON Encoder
-- Sorts object keys.
-- No whitespace (default aeson encode is compact).
canonicalEncode :: ToJSON a => a -> BL.ByteString
canonicalEncode v = encode (toCanonical (toJSON v))

-- | Convert Value to a form that encodes with sorted keys (Map)
toCanonical :: Value -> Value
toCanonical (Object m) = 
    let 
        -- Convert KeyMap to Map (which sorts by key)
        -- KM.toMap returns Map Key Value. We need Map Text Value for JSON encoding to work as expected?
        -- Actually Aeson encodes Map Key Value just fine.
        sortedMap = Map.map toCanonical (KM.toMap m)
    in toJSON sortedMap
toCanonical (Array v) = Array (V.map toCanonical v)
toCanonical x = x

-- | Compute SHA256 of canonical JSON bytes
hashCanonical :: ToJSON a => a -> BS.ByteString
hashCanonical = hash . BL.toStrict . canonicalEncode

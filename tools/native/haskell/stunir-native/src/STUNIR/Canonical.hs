{-# LANGUAGE OverloadedStrings #-}
module Stunir.Canonical (
    canonicalEncode,
    hashCanonical
) where

import Data.Aeson (Value(..), ToJSON(..), encode)
import qualified Data.Aeson.KeyMap as KM
import qualified Data.ByteString.Lazy as BL
import qualified Data.ByteString as BS
import qualified Data.Map.Strict as Map
import qualified Data.Vector as V
import Crypto.Hash.SHA256 (hash)

canonicalEncode :: ToJSON a => a -> BL.ByteString
canonicalEncode v = encode (toCanonical (toJSON v))

toCanonical :: Value -> Value
toCanonical (Object m) = 
    let sortedMap = Map.map toCanonical (KM.toMap m)
    in toJSON sortedMap
toCanonical (Array v) = Array (V.map toCanonical v)
toCanonical x = x

hashCanonical :: ToJSON a => a -> BS.ByteString
hashCanonical = hash . BL.toStrict . canonicalEncode

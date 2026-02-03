{-# LANGUAGE OverloadedStrings #-}
module StunirPack where

import Data.Aeson
import Data.ByteString (ByteString)
import qualified Data.ByteString.Base64.URL as B64
import Crypto.Hash.SHA256

data Pack = Pack {
    rootHash :: ByteString,
    artifacts :: [Artifact],
    receipts :: [Receipt]
} deriving (Generic, Show)

data Artifact = Artifact {
    path :: String,
    digest :: ByteString
} deriving (Generic, Show)

data Receipt = Receipt {
    schema :: String,
    verifier :: String,
    signature :: ByteString
} deriving (Generic, Show)

instance FromJSON Pack
instance ToJSON Pack
instance FromJSON Artifact
instance ToJSON Artifact  
instance FromJSON Receipt
instance ToJSON Receipt

verifyPack :: Pack -> IO Bool
verifyPack pack = do
    let expected = rootHash pack
    let actual = computeMerkleRoot (artifacts pack)
    pure $ expected == actual

computeMerkleRoot :: [Artifact] -> ByteString
computeMerkleRoot arts = hash $ mconcat [ B64.encode (digest a) | a <- arts ]

-- Profile-3 contract integration
profile3Contract :: Value
profile3Contract = object [
    "version" .= ("3.0" :: String),
    "rules" .= array [ "integer_only", "sorted_keys", "no_floats" ]
]
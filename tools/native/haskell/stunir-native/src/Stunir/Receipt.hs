{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}
module Stunir.Receipt where

import GHC.Generics
import Data.Aeson
import Data.Text (Text)
import Data.Map.Strict (Map)
import qualified Data.ByteString.Base16 as Hex
import qualified Data.Text.Encoding as TE

import Stunir.Canonical (hashCanonical)

data ToolInfo = ToolInfo {
    tool_name :: Text,
    tool_path :: Text,
    tool_sha256 :: Text,
    tool_version :: Text
} deriving (Show, Eq, Generic)

instance ToJSON ToolInfo where
    toJSON = genericToJSON defaultOptions

data Receipt = Receipt {
    receipt_schema :: Text,
    receipt_target :: Text,
    receipt_status :: Text,
    receipt_build_epoch :: Integer,
    receipt_epoch_json :: Text,
    receipt_inputs :: Map Text Text,
    receipt_tool :: ToolInfo,
    receipt_argv :: [Text]
} deriving (Show, Eq, Generic)

instance ToJSON Receipt where
    toJSON = genericToJSON defaultOptions

-- | Generate the Core ID (SHA256 of canonical JSON)
computeCoreId :: Receipt -> Text
computeCoreId r = 
    let h = hashCanonical r
    in TE.decodeUtf8 (Hex.encode h)

{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

module IR.Provenance where

import Data.Aeson (FromJSON, ToJSON, Value)
import Data.Text (Text)
import GHC.Generics (Generic)

data Provenance = Provenance
  { epoch   :: Value
  , ir_hash :: Text
  , schema  :: Text
  , status  :: Text
  } deriving (Show, Eq, Generic)

instance FromJSON Provenance
instance ToJSON Provenance

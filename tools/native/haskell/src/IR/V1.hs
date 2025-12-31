{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

module IR.V1 where

import Data.Aeson (FromJSON, ToJSON)
import Data.Text (Text)
import GHC.Generics (Generic)

-- | Represents a single instruction in the IR.
-- Matches Rust's IrInstruction struct.
data IrInstruction = IrInstruction
  { op   :: Text
  , args :: [Text]
  } deriving (Show, Eq, Generic)

instance FromJSON IrInstruction
instance ToJSON IrInstruction

-- | Represents a function definition in the IR.
-- Matches Rust's IrFunction struct.
data IrFunction = IrFunction
  { name :: Text
  , body :: [IrInstruction]
  } deriving (Show, Eq, Generic)

instance FromJSON IrFunction
instance ToJSON IrFunction

-- | The top-level IR structure.
-- Updated 'functions' to be a list of IrFunction instead of [Text].
data IrV1 = IrV1
  { version   :: Text
  , functions :: [IrFunction]
  } deriving (Show, Eq, Generic)

instance FromJSON IrV1
instance ToJSON IrV1

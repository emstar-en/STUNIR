{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}
module Stunir.IR where

import GHC.Generics
import Data.Aeson
import Data.Text (Text)
import Stunir.Spec (SpecModule)

data IRSource = IRSource {
    irs_spec_sha256 :: Text,
    irs_spec_path :: Text
} deriving (Show, Eq, Generic)

instance ToJSON IRSource where
    toJSON (IRSource s p) = object ["spec_sha256" .= s, "spec_path" .= p]

data IR = IR {
    ir_version :: Text,
    ir_module_name :: Text,
    ir_types :: [Text], -- Placeholder
    ir_functions :: [Text], -- Placeholder
    ir_spec_sha256 :: Text,
    ir_source :: IRSource,
    ir_source_modules :: [SpecModule]
} deriving (Show, Eq, Generic)

instance ToJSON IR where
    toJSON ir = object [
        "ir_version" .= ir_version ir,
        "module_name" .= ir_module_name ir,
        "types" .= ir_types ir,
        "functions" .= ir_functions ir,
        "spec_sha256" .= ir_spec_sha256 ir,
        "source" .= ir_source ir,
        "source_modules" .= ir_source_modules ir
        ]

{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}
module Stunir.Provenance where

import GHC.Generics
import Data.Aeson
import Data.Text (Text)
import qualified Data.Text as T
import Stunir.Spec (Spec(..), SpecModule(..))

data Provenance = Provenance {
    prov_epoch :: Integer,
    prov_spec_sha256 :: Text,
    prov_modules :: [Text] -- List of module names
} deriving (Show, Eq, Generic)

instance ToJSON Provenance where
    toJSON (Provenance e s m) = object [
        "epoch" .= e,
        "spec_sha256" .= s,
        "modules" .= m
        ]

-- | Generate C Header content
generateCHeader :: Provenance -> Text
generateCHeader p = T.unlines [
    "#ifndef STUNIR_PROVENANCE_H",
    "#define STUNIR_PROVENANCE_H",
    "",
    "#define STUNIR_EPOCH " <> T.pack (show (prov_epoch p)),
    "#define STUNIR_SPEC_SHA256 \"" <> prov_spec_sha256 p <> "\"",
    "",
    "#endif // STUNIR_PROVENANCE_H"
    ]

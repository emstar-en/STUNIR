{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}
module Stunir.Spec where

import GHC.Generics
import Data.Aeson
import Data.Text (Text)

data SpecModule = SpecModule {
    sm_name :: Text,
    sm_code :: Text,
    sm_lang :: Text
} deriving (Show, Eq, Generic)

instance ToJSON SpecModule where
    toJSON (SpecModule n c l) = object ["name" .= n, "code" .= c, "lang" .= l]
instance FromJSON SpecModule where
    parseJSON = withObject "SpecModule" $ \v -> SpecModule
        <$> v .: "name"
        <*> v .: "code"
        <*> v .: "lang"

data Spec = Spec {
    sp_kind :: Text,
    sp_modules :: [SpecModule]
} deriving (Show, Eq, Generic)

instance ToJSON Spec where
    toJSON (Spec k m) = object ["kind" .= k, "modules" .= m]
instance FromJSON Spec where
    parseJSON = withObject "Spec" $ \v -> Spec
        <$> v .: "kind"
        <*> v .: "modules"

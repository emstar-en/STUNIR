{-# LANGUAGE DeriveGeneric, OverloadedStrings #-}
module STUNIR.Spec where
import Data.Aeson
import Data.Text (Text)
import GHC.Generics
data SpecModule = SpecModule { name :: Text, code :: Text } deriving (Show, Generic)
instance FromJSON SpecModule
instance ToJSON SpecModule
data Spec = Spec { kind :: Text, modules :: [SpecModule] } deriving (Show, Generic)
instance FromJSON Spec
instance ToJSON Spec

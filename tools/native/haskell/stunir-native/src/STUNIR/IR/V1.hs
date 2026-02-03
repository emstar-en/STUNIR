{-# LANGUAGE DeriveGeneric, OverloadedStrings #-}
module STUNIR.IR.V1 where
import Data.Aeson
import Data.Text (Text)
import GHC.Generics
import qualified STUNIR.Spec as S
data IrMetadata = IrMetadata { kind :: Text, modules :: [S.SpecModule] } deriving (Show, Generic)
instance FromJSON IrMetadata
instance ToJSON IrMetadata
data IrV1 = IrV1 { ir_kind :: Text, generator :: Text, ir_version :: Text, module_name :: Text, functions :: [Text], modules :: [Text], metadata :: IrMetadata } deriving (Show, Generic)
instance ToJSON IrV1 where
  toJSON (IrV1 k g v m fs ms meta) = object ["kind" .= k, "generator" .= g, "ir_version" .= v, "module_name" .= m, "functions" .= fs, "modules" .= ms, "metadata" .= meta]

{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

-- | STUNIR Semantic IR Node Structures
--   Base node types and type references

module STUNIR.SemanticIR.Nodes
    ( IRNodeBase(..)
    , TypeKind(..)
    , TypeReference(..)
    , isValidNodeID
    , isValidHash
    ) where

import Data.Aeson
import qualified Data.Text as T
import qualified Data.HashMap.Strict as HM
import GHC.Generics

import STUNIR.SemanticIR.Types

-- Type kind
data TypeKind
    = TKPrimitive
    | TKArray
    | TKPointer
    | TKStruct
    | TKFunction
    | TKRef
    deriving (Show, Eq, Generic)

instance ToJSON TypeKind
instance FromJSON TypeKind

-- Type reference
data TypeReference
    = PrimitiveType { primType :: IRPrimitiveType }
    | TypeRef { typeName :: IRName, typeBinding :: Maybe NodeID }
    deriving (Show, Eq, Generic)

instance ToJSON TypeReference where
    toJSON (PrimitiveType pt) = object
        [ "kind" .= ("primitive_type" :: T.Text)
        , "primitive" .= pt
        ]
    toJSON (TypeRef name binding) = object $
        [ "kind" .= ("type_ref" :: T.Text)
        , "name" .= name
        ] ++ maybe [] (\b -> ["binding" .= b]) binding

instance FromJSON TypeReference

-- Base IR node
data IRNodeBase = IRNodeBase
    { nodeID     :: NodeID
    , nodeKind   :: IRNodeKind
    , nodeLocation :: Maybe SourceLocation
    , nodeType   :: Maybe TypeReference
    , nodeAttrs  :: HM.HashMap T.Text Value
    , nodeHash   :: Maybe IRHash
    } deriving (Show, Eq, Generic)

instance ToJSON IRNodeBase where
    toJSON node = object $
        [ "node_id" .= nodeID node
        , "kind" .= nodeKind node
        ] ++
        maybe [] (\l -> ["location" .= l]) (nodeLocation node) ++
        maybe [] (\t -> ["type" .= t]) (nodeType node) ++
        (if HM.null (nodeAttrs node) then [] else ["attributes" .= nodeAttrs node]) ++
        maybe [] (\h -> ["hash" .= h]) (nodeHash node)

instance FromJSON IRNodeBase

-- Validation functions
isValidNodeID :: NodeID -> Bool
isValidNodeID nid =
    let s = T.unpack nid
    in length s > 2 && take 2 s == "n_"

isValidHash :: IRHash -> Bool
isValidHash h =
    let s = T.unpack h
    in length s == 71 && take 7 s == "sha256:"

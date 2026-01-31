{-# LANGUAGE DeriveGeneric #-}

-- | STUNIR Semantic IR Declaration Nodes

module STUNIR.SemanticIR.Declarations where

import GHC.Generics
import Data.Aeson

import STUNIR.SemanticIR.Types
import STUNIR.SemanticIR.Nodes

-- Declaration node (simplified)
data DeclarationNode = DeclarationNode
    { declBase :: IRNodeBase
    , declName :: IRName
    , declVisibility :: VisibilityKind
    } deriving (Show, Eq, Generic)

instance ToJSON DeclarationNode
instance FromJSON DeclarationNode

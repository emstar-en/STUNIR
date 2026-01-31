{-# LANGUAGE DeriveGeneric #-}

-- | STUNIR Semantic IR Expression Nodes

module STUNIR.SemanticIR.Expressions where

import GHC.Generics
import Data.Aeson

import STUNIR.SemanticIR.Types
import STUNIR.SemanticIR.Nodes

-- Expression node (simplified)
data ExpressionNode = ExpressionNode
    { exprBase :: IRNodeBase
    , exprType :: TypeReference
    } deriving (Show, Eq, Generic)

instance ToJSON ExpressionNode
instance FromJSON ExpressionNode

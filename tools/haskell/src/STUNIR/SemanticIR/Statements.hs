{-# LANGUAGE DeriveGeneric #-}

-- | STUNIR Semantic IR Statement Nodes

module STUNIR.SemanticIR.Statements where

import GHC.Generics
import Data.Aeson

import STUNIR.SemanticIR.Types
import STUNIR.SemanticIR.Nodes

-- Statement node (simplified)
data StatementNode = StatementNode
    { stmtBase :: IRNodeBase
    } deriving (Show, Eq, Generic)

instance ToJSON StatementNode
instance FromJSON StatementNode

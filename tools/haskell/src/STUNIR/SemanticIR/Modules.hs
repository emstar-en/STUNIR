{-# LANGUAGE DeriveGeneric #-}

-- | STUNIR Semantic IR Module Structures

module STUNIR.SemanticIR.Modules where

import GHC.Generics
import Data.Aeson
import qualified Data.Text as T

import STUNIR.SemanticIR.Types
import STUNIR.SemanticIR.Nodes

-- Module (simplified)
data IRModule = IRModule
    { modBase :: IRNodeBase
    , modName :: IRName
    , modImports :: [T.Text]
    , modExports :: [IRName]
    , modDecls :: [NodeID]
    } deriving (Show, Eq, Generic)

instance ToJSON IRModule
instance FromJSON IRModule

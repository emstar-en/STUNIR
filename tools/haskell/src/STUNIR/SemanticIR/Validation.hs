{-# LANGUAGE DeriveGeneric #-}

-- | STUNIR Semantic IR Validation

module STUNIR.SemanticIR.Validation
    ( ValidationStatus(..)
    , ValidationResult(..)
    , validateNodeID
    , validateHash
    , makeValid
    , makeInvalid
    ) where

import GHC.Generics
import Data.Aeson
import qualified Data.Text as T

import STUNIR.SemanticIR.Types
import STUNIR.SemanticIR.Nodes

-- Validation status
data ValidationStatus
    = Valid
    | Invalid
    | Warning
    deriving (Show, Eq, Generic)

instance ToJSON ValidationStatus
instance FromJSON ValidationStatus

-- Validation result
data ValidationResult = ValidationResult
    { valStatus  :: ValidationStatus
    , valMessage :: T.Text
    } deriving (Show, Eq, Generic)

instance ToJSON ValidationResult
instance FromJSON ValidationResult

-- Validation functions
makeValid :: ValidationResult
makeValid = ValidationResult Valid T.empty

makeInvalid :: T.Text -> ValidationResult
makeInvalid msg = ValidationResult Invalid msg

validateNodeID :: NodeID -> ValidationResult
validateNodeID nid
    | isValidNodeID nid = makeValid
    | otherwise = makeInvalid $ T.pack $ "Invalid node ID: " ++ T.unpack nid

validateHash :: IRHash -> ValidationResult
validateHash h
    | isValidHash h = makeValid
    | otherwise = makeInvalid $ T.pack $ "Invalid hash: " ++ T.unpack h

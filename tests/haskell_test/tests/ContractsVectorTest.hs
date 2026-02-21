{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{-|
Module      : ContractsVectorTest
Description : Contract test vector validation
Copyright   : (c) STUNIR Authors, 2026
License     : MIT

Tests for validating contract test vectors, matching Python
test_vectors/contracts/ functionality.
-}

module ContractsVectorTest (suite) where

import Test.Harness
import Test.Utils
import Test.Vectors

import Data.Aeson
import Data.Aeson.Key (Key)
import qualified Data.Aeson.Key as Key
import qualified Data.Aeson.KeyMap as KM
import Data.Text (Text)
import qualified Data.Text as T

-- | Contracts test vector suite
suite :: TestSuite
suite = testSuite "Contracts Vectors"
    [ testProfile2SchemaCompliance
    , testInvalidContractDetection
    , testContractValidationDeterminism
    , testMultiStageContract
    ]

-- | Test Profile 2 contract schema compliance (tv_contracts_001)
testProfile2SchemaCompliance :: TestCase
testProfile2SchemaCompliance = testCase "Profile 2 Schema Compliance"
    "Verify Profile 2 contract schema compliance" $ do
    let contract = object
            [ "id" .= ("contract_p2_001" :: Text)
            , "profile" .= ("profile2" :: Text)
            , "schema" .= ("stunir.contract.profile2.v1" :: Text)
            , "epoch" .= (1735500000 :: Int)
            , "stages" .= (["build"] :: [Text])
            ]
        expectedHash = "1b7e1324b440813a98753ae92d98327aa53e99d920ac46a5827d7a2b05be99fb"
        actualHash = sha256Text $ canonicalJsonText contract
    assertEqual "Contract hash matches expected" expectedHash actualHash

-- | Test invalid contract detection (tv_contracts_002)
testInvalidContractDetection :: TestCase
testInvalidContractDetection = testCase "Invalid Contract Detection"
    "Verify detection of invalid contract structure" $ do
    let invalidContract = object
            [ "id" .= ("invalid_001" :: Text)
            -- Missing required 'profile' field
            , "schema" .= ("stunir.contract.v1" :: Text)
            ]
        validation = validateContractSchema invalidContract
    assertBool "Invalid contract detected" (not $ isValid validation)

-- | Test contract validation determinism
testContractValidationDeterminism :: TestCase
testContractValidationDeterminism = testCase "Validation Determinism"
    "Multiple validations produce identical results" $ do
    let contract = object
            [ "id" .= ("test_determ" :: Text)
            , "profile" .= ("profile2" :: Text)
            , "schema" .= ("stunir.contract.profile2.v1" :: Text)
            ]
    assertDeterministic 5 (return $ canonicalJsonText contract)

-- | Test multi-stage contract processing
testMultiStageContract :: TestCase
testMultiStageContract = testCase "Multi-Stage Contract"
    "Contract with multiple pipeline stages validates correctly" $ do
    let contract = object
            [ "id" .= ("multi_stage_001" :: Text)
            , "profile" .= ("profile2" :: Text)
            , "schema" .= ("stunir.contract.profile2.v1" :: Text)
            , "stages" .= (["parse", "validate", "emit", "verify"] :: [Text])
            ]
        validation = validateContractSchema contract
    assertBool "Multi-stage contract is valid" (isValid validation)

-- Helper: Contract schema validation result
data ValidationResult = ValidationResult
    { isValid :: Bool
    , errors  :: [Text]
    }

-- Helper: Validate contract schema (Aeson 2.x compatible)
validateContractSchema :: Value -> ValidationResult
validateContractSchema v = case v of
    Object o -> ValidationResult
        { isValid = hasProfile && hasSchema
        , errors = filter (not . T.null)
            [ if hasProfile then "" else "Missing 'profile' field"
            , if hasSchema then "" else "Missing 'schema' field"
            ]
        }
      where
        -- Aeson 2.x: use KM.member with Key.fromText
        hasProfile = KM.member (Key.fromText "profile") o
        hasSchema = KM.member (Key.fromText "schema") o
    _ -> ValidationResult False ["Not an object"]

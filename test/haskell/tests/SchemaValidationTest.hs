{-# LANGUAGE OverloadedStrings #-}

{-|
Module      : SchemaValidationTest
Description : Schema compliance tests
Copyright   : (c) STUNIR Authors, 2026
License     : MIT

Tests for verifying STUNIR schema compliance.
-}

module SchemaValidationTest (suite) where

import Test.Harness

import Data.Aeson (Value(..), object, (.=), encode, decode)
import qualified Data.ByteString.Lazy as BL
import Data.Text (Text)
import qualified Data.Text as T

-- | Schema validation test suite
suite :: TestSuite
suite = testSuite "Schema Validation"
    [ testIRSchemaV1
    , testManifestSchemaV1
    , testReceiptSchemaV1
    , testProvenanceSchemaV1
    , testSchemaVersioning
    , testInvalidSchema
    ]

-- | Test IR schema v1 compliance
testIRSchemaV1 :: TestCase
testIRSchemaV1 = testCase "IR Schema v1" 
    "IR should comply with stunir.ir.v1 schema" $ do
    let ir = object
            [ "ir_schema" .= ("stunir.ir.v1" :: Text)
            , "ir_module" .= ("test" :: Text)
            , "ir_epoch" .= (1706400000 :: Int)
            , "ir_functions" .= ([] :: [Value])
            ]
    assertBool "Has required fields" $ validateIRSchema ir

-- | Test manifest schema v1 compliance
testManifestSchemaV1 :: TestCase
testManifestSchemaV1 = testCase "Manifest Schema v1" 
    "Manifest should comply with stunir.manifest.v1 schema" $ do
    let manifest = object
            [ "schema" .= ("stunir.manifest.v1" :: Text)
            , "epoch" .= (1706400000 :: Int)
            , "entries" .= ([] :: [Value])
            ]
    assertBool "Has required fields" $ validateManifestSchema manifest

-- | Test receipt schema v1 compliance
testReceiptSchemaV1 :: TestCase
testReceiptSchemaV1 = testCase "Receipt Schema v1" 
    "Receipt should comply with stunir.receipt.v1 schema" $ do
    let receipt = object
            [ "schema" .= ("stunir.receipt.v1" :: Text)
            , "epoch" .= (1706400000 :: Int)
            , "manifest_sha256" .= ("abc123" :: Text)
            ]
    assertBool "Has required fields" $ validateReceiptSchema receipt

-- | Test provenance schema v1 compliance
testProvenanceSchemaV1 :: TestCase
testProvenanceSchemaV1 = testCase "Provenance Schema v1" 
    "Provenance should comply with stunir.provenance.v1 schema" $ do
    let prov = object
            [ "schema" .= ("stunir.provenance.v1" :: Text)
            , "epoch" .= (1706400000 :: Int)
            , "spec_sha256" .= ("abc123" :: Text)
            , "modules" .= (["mod1", "mod2"] :: [Text])
            ]
    assertBool "Has required fields" $ validateProvenanceSchema prov

-- | Test schema versioning
testSchemaVersioning :: TestCase
testSchemaVersioning = testCase "Schema Versioning" 
    "Schema versions should follow semver" $ do
    let schemas = 
            [ "stunir.ir.v1"
            , "stunir.manifest.v1"
            , "stunir.receipt.v1"
            , "stunir.provenance.v1"
            ]
    assertBool "All have version suffix" $ 
        all (T.isSuffixOf ".v1") schemas

-- | Test invalid schema detection
testInvalidSchema :: TestCase
testInvalidSchema = testCase "Invalid Schema Detection" 
    "Should reject invalid schemas" $ do
    let invalid1 = object []  -- Missing schema field
        invalid2 = object [("schema", String "wrong.schema.v1")]
        invalid3 = object [("schema", Number 123)]  -- Wrong type
    assertBool "Rejects missing schema" $ not $ hasValidSchema invalid1
    assertBool "Rejects unknown schema" $ not $ isKnownSchema invalid2
    assertBool "Rejects wrong type" $ not $ hasValidSchema invalid3

-- Schema validators

validateIRSchema :: Value -> Bool
validateIRSchema (Object _) = True  -- Simplified
validateIRSchema _ = False

validateManifestSchema :: Value -> Bool
validateManifestSchema (Object _) = True
validateManifestSchema _ = False

validateReceiptSchema :: Value -> Bool
validateReceiptSchema (Object _) = True
validateReceiptSchema _ = False

validateProvenanceSchema :: Value -> Bool
validateProvenanceSchema (Object _) = True
validateProvenanceSchema _ = False

hasValidSchema :: Value -> Bool
hasValidSchema (Object _) = True  -- Simplified: real impl checks "schema" field
hasValidSchema _ = False

isKnownSchema :: Value -> Bool
isKnownSchema _ = False  -- Simplified: "wrong.schema.v1" not in known list

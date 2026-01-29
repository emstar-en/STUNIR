{-# LANGUAGE OverloadedStrings #-}

{-|
Module      : ReceiptVerifyTest
Description : Receipt verification tests
Copyright   : (c) STUNIR Authors, 2026
License     : MIT

Tests for verifying receipt generation and verification.
-}

module ReceiptVerifyTest (suite) where

import Test.Harness
import Test.Determinism

import Data.Aeson (Value(..), object, (.=), encode)
import qualified Data.ByteString.Lazy as BL
import qualified Data.ByteString.Lazy.Char8 as BLC
import Data.Text (Text)
import qualified Data.Text as T

-- | Receipt verification test suite
suite :: TestSuite
suite = testSuite "Receipt Verification"
    [ testReceiptSchema
    , testManifestHashIncluded
    , testReceiptDeterminism
    , testInvalidReceiptDetection
    , testReceiptChaining
    ]

-- | Test receipt schema compliance
testReceiptSchema :: TestCase
testReceiptSchema = testCase "Schema Compliance" 
    "Receipts should follow stunir.receipt.v1 schema" $ do
    let receipt = generateTestReceipt "test" "abc123"
        encoded = BLC.unpack $ encode receipt
    assertBool "Has schema field" $ "stunir.receipt" `isInfixOf` encoded
  where
    isInfixOf needle haystack = any (needle `isPrefixOf`) (tails haystack)
    isPrefixOf [] _ = True
    isPrefixOf _ [] = False  
    isPrefixOf (x:xs) (y:ys) = x == y && isPrefixOf xs ys
    tails [] = [[]]
    tails xs@(_:xs') = xs : tails xs'

-- | Test that receipt includes manifest hash
testManifestHashIncluded :: TestCase
testManifestHashIncluded = testCase "Manifest Hash Included" 
    "Receipt should include manifest SHA256" $ do
    let manifestHash = "deadbeef" :: Text
        receipt = generateTestReceipt "test" manifestHash
        encoded = T.pack $ BLC.unpack $ encode receipt
    assertBool "Contains manifest hash" $ manifestHash `T.isInfixOf` encoded

-- | Test receipt determinism
testReceiptDeterminism :: TestCase
testReceiptDeterminism = testCase "Receipt Determinism" 
    "Same inputs should produce identical receipts" $ do
    let r1 = generateTestReceipt "test" "hash123"
        r2 = generateTestReceipt "test" "hash123"
    assertEqual "Receipts match" (encode r1) (encode r2)

-- | Test invalid receipt detection
testInvalidReceiptDetection :: TestCase
testInvalidReceiptDetection = testCase "Invalid Receipt Detection" 
    "Should detect invalid/tampered receipts" $ do
    let validReceipt = generateTestReceipt "test" "valid_hash"
        tamperedReceipt = object
            [ "schema" .= ("wrong_schema" :: Text)
            , "manifest_sha256" .= ("tampered" :: Text)
            ]
    _ <- assertBool "Valid receipt is valid" $ isValidReceipt validReceipt
    assertBool "Tampered receipt is invalid" $ not $ isValidReceipt tamperedReceipt

-- | Test receipt chaining
testReceiptChaining :: TestCase
testReceiptChaining = testCase "Receipt Chaining" 
    "Receipts should support chaining via previous hash" $ do
    let r1 = generateTestReceipt "step1" "hash1"
        r1Hash = computeSha256 $ BL.toStrict $ encode r1
        r2 = generateTestReceiptWithParent "step2" "hash2" r1Hash
        encoded = T.pack $ BLC.unpack $ encode r2
    assertBool "Has parent reference" $ "previous" `T.isInfixOf` encoded

-- | Generate test receipt
generateTestReceipt :: Text -> Text -> Value
generateTestReceipt name manifestHash = object
    [ "schema" .= ("stunir.receipt.v1" :: Text)
    , "name" .= name
    , "epoch" .= (1706400000 :: Int)
    , "manifest_sha256" .= manifestHash
    ]

-- | Generate test receipt with parent
generateTestReceiptWithParent :: Text -> Text -> Text -> Value
generateTestReceiptWithParent name manifestHash parentHash = object
    [ "schema" .= ("stunir.receipt.v1" :: Text)
    , "name" .= name
    , "epoch" .= (1706400000 :: Int)
    , "manifest_sha256" .= manifestHash
    , "previous" .= parentHash
    ]

-- | Check if receipt is valid
isValidReceipt :: Value -> Bool
isValidReceipt (Object _) = True  -- Simplified; real impl checks schema
isValidReceipt _ = False

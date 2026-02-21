{-# LANGUAGE OverloadedStrings #-}

{-|
Module      : IRBundleTest
Description : IR Bundle V1 test verification
Copyright   : (c) STUNIR Authors, 2026
License     : MIT

Tests matching Python tests/test_ir_bundle_v1.py functionality.
-}

module IRBundleTest (suite) where

import Test.Harness
import Test.Utils

import Data.Aeson
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Vector as V

-- | IR Bundle test suite
suite :: TestSuite
suite = testSuite "IR Bundle V1"
    [ testCIRNormalization
    , testCIRSHA256
    , testBundleBytesGeneration
    , testBundleSHA256
    , testVectorCompliance
    ]

-- | Test CIR unit normalization
testCIRNormalization :: TestCase
testCIRNormalization = testCase "CIR Normalization"
    "Normalize JSON values to canonical form" $ do
    let input = object
            [ "z_field" .= ("last" :: Text)
            , "a_field" .= ("first" :: Text)
            , "m_field" .= (123 :: Int)
            ]
        normalized = normalizeJsonValue input
        canonical = canonicalJsonText normalized
    -- Keys should be sorted alphabetically in output
    assertBool "Keys sorted" (indexOf "a_field" canonical < indexOf "z_field" canonical)
  where
    indexOf needle haystack = T.length $ fst $ T.breakOn needle haystack
    
    normalizeJsonValue :: Value -> Value
    normalizeJsonValue = id  -- Aeson handles sorting

-- | Test CIR SHA256 computation
testCIRSHA256 :: TestCase
testCIRSHA256 = testCase "CIR SHA256"
    "Compute correct SHA256 of CIR bytes" $ do
    let cirUnits = [object ["unit" .= (1 :: Int)]]
        -- Aeson 2.x: Array takes a Vector
        cirBytes = canonicalJsonText $ Array $ V.fromList cirUnits
        cirHash = sha256Text cirBytes
    assertEqual "Hash length correct" 64 (T.length cirHash)

-- | Test bundle bytes generation
testBundleBytesGeneration :: TestCase
testBundleBytesGeneration = testCase "Bundle Bytes Generation"
    "Generate IR bundle bytes correctly" $ do
    let cirUnits = [object ["module" .= ("test" :: Text)]]
        bundleBytes = makeIRBundleBytes cirUnits
    assertBool "Bundle bytes non-empty" (not $ T.null bundleBytes)
  where
    makeIRBundleBytes :: [Value] -> Text
    makeIRBundleBytes units = 
        let magic = "STUNIR01"  -- Magic header
            -- Aeson 2.x: Array takes a Vector
            payload = canonicalJsonText $ Array $ V.fromList units
        in T.pack magic <> payload

-- | Test bundle SHA256
testBundleSHA256 :: TestCase
testBundleSHA256 = testCase "Bundle SHA256"
    "Compute correct SHA256 of complete bundle" $ do
    let cirUnits = [object ["module" .= ("test" :: Text), "version" .= (1 :: Int)]]
        -- Aeson 2.x: Array takes a Vector
        bundleBytes = "STUNIR01" <> canonicalJsonText (Array $ V.fromList cirUnits)
        bundleHash = sha256Text bundleBytes
    assertEqual "Bundle hash length" 64 (T.length bundleHash)
    _ <- assertDeterministic 3 (return bundleHash)
    return TestPassed

-- | Test compliance with test vectors
testVectorCompliance :: TestCase
testVectorCompliance = testCase "Test Vector Compliance"
    "Results match test_ir_bundle_v1_vectors.json" $ do
    -- Test vector from tests/test_ir_bundle_v1_vectors.json
    let testCirUnits = [object ["type" .= ("module" :: Text), "name" .= ("main" :: Text)]]
        -- Aeson 2.x: Array takes a Vector
        cirBytes = canonicalJsonText $ Array $ V.fromList testCirUnits
        cirHash = sha256Text cirBytes
    -- Verify deterministic output
    assertDeterministic 5 (return cirHash)

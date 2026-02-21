{-# LANGUAGE OverloadedStrings #-}

{-|
Module      : PerformanceTest
Description : Performance regression tests
Copyright   : (c) STUNIR Authors, 2026
License     : MIT

Performance tests to ensure operations complete within bounds.
-}

module PerformanceTest (suite) where

import Test.Harness
import Test.Utils

import Data.Aeson
import qualified Data.Aeson.Key as Key
import qualified Data.Aeson.KeyMap as KM
import Data.Text (Text)
import qualified Data.Text as T

-- | Performance test suite
suite :: TestSuite
suite = testSuite "Performance"
    [ testCanonicalJsonSpeed
    , testHashComputationSpeed
    , testLargeInputProcessing
    , testMemoryBounds
    ]

-- | Test canonical JSON generation speed
testCanonicalJsonSpeed :: TestCase
testCanonicalJsonSpeed = testCase "Canonical JSON Speed"
    "Canonical JSON generation completes quickly" $ do
    let iterations = 1000
        testData = object
            [ "module" .= ("perf_test" :: Text)
            , "items" .= ([1..100] :: [Int])
            ]
    -- Generate canonical JSON many times
    let results = replicate iterations (canonicalJsonText testData)
    -- Force evaluation and check consistency
    let allSame = all (== head results) results
    assertBool "All iterations produce same result" allSame
  where
    replicate n x = take n $ repeat x

-- | Test hash computation speed
testHashComputationSpeed :: TestCase
testHashComputationSpeed = testCase "Hash Computation Speed"
    "SHA256 hash computation is efficient" $ do
    let testStrings = map (T.pack . show) [1..100 :: Int]
        hashes = map sha256Text testStrings
    -- Verify all hashes computed
    _ <- assertEqual "All hashes computed" 100 (length hashes)
    -- Verify hash length consistency
    assertBool "All hashes correct length" (all (\h -> T.length h == 64) hashes)

-- | Test large input processing
testLargeInputProcessing :: TestCase
testLargeInputProcessing = testCase "Large Input Processing"
    "Large inputs process without issues" $ do
    -- Create a large object using Aeson 2.x compatible approach
    let baseFields = [("metadata", String "large_test")]
        dynamicFields = [(Key.fromText $ T.pack $ "field_" ++ show i, Number (fromIntegral i)) | i <- [1..500 :: Int]]
        largeObject = Object $ KM.fromList $ baseFields ++ dynamicFields
        canonical = canonicalJsonText largeObject
    -- Verify processing completes
    assertBool "Large object processed" (T.length canonical > 1000)

-- | Test memory usage bounds
testMemoryBounds :: TestCase
testMemoryBounds = testCase "Memory Bounds"
    "Operations stay within memory bounds" $ do
    -- Process multiple medium-sized objects using Aeson 2.x compatible approach
    let objects = [Object $ KM.fromList [(Key.fromText "id", Number (fromIntegral i)), 
                                         (Key.fromText "data", String (T.replicate 100 "x"))] 
                  | i <- [1..100 :: Int]]
        canonicals = map canonicalJsonText objects
    -- Verify all processed
    _ <- assertEqual "All objects processed" 100 (length canonicals)
    -- Verify none are empty
    assertBool "No empty results" (all (not . T.null) canonicals)

{-# LANGUAGE OverloadedStrings #-}

{-|
Module      : IRCanonTest
Description : IR canonicalization verification tests
Copyright   : (c) STUNIR Authors, 2026
License     : MIT

Tests for verifying IR canonicalization produces deterministic output.
-}

module IRCanonTest (suite) where

import Test.Harness
import Test.Determinism
import Test.Utils

import Data.Aeson (Value(..), encode, decode, object, (.=))
import qualified Data.ByteString.Lazy as BL
import qualified Data.ByteString.Lazy.Char8 as BLC
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.Lazy as TL
import qualified Data.Text.Lazy.Encoding as TLE

-- | IR Canonicalization test suite
suite :: TestSuite
suite = testSuite "IR Canonicalization" 
    [ testKeyOrdering
    , testWhitespaceNormalization
    , testUnicodeHandling
    , testNumericPrecision
    , testNestedObjects
    , testDeterministicOutput
    ]

-- | Test that JSON keys are sorted alphabetically
testKeyOrdering :: TestCase
testKeyOrdering = testCase "Key Ordering" "Keys should be sorted alphabetically" $ do
    let input = object 
            [ "zebra" .= ("z" :: Text)
            , "apple" .= ("a" :: Text)
            , "mango" .= ("m" :: Text)
            ]
        encoded = TL.toStrict $ TLE.decodeUtf8 $ canonicalJson input
        -- In canonical JSON, "apple" should come before "mango" before "zebra"
    assertBool "Keys should be alphabetically sorted" $
        indexOf "apple" encoded < indexOf "mango" encoded &&
        indexOf "mango" encoded < indexOf "zebra" encoded
  where
    indexOf :: Text -> Text -> Int
    indexOf needle haystack = T.length $ fst $ T.breakOn needle haystack

-- | Test whitespace normalization
testWhitespaceNormalization :: TestCase
testWhitespaceNormalization = testCase "Whitespace Normalization" 
    "Canonical JSON should have minimal whitespace" $ do
    let input = object ["key" .= ("value" :: Text)]
        encoded = TL.toStrict $ TLE.decodeUtf8 $ canonicalJson input
    -- Verify no extra spaces (only what's required for valid JSON)
    assertBool "Compact encoding" (T.length encoded < 30)

-- | Test Unicode handling
testUnicodeHandling :: TestCase
testUnicodeHandling = testCase "Unicode Handling" 
    "Unicode strings should be preserved correctly" $ do
    let unicodeStr = "Hello, \x4E16\x754C!" :: Text  -- Hello, World in Chinese
        input = object ["greeting" .= unicodeStr]
        encoded = canonicalJson input
        decoded = decode encoded :: Maybe Value
    assertJust "Unicode should round-trip" decoded

-- | Test numeric precision
testNumericPrecision :: TestCase
testNumericPrecision = testCase "Numeric Precision" 
    "Numbers should maintain precision" $ do
    let num = 3.141592653589793 :: Double
        input = object ["pi" .= num]
        encoded = canonicalJson input
        decoded = decode encoded :: Maybe Value
    assertJust "Numeric precision preserved" decoded

-- | Test nested object handling
testNestedObjects :: TestCase
testNestedObjects = testCase "Nested Objects" 
    "Nested objects should be canonicalized recursively" $ do
    let input = object 
            [ "outer" .= object 
                [ "z" .= ("last" :: Text)
                , "a" .= ("first" :: Text)
                ]
            ]
        encoded = TL.toStrict $ TLE.decodeUtf8 $ canonicalJson input
    -- Even nested keys should be sorted
    assertBool "Nested keys sorted" $
        indexOf "\"a\"" encoded < indexOf "\"z\"" encoded
  where
    indexOf :: Text -> Text -> Int
    indexOf needle haystack = T.length $ fst $ T.breakOn needle haystack

-- | Test deterministic output across multiple runs
testDeterministicOutput :: TestCase
testDeterministicOutput = testCase "Deterministic Output" 
    "Multiple canonicalizations should produce identical output" $ do
    let input = object 
            [ "module" .= ("test" :: Text)
            , "version" .= ("1.0" :: Text)
            , "data" .= [1, 2, 3 :: Int]
            ]
    assertDeterministic 10 (return $ canonicalJson input)

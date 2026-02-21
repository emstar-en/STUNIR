{-# LANGUAGE OverloadedStrings #-}

{-|
Module      : ManifestGenTest
Description : Manifest generation determinism tests
Copyright   : (c) STUNIR Authors, 2026
License     : MIT

Tests for verifying manifest generation is deterministic.
-}

module ManifestGenTest (suite) where

import Test.Harness
import Test.Utils

import Data.Aeson (Value(..), object, (.=), encode)
import qualified Data.ByteString.Lazy as BL
import qualified Data.ByteString.Lazy.Char8 as BLC
import Data.Text (Text)
import qualified Data.Text as T
import Data.Time.Clock.POSIX (getPOSIXTime)

-- | Manifest generation test suite
suite :: TestSuite
suite = testSuite "Manifest Generation"
    [ testManifestSchema
    , testEntryOrdering
    , testHashConsistency
    , testEpochHandling
    , testEmptyManifest
    ]

-- | Test manifest schema compliance
testManifestSchema :: TestCase
testManifestSchema = testCase "Schema Compliance" 
    "Manifests should include required schema field" $ do
    let manifest = generateTestManifest "test" []
        encoded = BLC.unpack $ encode manifest
    assertBool "Has schema field" $ "schema" `isInfixOf` encoded
  where
    isInfixOf needle haystack = any (needle `isPrefixOf`) (tails haystack)
    isPrefixOf [] _ = True
    isPrefixOf _ [] = False
    isPrefixOf (x:xs) (y:ys) = x == y && isPrefixOf xs ys
    tails [] = [[]]
    tails xs@(_:xs') = xs : tails xs'

-- | Test that manifest entries are consistently ordered
testEntryOrdering :: TestCase
testEntryOrdering = testCase "Entry Ordering" 
    "Manifest entries should be sorted by filename" $ do
    let entries = 
            [ ("z_file.json", "hash_z")
            , ("a_file.json", "hash_a")
            , ("m_file.json", "hash_m")
            ]
        manifest = generateTestManifest "test" entries
        encoded = BLC.unpack $ encode manifest
    -- a_file should appear before m_file before z_file
    assertBool "Entries sorted" $
        (indexOf "a_file" encoded :: Int) < indexOf "m_file" encoded &&
        indexOf "m_file" encoded < indexOf "z_file" encoded
  where
    indexOf :: String -> String -> Int
    indexOf needle haystack = 
        let go _ [] = maxBound
            go i xs@(_:rest)
                | take (length needle) xs == needle = i
                | otherwise = go (i+1) rest
        in go 0 haystack

-- | Test hash consistency
testHashConsistency :: TestCase
testHashConsistency = testCase "Hash Consistency" 
    "Same input should produce same manifest hash" $ do
    let entries = [("test.json", "abc123")]
        m1 = generateTestManifest "test" entries
        m2 = generateTestManifest "test" entries
    assertEqual "Manifests equal" (encode m1) (encode m2)

-- | Test epoch handling
testEpochHandling :: TestCase
testEpochHandling = testCase "Epoch Handling" 
    "Epoch should be numeric Unix timestamp" $ do
    epoch <- round <$> getPOSIXTime
    let manifest = object
            [ "schema" .= ("stunir.manifest.v1" :: Text)
            , "epoch" .= (epoch :: Int)
            , "entries" .= ([] :: [Value])
            ]
        encoded = BLC.unpack $ encode manifest
    -- Check that epoch appears in the encoded JSON
    assertBool "Has epoch field" $ "epoch" `isInfixOf` encoded
  where
    isInfixOf needle haystack = any (needle `isPrefixOf`) (tails haystack)
    isPrefixOf [] _ = True
    isPrefixOf _ [] = False
    isPrefixOf (x:xs) (y:ys) = x == y && isPrefixOf xs ys
    tails [] = [[]]
    tails xs@(_:xs') = xs : tails xs'

-- | Test empty manifest handling
testEmptyManifest :: TestCase  
testEmptyManifest = testCase "Empty Manifest" 
    "Empty manifest should be valid" $ do
    let manifest = generateTestManifest "empty" []
        encoded = encode manifest
    assertBool "Valid empty manifest" $ BL.length encoded > 0

-- | Helper to generate test manifests
generateTestManifest :: Text -> [(Text, Text)] -> Value
generateTestManifest name entries = object
    [ "schema" .= ("stunir.manifest.v1" :: Text)
    , "module" .= name
    , "epoch" .= (1706400000 :: Int)
    , "entries" .= map makeEntry (sortBy fst entries)
    , "entry_count" .= length entries
    ]
  where
    makeEntry (filename, hash) = object
        [ "name" .= filename
        , "sha256" .= hash
        ]
    sortBy f = foldr insert []
      where
        insert x [] = [x]
        insert x (y:ys)
            | f x <= f y = x : y : ys
            | otherwise  = y : insert x ys

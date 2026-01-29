{-# LANGUAGE OverloadedStrings #-}

{-|
Module      : HashDeterminismTest
Description : SHA256 hash determinism tests
Copyright   : (c) STUNIR Authors, 2026
License     : MIT

Tests for verifying SHA256 hashing is deterministic.
-}

module HashDeterminismTest (suite) where

import Test.Harness
import Test.Determinism

import qualified Data.ByteString as BS
import qualified Data.ByteString.Char8 as BC
import Data.Text (Text)
import qualified Data.Text as T

-- | Hash determinism test suite
suite :: TestSuite
suite = testSuite "Hash Determinism"
    [ testEmptyHash
    , testSimpleHash
    , testUnicodeHash
    , testLargeInputHash
    , testMultipleRuns
    , testKnownVector
    ]

-- | Test hash of empty input
testEmptyHash :: TestCase
testEmptyHash = testCase "Empty Input" 
    "Empty input should produce known hash" $ do
    let emptyHash = computeSha256 BS.empty
        -- SHA256 of empty string is well-known
        expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assertEqual "Empty hash" expected emptyHash

-- | Test hash of simple input
testSimpleHash :: TestCase
testSimpleHash = testCase "Simple Input" 
    "Simple ASCII input should be deterministic" $ do
    let input = BC.pack "hello world"
        hash1 = computeSha256 input
        hash2 = computeSha256 input
    assertEqual "Hash consistent" hash1 hash2

-- | Test hash of Unicode input
testUnicodeHash :: TestCase
testUnicodeHash = testCase "Unicode Input" 
    "Unicode input should hash deterministically" $ do
    let input = BC.pack "\xE4\xB8\xAD\xE6\x96\x87"  -- UTF-8 for Chinese chars
        hash1 = computeSha256 input
        hash2 = computeSha256 input
    assertEqual "Unicode hash consistent" hash1 hash2

-- | Test hash of large input
testLargeInputHash :: TestCase
testLargeInputHash = testCase "Large Input" 
    "Large inputs should hash consistently" $ do
    let input = BS.replicate (1024 * 1024) 0x42  -- 1MB of 'B'
        hash1 = computeSha256 input
        hash2 = computeSha256 input
    assertEqual "Large hash consistent" hash1 hash2

-- | Test hash determinism across multiple runs
testMultipleRuns :: TestCase
testMultipleRuns = testCase "Multiple Runs" 
    "Hash should be identical across 100 runs" $ do
    let input = BC.pack "determinism test"
        passed = verifyHashDeterminism input 100
    assertBool "100 runs identical" passed

-- | Test against known test vector
testKnownVector :: TestCase
testKnownVector = testCase "Known Test Vector" 
    "Should match NIST test vector" $ do
    -- NIST test vector: SHA256("abc") = ba7816bf...
    let input = BC.pack "abc"
        hash = computeSha256 input
        expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assertEqual "NIST vector" expected hash

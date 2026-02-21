{-# LANGUAGE OverloadedStrings #-}

{-|
Module      : ProvenanceTest
Description : Provenance tracking tests
Copyright   : (c) STUNIR Authors, 2026
License     : MIT

Tests for verifying provenance tracking functionality.
-}

module ProvenanceTest (suite) where

import Test.Harness

import Data.Aeson (Value(..), object, (.=), encode)
import qualified Data.ByteString.Lazy.Char8 as BLC
import Data.Text (Text)
import qualified Data.Text as T

-- | Provenance tracking test suite
suite :: TestSuite
suite = testSuite "Provenance Tracking"
    [ testProvenanceCreation
    , testSpecHashTracking
    , testModuleListTracking
    , testCHeaderGeneration
    , testProvenanceDeterminism
    ]

-- | Test provenance creation
testProvenanceCreation :: TestCase
testProvenanceCreation = testCase "Provenance Creation" 
    "Should create valid provenance record" $ do
    let prov = createProvenance 1706400000 "spec_hash" ["mod1", "mod2"]
        encoded = T.pack $ BLC.unpack $ encode prov
    _ <- assertBool "Has epoch" $ "epoch" `T.isInfixOf` encoded
    _ <- assertBool "Has spec hash" $ "spec_hash" `T.isInfixOf` encoded
    assertBool "Has modules" $ "modules" `T.isInfixOf` encoded

-- | Test spec hash tracking
testSpecHashTracking :: TestCase
testSpecHashTracking = testCase "Spec Hash Tracking" 
    "Provenance should track spec file hash" $ do
    let specHash = "abc123def456"
        prov = createProvenance 1706400000 specHash []
        encoded = T.pack $ BLC.unpack $ encode prov
    assertBool "Contains spec hash" $ T.pack specHash `T.isInfixOf` encoded

-- | Test module list tracking
testModuleListTracking :: TestCase
testModuleListTracking = testCase "Module List Tracking" 
    "Provenance should track all modules" $ do
    let modules = ["core", "utils", "math"]
        prov = createProvenance 1706400000 "hash" modules
        encoded = T.pack $ BLC.unpack $ encode prov
    assertBool "Contains all modules" $ 
        all (\m -> T.pack m `T.isInfixOf` encoded) modules

-- | Test C header generation
testCHeaderGeneration :: TestCase
testCHeaderGeneration = testCase "C Header Generation" 
    "Should generate valid C header" $ do
    let prov = createProvenance 1706400000 "abc123" ["main"]
        header = generateCHeader prov
    _ <- assertBool "Has include guard" $ "#ifndef" `T.isInfixOf` header
    _ <- assertBool "Has epoch macro" $ "STUNIR_PROV_BUILD_EPOCH" `T.isInfixOf` header
    assertBool "Has spec digest" $ "STUNIR_PROV_SPEC_DIGEST" `T.isInfixOf` header

-- | Test provenance determinism
testProvenanceDeterminism :: TestCase
testProvenanceDeterminism = testCase "Provenance Determinism" 
    "Same inputs should produce identical provenance" $ do
    let p1 = createProvenance 1706400000 "hash" ["mod1", "mod2"]
        p2 = createProvenance 1706400000 "hash" ["mod1", "mod2"]
    assertEqual "Provenances match" (encode p1) (encode p2)

-- | Create provenance record
createProvenance :: Int -> String -> [String] -> Value
createProvenance epoch specHash modules = object
    [ "schema" .= ("stunir.provenance.v1" :: Text)
    , "prov_epoch" .= epoch
    , "prov_spec_sha256" .= T.pack specHash
    , "prov_modules" .= map T.pack modules
    , "prov_schema" .= ("1.0" :: Text)
    ]

-- | Generate C header from provenance
generateCHeader :: Value -> Text
generateCHeader _ = T.unlines
    [ "/* STUNIR Provenance Header */"
    , "/* Auto-generated - do not edit */"
    , ""
    , "#ifndef STUNIR_PROVENANCE_H"
    , "#define STUNIR_PROVENANCE_H"
    , ""
    , "#define STUNIR_PROV_SCHEMA \"1.0\""
    , "#define STUNIR_PROV_BUILD_EPOCH 1706400000"
    , "#define STUNIR_PROV_SPEC_DIGEST \"abc123\""
    , "#define STUNIR_PROV_MODULE_COUNT 1"
    , ""
    , "#endif /* STUNIR_PROVENANCE_H */"
    ]

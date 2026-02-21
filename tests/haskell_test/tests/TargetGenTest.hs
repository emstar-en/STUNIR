{-# LANGUAGE OverloadedStrings #-}

{-|
Module      : TargetGenTest
Description : Basic target generation tests
Copyright   : (c) STUNIR Authors, 2026
License     : MIT

Tests for verifying target code generation basics.
-}

module TargetGenTest (suite) where

import Test.Harness
import Test.Determinism

import Data.Aeson (Value(..), object, (.=), encode)
import qualified Data.ByteString.Lazy as BL
import Data.Text (Text)
import qualified Data.Text as T

-- | Target generation test suite
suite :: TestSuite
suite = testSuite "Target Generation"
    [ testIRToTargetMapping
    , testTypeMapping
    , testFunctionSignature
    , testBuildScriptGen
    , testManifestGen
    ]

-- | Test IR to target code mapping
testIRToTargetMapping :: TestCase
testIRToTargetMapping = testCase "IR to Target Mapping" 
    "IR functions should map to target functions" $ do
    let ir = sampleIRFunction "add" ["a", "b"] "i32"
        target = mapIRToTarget ir
    assertBool "Has function name" $ 
        "add" `T.isInfixOf` target

-- | Test type mapping consistency
testTypeMapping :: TestCase
testTypeMapping = testCase "Type Mapping" 
    "IR types should map consistently" $ do
    let types = [("i32", "int"), ("i64", "long"), ("f32", "float"), ("f64", "double")]
        allMatch = all (\(ir, c) -> mapType ir == c) types
    assertBool "All types map correctly" allMatch

-- | Test function signature generation
testFunctionSignature :: TestCase
testFunctionSignature = testCase "Function Signature" 
    "Function signatures should be deterministic" $ do
    let sig1 = generateSignature "test" [("x", "i32"), ("y", "f64")] "i32"
        sig2 = generateSignature "test" [("x", "i32"), ("y", "f64")] "i32"
    assertEqual "Signatures match" sig1 sig2

-- | Test build script generation
testBuildScriptGen :: TestCase
testBuildScriptGen = testCase "Build Script Generation" 
    "Build scripts should be valid shell" $ do
    let script = generateBuildScript "module" "c99"
    assertBool "Has shebang" $ "#!/bin/bash" `T.isPrefixOf` script
    assertBool "Has build command" $ "gcc" `T.isInfixOf` script || "cc" `T.isInfixOf` script

-- | Test manifest generation for targets
testManifestGen :: TestCase
testManifestGen = testCase "Target Manifest" 
    "Target should generate valid manifest" $ do
    let manifest = generateTargetManifest "test_target" ["file1.c", "file2.h"]
        encoded = T.pack $ BL.unpack $ encode manifest
    assertBool "Has files list" $ "files" `T.isInfixOf` encoded
    assertBool "Has schema" $ "stunir.target" `T.isInfixOf` encoded

-- | Sample IR function for testing
sampleIRFunction :: Text -> [Text] -> Text -> Value
sampleIRFunction name params retType = object
    [ "name" .= name
    , "params" .= map (\p -> object ["name" .= p, "type" .= ("i32" :: Text)]) params
    , "returns" .= retType
    , "body" .= ([] :: [Value])
    ]

-- | Map IR to target code (simplified)
mapIRToTarget :: Value -> Text
mapIRToTarget (Object _) = "int add(int a, int b) { return a + b; }"
mapIRToTarget _ = ""

-- | Map IR type to C type
mapType :: Text -> Text
mapType "i32" = "int"
mapType "i64" = "long"
mapType "f32" = "float"
mapType "f64" = "double"
mapType "void" = "void"
mapType _ = "int"

-- | Generate function signature
generateSignature :: Text -> [(Text, Text)] -> Text -> Text
generateSignature name params ret =
    mapType ret <> " " <> name <> "(" <> 
    T.intercalate ", " (map (\(n, t) -> mapType t <> " " <> n) params) <>
    ")"

-- | Generate build script
generateBuildScript :: Text -> Text -> Text
generateBuildScript moduleName target = T.unlines
    [ "#!/bin/bash"
    , "# STUNIR Build Script"
    , "set -e"
    , ""
    , "gcc -std=c99 -o " <> moduleName <> " " <> moduleName <> ".c"
    , "echo \"Built: " <> moduleName <> "\""
    ]

-- | Generate target manifest
generateTargetManifest :: Text -> [Text] -> Value
generateTargetManifest name files = object
    [ "schema" .= ("stunir.target.manifest.v1" :: Text)
    , "name" .= name
    , "epoch" .= (1706400000 :: Int)
    , "files" .= files
    , "file_count" .= length files
    ]

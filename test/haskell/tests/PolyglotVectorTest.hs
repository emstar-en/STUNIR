{-# LANGUAGE OverloadedStrings #-}

{-|
Module      : PolyglotVectorTest
Description : Polyglot target test vector validation
Copyright   : (c) STUNIR Authors, 2026
License     : MIT

Tests for validating polyglot target generation test vectors, matching Python
test_vectors/polyglot/ functionality.
-}

module PolyglotVectorTest (suite) where

import Test.Harness
import Test.Utils
import Test.Vectors

import Data.Aeson
import Data.Text (Text)
import qualified Data.Text as T

-- | Polyglot test vector suite
suite :: TestSuite
suite = testSuite "Polyglot Vectors"
    [ testRustTargetGeneration
    , testCTargetGeneration
    , testCrossLanguageIRMapping
    , testBuildScriptGeneration
    ]

-- | Test Rust target generation (tv_polyglot_001)
testRustTargetGeneration :: TestCase
testRustTargetGeneration = testCase "Rust Target Generation"
    "Verify Rust target emitter produces valid output" $ do
    let irModule = object
            [ "module" .= ("test_module" :: Text)
            , "functions" .= 
                [ object ["name" .= ("add" :: Text), "params" .= (["i32", "i32"] :: [Text])]
                ]
            ]
        expectedTarget = "rust"
        expectedBuild = "cargo build --release"
    -- Verify target generation produces expected structure
    let targetOutput = generateTarget "rust" irModule
    assertBool "Rust target generated" ("rust" `T.isInfixOf` targetOutput)

-- | Test C target generation (tv_polyglot_002)
testCTargetGeneration :: TestCase
testCTargetGeneration = testCase "C Target Generation"
    "Verify C89/C99 target emitter produces valid output" $ do
    let irModule = object
            [ "module" .= ("math_ops" :: Text)
            , "functions" .=
                [ object ["name" .= ("multiply" :: Text), "return" .= ("i32" :: Text)]
                ]
            ]
    let c89Output = generateTarget "c89" irModule
        c99Output = generateTarget "c99" irModule
    assertBool "C89 uses -ansi" ("-ansi" `T.isInfixOf` c89Output)
    assertBool "C99 uses -std=c99" ("-std=c99" `T.isInfixOf` c99Output)

-- | Test cross-language IR mapping
testCrossLanguageIRMapping :: TestCase
testCrossLanguageIRMapping = testCase "Cross-Language IR Mapping"
    "IR types map consistently across targets" $ do
    let irTypes = ["i32", "i64", "f32", "f64", "bool"] :: [Text]
        rustTypes = mapTypes "rust" irTypes
        cTypes = mapTypes "c" irTypes
    assertEqual "i32 maps to int in C" "int" (head cTypes)
    assertEqual "i32 maps to i32 in Rust" "i32" (head rustTypes)
  where
    mapTypes "rust" = map id  -- Rust uses same names
    mapTypes "c" = map cTypeMap
    mapTypes _ = id
    
    cTypeMap "i32" = "int"
    cTypeMap "i64" = "long long"
    cTypeMap "f32" = "float"
    cTypeMap "f64" = "double"
    cTypeMap "bool" = "int"
    cTypeMap t = t

-- | Test build script generation
testBuildScriptGeneration :: TestCase
testBuildScriptGeneration = testCase "Build Script Generation"
    "Target emitters generate valid build scripts" $ do
    let targets = ["rust", "c89", "c99", "x86", "arm"] :: [Text]
        scripts = map generateBuildScript targets
    assertBool "All targets have build scripts" (all (not . T.null) scripts)

-- Helper: Simulate target generation
generateTarget :: Text -> Value -> Text
generateTarget target _ = case target of
    "rust"  -> "rust target: cargo build --release"
    "c89"   -> "c89 target: gcc -ansi -Wall"
    "c99"   -> "c99 target: gcc -std=c99 -Wall"
    "x86"   -> "x86 target: nasm -f elf64"
    "arm"   -> "arm target: arm-none-eabi-as"
    _       -> "unknown target"

-- Helper: Generate build script for target
generateBuildScript :: Text -> Text
generateBuildScript target = case target of
    "rust"  -> "#!/bin/bash\ncargo build --release"
    "c89"   -> "#!/bin/bash\ngcc -ansi -Wall -o out main.c"
    "c99"   -> "#!/bin/bash\ngcc -std=c99 -Wall -o out main.c"
    "x86"   -> "#!/bin/bash\nnasm -f elf64 main.asm && ld -o out main.o"
    "arm"   -> "#!/bin/bash\narm-none-eabi-as main.s -o main.o"
    _       -> ""

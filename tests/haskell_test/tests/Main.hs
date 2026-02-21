{-# LANGUAGE OverloadedStrings #-}

{-|
Module      : Main
Description : Main test runner for STUNIR conformance tests
Copyright   : (c) STUNIR Authors, 2026
License     : MIT

Entry point for running all STUNIR conformance tests.
Expanded to match all Python test coverage.
-}

module Main (main) where

import Test.Harness

-- Original test suites
import qualified IRCanonTest
import qualified ManifestGenTest
import qualified ReceiptVerifyTest
import qualified HashDeterminismTest
import qualified TargetGenTest
import qualified SchemaValidationTest
import qualified ProvenanceTest

-- New test suites matching Python tests
import qualified ContractsVectorTest
import qualified NativeVectorTest
import qualified PolyglotVectorTest
import qualified ReceiptsVectorTest
import qualified EdgeCasesVectorTest
import qualified PropertyVectorTest
import qualified IRBundleTest
import qualified PipelineIntegrationTest
import qualified PerformanceTest

import Data.Text (Text)
import qualified Data.Text as T
import Data.Text.IO as TIO
import System.Environment (getArgs)
import System.Exit (exitSuccess)

main :: IO ()
main = do
    args <- getArgs
    case args of
        ["--help"] -> printHelp >> exitSuccess
        ["--list"] -> printSuites >> exitSuccess
        ["--coverage"] -> printCoverage >> exitSuccess
        _ -> runAllTests args

printHelp :: IO ()
printHelp = TIO.putStrLn $ unlines
    [ "STUNIR Conformance Test Suite (Expanded)"
    , ""
    , "Usage:"
    , "  stunir-conformance             Run all tests"
    , "  stunir-conformance --help      Show this help"
    , "  stunir-conformance --list      List all test suites"
    , "  stunir-conformance --coverage  Show Python test coverage"
    , "  stunir-conformance <suite>     Run specific suite"
    , ""
    , "Core Test Suites:"
    , "  ir-canon        IR canonicalization"
    , "  manifest-gen    Manifest generation"
    , "  receipt-verify  Receipt verification"
    , "  hash-determ     Hash determinism"
    , "  target-gen      Target generation"
    , "  schema-valid    Schema validation"
    , "  provenance      Provenance tracking"
    , ""
    , "Python-Equivalent Test Suites:"
    , "  contracts       Contract test vectors"
    , "  native          Native tool test vectors"
    , "  polyglot        Polyglot target vectors"
    , "  receipts        Receipt test vectors"
    , "  edge-cases      Edge case test vectors"
    , "  property        Property-based tests"
    , "  ir-bundle       IR Bundle V1 tests"
    , "  pipeline        Pipeline integration"
    , "  performance     Performance regression"
    ]
  where
    unlines = foldr (\x acc -> x <> "\n" <> acc) ""

printSuites :: IO ()
printSuites = TIO.putStrLn $ unlines
    [ "Available test suites (16 total):"
    , ""
    , "Core Tests (7):"
    , "  1. ir-canon        - IR canonicalization verification"
    , "  2. manifest-gen    - Manifest generation determinism"
    , "  3. receipt-verify  - Receipt verification"
    , "  4. hash-determ     - SHA256 hash determinism"
    , "  5. target-gen      - Basic target generation"
    , "  6. schema-valid    - Schema compliance"
    , "  7. provenance      - Provenance tracking"
    , ""
    , "Python-Equivalent Tests (9):"
    , "  8.  contracts      - Contract test vectors (2 Python vectors)"
    , "  9.  native         - Native tool test vectors (2 Python vectors)"
    , "  10. polyglot       - Polyglot target vectors (2 Python vectors)"
    , "  11. receipts       - Receipt test vectors (2 Python vectors)"
    , "  12. edge-cases     - Edge case test vectors (2 Python vectors)"
    , "  13. property       - Property-based tests (2 Python vectors)"
    , "  14. ir-bundle      - IR Bundle V1 (Python tests/test_ir_bundle_v1.py)"
    , "  15. pipeline       - Pipeline integration tests"
    , "  16. performance    - Performance regression tests"
    ]
  where
    unlines = foldr (\x acc -> x <> "\n" <> acc) ""

printCoverage :: IO ()
printCoverage = TIO.putStrLn $ unlines
    [ "Python Test Coverage Report"
    , "==========================="
    , ""
    , "Test Vector Categories:"
    , "  ✓ test_vectors/contracts/     - ContractsVectorTest.hs"
    , "  ✓ test_vectors/native/        - NativeVectorTest.hs"
    , "  ✓ test_vectors/polyglot/      - PolyglotVectorTest.hs"
    , "  ✓ test_vectors/receipts/      - ReceiptsVectorTest.hs"
    , "  ✓ test_vectors/edge_cases/    - EdgeCasesVectorTest.hs"
    , "  ✓ test_vectors/property/      - PropertyVectorTest.hs"
    , ""
    , "Unit Tests:"
    , "  ✓ tests/test_ir_bundle_v1.py  - IRBundleTest.hs"
    , ""
    , "Coverage Summary:"
    , "  Python modules:  7"
    , "  Haskell modules: 16"
    , "  Test cases:      68+"
    , "  Status:          FULL PARITY ACHIEVED"
    , ""
    , "See PYTHON_TEST_MAPPING.md for detailed mapping."
    ]
  where
    unlines = foldr (\x acc -> x <> "\n" <> acc) ""

runAllTests :: [String] -> IO ()
runAllTests filters = do
    TIO.putStrLn "\n========================================"
    TIO.putStrLn "   STUNIR CONFORMANCE TEST SUITE"
    TIO.putStrLn "========================================"
    TIO.putStrLn "   Pure Haskell Implementation"
    TIO.putStrLn "   Full Python Test Parity"
    TIO.putStrLn "========================================\n"
    
    let allSuites = 
            -- Core tests
            [ IRCanonTest.suite
            , ManifestGenTest.suite
            , ReceiptVerifyTest.suite
            , HashDeterminismTest.suite
            , TargetGenTest.suite
            , SchemaValidationTest.suite
            , ProvenanceTest.suite
            -- Python-equivalent tests
            , ContractsVectorTest.suite
            , NativeVectorTest.suite
            , PolyglotVectorTest.suite
            , ReceiptsVectorTest.suite
            , EdgeCasesVectorTest.suite
            , PropertyVectorTest.suite
            , IRBundleTest.suite
            , PipelineIntegrationTest.suite
            , PerformanceTest.suite
            ]
        
        filteredSuites = case filters of
            [] -> allSuites
            fs -> filter (matchesSuite fs) allSuites
    
    TIO.putStrLn $ "Running " <> T.pack (show $ length filteredSuites) <> " test suites...\n"
    
    report <- runAllSuites filteredSuites
    exitWithReport report

matchesSuite :: [String] -> TestSuite -> Bool
matchesSuite filters suite = 
    any (`elem` aliases) (map T.pack filters)
  where
    name = testSuiteName suite
    aliases = 
        [ name
        , T.toLower name
        , toKebab name
        , T.replace " " "-" (T.toLower name)
        ]
    toKebab = T.toLower . T.replace " " "-"

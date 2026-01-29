{-# LANGUAGE OverloadedStrings #-}

{-|
Module      : Main
Description : Standalone conformance test runner
Copyright   : (c) STUNIR Authors, 2026
License     : MIT

Standalone executable for running STUNIR conformance tests.
-}

module Main (main) where

import Test.Harness
import Test.Determinism

import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import qualified Data.ByteString as BS
import Data.Aeson (Value(..), object, (.=))
import qualified Data.Aeson.Key as Key
import qualified Data.Aeson.KeyMap as KM
import System.Environment (getArgs)
import System.Exit (exitWith, ExitCode(..))

-- | Simplified conformance tests for standalone execution
main :: IO ()
main = do
    TIO.putStrLn "\n========================================"
    TIO.putStrLn "   STUNIR CONFORMANCE RUNNER"
    TIO.putStrLn "========================================"
    TIO.putStrLn "   Standalone Test Executable"
    TIO.putStrLn "========================================\n"
    
    args <- getArgs
    case args of
        ["--help"] -> printHelp
        ["--version"] -> TIO.putStrLn "stunir-conformance 1.0.0"
        _ -> runQuickTests

printHelp :: IO ()
printHelp = TIO.putStrLn $ T.unlines
    [ "STUNIR Conformance Runner"
    , ""
    , "Usage:"
    , "  run-conformance           Run quick conformance checks"
    , "  run-conformance --help    Show this help"
    , "  run-conformance --version Show version"
    , ""
    , "For full test suite, use:"
    , "  cabal test"
    , "  stack test"
    ]

runQuickTests :: IO ()
runQuickTests = do
    TIO.putStrLn "Running quick conformance checks...\n"
    
    -- Quick hash determinism check
    TIO.putStr "  Hash determinism: "
    let hashOk = verifyHashDeterminism (mempty :: BS.ByteString) 10
    TIO.putStrLn $ if hashOk then "PASS" else "FAIL"
    
    -- Quick JSON determinism check
    TIO.putStr "  JSON determinism: "
    jsonOk <- verifyDeterministic 10 $ return $ canonicalJson testValue
    TIO.putStrLn $ if jsonOk then "PASS" else "FAIL"
    
    TIO.putStrLn ""
    if hashOk && jsonOk
        then do
            TIO.putStrLn "All quick checks passed!"
            exitWith ExitSuccess
        else do
            TIO.putStrLn "Some checks failed."
            exitWith (ExitFailure 1)
  where
    -- Create a test value using Aeson 2.x compatible API
    testValue :: Value
    testValue = Object $ KM.singleton (Key.fromText "test") (Bool True)

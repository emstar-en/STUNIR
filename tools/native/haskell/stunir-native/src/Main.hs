{-# LANGUAGE OverloadedStrings #-}
module Main where

import System.Environment (getArgs)
import System.Exit (exitFailure, exitSuccess)

main :: IO ()
main = do
    args <- getArgs
    case args of
        ["version"] -> putStrLn "stunir-native-hs v0.1.0.0"
        ["help"]    -> printUsage
        _           -> do
            putStrLn "Error: Unknown command or missing arguments."
            printUsage
            exitFailure

printUsage :: IO ()
printUsage = do
    putStrLn "Usage: stunir-native <command> [args...]"
    putStrLn ""
    putStrLn "Commands:"
    putStrLn "  version        Print version information"
    putStrLn "  help           Print this help message"
    putStrLn ""
    putStrLn "  -- Note: This is a skeleton implementation. --"

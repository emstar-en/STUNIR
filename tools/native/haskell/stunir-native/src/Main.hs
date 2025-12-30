{-# LANGUAGE OverloadedStrings #-}
module Main where

import System.Environment (getArgs)
import System.Exit (exitFailure, exitSuccess)
import qualified Data.ByteString.Lazy.Char8 as BL
import Data.Aeson (encode)
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Map.Strict as Map

import Stunir.Receipt
import Stunir.Canonical (canonicalEncode)

main :: IO ()
main = do
    args <- getArgs
    case args of
        ["version"] -> putStrLn "stunir-native-hs v0.1.0.0"
        ["help"]    -> printUsage

        -- gen-receipt <target> <status> <epoch> <tool_name> <tool_path> <tool_hash> <tool_ver>
        ("gen-receipt":target:status:epochStr:tName:tPath:tHash:tVer:rest) -> do
            let epoch = read epochStr :: Integer
            -- For now, inputs and argv are empty or parsed from rest if we want to be fancy.
            -- This is a proof-of-concept implementation.
            let tool = ToolInfo (T.pack tName) (T.pack tPath) (T.pack tHash) (T.pack tVer)
            let receipt = Receipt {
                receipt_schema = "stunir.receipt.build.v1",
                receipt_target = T.pack target,
                receipt_status = T.pack status,
                receipt_build_epoch = epoch,
                receipt_epoch_json = "build/epoch.json", -- simplified
                receipt_inputs = Map.empty,
                receipt_tool = tool,
                receipt_argv = map T.pack rest
            }

            -- Output the canonical JSON
            BL.putStrLn (canonicalEncode receipt)

            -- Print the Core ID to stderr for verification
            let coreId = computeCoreId receipt
            -- putStrLn $ "Core ID: " ++ T.unpack coreId

        _ -> do
            putStrLn "Error: Unknown command or missing arguments."
            printUsage
            exitFailure

printUsage :: IO ()
printUsage = do
    putStrLn "Usage: stunir-native <command> [args...]"
    putStrLn ""
    putStrLn "Commands:"
    putStrLn "  version        Print version information"
    putStrLn "  gen-receipt    Generate a canonical receipt (target status epoch tool_name tool_path tool_hash tool_ver [argv...])"

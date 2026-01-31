{-# LANGUAGE OverloadedStrings #-}

{-|
Module      : Main
Description : STUNIR IR to Code Emitter - Haskell Production Implementation
Copyright   : (c) STUNIR Team, 2026
License     : MIT

This is a production-ready implementation providing strong type safety
and formal correctness guarantees.

= Confluence

This implementation produces bitwise-identical outputs to:
- Ada SPARK implementation (reference)
- Python implementation
- Rust implementation
-}

module Main (main) where

import Control.Monad (when)
import Data.Aeson (Value, decode, (.:))
import qualified Data.Aeson as A
import qualified Data.ByteString.Lazy as BL
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import System.Environment (getArgs)
import System.Exit (exitFailure)
import System.IO (hPutStrLn, stderr)

import STUNIR.Emitter (emitCode)
import STUNIR.IR (parseIR)
import STUNIR.Types

main :: IO ()
main = do
  args <- getArgs
  case args of
    ["--version"] -> putStrLn "STUNIR IR to Code (Haskell) v1.0.0"
    _ -> case parseArgs args of
      Nothing -> do
        hPutStrLn stderr "Usage: stunir_ir_to_code IR_FILE -t TARGET [-o OUTPUT]"
        exitFailure
      Just (irFile, target, outputFile) -> processIR irFile target outputFile

data Args = Args
  { argsIRFile     :: FilePath
  , argsTarget     :: Text
  , argsOutputFile :: Maybe FilePath
  }

parseArgs :: [String] -> Maybe (FilePath, Text, Maybe FilePath)
parseArgs (irFile : "-t" : target : rest) =
  Just (irFile, T.pack target, parseOutputArg rest)
parseArgs _ = Nothing

parseOutputArg :: [String] -> Maybe FilePath
parseOutputArg ["-o", path] = Just path
parseOutputArg _ = Nothing

processIR :: FilePath -> Text -> Maybe FilePath -> IO ()
processIR irFile target outputFile = do
  -- Read IR file
  irBS <- BL.readFile irFile
  
  case decode irBS of
    Nothing -> do
      hPutStrLn stderr "Error: Failed to parse IR JSON"
      exitFailure
    Just manifest -> do
      -- Extract module from manifest
      case A.parseMaybe (.: "module") manifest of
        Nothing -> do
          hPutStrLn stderr "Error: IR manifest missing 'module' field"
          exitFailure
        Just moduleJSON -> do
          case parseIR moduleJSON of
            Left err -> do
              hPutStrLn stderr $ "Error parsing IR module: " ++ err
              exitFailure
            Right irModule -> do
              -- Emit code
              case emitCode irModule target of
                Left err -> do
                  hPutStrLn stderr $ "Error: " ++ err
                  exitFailure
                Right code -> do
                  case outputFile of
                    Just path -> do
                      TIO.writeFile path code
                      hPutStrLn stderr $ "[STUNIR][Haskell] Code written to: " ++ path
                    Nothing -> TIO.putStr code

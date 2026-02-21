{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}
module Stunir.Toolchain where

import GHC.Generics
import Data.Aeson
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import qualified Data.ByteString.Lazy as BL
import qualified Data.Map.Strict as Map
import System.Directory (doesFileExist)
import Control.Monad (forM_, when)
import System.Exit (exitFailure)
import Crypto.Hash.SHA256 (hash)
import qualified Data.ByteString.Base16 as Hex
import qualified Data.Text.Encoding as TE

-- | Tool definition in lockfile
data ToolDef = ToolDef {
    td_path :: Text,
    td_sha256 :: Text,
    td_version :: Text
} deriving (Show, Eq, Generic)

instance FromJSON ToolDef where
    parseJSON = withObject "ToolDef" $ \v -> ToolDef
        <$> v .: "path"
        <*> v .: "sha256"
        <*> v .: "version"

-- | The Lockfile is a Map of ToolName -> ToolDef
type ToolchainLock = Map.Map Text ToolDef

-- | Helper to compute SHA256 of a file
hashFile :: FilePath -> IO Text
hashFile path = do
    content <- BL.readFile path
    let h = hash (BL.toStrict content)
    return $ TE.decodeUtf8 (Hex.encode h)

-- | Verify the toolchain
verifyToolchain :: FilePath -> IO ()
verifyToolchain lockPath = do
    exists <- doesFileExist lockPath
    if not exists then do
        putStrLn $ "Error: Lockfile not found at " ++ lockPath
        exitFailure
    else do
        content <- BL.readFile lockPath
        case decode content :: Maybe ToolchainLock of
            Nothing -> do
                putStrLn "Error: Failed to parse toolchain lockfile."
                exitFailure
            Just tools -> do
                putStrLn $ "Verifying " ++ show (Map.size tools) ++ " tools from " ++ lockPath

                -- Iterate over tools
                mapM_ verifyTool (Map.toList tools)

                putStrLn "Toolchain verification passed."

verifyTool :: (Text, ToolDef) -> IO ()
verifyTool (name, def) = do
    let path = T.unpack (td_path def)
    let expectedHash = td_sha256 def

    -- 1. Check Existence
    exists <- doesFileExist path
    if not exists then do
        putStrLn $ "Error: Tool '" ++ T.unpack name ++ "' not found at " ++ path
        exitFailure
    else do
        -- 2. Check Hash
        -- Note: In a real large-scale build, we might skip hashing for performance unless strict mode is on.
        -- For STUNIR, correctness is paramount, so we hash.
        actualHash <- hashFile path

        if actualHash /= expectedHash then do
            putStrLn $ "Error: Hash mismatch for tool '" ++ T.unpack name ++ "'"
            putStrLn $ "  Path: " ++ path
            putStrLn $ "  Expected: " ++ T.unpack expectedHash
            putStrLn $ "  Actual:   " ++ T.unpack actualHash
            exitFailure
        else do
            putStrLn $ "  [OK] " ++ T.unpack name

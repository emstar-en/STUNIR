{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
-- |
-- Module      : Stunir.Manifest
-- Description : Deterministic IR Bundle Manifest Generation
-- 
-- This module implements Issue native/haskell/1205:
-- Generates deterministic IR bundle manifests with SHA256 hashes
-- and alphabetically sorted keys for reproducible builds.

module Stunir.Manifest (
    IrManifest(..),
    ManifestEntry(..),
    generateIrManifest,
    writeIrManifest,
    computeFileHash
) where

import GHC.Generics
import Data.Aeson
import Data.Aeson.Types (Pair)
import qualified Data.Aeson.Encode.Pretty as AP
import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as BSL
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import qualified Data.Text.IO as TIO
import Crypto.Hash.SHA256 (hash)
import qualified Data.ByteString.Base16 as Hex
import System.Directory (listDirectory, doesFileExist, doesDirectoryExist, createDirectoryIfMissing)
import System.FilePath ((</>), takeExtension, takeFileName)
import Control.Monad (filterM)
import Data.List (sort)
import Data.Time.Clock.POSIX (getPOSIXTime)

-- | A single file entry in the manifest
data ManifestEntry = ManifestEntry {
    me_filename :: Text,        -- ^ File name (without path)
    me_sha256 :: Text,          -- ^ SHA256 hash of file contents
    me_size :: Int              -- ^ File size in bytes
} deriving (Show, Eq, Generic)

-- | Custom ToJSON for deterministic output
instance ToJSON ManifestEntry where
    toJSON ManifestEntry{..} = object $ sort [
        "filename" .= me_filename,
        "sha256" .= me_sha256,
        "size" .= me_size
        ]

-- | The IR bundle manifest
data IrManifest = IrManifest {
    im_schema :: Text,          -- ^ Schema identifier
    im_version :: Text,         -- ^ Manifest version
    im_generated_epoch :: Integer,  -- ^ Unix timestamp of generation
    im_ir_count :: Int,         -- ^ Number of IR files
    im_files :: [ManifestEntry] -- ^ List of IR file entries (sorted by filename)
} deriving (Show, Eq, Generic)

-- | Custom ToJSON for deterministic output with sorted keys
instance ToJSON IrManifest where
    toJSON IrManifest{..} = object $ sort [
        "files" .= im_files,
        "generated_epoch" .= im_generated_epoch,
        "ir_count" .= im_ir_count,
        "schema" .= im_schema,
        "version" .= im_version
        ]

-- | Compute SHA256 hash of a file, returning hex-encoded text
computeFileHash :: FilePath -> IO Text
computeFileHash path = do
    contents <- BS.readFile path
    let hashBytes = hash contents
    return $ TE.decodeUtf8 $ Hex.encode hashBytes

-- | Generate IR manifest from a directory
-- Files are sorted alphabetically for deterministic output
generateIrManifest :: FilePath -> IO IrManifest
generateIrManifest irDir = do
    dirExists <- doesDirectoryExist irDir
    
    entries <- if dirExists
        then do
            allFiles <- listDirectory irDir
            -- Filter to .dcbor files only
            let dcborFiles = filter (\f -> takeExtension f == ".dcbor") allFiles
            -- Sort alphabetically for determinism
            let sortedFiles = sort dcborFiles
            mapM (createEntry irDir) sortedFiles
        else return []
    
    epoch <- round <$> getPOSIXTime
    
    return IrManifest {
        im_schema = "stunir.ir_manifest.v2",
        im_version = "1.0.0",
        im_generated_epoch = epoch,
        im_ir_count = length entries,
        im_files = entries
    }

-- | Create a manifest entry for a single file
createEntry :: FilePath -> FilePath -> IO ManifestEntry
createEntry baseDir filename = do
    let fullPath = baseDir </> filename
    sha256 <- computeFileHash fullPath
    contents <- BS.readFile fullPath
    return ManifestEntry {
        me_filename = T.pack filename,
        me_sha256 = sha256,
        me_size = BS.length contents
    }

-- | Pretty print config for canonical JSON output
prettyConfig :: AP.Config
prettyConfig = AP.defConfig {
    AP.confIndent = AP.Spaces 2,
    AP.confCompare = compare  -- Alphabetical key ordering
}

-- | Write manifest to file in canonical JSON format
writeIrManifest :: FilePath -> IrManifest -> IO ()
writeIrManifest outPath manifest = do
    createDirectoryIfMissing True (takeDirectory outPath)
    BSL.writeFile outPath (AP.encodePretty' prettyConfig manifest)
  where
    takeDirectory p = 
        let parts = T.splitOn "/" (T.pack p)
        in T.unpack $ T.intercalate "/" (init parts)

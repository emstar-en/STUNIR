{-# LANGUAGE OverloadedStrings #-}

{-|
Module      : Test.Utils
Description : Testing utilities for STUNIR conformance tests
Copyright   : (c) STUNIR Authors, 2026
License     : MIT

Common utilities for STUNIR conformance testing.
-}

module Test.Utils
    ( -- * File Operations
      withTempDir
    , withTempFile
    , writeTestFile
    , readTestFile
      -- * JSON Helpers
    , parseJsonFile
    , parseJsonText
    , jsonField
    , jsonArrayLength
    , canonicalJsonText
      -- * Hashing
    , sha256Text
      -- * Test Data
    , sampleIR
    , sampleSpec
    , sampleManifest
      -- * Path Utilities
    , projectRoot
    , testDataDir
    ) where

import Control.Exception (bracket)
import qualified Crypto.Hash.SHA256 as SHA256
import Data.Aeson (Value(..), decode, encode, (.:), (.=), object)
import Data.Aeson.Types (parseMaybe)
import Data.Aeson.Encode.Pretty (encodePretty', defConfig, Config(..), Indent(..))
import qualified Data.Aeson.Key as Key
import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as BL
import qualified Data.ByteString.Base16 as B16
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import qualified Data.Text.Encoding as TE
import qualified Data.Text.Lazy as TL
import qualified Data.Text.Lazy.Encoding as TLE
import Data.Text (Text)
import qualified Data.Vector as V
import System.Directory (createDirectoryIfMissing, removeDirectoryRecursive,
                         getTemporaryDirectory, doesDirectoryExist)
import System.FilePath ((</>))
import System.IO (hClose, openTempFile)

-- | Compute SHA256 hash of text, returning hex string
sha256Text :: Text -> Text
sha256Text txt = TE.decodeUtf8 $ B16.encode $ SHA256.hash $ TE.encodeUtf8 txt

-- | Convert JSON value to canonical JSON text (sorted keys, no extra whitespace)
canonicalJsonText :: Value -> Text
canonicalJsonText v = TL.toStrict $ TLE.decodeUtf8 $ encodePretty' config v
  where
    config = defConfig
        { confIndent = Spaces 0
        , confCompare = compare
        }

-- | Run action with temporary directory, cleanup afterwards
withTempDir :: String -> (FilePath -> IO a) -> IO a
withTempDir prefix action = bracket setup cleanup action
  where
    setup = do
        tmpBase <- getTemporaryDirectory
        let tmpDir = tmpBase </> prefix
        createDirectoryIfMissing True tmpDir
        return tmpDir
    cleanup dir = do
        exists <- doesDirectoryExist dir
        when exists $ removeDirectoryRecursive dir
    when True a = a
    when False _ = return ()

-- | Run action with temporary file
withTempFile :: String -> (FilePath -> IO a) -> IO a
withTempFile prefix action = do
    tmpDir <- getTemporaryDirectory
    (path, handle) <- openTempFile tmpDir prefix
    hClose handle
    action path

-- | Write text to a test file
writeTestFile :: FilePath -> Text -> IO ()
writeTestFile = TIO.writeFile

-- | Read text from a test file
readTestFile :: FilePath -> IO Text
readTestFile = TIO.readFile

-- | Parse JSON from file
parseJsonFile :: FilePath -> IO (Maybe Value)
parseJsonFile path = decode <$> BL.readFile path

-- | Parse JSON from text
parseJsonText :: Text -> Maybe Value
parseJsonText = decode . BL.fromStrict . TE.encodeUtf8

-- | Extract field from JSON object
jsonField :: Text -> Value -> Maybe Value
jsonField key (Object obj) = parseMaybe (.: Key.fromText key) obj
jsonField _ _ = Nothing

-- | Get length of JSON array
jsonArrayLength :: Value -> Maybe Int
jsonArrayLength (Array arr) = Just $ V.length arr
jsonArrayLength _ = Nothing

-- | Sample IR for testing
sampleIR :: Value
sampleIR = object
    [ "ir_module" .= ("test_module" :: Text)
    , "ir_schema" .= ("stunir.ir.v1" :: Text)
    , "ir_epoch" .= (1706400000 :: Int)
    , "ir_functions" .= 
        [ object
            [ "name" .= ("add" :: Text)
            , "params" .= 
                [ object ["name" .= ("a" :: Text), "type" .= ("i32" :: Text)]
                , object ["name" .= ("b" :: Text), "type" .= ("i32" :: Text)]
                ]
            , "returns" .= ("i32" :: Text)
            , "body" .= ([] :: [Value])
            ]
        ]
    ]

-- | Sample spec for testing (simplified)
sampleSpec :: Value
sampleSpec = object
    [ "module" .= ("test_spec" :: Text)
    , "version" .= ("1.0.0" :: Text)
    , "functions" .= ([] :: [Value])
    ]

-- | Sample manifest for testing
sampleManifest :: Value
sampleManifest = object
    [ "schema" .= ("stunir.manifest.v1" :: Text)
    , "epoch" .= (1706400000 :: Int)
    , "entries" .= ([] :: [Value])
    ]

-- | Get project root (relative to test directory)
projectRoot :: FilePath
projectRoot = "../.."  -- test/haskell -> repo root

-- | Get test data directory
testDataDir :: FilePath
testDataDir = projectRoot </> "test" </> "data"

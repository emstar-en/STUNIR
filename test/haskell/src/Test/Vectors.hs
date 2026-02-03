{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{-|
Module      : Test.Vectors
Description : Test vector loading and validation utilities
Copyright   : (c) STUNIR Authors, 2026
License     : MIT

Provides utilities for loading and validating test vectors,
mirroring Python test_vectors/base.py functionality.
-}

module Test.Vectors
    ( -- * Test Vector Types
      TestVector(..)
    , TestVectorSet(..)
      -- * Loading
    , loadTestVector
    , loadTestVectorSet
    , loadVectorFromFile
      -- * Validation
    , validateVector
    , verifyVectorHash
      -- * Generation
    , generateTestId
    , createTestVector
      -- * Constants
    , defaultEpoch
    , testSeed
    ) where

import Data.Aeson
import Data.Aeson.Types (parseMaybe)
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import qualified Data.Text.Encoding as TE
import qualified Data.ByteString.Lazy as BL
import Test.Utils (sha256Text, canonicalJsonText)
import Control.Exception (try, SomeException)

-- | Fixed epoch for determinism (matches Python DEFAULT_EPOCH)
defaultEpoch :: Int
defaultEpoch = 1735500000

-- | Test seed for reproducibility (matches Python TEST_SEED)
testSeed :: Int
testSeed = 42

-- | A single test vector
data TestVector = TestVector
    { tvId          :: !Text
    , tvName        :: !Text
    , tvDescription :: !Text
    , tvSchema      :: !Text
    , tvEpoch       :: !Int
    , tvInput       :: !Value
    , tvExpected    :: !Value
    , tvExpectedHash :: !Text
    , tvTags        :: ![Text]
    } deriving (Show, Eq)

instance FromJSON TestVector where
    parseJSON = withObject "TestVector" $ \o -> TestVector
        <$> o .: "id"
        <*> o .: "name"
        <*> o .: "description"
        <*> o .: "schema"
        <*> o .:? "created_epoch" .!= defaultEpoch
        <*> o .: "input"
        <*> o .: "expected_output"
        <*> o .: "expected_hash"
        <*> o .:? "tags" .!= []

instance ToJSON TestVector where
    toJSON TestVector{..} = object
        [ "id" .= tvId
        , "name" .= tvName
        , "description" .= tvDescription
        , "schema" .= tvSchema
        , "created_epoch" .= tvEpoch
        , "input" .= tvInput
        , "expected_output" .= tvExpected
        , "expected_hash" .= tvExpectedHash
        , "tags" .= tvTags
        ]

-- | A set of test vectors with metadata
data TestVectorSet = TestVectorSet
    { tvsCategory :: !Text
    , tvsSchema   :: !Text
    , tvsVectors  :: ![TestVector]
    } deriving (Show, Eq)

-- | Load a single test vector from JSON text
loadTestVector :: Text -> Maybe TestVector
loadTestVector jsonText = decode (BL.fromStrict $ TE.encodeUtf8 jsonText)

-- | Load a test vector set from a directory
loadTestVectorSet :: Text -> [Text] -> TestVectorSet
loadTestVectorSet category vectors = TestVectorSet
    { tvsCategory = category
    , tvsSchema = "stunir.test_vector." <> category <> ".v1"
    , tvsVectors = mapMaybe loadTestVector vectors
    }
  where
    mapMaybe f = foldr (\x acc -> case f x of Just y -> y:acc; Nothing -> acc) []

-- | Load a test vector from a file path
loadVectorFromFile :: FilePath -> IO (Maybe TestVector)
loadVectorFromFile path = do
    result <- try (BL.readFile path) :: IO (Either SomeException BL.ByteString)
    return $ case result of
        Right bytes -> decode bytes
        Left _ -> Nothing

-- | Validate a test vector structure
validateVector :: TestVector -> (Bool, [Text])
validateVector tv =
    let errors = filter (not . T.null)
            [ if T.null (tvId tv) then "Missing id" else ""
            , if T.null (tvSchema tv) then "Missing schema" else ""
            , if T.length (tvExpectedHash tv) /= 64 then "Invalid hash length" else ""
            ]
    in (null errors, errors)

-- | Verify a test vector's expected hash
verifyVectorHash :: TestVector -> Bool
verifyVectorHash tv =
    let inputHash = sha256Text $ canonicalJsonText (tvInput tv)
    in inputHash == tvExpectedHash tv

-- | Generate a unique test ID
generateTestId :: Text -> Int -> Text
generateTestId category index =
    "tv_" <> category <> "_" <> T.pack (padLeft 3 '0' (show index))
  where
    padLeft n c s = replicate (n - length s) c ++ s

-- | Create a new test vector
createTestVector :: Text -> Text -> Text -> Value -> Value -> TestVector
createTestVector category name desc input expected = TestVector
    { tvId = generateTestId category 1
    , tvName = name
    , tvDescription = desc
    , tvSchema = "stunir.test_vector." <> category <> ".v1"
    , tvEpoch = defaultEpoch
    , tvInput = input
    , tvExpected = expected
    , tvExpectedHash = sha256Text $ canonicalJsonText input
    , tvTags = [category]
    }

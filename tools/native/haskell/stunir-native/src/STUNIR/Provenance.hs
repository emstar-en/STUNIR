{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
-- |
-- Module      : Stunir.Provenance
-- Description : Standardized C Provenance Template Management
--
-- This module implements Issue provenance/1401:
-- Provides standardized C header generation for provenance tracking.

module Stunir.Provenance (
    Provenance(..),
    generateCHeader,
    generateCHeaderExtended
) where

import GHC.Generics
import Data.Aeson
import Data.Text (Text)
import qualified Data.Text as T
import Stunir.Spec (Spec(..), SpecModule(..))

-- | Provenance data including build epoch, hashes, and module list
data Provenance = Provenance {
    prov_epoch :: Integer,         -- ^ Build epoch timestamp
    prov_spec_sha256 :: Text,      -- ^ SHA256 of spec.json
    prov_modules :: [Text],        -- ^ List of module names
    prov_asm_sha256 :: Maybe Text, -- ^ Optional ASM bundle digest
    prov_schema :: Text            -- ^ Schema version
} deriving (Show, Eq, Generic)

instance ToJSON Provenance where
    toJSON Provenance{..} = object [
        "schema" .= prov_schema,
        "epoch" .= prov_epoch,
        "spec_sha256" .= prov_spec_sha256,
        "asm_sha256" .= prov_asm_sha256,
        "modules" .= prov_modules,
        "module_count" .= length prov_modules
        ]

-- | Generate minimal C header for backward compatibility
generateCHeader :: Provenance -> Text
generateCHeader p = T.unlines [
    "/* STUNIR Provenance Header - Auto-generated */",
    "#ifndef STUNIR_PROVENANCE_H",
    "#define STUNIR_PROVENANCE_H",
    "",
    "#define STUNIR_PROV_BUILD_EPOCH " <> T.pack (show (prov_epoch p)),
    "#define STUNIR_PROV_SPEC_DIGEST \"" <> prov_spec_sha256 p <> "\"",
    "#define STUNIR_PROV_ASM_DIGEST \"" <> maybe "" id (prov_asm_sha256 p) <> "\"",
    "#define STUNIR_PROV_MODULE_COUNT " <> T.pack (show (length $ prov_modules p)),
    "",
    "/* Legacy aliases */",
    "#define STUNIR_EPOCH STUNIR_PROV_BUILD_EPOCH",
    "#define STUNIR_SPEC_SHA256 STUNIR_PROV_SPEC_DIGEST",
    "",
    "#endif /* STUNIR_PROVENANCE_H */"
    ]

-- | Generate extended C header with module definitions (Issue: provenance/1401)
generateCHeaderExtended :: Provenance -> Text
generateCHeaderExtended p = T.unlines $ [
    "/* STUNIR Provenance Header - Extended Template */",
    "/* Schema: " <> prov_schema p <> " */",
    "#ifndef STUNIR_PROVENANCE_H",
    "#define STUNIR_PROVENANCE_H",
    "",
    "/* Core Provenance Macros */",
    "#define STUNIR_PROV_SCHEMA \"" <> prov_schema p <> "\"",
    "#define STUNIR_PROV_BUILD_EPOCH " <> T.pack (show (prov_epoch p)),
    "#define STUNIR_PROV_SPEC_DIGEST \"" <> prov_spec_sha256 p <> "\"",
    "#define STUNIR_PROV_ASM_DIGEST \"" <> maybe "PENDING" id (prov_asm_sha256 p) <> "\"",
    "#define STUNIR_PROV_MODULE_COUNT " <> T.pack (show (length $ prov_modules p)),
    "",
    "/* Module List */",
    "#define STUNIR_PROV_MODULES \\"
    ] ++ moduleLines ++ [
    "",
    "/* Legacy Compatibility */",
    "#ifndef _STUNIR_BUILD_EPOCH",
    "#define _STUNIR_BUILD_EPOCH STUNIR_PROV_BUILD_EPOCH",
    "#endif",
    "",
    "#endif /* STUNIR_PROVENANCE_H */"
    ]
  where
    moduleLines = case prov_modules p of
        [] -> ["    \"(no modules)\""]
        ms -> zipWith formatModule [0..] ms
    formatModule :: Int -> Text -> Text
    formatModule i m = "    \"" <> m <> "\"" <> 
        (if i < length (prov_modules p) - 1 then ", \\" else "")

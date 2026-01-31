{-# LANGUAGE OverloadedStrings #-}

{-|
Module      : STUNIR.SemanticIR.Emitters.CodeGen
Description : Code generation utilities
Copyright   : (c) STUNIR Team, 2026
License     : MIT
Maintainer  : stunir@example.com

Common code generation utilities for all emitters.
-}

module STUNIR.SemanticIR.Emitters.CodeGen
  ( -- * Function Generation
    generateFunctionSignature
  , generateFunctionBody
  , -- * Type Mapping
    mapTypeToLanguage
  , -- * Comment Generation
    formatComment
  , CommentStyle(..)
  , -- * String Utilities
    escapeString
  , wrapLine
  , joinLines
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import Data.List (intersperse)
import STUNIR.SemanticIR.Emitters.Types

-- | Comment style for different languages
data CommentStyle
  = CommentC        -- ^ /* ... */
  | CommentCPP      -- ^ //
  | CommentPython   -- ^ #
  | CommentLisp     -- ^ ;
  | CommentHaskell  -- ^ --
  deriving (Eq, Show)

-- | Generate function signature for a language
generateFunctionSignature
  :: Text                    -- ^ Function name
  -> [(Text, Text)]          -- ^ Parameters (name, type)
  -> Text                    -- ^ Return type
  -> Text                    -- ^ Language ("c", "rust", "python", etc.)
  -> Text
generateFunctionSignature name params returnType lang
  | lang == "c" || lang == "c99" || lang == "c89" =
      returnType <> " " <> name <> "(" <> formatParams params ", " <> ")"
  | lang == "rust" =
      "fn " <> name <> "(" <> formatParamsRust params <> ") -> " <> returnType
  | lang == "python" =
      "def " <> name <> "(" <> formatParamsPython params <> "):"
  | lang == "haskell" =
      name <> " :: " <> T.intercalate " -> " (map snd params ++ [returnType])
  | otherwise =
      name <> "(" <> formatParams params ", " <> ")"
  where
    formatParams [] _ = ""
    formatParams ps sep =
      T.intercalate sep [pType <> " " <> pName | (pName, pType) <- ps]
    
    formatParamsRust ps =
      T.intercalate ", " [pName <> ": " <> pType | (pName, pType) <- ps]
    
    formatParamsPython ps =
      T.intercalate ", " [pName | (pName, _) <- ps]

-- | Generate function body stub
generateFunctionBody :: Text -> [Text] -> Text
generateFunctionBody indent bodyLines =
  T.unlines [indent <> line | line <- bodyLines]

-- | Map IR data type to target language type
mapTypeToLanguage :: IRDataType -> Text -> Text
mapTypeToLanguage irType lang
  | lang == "c" || lang == "c99" || lang == "c89" = mapIRTypeToC irType
  | lang == "rust" = mapIRTypeToRust irType
  | lang == "python" = mapIRTypeToPython irType
  | otherwise = T.pack $ show irType

-- | Format comment in specified style
formatComment :: Text -> CommentStyle -> [Text]
formatComment text style =
  case style of
    CommentC ->
      ["/*"] ++ map (" * " <>) (T.lines text) ++ [" */"]
    CommentCPP ->
      map ("// " <>) (T.lines text)
    CommentPython ->
      map ("# " <>) (T.lines text)
    CommentLisp ->
      map ("; " <>) (T.lines text)
    CommentHaskell ->
      map ("-- " <>) (T.lines text)

-- | Escape string for target language
escapeString :: Text -> Text -> Text
escapeString str lang
  | lang == "c" || lang == "c99" || lang == "c89" =
      "\"" <> T.replace "\"" "\\\"" str <> "\""
  | lang == "rust" =
      "\"" <> T.replace "\"" "\\\"" str <> "\""
  | lang == "python" =
      "\"" <> T.replace "\"" "\\\"" str <> "\""
  | otherwise =
      "\"" <> str <> "\""

-- | Wrap line to maximum length
wrapLine :: Int -> Text -> [Text]
wrapLine maxLen text
  | T.length text <= maxLen = [text]
  | otherwise =
      let (prefix, rest) = T.splitAt maxLen text
      in prefix : wrapLine maxLen rest

-- | Join lines with separator
joinLines :: Text -> [Text] -> Text
joinLines sep = T.intercalate sep

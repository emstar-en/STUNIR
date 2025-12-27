{-# LANGUAGE OverloadedStrings #-}

module STUNIR.JCS
  ( canonicalize
  ) where

import qualified Data.Aeson as A
import qualified Data.Aeson.Key as Key
import qualified Data.Aeson.KeyMap as KM
import qualified Data.ByteString as BS
import qualified Data.ByteString.Builder as B
import qualified Data.ByteString.Lazy as LBS
import qualified Data.List as L
import qualified Data.Scientific as Sci
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import qualified Data.Vector as V

canonicalize :: A.Value -> Either String BS.ByteString
canonicalize v = do
  b <- goE v
  pure (LBS.toStrict (B.toLazyByteString b))

-- Determinism policy: numbers must be integers.
goE :: A.Value -> Either String B.Builder
goE v = case v of
  A.Null -> Right (B.string7 "null")
  A.Bool True -> Right (B.string7 "true")
  A.Bool False -> Right (B.string7 "false")
  A.Number n ->
    case Sci.floatingOrInteger n :: Either Double Integer of
      Left _ -> Left "bad_digest"
      Right i -> Right (B.string7 (show i))
  A.String s -> Right (buildString s)
  A.Array arr -> do
    xs <- mapM goE (V.toList arr)
    Right (B.char7 '[' <> joinWithComma xs <> B.char7 ']')
  A.Object obj -> do
    let kvs = KM.toList obj
        kvsSorted = L.sortOn (TE.encodeUtf8 . Key.toText . fst) kvs
    pairs <- mapM (\(k,val) -> do
      vb <- goE val
      Right (buildString (Key.toText k) <> B.char7 ':' <> vb)
      ) kvsSorted
    Right (B.char7 '{' <> joinWithComma pairs <> B.char7 '}')

joinWithComma :: [B.Builder] -> B.Builder
joinWithComma bs = case bs of
  [] -> mempty
  (x:xs) -> foldl (\acc y -> acc <> B.char7 ',' <> y) x xs

buildString :: T.Text -> B.Builder
buildString t =
  let s = T.unpack t
  in B.char7 '"' <> mconcat (map esc s) <> B.char7 '"'

esc :: Char -> B.Builder
esc c = case c of
  '"' -> B.string7 "\\\""
  '\\' -> B.string7 "\\\\"
  '\b' -> B.string7 "\\b"
  '\f' -> B.string7 "\\f"
  '\n' -> B.string7 "\\n"
  '\r' -> B.string7 "\\r"
  '\t' -> B.string7 "\\t"
  _ | fromEnum c <= 0x1F ->
      let hex4 = pad4 (showHex (fromEnum c))
      in B.string7 "\\u" <> B.string7 hex4
    | otherwise -> TE.encodeUtf8Builder (T.singleton c)

showHex :: Int -> String
showHex n =
  let digits = "0123456789abcdef"
      go' 0 acc = acc
      go' x acc =
        let (q,r) = x `divMod` 16
        in go' q ((digits !! r) : acc)
  in if n == 0 then "0" else go' n ""

pad4 :: String -> String
pad4 s = replicate (4 - length s) '0' ++ s

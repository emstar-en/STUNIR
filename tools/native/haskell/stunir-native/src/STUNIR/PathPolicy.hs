module STUNIR.PathPolicy
  ( checkRelpathSafe
  , isScopeExcluded
  ) where

import qualified Data.Char as C

checkRelpathSafe :: String -> Either String ()
checkRelpathSafe p
  | null p = Left "unsafe_filename"
  | head p == '/' = Left "unsafe_filename"
  | take 2 p == "./" = Left "unsafe_filename"
  | any (== '\\') p = Left "unsafe_filename"
  | any C.isSpace p = Left "unsafe_filename"
  | any (not . isAllowed) p = Left "unsafe_filename"
  | any null (splitOnSlash p) = Left "unsafe_filename"
  | any (== ".") (splitOnSlash p) || any (== "..") (splitOnSlash p) = Left "unsafe_filename"
  | any startsWithDash (splitOnSlash p) = Left "unsafe_filename"
  | otherwise = Right ()
  where
    isAllowed ch = C.isAsciiAlphaNum ch || ch == '.' || ch == '_' || ch == '-' || ch == '/'
    splitOnSlash s = case break (== '/') s of
      (a, []) -> [a]
      (a, _:rest) -> a : splitOnSlash rest
    startsWithDash s = not (null s) && head s == '-'

isScopeExcluded :: String -> Bool
isScopeExcluded p = p == "pack_manifest.tsv" || take (length "objects/sha256/") p == "objects/sha256/"

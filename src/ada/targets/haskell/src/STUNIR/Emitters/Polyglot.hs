{-# LANGUAGE OverloadedStrings #-}

-- | Polyglot code emitters (C, C++, Rust)
module STUNIR.Emitters.Polyglot
  ( emitC89
  , emitC99
  , emitRust
  , PolyglotLanguage(..)
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import STUNIR.Emitters.Types

-- | Polyglot language
data PolyglotLanguage
  = C89
  | C99
  | C11
  | Rust
  | CPlusPlus
  deriving (Show, Eq)

-- | Emit C89 code
emitC89 :: Text -> EmitterResult Text
emitC89 moduleName = Right $ T.unlines
  [ "/* STUNIR Generated C89 */"
  , "/* Module: " <> moduleName <> " */"
  , "/* Generator: Haskell Pipeline */"
  , ""
  , "#include <stdio.h>"
  , ""
  , "typedef long int32_t;"
  , "typedef unsigned long uint32_t;"
  , ""
  , "int main(void) {"
  , "    printf(\"STUNIR Generated\\n\");"
  , "    return 0;"
  , "}"
  ]

-- | Emit C99 code
emitC99 :: Text -> EmitterResult Text
emitC99 moduleName = Right $ T.unlines
  [ "/* STUNIR Generated C99 */"
  , "/* Module: " <> moduleName <> " */"
  , "/* Generator: Haskell Pipeline */"
  , ""
  , "#include <stdint.h>"
  , "#include <stdbool.h>"
  , "#include <stdio.h>"
  , ""
  , "int main(void) {"
  , "    printf(\"STUNIR Generated\\n\");"
  , "    return 0;"
  , "}"
  ]

-- | Emit Rust code
emitRust :: Text -> EmitterResult Text
emitRust moduleName = Right $ T.unlines
  [ "// STUNIR Generated Rust"
  , "// Module: " <> moduleName
  , "// Generator: Haskell Pipeline"
  , ""
  , "fn main() {"
  , "    println!(\"STUNIR Generated\");"
  , "}"
  , ""
  , "#[cfg(test)]"
  , "mod tests {"
  , "    #[test]"
  , "    fn it_works() {"
  , "        assert_eq!(2 + 2, 4);"
  , "    }"
  , "}"
  ]

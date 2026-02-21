{-# LANGUAGE OverloadedStrings #-}

-- | Object-Oriented Programming emitters
module STUNIR.Emitters.OOP
  ( emitJava
  , emitCPlusPlus
  , emitCSharp
  , OOPLanguage(..)
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import STUNIR.Emitters.Types

-- | OOP language
data OOPLanguage
  = Java
  | CPlusPlus
  | CSharp
  | TypeScript
  deriving (Show, Eq)

-- | Emit Java code
emitJava :: Text -> EmitterResult Text
emitJava className = Right $ T.unlines
  [ "// STUNIR Generated Java"
  , "// Class: " <> className
  , "// Generator: Haskell Pipeline"
  , ""
  , "public class " <> className <> " {"
  , "    private int value;"
  , "    "
  , "    public " <> className <> "(int value) {"
  , "        this.value = value;"
  , "    }"
  , "    "
  , "    public int getValue() {"
  , "        return value;"
  , "    }"
  , "    "
  , "    public void setValue(int value) {"
  , "        this.value = value;"
  , "    }"
  , "}"
  ]

-- | Emit C++ code
emitCPlusPlus :: Text -> EmitterResult Text
emitCPlusPlus className = Right $ T.unlines
  [ "// STUNIR Generated C++"
  , "// Class: " <> className
  , "// Generator: Haskell Pipeline"
  , ""
  , "#include <iostream>"
  , ""
  , "class " <> className <> " {"
  , "private:"
  , "    int value;"
  , ""
  , "public:"
  , "    " <> className <> "(int val) : value(val) {}"
  , "    "
  , "    int getValue() const {"
  , "        return value;"
  , "    }"
  , "    "
  , "    void setValue(int val) {"
  , "        value = val;"
  , "    }"
  , "};"
  ]

-- | Emit C# code
emitCSharp :: Text -> EmitterResult Text
emitCSharp className = Right $ T.unlines
  [ "// STUNIR Generated C#"
  , "// Class: " <> className
  , "// Generator: Haskell Pipeline"
  , ""
  , "public class " <> className
  , "{"
  , "    private int value;"
  , "    "
  , "    public " <> className <> "(int value)"
  , "    {"
  , "        this.value = value;"
  , "    }"
  , "    "
  , "    public int Value"
  , "    {"
  , "        get { return value; }"
  , "        set { this.value = value; }"
  , "    }"
  , "}"
  ]

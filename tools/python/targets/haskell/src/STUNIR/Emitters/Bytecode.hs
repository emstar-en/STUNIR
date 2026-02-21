{-# LANGUAGE OverloadedStrings #-}

-- | Bytecode emitters
module STUNIR.Emitters.Bytecode
  ( emitJVMBytecode
  , emitDotNetIL
  , BytecodeFormat(..)
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import STUNIR.Emitters.Types

-- | Bytecode format
data BytecodeFormat
  = JVM
  | DotNetIL
  | PythonBytecode
  deriving (Show, Eq)

-- | Emit JVM bytecode (Jasmin format)
emitJVMBytecode :: Text -> EmitterResult Text
emitJVMBytecode className = Right $ T.unlines
  [ "; STUNIR Generated JVM Bytecode"
  , "; Class: " <> className
  , "; Generator: Haskell Pipeline"
  , ""
  , ".class public " <> className
  , ".super java/lang/Object"
  , ""
  , ".method public <init>()V"
  , "    .limit stack 1"
  , "    .limit locals 1"
  , "    aload_0"
  , "    invokespecial java/lang/Object/<init>()V"
  , "    return"
  , ".end method"
  , ""
  , ".method public static main([Ljava/lang/String;)V"
  , "    .limit stack 2"
  , "    .limit locals 1"
  , "    getstatic java/lang/System/out Ljava/io/PrintStream;"
  , "    ldc \"STUNIR Generated\""
  , "    invokevirtual java/io/PrintStream/println(Ljava/lang/String;)V"
  , "    return"
  , ".end method"
  ]

-- | Emit .NET IL bytecode
emitDotNetIL :: Text -> EmitterResult Text
emitDotNetIL className = Right $ T.unlines
  [ "// STUNIR Generated .NET IL"
  , "// Class: " <> className
  , "// Generator: Haskell Pipeline"
  , ""
  , ".assembly extern mscorlib {}"
  , ".assembly " <> className <> " {}"
  , ""
  , ".class public auto ansi beforefieldinit " <> className
  , "       extends [mscorlib]System.Object"
  , "{"
  , "    .method public hidebysig static void Main() cil managed"
  , "    {"
  , "        .entrypoint"
  , "        .maxstack 1"
  , "        ldstr \"STUNIR Generated\""
  , "        call void [mscorlib]System.Console::WriteLine(string)"
  , "        ret"
  , "    }"
  , "}"
  ]

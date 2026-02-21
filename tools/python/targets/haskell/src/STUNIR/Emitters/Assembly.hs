{-# LANGUAGE OverloadedStrings #-}

-- | Assembly code emitters
module STUNIR.Emitters.Assembly
  ( emitARM
  , emitX86
  , AssemblyFlavor(..)
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import STUNIR.Emitters.Types

-- | Assembly flavor
data AssemblyFlavor
  = ARM32
  | ARM64
  | X86_32
  | X86_64
  deriving (Show, Eq)

-- | Emit ARM assembly
emitARM :: Text -> EmitterResult Text
emitARM moduleName = Right $ T.unlines
  [ "@ STUNIR Generated ARM Assembly"
  , "@ Module: " <> moduleName
  , "@ Generator: Haskell Pipeline"
  , ""
  , "    .text"
  , "    .global _start"
  , ""
  , "_start:"
  , "    mov r0, #42"
  , "    mov r7, #1"
  , "    swi 0"
  ]

-- | Emit x86 assembly
emitX86 :: Text -> EmitterResult Text
emitX86 moduleName = Right $ T.unlines
  [ "; STUNIR Generated x86 Assembly"
  , "; Module: " <> moduleName
  , "; Generator: Haskell Pipeline"
  , ""
  , "section .text"
  , "    global _start"
  , ""
  , "_start:"
  , "    mov eax, 1"
  , "    mov ebx, 42"
  , "    int 0x80"
  ]

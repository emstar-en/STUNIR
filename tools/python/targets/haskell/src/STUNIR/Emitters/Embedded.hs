{-# LANGUAGE OverloadedStrings #-}

-- | Embedded systems code emitters
module STUNIR.Emitters.Embedded
  ( emitEmbedded
  , EmbeddedArch(..)
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import STUNIR.Emitters.Types

-- | Embedded architecture
data EmbeddedArch
  = CortexM
  | AVRTiny
  | RISCV32
  deriving (Show, Eq)

-- | Emit embedded code
emitEmbedded :: EmbeddedArch -> Text -> EmitterResult Text
emitEmbedded arch moduleName = Right $ T.unlines $
  [ "/* STUNIR Generated Embedded C */"
  , "/* Module: " <> moduleName <> " */"
  , "/* Architecture: " <> T.pack (show arch) <> " */"
  , "/* Generator: Haskell Pipeline */"
  , ""
  , "#include <stdint.h>"
  , ""
  ] ++ archSpecific arch ++
  [ ""
  , "int main(void) {"
  , "    while(1) {"
  , "        // Main loop"
  , "    }"
  , "    return 0;"
  , "}"
  ]
  where
    archSpecific CortexM =
      [ "// ARM Cortex-M specific"
      , "#define NVIC_BASE 0xE000E100"
      , "volatile uint32_t *const NVIC_ISER = (uint32_t*)NVIC_BASE;"
      ]
    archSpecific AVRTiny =
      [ "// AVR specific"
      , "#include <avr/io.h>"
      , "#include <avr/interrupt.h>"
      ]
    archSpecific RISCV32 =
      [ "// RISC-V 32 specific"
      , "#define MSTATUS 0x300"
      ]

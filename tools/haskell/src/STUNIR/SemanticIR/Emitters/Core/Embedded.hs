{-# LANGUAGE OverloadedStrings #-}

{-|
Module      : STUNIR.SemanticIR.Emitters.Core.Embedded
Description : Embedded systems emitter (ARM, ARM64, RISC-V, MIPS, AVR, x86)
Copyright   : (c) STUNIR Team, 2026
License     : MIT

Emits bare-metal C code for embedded systems.
Supports ARM, ARM64, RISC-V, MIPS, AVR, and x86 targets.
Based on Ada SPARK embedded_emitter implementation.
-}

module STUNIR.SemanticIR.Emitters.Core.Embedded
  ( EmbeddedEmitter
  , EmbeddedConfig(..)
  , EmbeddedTarget(..)
  , defaultEmbeddedConfig
  , emitEmbedded
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import Data.Maybe (fromMaybe)
import STUNIR.SemanticIR.Emitters.Base
import STUNIR.SemanticIR.Emitters.Types
import STUNIR.SemanticIR.Emitters.CodeGen

-- | Embedded target architectures
data EmbeddedTarget
  = TargetARM
  | TargetARM64
  | TargetRISCV
  | TargetMIPS
  | TargetAVR
  | TargetX86
  deriving (Eq, Show)

-- | Embedded emitter configuration
data EmbeddedConfig = EmbeddedConfig
  { embBaseConfig :: !EmitterConfig
  , embTarget     :: !EmbeddedTarget
  } deriving (Show)

-- | Default embedded configuration
defaultEmbeddedConfig :: FilePath -> Text -> EmbeddedTarget -> EmbeddedConfig
defaultEmbeddedConfig outputDir moduleName target = EmbeddedConfig
  { embBaseConfig = defaultEmitterConfig outputDir moduleName
  , embTarget = target
  }

-- | Embedded emitter
data EmbeddedEmitter = EmbeddedEmitter EmbeddedConfig

instance Emitter EmbeddedEmitter where
  emit (EmbeddedEmitter config) irModule
    | not (validateIR irModule) = Left "Invalid IR module structure"
    | otherwise = Right $ EmitterResult
        { erStatus = Success
        , erFiles = [mainFile]
        , erTotalSize = gfSize mainFile
        , erErrorMessage = Nothing
        }
    where
      mainFile = generateEmbeddedFile config irModule

-- | Generate embedded C file
generateEmbeddedFile :: EmbeddedConfig -> IRModule -> GeneratedFile
generateEmbeddedFile config irModule =
  let content = generateEmbeddedCode config irModule
      fileName = imModuleName irModule <> ".c"
  in GeneratedFile
       { gfPath = T.unpack fileName
       , gfHash = computeFileHash content
       , gfSize = T.length content
       }

-- | Generate embedded C code
generateEmbeddedCode :: EmbeddedConfig -> IRModule -> Text
generateEmbeddedCode config irModule = T.unlines $
  [ getDO178CHeader (ecAddDO178CHeaders baseConfig)
                    ("Embedded Code for " <> targetName (embTarget config))
  ] ++
  generateIncludes ++
  [""] ++
  generateTargetDefines config ++
  [""] ++
  concatMap (generateFunction baseConfig) (imFunctions irModule)
  where
    baseConfig = embBaseConfig config

-- | Generate includes
generateIncludes :: [Text]
generateIncludes =
  [ "#include <stdint.h>"
  , "#include <stdbool.h>"
  , "#include <stddef.h>"
  ]

-- | Generate target-specific defines
generateTargetDefines :: EmbeddedConfig -> [Text]
generateTargetDefines config =
  case embTarget config of
    TargetARM -> ["#define TARGET_ARM 1"]
    TargetARM64 -> ["#define TARGET_ARM64 1"]
    TargetRISCV -> ["#define TARGET_RISCV 1"]
    TargetMIPS -> ["#define TARGET_MIPS 1"]
    TargetAVR -> ["#define TARGET_AVR 1"]
    TargetX86 -> ["#define TARGET_X86 1"]

-- | Generate function code
generateFunction :: EmitterConfig -> IRFunction -> [Text]
generateFunction config func =
  let signature = generateFunctionSignature
                    (ifName func)
                    [(ipName p, mapIRTypeToC (ipType p)) | p <- ifParameters func]
                    (mapIRTypeToC (ifReturnType func))
                    "c"
      indent = indentString (ecIndentSize config) 1
  in [ signature
     , "{"
     , indent <> "/* Function body */"
     , indent <> "/* Generated from STUNIR Semantic IR */"
     ] ++
     (if ifReturnType func /= TypeVoid
      then [indent <> "return 0; /* TODO */"]
      else []) ++
     ["}", ""]

-- | Get target name
targetName :: EmbeddedTarget -> Text
targetName TargetARM = "ARM"
targetName TargetARM64 = "ARM64"
targetName TargetRISCV = "RISC-V"
targetName TargetMIPS = "MIPS"
targetName TargetAVR = "AVR"
targetName TargetX86 = "x86"

-- | Convenience function for direct usage
emitEmbedded
  :: IRModule
  -> FilePath
  -> EmbeddedTarget
  -> Either Text EmitterResult
emitEmbedded irModule outputDir target =
  let config = defaultEmbeddedConfig outputDir (imModuleName irModule) target
      emitter = EmbeddedEmitter config
  in emit emitter irModule

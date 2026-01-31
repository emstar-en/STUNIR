{-# LANGUAGE OverloadedStrings #-}

{-|
Module      : STUNIR.SemanticIR.Emitters.Core.Assembly
Description : Assembly emitter (x86, x86_64, ARM, ARM64)
Copyright   : (c) STUNIR Team, 2026
License     : MIT

Emits assembly code for various architectures.
Supports x86, x86_64, ARM, and ARM64 with multiple syntaxes.
Based on Ada SPARK assembly emitters.
-}

module STUNIR.SemanticIR.Emitters.Core.Assembly
  ( AssemblyEmitter
  , AssemblyConfig(..)
  , AssemblyTarget(..)
  , AssemblySyntax(..)
  , defaultAssemblyConfig
  , emitAssembly
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import STUNIR.SemanticIR.Emitters.Base
import STUNIR.SemanticIR.Emitters.Types
import STUNIR.SemanticIR.Emitters.CodeGen

-- | Assembly target architectures
data AssemblyTarget
  = AsmX86
  | AsmX86_64
  | AsmARM
  | AsmARM64
  deriving (Eq, Show)

-- | Assembly syntax variants
data AssemblySyntax
  = SyntaxIntel    -- ^ Intel syntax (x86/x86_64)
  | SyntaxATT      -- ^ AT&T syntax (x86/x86_64)
  | SyntaxARM      -- ^ ARM assembly syntax
  deriving (Eq, Show)

-- | Assembly emitter configuration
data AssemblyConfig = AssemblyConfig
  { asmBaseConfig :: !EmitterConfig
  , asmTarget     :: !AssemblyTarget
  , asmSyntax     :: !AssemblySyntax
  } deriving (Show)

-- | Default assembly configuration
defaultAssemblyConfig :: FilePath -> Text -> AssemblyTarget -> AssemblySyntax -> AssemblyConfig
defaultAssemblyConfig outputDir moduleName target syntax = AssemblyConfig
  { asmBaseConfig = defaultEmitterConfig outputDir moduleName
  , asmTarget = target
  , asmSyntax = syntax
  }

-- | Assembly emitter
data AssemblyEmitter = AssemblyEmitter AssemblyConfig

instance Emitter AssemblyEmitter where
  emit (AssemblyEmitter config) irModule
    | not (validateIR irModule) = Left "Invalid IR module structure"
    | otherwise = Right $ EmitterResult
        { erStatus = Success
        , erFiles = [mainFile]
        , erTotalSize = gfSize mainFile
        , erErrorMessage = Nothing
        }
    where
      mainFile = generateAssemblyFile config irModule

-- | Generate assembly file
generateAssemblyFile :: AssemblyConfig -> IRModule -> GeneratedFile
generateAssemblyFile config irModule =
  let content = generateAssemblyCode config irModule
      fileName = imModuleName irModule <> ".s"
  in GeneratedFile
       { gfPath = T.unpack fileName
       , gfHash = computeFileHash content
       , gfSize = T.length content
       }

-- | Generate assembly code
generateAssemblyCode :: AssemblyConfig -> IRModule -> Text
generateAssemblyCode config irModule = T.unlines $
  [ "; STUNIR Generated Assembly"
  , "; Target: " <> targetName (asmTarget config)
  , "; Syntax: " <> syntaxName (asmSyntax config)
  , ""
  , ".section .text"
  , ".global _start"
  , ""
  ] ++
  concatMap (generateAssemblyFunction config) (imFunctions irModule)

-- | Generate assembly function
generateAssemblyFunction :: AssemblyConfig -> IRFunction -> [Text]
generateAssemblyFunction config func =
  case asmTarget config of
    AsmX86 -> generateX86Function config func
    AsmX86_64 -> generateX86_64Function config func
    AsmARM -> generateARMFunction config func
    AsmARM64 -> generateARM64Function config func

-- | Generate x86 function
generateX86Function :: AssemblyConfig -> IRFunction -> [Text]
generateX86Function config func =
  case asmSyntax config of
    SyntaxIntel ->
      [ ifName func <> ":"
      , "    push ebp"
      , "    mov ebp, esp"
      , "    ; Function body"
      , "    mov esp, ebp"
      , "    pop ebp"
      , "    ret"
      , ""
      ]
    SyntaxATT ->
      [ ifName func <> ":"
      , "    pushl %ebp"
      , "    movl %esp, %ebp"
      , "    # Function body"
      , "    movl %ebp, %esp"
      , "    popl %ebp"
      , "    ret"
      , ""
      ]
    _ -> []

-- | Generate x86_64 function
generateX86_64Function :: AssemblyConfig -> IRFunction -> [Text]
generateX86_64Function config func =
  case asmSyntax config of
    SyntaxIntel ->
      [ ifName func <> ":"
      , "    push rbp"
      , "    mov rbp, rsp"
      , "    ; Function body"
      , "    mov rsp, rbp"
      , "    pop rbp"
      , "    ret"
      , ""
      ]
    SyntaxATT ->
      [ ifName func <> ":"
      , "    pushq %rbp"
      , "    movq %rsp, %rbp"
      , "    # Function body"
      , "    movq %rbp, %rsp"
      , "    popq %rbp"
      , "    ret"
      , ""
      ]
    _ -> []

-- | Generate ARM function
generateARMFunction :: AssemblyConfig -> IRFunction -> [Text]
generateARMFunction config func =
  [ ifName func <> ":"
  , "    push {fp, lr}"
  , "    add fp, sp, #4"
  , "    @ Function body"
  , "    sub sp, fp, #4"
  , "    pop {fp, pc}"
  , ""
  ]

-- | Generate ARM64 function
generateARM64Function :: AssemblyConfig -> IRFunction -> [Text]
generateARM64Function config func =
  [ ifName func <> ":"
  , "    stp x29, x30, [sp, #-16]!"
  , "    mov x29, sp"
  , "    // Function body"
  , "    mov sp, x29"
  , "    ldp x29, x30, [sp], #16"
  , "    ret"
  , ""
  ]

-- | Get target name
targetName :: AssemblyTarget -> Text
targetName AsmX86 = "x86"
targetName AsmX86_64 = "x86_64"
targetName AsmARM = "ARM"
targetName AsmARM64 = "ARM64"

-- | Get syntax name
syntaxName :: AssemblySyntax -> Text
syntaxName SyntaxIntel = "Intel"
syntaxName SyntaxATT = "AT&T"
syntaxName SyntaxARM = "ARM"

-- | Convenience function for direct usage
emitAssembly
  :: IRModule
  -> FilePath
  -> AssemblyTarget
  -> AssemblySyntax
  -> Either Text EmitterResult
emitAssembly irModule outputDir target syntax =
  let config = defaultAssemblyConfig outputDir (imModuleName irModule) target syntax
      emitter = AssemblyEmitter config
  in emit emitter irModule

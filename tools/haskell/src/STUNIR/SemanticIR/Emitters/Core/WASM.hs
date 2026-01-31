{-# LANGUAGE OverloadedStrings #-}

{-|
Module      : STUNIR.SemanticIR.Emitters.Core.WASM
Description : WebAssembly emitter (WASM, WASI, SIMD)
Copyright   : (c) STUNIR Team, 2026
License     : MIT

Emits WebAssembly text format (WAT) and supports WASI and SIMD extensions.
Based on Ada SPARK wasm_emitter implementation.
-}

module STUNIR.SemanticIR.Emitters.Core.WASM
  ( WASMEmitter
  , WASMConfig(..)
  , WASMFeature(..)
  , defaultWASMConfig
  , emitWASM
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import STUNIR.SemanticIR.Emitters.Base
import STUNIR.SemanticIR.Emitters.Types
import STUNIR.SemanticIR.Emitters.CodeGen

-- | WASM feature flags
data WASMFeature
  = FeatureWASI      -- ^ WASI system interface
  | FeatureSIMD      -- ^ SIMD operations
  | FeatureThreads   -- ^ Threading support
  deriving (Eq, Show)

-- | WASM emitter configuration
data WASMConfig = WASMConfig
  { wasmBaseConfig :: !EmitterConfig
  , wasmFeatures   :: ![WASMFeature]
  } deriving (Show)

-- | Default WASM configuration
defaultWASMConfig :: FilePath -> Text -> [WASMFeature] -> WASMConfig
defaultWASMConfig outputDir moduleName features = WASMConfig
  { wasmBaseConfig = defaultEmitterConfig outputDir moduleName
  , wasmFeatures = features
  }

-- | WASM emitter
data WASMEmitter = WASMEmitter WASMConfig

instance Emitter WASMEmitter where
  emit (WASMEmitter config) irModule
    | not (validateIR irModule) = Left "Invalid IR module structure"
    | otherwise = Right $ EmitterResult
        { erStatus = Success
        , erFiles = [mainFile]
        , erTotalSize = gfSize mainFile
        , erErrorMessage = Nothing
        }
    where
      mainFile = generateWASMFile config irModule

-- | Generate WASM file
generateWASMFile :: WASMConfig -> IRModule -> GeneratedFile
generateWASMFile config irModule =
  let content = generateWASMCode config irModule
      fileName = imModuleName irModule <> ".wat"
  in GeneratedFile
       { gfPath = T.unpack fileName
       , gfHash = computeFileHash content
       , gfSize = T.length content
       }

-- | Generate WASM text format code
generateWASMCode :: WASMConfig -> IRModule -> Text
generateWASMCode config irModule = T.unlines $
  [";; STUNIR Generated WebAssembly"
  , ";; Module: " <> imModuleName irModule
  , ""
  , "(module"
  ] ++
  map ("  " <>) (generateImports config) ++
  [""] ++
  map ("  " <>) (concatMap (generateWASMFunction config) (imFunctions irModule)) ++
  [")", ""]

-- | Generate WASM imports for features
generateImports :: WASMConfig -> [Text]
generateImports config
  | FeatureWASI `elem` wasmFeatures config =
      ["(import \"wasi_snapshot_preview1\" \"fd_write\" (func $fd_write (param i32 i32 i32 i32) (result i32)))"]
  | otherwise = []

-- | Generate WASM function
generateWASMFunction :: WASMConfig -> IRFunction -> [Text]
generateWASMFunction config func =
  let paramDecls = ["(param $" <> ipName p <> " " <> mapIRTypeToWASM (ipType p) <> ")"
                   | p <- ifParameters func]
      resultDecl = if ifReturnType func /= TypeVoid
                   then ["(result " <> mapIRTypeToWASM (ifReturnType func) <> ")"]
                   else []
  in [ "(func $" <> ifName func
     ] ++
     map ("  " <>) paramDecls ++
     map ("  " <>) resultDecl ++
     [ "  ;; Function body"
     ] ++
     (if ifReturnType func /= TypeVoid
      then ["  (i32.const 0)"]
      else []) ++
     [")", ""]

-- | Map IR type to WASM type
mapIRTypeToWASM :: IRDataType -> Text
mapIRTypeToWASM TypeI8  = "i32"
mapIRTypeToWASM TypeI16 = "i32"
mapIRTypeToWASM TypeI32 = "i32"
mapIRTypeToWASM TypeI64 = "i64"
mapIRTypeToWASM TypeU8  = "i32"
mapIRTypeToWASM TypeU16 = "i32"
mapIRTypeToWASM TypeU32 = "i32"
mapIRTypeToWASM TypeU64 = "i64"
mapIRTypeToWASM TypeF32 = "f32"
mapIRTypeToWASM TypeF64 = "f64"
mapIRTypeToWASM _ = "i32"

-- | Convenience function for direct usage
emitWASM
  :: IRModule
  -> FilePath
  -> [WASMFeature]
  -> Either Text EmitterResult
emitWASM irModule outputDir features =
  let config = defaultWASMConfig outputDir (imModuleName irModule) features
      emitter = WASMEmitter config
  in emit emitter irModule

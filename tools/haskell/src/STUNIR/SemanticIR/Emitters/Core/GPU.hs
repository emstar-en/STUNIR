{-# LANGUAGE OverloadedStrings #-}

{-|
Module      : STUNIR.SemanticIR.Emitters.Core.GPU
Description : GPU computing emitter (CUDA, OpenCL, Metal, ROCm, Vulkan)
Copyright   : (c) STUNIR Team, 2026
License     : MIT

Emits GPU computing code for various platforms.
Supports CUDA, OpenCL, Metal, ROCm, and Vulkan Compute.
Based on Ada SPARK gpu_emitter implementation.
-}

module STUNIR.SemanticIR.Emitters.Core.GPU
  ( GPUEmitter
  , GPUConfig(..)
  , GPUBackend(..)
  , defaultGPUConfig
  , emitGPU
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import STUNIR.SemanticIR.Emitters.Base
import STUNIR.SemanticIR.Emitters.Types
import STUNIR.SemanticIR.Emitters.CodeGen

-- | GPU backend types
data GPUBackend
  = BackendCUDA
  | BackendOpenCL
  | BackendMetal
  | BackendROCm
  | BackendVulkan
  deriving (Eq, Show)

-- | GPU emitter configuration
data GPUConfig = GPUConfig
  { gpuBaseConfig :: !EmitterConfig
  , gpuBackend    :: !GPUBackend
  } deriving (Show)

-- | Default GPU configuration
defaultGPUConfig :: FilePath -> Text -> GPUBackend -> GPUConfig
defaultGPUConfig outputDir moduleName backend = GPUConfig
  { gpuBaseConfig = defaultEmitterConfig outputDir moduleName
  , gpuBackend = backend
  }

-- | GPU emitter
data GPUEmitter = GPUEmitter GPUConfig

instance Emitter GPUEmitter where
  emit (GPUEmitter config) irModule
    | not (validateIR irModule) = Left "Invalid IR module structure"
    | otherwise = Right $ EmitterResult
        { erStatus = Success
        , erFiles = [mainFile]
        , erTotalSize = gfSize mainFile
        , erErrorMessage = Nothing
        }
    where
      mainFile = generateGPUFile config irModule

-- | Generate GPU file
generateGPUFile :: GPUConfig -> IRModule -> GeneratedFile
generateGPUFile config irModule =
  let content = generateGPUCode config irModule
      extension = getFileExtension (gpuBackend config)
      fileName = imModuleName irModule <> extension
  in GeneratedFile
       { gfPath = T.unpack fileName
       , gfHash = computeFileHash content
       , gfSize = T.length content
       }

-- | Generate GPU code
generateGPUCode :: GPUConfig -> IRModule -> Text
generateGPUCode config irModule = T.unlines $
  [ getDO178CHeader (ecAddDO178CHeaders baseConfig)
                    ("GPU Code for " <> backendName (gpuBackend config))
  ] ++
  generateGPUIncludes (gpuBackend config) ++
  [""] ++
  concatMap (generateKernel config) (imFunctions irModule)
  where
    baseConfig = gpuBaseConfig config

-- | Generate GPU includes
generateGPUIncludes :: GPUBackend -> [Text]
generateGPUIncludes BackendCUDA =
  ["#include <cuda_runtime.h>"]
generateGPUIncludes BackendOpenCL =
  ["#include <CL/cl.h>"]
generateGPUIncludes BackendMetal =
  ["#include <metal_stdlib>", "using namespace metal;"]
generateGPUIncludes BackendROCm =
  ["#include <hip/hip_runtime.h>"]
generateGPUIncludes BackendVulkan =
  ["#version 450"]

-- | Generate GPU kernel
generateKernel :: GPUConfig -> IRFunction -> [Text]
generateKernel config func =
  let kernelAttr = getKernelAttribute (gpuBackend config)
      signature = generateFunctionSignature
                    (ifName func)
                    [(ipName p, mapIRTypeToC (ipType p)) | p <- ifParameters func]
                    (mapIRTypeToC (ifReturnType func))
                    "c"
      baseConfig = gpuBaseConfig config
      indent = indentString (ecIndentSize baseConfig) 1
  in [ kernelAttr <> " " <> signature
     , "{"
     , indent <> "/* Kernel body */"
     , "}"
     , ""
     ]

-- | Get kernel attribute for backend
getKernelAttribute :: GPUBackend -> Text
getKernelAttribute BackendCUDA = "__global__ void"
getKernelAttribute BackendOpenCL = "__kernel void"
getKernelAttribute BackendMetal = "kernel void"
getKernelAttribute BackendROCm = "__global__ void"
getKernelAttribute BackendVulkan = "void"

-- | Get file extension for backend
getFileExtension :: GPUBackend -> Text
getFileExtension BackendCUDA = ".cu"
getFileExtension BackendOpenCL = ".cl"
getFileExtension BackendMetal = ".metal"
getFileExtension BackendROCm = ".hip"
getFileExtension BackendVulkan = ".comp"

-- | Get backend name
backendName :: GPUBackend -> Text
backendName BackendCUDA = "CUDA"
backendName BackendOpenCL = "OpenCL"
backendName BackendMetal = "Metal"
backendName BackendROCm = "ROCm"
backendName BackendVulkan = "Vulkan Compute"

-- | Convenience function for direct usage
emitGPU
  :: IRModule
  -> FilePath
  -> GPUBackend
  -> Either Text EmitterResult
emitGPU irModule outputDir backend =
  let config = defaultGPUConfig outputDir (imModuleName irModule) backend
      emitter = GPUEmitter config
  in emit emitter irModule

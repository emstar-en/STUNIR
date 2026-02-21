{-# LANGUAGE OverloadedStrings #-}

-- | GPU code emitters
module STUNIR.Emitters.GPU
  ( emitCUDA
  , emitOpenCL
  , GPUBackend(..)
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import STUNIR.Emitters.Types

-- | GPU backend
data GPUBackend
  = CUDA
  | OpenCL
  | ROCm
  | Metal
  | Vulkan
  deriving (Show, Eq)

-- | Emit CUDA code
emitCUDA :: Text -> EmitterResult Text
emitCUDA moduleName = Right $ T.unlines
  [ "// STUNIR Generated CUDA"
  , "// Module: " <> moduleName
  , "// Generator: Haskell Pipeline"
  , ""
  , "#include <cuda_runtime.h>"
  , ""
  , "__global__ void kernel(float *data, int n) {"
  , "    int idx = blockIdx.x * blockDim.x + threadIdx.x;"
  , "    if (idx < n) {"
  , "        data[idx] = data[idx] * 2.0f;"
  , "    }"
  , "}"
  , ""
  , "extern \"C\" {"
  , "    void launch_kernel(float *data, int n) {"
  , "        int threads = 256;"
  , "        int blocks = (n + threads - 1) / threads;"
  , "        kernel<<<blocks, threads>>>(data, n);"
  , "    }"
  , "}"
  ]

-- | Emit OpenCL code
emitOpenCL :: Text -> EmitterResult Text
emitOpenCL moduleName = Right $ T.unlines
  [ "// STUNIR Generated OpenCL"
  , "// Module: " <> moduleName
  , "// Generator: Haskell Pipeline"
  , ""
  , "__kernel void compute(__global float *data, int n) {"
  , "    int gid = get_global_id(0);"
  , "    if (gid < n) {"
  , "        data[gid] = data[gid] * 2.0f;"
  , "    }"
  , "}"
  ]

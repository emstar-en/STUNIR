{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}

-- | Common types for STUNIR emitters
module STUNIR.Emitters.Types
  ( Architecture(..)
  , EmitterError(..)
  , EmitterResult
  , IRData(..)
  , GeneratedFile(..)
  ) where

import Data.Text (Text)
import GHC.Generics (Generic)
import Control.Exception (Exception)

-- | Target architectures
data Architecture
  = ARM
  | ARM64
  | X86
  | X86_64
  | RISCV
  | MIPS
  | AVR
  deriving (Show, Eq, Ord, Generic)

-- | Emitter errors
data EmitterError
  = UnsupportedTarget Text
  | InvalidConfiguration Text
  | GenerationFailed Text
  deriving (Show, Eq, Generic, Exception)

-- | Result type for emitters
type EmitterResult a = Either EmitterError a

-- | IR data structure
data IRData = IRData
  { irName :: Text
  , irFunctions :: [Text]
  , irTypes :: [Text]
  } deriving (Show, Eq, Generic)

-- | Generated file information
data GeneratedFile = GeneratedFile
  { filePath :: Text
  , fileContent :: Text
  , fileHash :: Text
  } deriving (Show, Eq, Generic)

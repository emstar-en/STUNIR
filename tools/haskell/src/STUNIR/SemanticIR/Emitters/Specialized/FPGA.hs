{-# LANGUAGE OverloadedStrings #-}
module STUNIR.SemanticIR.Emitters.Specialized.FPGA
  ( FPGAEmitter, FPGAConfig(..), FPGAHDL(..)
  , defaultFPGAConfig, emitFPGA
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import STUNIR.SemanticIR.Emitters.Base
import STUNIR.SemanticIR.Emitters.Types

data FPGAHDL = VHDL | Verilog | SystemVerilog deriving (Eq, Show)
data FPGAConfig = FPGAConfig
  { fpgaBaseConfig :: !EmitterConfig
  , fpgaHDL        :: !FPGAHDL
  } deriving (Show)

defaultFPGAConfig :: FilePath -> Text -> FPGAHDL -> FPGAConfig
defaultFPGAConfig outputDir moduleName hdl = FPGAConfig
  { fpgaBaseConfig = defaultEmitterConfig outputDir moduleName
  , fpgaHDL = hdl
  }

data FPGAEmitter = FPGAEmitter FPGAConfig

instance Emitter FPGAEmitter where
  emit (FPGAEmitter config) irModule
    | not (validateIR irModule) = Left "Invalid IR module"
    | otherwise = Right $ EmitterResult Success [mainFile] (gfSize mainFile) Nothing
    where mainFile = GeneratedFile (T.unpack $ imModuleName irModule <> ".vhd")
                                    (computeFileHash content) (T.length content)
          content = T.unlines ["-- STUNIR Generated VHDL", "entity " <> imModuleName irModule <> " is", "end " <> imModuleName irModule <> ";"]

emitFPGA :: IRModule -> FilePath -> FPGAHDL -> Either Text EmitterResult
emitFPGA irModule outputDir hdl =
  emit (FPGAEmitter $ defaultFPGAConfig outputDir (imModuleName irModule) hdl) irModule

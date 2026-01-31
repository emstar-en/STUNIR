{-# LANGUAGE OverloadedStrings #-}

-- | WebAssembly code emitters
module STUNIR.Emitters.WASM
  ( emitWASM
  , emitWAT
  , WASMFormat(..)
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import STUNIR.Emitters.Types

-- | WASM format
data WASMFormat
  = WAT  -- WebAssembly Text
  | WASM -- WebAssembly Binary (represented as text)
  deriving (Show, Eq)

-- | Emit WASM (as WAT text format)
emitWASM :: Text -> EmitterResult Text
emitWASM = emitWAT

-- | Emit WebAssembly Text format
emitWAT :: Text -> EmitterResult Text
emitWAT moduleName = Right $ T.unlines
  [ ";; STUNIR Generated WebAssembly"
  , ";; Module: " <> moduleName
  , ";; Generator: Haskell Pipeline"
  , ""
  , "(module"
  , "  (func $add (param $a i32) (param $b i32) (result i32)"
  , "    local.get $a"
  , "    local.get $b"
  , "    i32.add"
  , "  )"
  , ""
  , "  (func $main (result i32)"
  , "    i32.const 10"
  , "    i32.const 20"
  , "    call $add"
  , "  )"
  , ""
  , "  (export \"add\" (func $add))"
  , "  (export \"main\" (func $main))"
  , ")"
  ]

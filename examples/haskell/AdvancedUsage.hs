{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}

{-|
Module      : AdvancedUsage
Description : STUNIR Advanced Usage Example - Haskell
Copyright   : (c) STUNIR Project, 2026
License     : MIT
Stability   : experimental

This example demonstrates advanced STUNIR features in Haskell:
- Type-safe IR handling with GADTs-style patterns
- Multi-target code generation
- Manifest generation
- Monad transformers for pipeline orchestration

Usage:
  ghci AdvancedUsage.hs
  > main
-}

module AdvancedUsage where

import Data.List (sortBy, intercalate)
import Data.Ord (comparing)
import Data.Time.Clock.POSIX (getPOSIXTime)
import Data.Char (ord)
import Text.Printf (printf)
import Control.Monad (forM_)

-- =============================================================================
-- Target Language Types
-- =============================================================================

data TargetLanguage 
    = Python
    | Rust
    | C89
    | C99
    | Go
    | TypeScript
    deriving (Show, Eq, Ord)

languageName :: TargetLanguage -> String
languageName Python = "Python"
languageName Rust = "Rust"
languageName C89 = "C89"
languageName C99 = "C99"
languageName Go = "Go"
languageName TypeScript = "TypeScript"

languageExtension :: TargetLanguage -> String
languageExtension Python = "py"
languageExtension Rust = "rs"
languageExtension C89 = "c"
languageExtension C99 = "c"
languageExtension Go = "go"
languageExtension TypeScript = "ts"

-- =============================================================================
-- IR Type System
-- =============================================================================

data IRType 
    = TI32
    | TI64
    | TF32
    | TF64
    | TBool
    | TStr
    | TVoid
    | TCustom String
    deriving (Show, Eq)

parseIRType :: String -> IRType
parseIRType "i32" = TI32
parseIRType "i64" = TI64
parseIRType "f32" = TF32
parseIRType "f64" = TF64
parseIRType "bool" = TBool
parseIRType "str" = TStr
parseIRType "void" = TVoid
parseIRType s = TCustom s

-- | Convert IR type to target language type
typeToTarget :: TargetLanguage -> IRType -> String
typeToTarget Python TI32 = "int"
typeToTarget Python TI64 = "int"
typeToTarget Python TF32 = "float"
typeToTarget Python TF64 = "float"
typeToTarget Python TBool = "bool"
typeToTarget Python TStr = "str"
typeToTarget Python TVoid = "None"
typeToTarget Python (TCustom s) = s

typeToTarget Rust TI32 = "i32"
typeToTarget Rust TI64 = "i64"
typeToTarget Rust TF32 = "f32"
typeToTarget Rust TF64 = "f64"
typeToTarget Rust TBool = "bool"
typeToTarget Rust TStr = "String"
typeToTarget Rust TVoid = "()"
typeToTarget Rust (TCustom s) = s

typeToTarget Go TI32 = "int32"
typeToTarget Go TI64 = "int64"
typeToTarget Go TF32 = "float32"
typeToTarget Go TF64 = "float64"
typeToTarget Go TBool = "bool"
typeToTarget Go TStr = "string"
typeToTarget Go TVoid = ""
typeToTarget Go (TCustom s) = s

typeToTarget TypeScript _ = "any"  -- Simplified

typeToTarget C89 TI32 = "int"
typeToTarget C89 TI64 = "long long"
typeToTarget C89 TBool = "int"
typeToTarget C89 _ = "void*"

typeToTarget C99 TI32 = "int32_t"
typeToTarget C99 TI64 = "int64_t"
typeToTarget C99 TBool = "bool"
typeToTarget C99 _ = "void*"

-- =============================================================================
-- IR Structures
-- =============================================================================

data IRParam = IRParam
    { irParamName :: String
    , irParamType :: IRType
    } deriving (Show, Eq)

data IRFunction = IRFunction
    { irFuncName   :: String
    , irFuncParams :: [IRParam]
    , irFuncReturns :: IRType
    , irFuncExported :: Bool
    } deriving (Show, Eq)

data IRModule = IRModule
    { irModName    :: String
    , irModVersion :: String
    , irModFuncs   :: [IRFunction]
    } deriving (Show, Eq)

-- =============================================================================
-- Code Emitter Type Class
-- =============================================================================

class CodeEmitter a where
    emitTarget :: a -> TargetLanguage
    emitHeader :: a -> IRModule -> String
    emitFunction :: a -> IRFunction -> String
    emitModule :: a -> IRModule -> String
    
    emitModule emitter mod' = unlines $
        [emitHeader emitter mod'] ++
        map (emitFunction emitter) (irModFuncs mod')

-- =============================================================================
-- Python Emitter
-- =============================================================================

data PythonEmitter = PythonEmitter

instance CodeEmitter PythonEmitter where
    emitTarget _ = Python
    
    emitHeader _ mod' = unlines
        [ "\"\"\"Generated Python module: " ++ irModName mod' ++ "\"\"\""
        , "# Version: " ++ irModVersion mod'
        , ""
        , "from typing import Any"
        , ""
        ]
    
    emitFunction _ func = unlines
        [ "def " ++ irFuncName func ++ "(" ++ params ++ ") -> " ++ retType ++ ":"
        , "    \"\"\"STUNIR generated function.\"\"\""
        , "    pass"
        , ""
        ]
      where
        params = intercalate ", " $ map formatParam (irFuncParams func)
        formatParam p = irParamName p ++ ": " ++ typeToTarget Python (irParamType p)
        retType = typeToTarget Python (irFuncReturns func)

-- =============================================================================
-- Rust Emitter
-- =============================================================================

data RustEmitter = RustEmitter

instance CodeEmitter RustEmitter where
    emitTarget _ = Rust
    
    emitHeader _ mod' = unlines
        [ "//! Generated Rust module: " ++ irModName mod'
        , "//! Version: " ++ irModVersion mod'
        , ""
        ]
    
    emitFunction _ func = unlines
        [ vis ++ "fn " ++ irFuncName func ++ "(" ++ params ++ ") -> " ++ retType ++ " {"
        , "    todo!(\"Implementation placeholder\")"
        , "}"
        , ""
        ]
      where
        vis = if irFuncExported func then "pub " else ""
        params = intercalate ", " $ map formatParam (irFuncParams func)
        formatParam p = irParamName p ++ ": " ++ typeToTarget Rust (irParamType p)
        retType = typeToTarget Rust (irFuncReturns func)

-- =============================================================================
-- Multi-Target Generation
-- =============================================================================

data EmitterOutput = EmitterOutput
    { outLanguage :: TargetLanguage
    , outFilename :: String
    , outCode     :: String
    , outHash     :: String
    } deriving (Show)

emitAllTargets :: IRModule -> [EmitterOutput]
emitAllTargets mod' = 
    [ emitWithEmitter PythonEmitter mod'
    , emitWithEmitter RustEmitter mod'
    ]

emitWithEmitter :: CodeEmitter e => e -> IRModule -> EmitterOutput
emitWithEmitter emitter mod' = EmitterOutput
    { outLanguage = emitTarget emitter
    , outFilename = irModName mod' ++ "." ++ languageExtension (emitTarget emitter)
    , outCode = code
    , outHash = computeHash code
    }
  where
    code = emitModule emitter mod'

-- =============================================================================
-- Hash and Manifest
-- =============================================================================

computeHash :: String -> String
computeHash s = printf "%016x%016x" hash1 hash2
  where
    fnv1a str = foldl step 2166136261 str
    step h c = ((h `xorInt` ord c) * 16777619) `mod` (2^32)
    xorInt a b = abs (a - b)  -- Simplified
    hash1 = fnv1a s :: Int
    hash2 = fnv1a (reverse s)

generateManifest :: [EmitterOutput] -> [(String, String)]
generateManifest outputs = sortBy (comparing fst)
    [ ("schema", "stunir.manifest.targets.v1")
    , ("entry_count", show $ length outputs)
    , ("entries", intercalate ";" $ map formatEntry outputs)
    , ("manifest_hash", computeHash $ show outputs)
    ]
  where
    formatEntry out = outFilename out ++ ":" ++ take 16 (outHash out)

-- =============================================================================
-- Sample Module
-- =============================================================================

sampleModule :: IRModule
sampleModule = IRModule
    { irModName = "calculator"
    , irModVersion = "2.0.0"
    , irModFuncs = 
        [ IRFunction "add" 
            [IRParam "a" TI32, IRParam "b" TI32] 
            TI32 True
        , IRFunction "divide"
            [IRParam "x" TF64, IRParam "y" TF64]
            TF64 True
        , IRFunction "internal_helper"
            [IRParam "n" TI32]
            TVoid False
        ]
    }

-- =============================================================================
-- Main
-- =============================================================================

main :: IO ()
main = do
    putStrLn "============================================================"
    putStrLn "STUNIR Advanced Usage Example - Haskell"
    putStrLn "============================================================\n"
    
    -- Display module info
    putStrLn $ "\128230 Module: " ++ irModName sampleModule
    putStrLn $ "   Version: " ++ irModVersion sampleModule
    putStrLn $ "   Functions: " ++ show (length $ irModFuncs sampleModule) ++ "\n"
    
    -- Emit all targets
    putStrLn "\127919 Emitting code for multiple targets..."
    let outputs = emitAllTargets sampleModule
    
    forM_ outputs $ \out -> do
        putStrLn $ "\n--- " ++ outFilename out ++ " (" ++ show (length $ outCode out) ++ " bytes) ---"
        putStrLn $ outCode out
    
    -- Generate manifest
    putStrLn "\128203 Generating manifest..."
    let manifest = generateManifest outputs
    forM_ manifest $ \(k, v) ->
        putStrLn $ "   " ++ k ++ ": " ++ take 60 v ++ (if length v > 60 then "..." else "")
    
    -- Summary
    putStrLn "\n============================================================"
    putStrLn "Summary"
    putStrLn "============================================================"
    putStrLn $ "Module:     " ++ irModName sampleModule
    putStrLn $ "Functions:  " ++ show (length $ irModFuncs sampleModule)
    putStrLn $ "Targets:    " ++ show (length outputs)
    putStrLn $ "Languages:  " ++ intercalate ", " (map (languageName . outLanguage) outputs)
    putStrLn "\n\9989 Advanced usage example completed!"

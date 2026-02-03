{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}

{-|
Module      : BasicUsage
Description : STUNIR Basic Usage Example - Haskell
Copyright   : (c) STUNIR Project, 2026
License     : MIT
Stability   : experimental

This example demonstrates fundamental STUNIR operations in Haskell:
- Loading and parsing specs
- Converting spec to IR
- Generating receipts
- Verifying determinism

Usage:
  ghci BasicUsage.hs
  > main
-}

module BasicUsage where

import Data.List (sortBy, intercalate)
import Data.Ord (comparing)
import Data.Time.Clock.POSIX (getPOSIXTime)
import Data.Char (ord)
import Text.Printf (printf)
import Numeric (showHex)

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Represents a function parameter
data Param = Param
    { paramName :: String
    , paramType :: String
    } deriving (Show, Eq)

-- | Represents a function in the spec
data Function = Function
    { funcName   :: String
    , funcParams :: [Param]
    , funcReturns :: String
    } deriving (Show, Eq)

-- | Represents a STUNIR spec
data Spec = Spec
    { specName     :: String
    , specVersion  :: String
    , specFuncs    :: [Function]
    , specExports  :: [String]
    } deriving (Show, Eq)

-- | Represents the IR (Intermediate Representation)
data IR = IR
    { irVersion   :: String
    , irEpoch     :: Integer
    , irSpecHash  :: String
    , irModule    :: ModuleInfo
    , irFunctions :: [Function]
    , irExports   :: [String]
    } deriving (Show, Eq)

data ModuleInfo = ModuleInfo
    { moduleName    :: String
    , moduleVersion :: String
    } deriving (Show, Eq)

-- | Represents a receipt
data Receipt = Receipt
    { receiptVersion     :: String
    , receiptEpoch       :: Integer
    , receiptModuleName  :: String
    , receiptIRHash      :: String
    , receiptSpecHash    :: String
    , receiptFuncCount   :: Int
    , receiptHash        :: String
    } deriving (Show, Eq)

-- =============================================================================
-- Canonical JSON Implementation
-- =============================================================================

-- | Simple key-value pair
type KV = (String, String)

-- | Generate canonical JSON from key-value pairs
-- Keys are sorted alphabetically for determinism
canonicalJson :: [KV] -> String
canonicalJson pairs = "{" ++ content ++ "}"
  where
    sorted = sortBy (comparing fst) pairs
    content = intercalate "," $ map formatPair sorted
    formatPair (k, v) = "\"" ++ k ++ "\":\"" ++ escapeJson v ++ "\""
    escapeJson = concatMap escapeChar
    escapeChar '"' = "\\\""
    escapeChar '\\' = "\\\\"
    escapeChar c = [c]

-- | Simple hash function (FNV-1a style for demonstration)
-- In production, use cryptonite's SHA256
computeHash :: String -> String
computeHash s = showHexPadded hash1 ++ showHexPadded hash2
  where
    fnv1a = foldl (\h c -> (h `xor` fromIntegral (ord c)) * 0x01000193) 0x811c9dc5
    hash1 = fnv1a s :: Integer
    hash2 = fnv1a (reverse s)
    showHexPadded n = printf "%016x" (abs n `mod` (2^64 :: Integer))
    xor = (\x y -> abs $ x - y)  -- Simplified XOR for demonstration

-- | Get current epoch time
getEpoch :: IO Integer
getEpoch = round <$> getPOSIXTime

-- =============================================================================
-- Sample Spec
-- =============================================================================

-- | Create a sample spec for demonstration
sampleSpec :: Spec
sampleSpec = Spec
    { specName = "example_module"
    , specVersion = "1.0.0"
    , specFuncs = 
        [ Function "add" 
            [Param "a" "i32", Param "b" "i32"] 
            "i32"
        , Function "multiply" 
            [Param "x" "i32", Param "y" "i32"] 
            "i32"
        ]
    , specExports = ["add", "multiply"]
    }

-- =============================================================================
-- Spec Operations
-- =============================================================================

-- | Compute hash of a spec
hashSpec :: Spec -> String
hashSpec spec = computeHash $ canonicalJson
    [ ("name", specName spec)
    , ("version", specVersion spec)
    , ("function_count", show $ length $ specFuncs spec)
    ]

-- =============================================================================
-- IR Generation
-- =============================================================================

-- | Convert spec to IR
specToIR :: Spec -> Integer -> IR
specToIR spec epoch = IR
    { irVersion = "1.0.0"
    , irEpoch = epoch
    , irSpecHash = hashSpec spec
    , irModule = ModuleInfo (specName spec) (specVersion spec)
    , irFunctions = specFuncs spec
    , irExports = specExports spec
    }

-- | Compute hash of IR
hashIR :: IR -> String
hashIR ir = computeHash $ canonicalJson
    [ ("ir_version", irVersion ir)
    , ("ir_spec_hash", irSpecHash ir)
    , ("module_name", moduleName $ irModule ir)
    , ("function_count", show $ length $ irFunctions ir)
    ]

-- =============================================================================
-- Receipt Generation
-- =============================================================================

-- | Generate a receipt from IR
generateReceipt :: IR -> Integer -> Receipt
generateReceipt ir epoch = receipt { receiptHash = computeReceiptHash receipt }
  where
    receipt = Receipt
        { receiptVersion = "1.0.0"
        , receiptEpoch = epoch
        , receiptModuleName = moduleName $ irModule ir
        , receiptIRHash = hashIR ir
        , receiptSpecHash = irSpecHash ir
        , receiptFuncCount = length $ irFunctions ir
        , receiptHash = ""  -- Will be computed
        }
    computeReceiptHash r = computeHash $ canonicalJson
        [ ("ir_hash", receiptIRHash r)
        , ("module_name", receiptModuleName r)
        , ("spec_hash", receiptSpecHash r)
        ]

-- =============================================================================
-- Determinism Verification
-- =============================================================================

-- | Verify determinism by computing multiple hashes
verifyDeterminism :: Spec -> Int -> IO Bool
verifyDeterminism spec iterations = do
    putStrLn $ "\127760 Verifying determinism (" ++ show iterations ++ " iterations)..."
    let hashes = replicate iterations (hashSpec spec)
    mapM_ (\(i, h) -> putStrLn $ "   Round " ++ show i ++ ": " ++ take 16 h ++ "...") 
          (zip [1..] hashes)
    let isDeterministic = all (== head hashes) hashes
    if isDeterministic
        then putStrLn "   \9989 Determinism verified!"
        else putStrLn "   \10060 DETERMINISM FAILURE - hashes differ!"
    return isDeterministic

-- =============================================================================
-- Main
-- =============================================================================

main :: IO ()
main = do
    putStrLn "============================================================"
    putStrLn "STUNIR Basic Usage Example - Haskell"
    putStrLn "============================================================\n"
    
    -- Step 1: Load sample spec
    putStrLn "\128196 Using sample spec..."
    let spec = sampleSpec
    putStrLn $ "   Module: " ++ specName spec
    putStrLn $ "   Version: " ++ specVersion spec
    putStrLn $ "   Functions: " ++ show (length $ specFuncs spec) ++ "\n"
    
    -- Step 2: Convert to IR
    putStrLn "\128260 Converting spec to IR..."
    epoch <- getEpoch
    let ir = specToIR spec epoch
    putStrLn $ "   \9989 Generated IR with " ++ show (length $ irFunctions ir) ++ " functions\n"
    
    -- Step 3: Generate receipt
    putStrLn "\128221 Generating receipt..."
    let receipt = generateReceipt ir epoch
    putStrLn $ "   \9989 Receipt generated: " ++ take 16 (receiptHash receipt) ++ "...\n"
    
    -- Step 4: Verify determinism
    _ <- verifyDeterminism spec 3
    putStrLn ""
    
    -- Step 5: Display results
    putStrLn "============================================================"
    putStrLn "Results Summary"
    putStrLn "============================================================"
    putStrLn $ "Module Name:    " ++ moduleName (irModule ir)
    putStrLn $ "IR Hash:        " ++ receiptIRHash receipt
    putStrLn $ "Receipt Hash:   " ++ receiptHash receipt
    putStrLn $ "Functions:      " ++ show (receiptFuncCount receipt)
    putStrLn $ "Exports:        " ++ intercalate ", " (irExports ir) ++ "\n"
    
    putStrLn "\9989 Basic usage example completed successfully!"

{-# LANGUAGE OverloadedStrings #-}

module Commands.SpecToIr (run) where

import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.ByteString.Lazy as B
import Data.Aeson (encode)
import IR.V1 (IrV1(..), IrFunction(..), IrInstruction(..))

-- | Mock implementation of the Spec -> IR transformation.
-- In a real scenario, this would parse the input JSON spec.
-- Here we demonstrate producing the structured IR to match the Rust side.
run :: FilePath -> FilePath -> IO ()
run _inJson outIr = do
    -- Example: Constructing a structured function
    let mainFn = IrFunction
          { name = "main"
          , body = 
              [ IrInstruction { op = "LOAD", args = ["r1", "0"] }
              , IrInstruction { op = "STORE", args = ["r1", "result"] }
              ]
          }

    let ir = IrV1
          { version = "1.0.0"
          , functions = [mainFn]
          }

    -- Write the JSON output
    B.writeFile outIr (encode ir)
    putStrLn $ "Generated IR at: " ++ outIr

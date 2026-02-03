{-# LANGUAGE OverloadedStrings #-}

module Commands.SpecToIr (run) where

import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.ByteString.Lazy as B
import Data.Aeson.Encode.Pretty (encodePretty', Config(..), defConfig, Indent(Spaces))
import IR.V1 (IrV1(..), IrFunction(..), IrInstruction(..))

-- | Mock implementation of the Spec -> IR transformation.
run :: FilePath -> FilePath -> IO ()
run _inJson outIr = do
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

    -- Configure pretty-printer to use 2 spaces (matches Rust serde_json default)
    let config = defConfig { confIndent = Spaces 2 }

    -- Write the JSON output
    B.writeFile outIr (encodePretty' config ir)
    putStrLn $ "Generated IR at: " ++ outIr

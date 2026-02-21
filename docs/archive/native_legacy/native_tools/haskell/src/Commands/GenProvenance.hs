{-# LANGUAGE OverloadedStrings #-}

module Commands.GenProvenance (run) where

import qualified Data.ByteString.Lazy as BL
import qualified Data.ByteString as BS
import Data.Aeson (decode, Value)
import Data.Aeson.Encode.Pretty (encodePretty', Config(..), defConfig, Indent(Spaces))
import qualified Crypto.Hash.SHA256 as SHA256
import qualified Data.ByteString.Base16 as Hex
import Data.Text.Encoding (decodeUtf8)
import Data.Text (Text)
import qualified Data.Text as T
import IR.Provenance (Provenance(..))

run :: FilePath -> FilePath -> FilePath -> IO ()
run inIr epochJson outProv = do
    -- 1. Read IR
    irContent <- BS.readFile inIr

    -- 2. Hash IR
    let hashBytes = SHA256.hash irContent
    let hashHex = decodeUtf8 (Hex.encode hashBytes)
    let hashStr = "sha256:" <> hashHex

    -- 3. Read Epoch
    epochContent <- BL.readFile epochJson
    let maybeEpoch = decode epochContent :: Maybe Value

    case maybeEpoch of
        Nothing -> error "Failed to parse Epoch JSON"
        Just epochData -> do
            -- 4. Create Provenance
            let prov = Provenance
                  { schema = "stunir.provenance.v1"
                  , ir_hash = hashStr
                  , epoch = epochData
                  , status = "SUCCESS"
                  }

            -- 5. Write Output (Pretty)
            let config = defConfig { confIndent = Spaces 2 }
            BL.writeFile outProv (encodePretty' config prov)
            putStrLn $ "Generated Provenance at: " ++ outProv

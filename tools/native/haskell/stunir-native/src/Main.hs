{-# LANGUAGE OverloadedStrings #-}
module Main where

import System.Environment (getArgs)
import System.Exit (exitFailure, exitSuccess)
import qualified Data.ByteString.Lazy.Char8 as BL
import qualified Data.ByteString as BS
import Data.Aeson (encode, decode)
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import qualified Data.Map.Strict as Map
import System.Directory (createDirectoryIfMissing)
import System.FilePath (takeDirectory)
import Data.Time.Clock.POSIX (getPOSIXTime)
import Crypto.Hash.SHA256 (hash)
import qualified Data.ByteString.Base16 as Hex
import qualified Data.Text.Encoding as TE

import Stunir.Receipt
import Stunir.Canonical (canonicalEncode)
import Stunir.Spec
import Stunir.IR
import Stunir.Import

-- Helper to write JSON
writeJson :: ToJSON a => FilePath -> a -> IO ()
writeJson path obj = do
    createDirectoryIfMissing True (takeDirectory path)
    BL.writeFile path (canonicalEncode obj)

-- Helper for SHA256
sha256Hex :: BL.ByteString -> Text
sha256Hex bs = TE.decodeUtf8 $ Hex.encode $ hash $ BL.toStrict bs

main :: IO ()
main = do
    args <- getArgs
    case args of
        ["version"] -> putStrLn "stunir-native-hs v0.2.0.0"

        -- epoch --out-json <path> --print-epoch
        ("epoch":"--out-json":outPath:"--print-epoch":_) -> do
            t <- getPOSIXTime
            let epoch = round t :: Integer
            writeJson outPath (object ["epoch" .= epoch])
            print epoch

        -- import-code --input-root <src> --out-spec <dest>
        ("import-code":"--input-root":inRoot:"--out-spec":outSpec:_) -> do
            modules <- scanDirectory inRoot
            let spec = Spec "spec" modules
            writeJson outSpec spec
            putStrLn $ "Imported " ++ show (length modules) ++ " modules to " ++ outSpec

        -- spec-to-ir --spec-root <root> --out <out>
        ("spec-to-ir":"--spec-root":specRoot:"--out":outPath:_) -> do
            let specPath = specRoot ++ "/spec.json"
            specContent <- BL.readFile specPath
            let specHash = sha256Hex specContent

            case decode specContent :: Maybe Spec of
                Nothing -> error "Failed to parse spec.json"
                Just spec -> do
                    let ir = IR {
                        ir_version = "v1",
                        ir_module_name = "stunir_module", -- Default
                        ir_types = [],
                        ir_functions = [],
                        ir_spec_sha256 = specHash,
                        ir_source = IRSource specHash (T.pack specPath),
                        ir_source_modules = sp_modules spec
                    }
                    writeJson outPath ir
                    putStrLn $ "Generated IR at " ++ outPath

        -- gen-receipt (existing)
        ("gen-receipt":target:status:epochStr:tName:tPath:tHash:tVer:rest) -> do
            let epoch = read epochStr :: Integer
            let tool = ToolInfo (T.pack tName) (T.pack tPath) (T.pack tHash) (T.pack tVer)
            let receipt = Receipt {
                receipt_schema = "stunir.receipt.build.v1",
                receipt_target = T.pack target,
                receipt_status = T.pack status,
                receipt_build_epoch = epoch,
                receipt_epoch_json = "build/epoch.json",
                receipt_inputs = Map.empty,
                receipt_tool = tool,
                receipt_argv = map T.pack rest
            }
            BL.putStrLn (canonicalEncode receipt)

        _ -> do
            putStrLn "Error: Unknown command or missing arguments."
            putStrLn "Commands: version, epoch, import-code, spec-to-ir, gen-receipt"
            exitFailure

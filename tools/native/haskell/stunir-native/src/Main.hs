{-# LANGUAGE OverloadedStrings #-}
module Main where

import System.Environment (getArgs, lookupEnv)
import System.Exit (exitFailure, exitSuccess)
import System.IO (hPutStrLn, stderr)
import qualified Data.ByteString.Lazy.Char8 as BL
import Data.Aeson (encode, decode, object, (.=))
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import qualified Data.Map.Strict as Map
import System.Directory (createDirectoryIfMissing, doesFileExist)
import System.FilePath (takeDirectory)
import Data.Time.Clock.POSIX (getPOSIXTime)
import Crypto.Hash.SHA256 (hash)
import qualified Data.ByteString.Base16 as Hex
import qualified Data.Text.Encoding as TE
import System.Process (callProcess)

import Stunir.Receipt
import Stunir.Canonical (canonicalEncode)
import Stunir.Spec
import Stunir.IR
import Stunir.Import
import Stunir.Provenance
import Stunir.Toolchain

-- | Helper to write JSON safely
writeJson :: ToJSON a => FilePath -> a -> IO ()
writeJson path obj = do
    createDirectoryIfMissing True (takeDirectory path)
    BL.writeFile path (canonicalEncode obj)

-- | Helper for SHA256
sha256Hex :: BL.ByteString -> Text
sha256Hex bs = TE.decodeUtf8 $ Hex.encode $ hash $ BL.toStrict bs

-- | Fail with a message
die :: String -> IO a
die msg = do
    hPutStrLn stderr $ "Error: " ++ msg
    exitFailure

main :: IO ()
main = do
    args <- getArgs
    case args of
        ["version"] -> putStrLn "stunir-native-hs v0.5.0.0"

        -- check-toolchain
        ("check-toolchain":"--lockfile":lockPath:_) -> do
            verifyToolchain lockPath

        -- epoch
        ("epoch":"--out-json":outPath:"--print-epoch":_) -> do
            t <- getPOSIXTime
            let epoch = round t :: Integer
            writeJson outPath (object ["epoch" .= epoch])
            print epoch

        -- import-code
        ("import-code":"--input-root":inRoot:"--out-spec":outSpec:_) -> do
            modules <- scanDirectory inRoot
            let spec = Spec "spec" modules
            writeJson outSpec spec
            putStrLn $ "Imported " ++ show (length modules) ++ " modules to " ++ outSpec

        -- spec-to-ir
        ("spec-to-ir":"--spec-root":specRoot:"--out":outPath:_) -> do
            let specPath = specRoot ++ "/spec.json"
            exists <- doesFileExist specPath
            if not exists then die $ "Spec file not found: " ++ specPath else do
                specContent <- BL.readFile specPath
                let specHash = sha256Hex specContent

                case decode specContent :: Maybe Spec of
                    Nothing -> die "Failed to parse spec.json"
                    Just spec -> do
                        let ir = IR {
                            ir_version = "v1",
                            ir_module_name = "stunir_module",
                            ir_types = [],
                            ir_functions = [],
                            ir_spec_sha256 = specHash,
                            ir_source = IRSource specHash (T.pack specPath),
                            ir_source_modules = sp_modules spec
                        }
                        writeJson outPath ir
                        putStrLn $ "Generated IR at " ++ outPath

        -- gen-provenance
        ("gen-provenance":"--epoch":epochStr:"--spec-root":specRoot:"--asm-root":_:"--out-json":outJson:"--out-header":outHeader:_) -> do
            let epoch = read epochStr :: Integer
            let specPath = specRoot ++ "/spec.json"
            exists <- doesFileExist specPath
            if not exists then die $ "Spec file not found: " ++ specPath else do
                specContent <- BL.readFile specPath
                let specHash = sha256Hex specContent

                case decode specContent :: Maybe Spec of
                    Nothing -> die "Failed to parse spec.json"
                    Just spec -> do
                        let modNames = map sm_name (sp_modules spec)
                        let prov = Provenance epoch specHash modNames

                        writeJson outJson prov
                        TIO.writeFile outHeader (generateCHeader prov)
                        putStrLn $ "Generated Provenance: " ++ outJson ++ ", " ++ outHeader

        -- compile-provenance
        ("compile-provenance":"--in-prov":_:"--out-bin":outBin:_) -> do
            let cFile = "tools/prov_emit.c"
            exists <- doesFileExist cFile
            if not exists then die ("Missing C source: " ++ cFile) else do
                -- Deterministic Compiler Lookup
                -- 1. Try STUNIR_TOOL_GCC
                -- 2. Fallback to 'gcc' (but warn)
                envGcc <- lookupEnv "STUNIR_TOOL_GCC"
                let gccCmd = case envGcc of
                        Just p -> p
                        Nothing -> "gcc"

                callProcess gccCmd [cFile, "-o", outBin, "-Ibuild"] 
                putStrLn $ "Compiled provenance binary to " ++ outBin ++ " using " ++ gccCmd

        -- gen-receipt
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

        _ -> die "Unknown command or missing arguments."

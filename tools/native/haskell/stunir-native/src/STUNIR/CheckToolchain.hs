{-# LANGUAGE OverloadedStrings, DeriveGeneric #-}
module STUNIR.CheckToolchain (run) where
import qualified Data.ByteString.Lazy as B
import qualified Data.ByteString as BS
import Data.Aeson
import Crypto.Hash (hash, SHA256, Digest)
import qualified Data.Text as T
import GHC.Generics
import System.Directory (doesFileExist)
import System.Exit (die, exitFailure)
import Control.Monad (forM_)
data ToolEntry = ToolEntry { name :: T.Text, path :: FilePath, hash :: Maybe T.Text } deriving (Show, Generic)
instance FromJSON ToolEntry
data ToolchainLock = ToolchainLock { kind :: T.Text, tools :: [ToolEntry] } deriving (Show, Generic)
instance FromJSON ToolchainLock
run :: FilePath -> IO ()
run lockPath = do
  putStrLn $ "Checking Toolchain Lock: " ++ lockPath
  input <- B.readFile lockPath
  case decode input of
    Nothing -> die "Invalid lockfile JSON"
    Just lock -> do
      if kind lock /= "toolchain_lock" then die "kind must be 'toolchain_lock'" else mapM_ checkTool (tools lock)
      putStrLn "Toolchain Verified."
checkTool :: ToolEntry -> IO ()
checkTool tool = do
  exists <- doesFileExist (path tool)
  if not exists then die $ "Tool not found: " ++ show (name tool) else do
    case STUNIR.CheckToolchain.hash tool of
      Nothing -> putStrLn $ "  [OK] " ++ T.unpack (name tool) ++ " (No hash)"
      Just expected -> do
        content <- BS.readFile (path tool)
        let digest = Crypto.Hash.hash content :: Digest SHA256
        let actual = T.pack $ show digest
        if actual /= expected then do
          putStrLn $ "MISMATCH: Tool " ++ T.unpack (name tool)
          exitFailure
        else putStrLn $ "  [OK] " ++ T.unpack (name tool)

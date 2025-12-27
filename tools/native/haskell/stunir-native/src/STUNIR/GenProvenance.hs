{-# LANGUAGE OverloadedStrings, DeriveGeneric #-}
module STUNIR.GenProvenance (run) where
import qualified Data.ByteString.Lazy as B
import qualified Data.ByteString as BS
import Data.Aeson
import Crypto.Hash (hash, SHA256, Digest)
import qualified Data.Text as T
import GHC.Generics
import System.Directory (createDirectoryIfMissing)
import System.FilePath (takeDirectory)
data Provenance = Provenance { kind :: T.Text, generator :: T.Text, epoch :: T.Text, input_ir_hash :: T.Text } deriving (Show, Generic)
instance ToJSON Provenance
run :: FilePath -> FilePath -> FilePath -> IO ()
run inIr epochJson outProv = do
  irBytes <- BS.readFile inIr
  let digest = hash irBytes :: Digest SHA256
  let irHash = T.pack $ show digest
  epochContent <- readFile epochJson
  let epochVal = T.strip $ T.pack epochContent
  let prov = Provenance "provenance" "stunir-native-haskell" epochVal irHash
  createDirectoryIfMissing True (takeDirectory outProv)
  B.writeFile outProv (encode prov)
  putStrLn $ "Generated Provenance at " ++ outProv

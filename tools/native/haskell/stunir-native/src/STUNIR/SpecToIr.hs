{-# LANGUAGE OverloadedStrings #-}
module STUNIR.SpecToIr (run) where
import qualified Data.ByteString.Lazy as B
import Data.Aeson
import qualified STUNIR.Spec as S
import qualified STUNIR.IR.V1 as IR
import System.Exit (die)
run :: FilePath -> FilePath -> IO ()
run inJson outIr = do
  input <- B.readFile inJson
  case decode input of
    Nothing -> die "Failed to parse Spec JSON"
    Just spec -> do
      let meta = IR.IrMetadata (S.kind spec) (S.modules spec)
      let ir = IR.IrV1 "ir" "stunir-native-haskell" "v1" "main" [] [] meta
      B.writeFile outIr (encode ir)
      putStrLn $ "Generated IR at " ++ outIr

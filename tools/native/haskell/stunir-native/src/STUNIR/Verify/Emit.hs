{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

module STUNIR.Verify.Emit
  ( run
  ) where

import GHC.Generics (Generic)
import qualified Data.Aeson as A
import qualified Data.ByteString as BS
import qualified Data.Text as T
import System.Exit (exitFailure, exitSuccess)
import System.FilePath ((</>))
import Control.Exception (catch)

import STUNIR.Hash (sha256Hex)
import STUNIR.PathPolicy (checkRelpathSafe)

data EmitReceipt = EmitReceipt
  { receipt_version :: T.Text
  , outputs :: [EmitOutput]
  } deriving (Show, Generic)

instance A.FromJSON EmitReceipt where
  parseJSON = A.genericParseJSON A.defaultOptions

data EmitOutput = EmitOutput
  { relpath :: String
  , sha256 :: String
  } deriving (Show, Generic)

instance A.FromJSON EmitOutput where
  parseJSON = A.genericParseJSON A.defaultOptions

run :: FilePath -> FilePath -> IO ()
run receiptFp root = do
  bytes <- BS.readFile receiptFp `orDie` "missing_object"
  receipt <- case A.eitherDecodeStrict' bytes of
    Left _ -> die "bad_digest"
    Right r -> pure r

  if receipt_version receipt /= "stunir.emit.v1" then die "bad_digest" else pure ()

  mapM_ (verifyOne root) (outputs receipt)

  putStrLn "OK verify.emit"
  exitSuccess

verifyOne :: FilePath -> EmitOutput -> IO ()
verifyOne root out = do
  case checkRelpathSafe (relpath out) of
    Left e -> die e
    Right () -> pure ()

  bs <- BS.readFile (root </> relpath out) `orDie` "missing_object"
  let h = sha256Hex bs
  if h /= map toLowerAscii (sha256 out) then die "bad_digest" else pure ()

toLowerAscii :: Char -> Char
toLowerAscii c
  | c >= 'A' && c <= 'F' = toEnum (fromEnum c + 32)
  | otherwise = c

orDie :: IO a -> String -> IO a
orDie action tag = action `catchIO` (\_ -> die tag)

catchIO :: IO a -> (IOError -> IO a) -> IO a
catchIO = catch

-- local die

die :: String -> IO a
die tag = do
  putStrLn tag
  exitFailure

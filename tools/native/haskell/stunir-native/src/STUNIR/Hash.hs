module STUNIR.Hash
  ( sha256Hex
  ) where

import qualified Crypto.Hash as CH
import qualified Data.ByteString as BS

sha256Hex :: BS.ByteString -> String
sha256Hex bs =
  let digest = (CH.hash bs :: CH.Digest CH.SHA256)
  in show digest

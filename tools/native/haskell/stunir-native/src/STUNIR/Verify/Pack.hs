{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}

module STUNIR.Verify.Pack
  ( run
  ) where

import Control.Exception (IOException, catch)
import qualified Crypto.Error as CE
import qualified Crypto.PubKey.Ed25519 as Ed
import qualified Data.ByteString as BS
import qualified Data.ByteString.Char8 as BSC
import qualified Data.ByteString.Base64 as B64
import qualified Data.List as L
import qualified Data.Map.Strict as M
import qualified Data.Set as S
import System.Directory
import System.Exit (exitFailure, exitSuccess)
import System.FilePath

import STUNIR.Hash (sha256Hex)
import STUNIR.PathPolicy (checkRelpathSafe, isScopeExcluded)

rootAttestationVersion :: BS.ByteString
rootAttestationVersion = "stunir.pack.root_attestation_text.v0"

run :: FilePath -> FilePath -> FilePath -> FilePath -> Bool -> Maybe String -> IO ()
run root objectsDir packManifest rootAttestation checkCompleteness pubkeyB64 = do
  let objDirPath = root </> objectsDir
  okObj <- doesDirectoryExist objDirPath
  if not okObj then die "missing_objects_dir" else pure ()

  let raPath = root </> rootAttestation
  raBytes <- readFileBS raPath `orDie` "missing_root_attestation"

  maybeVerifySig root raBytes pubkeyB64

  let raLines = filter (not . BS.null) (map stripCR (BSC.lines raBytes))
  if null raLines || head raLines /= rootAttestationVersion then die "bad_version_line" else pure ()

  let records = map parseRecord (tail raLines)
  if any (\(rt,_) -> rt /= "artifact" && rt /= "ir") records then die "unknown_record_type" else pure ()

  let irCount = length [() | (rt,_) <- records, rt == "ir"]
  if irCount /= 1 then die "ir_count_not_one" else pure ()

  kv <- case
          [ m
          | (rt, m) <- records
          , rt == "artifact"
          , M.lookup "kind" m == Just "manifest"
          , M.lookup "logical_path" m == Just "pack_manifest.tsv"
          ]
        of
          [x] -> pure x
          _ -> die "missing_manifest_binding"

  digest <- case M.lookup "digest" kv of
    Nothing -> die "missing_manifest_binding"
    Just d -> pure d

  (algo, hex) <- parseDigest digest
  if algo /= "sha256" then die "bad_digest" else pure ()

  let pmPath = root </> packManifest
  pmBytes <- readFileBS pmPath `orDie` "missing_pack_manifest"
  if BS.any (== 13) pmBytes then die "crlf_detected_in_pack_manifest" else pure ()

  if sha256Hex pmBytes /= hex then die "pack_manifest_hash_mismatch" else pure ()

  verifyObjectStoreBlob objDirPath hex pmBytes

  entries <- case traverse parseManifestLine (filter (not . BS.null) (BSC.lines pmBytes)) of
    Left _ -> die "bad_digest"
    Right xs -> pure xs

  let paths = map snd entries

  mapM_ (\p ->
    if isScopeExcluded p then die "manifest_scope_violation" else
      case checkRelpathSafe p of
        Left e -> die e
        Right () -> pure ()
    ) paths

  if paths /= L.sort paths || hasDup paths then die "manifest_not_sorted" else pure ()

  mapM_ (verifyEntry root objDirPath) entries

  mapM_ (verifyRecordDigestExists objDirPath) records

  if checkCompleteness then do
    actual <- listAllFiles root objectsDir packManifest
    let manifestSet = S.fromList paths
    if actual /= manifestSet then die "manifest_incomplete_or_extra_files" else pure ()
  else pure ()

  putStrLn "OK verify.pack"
  exitSuccess

verifyEntry :: FilePath -> FilePath -> (String,String) -> IO ()
verifyEntry root objDirPath (hex, rel) = do
  let fp = root </> rel
  isF <- doesFileExist fp
  if not isF then die "manifest_file_missing" else pure ()
  bs <- readFileBS fp `orDie` "manifest_file_missing"
  if sha256Hex bs /= hex then die "manifest_file_hash_mismatch" else pure ()
  verifyObjectStoreBlob objDirPath hex bs

verifyRecordDigestExists :: FilePath -> (String, M.Map String String) -> IO ()
verifyRecordDigestExists objDirPath (_, kv) =
  case M.lookup "digest" kv of
    Nothing -> pure ()
    Just d -> do
      (_, hex) <- parseDigest d
      let obj = objDirPath </> hex
      bs <- readFileBS obj `orDie` "missing_object"
      if sha256Hex bs /= hex then die "object_hash_mismatch" else pure ()

listAllFiles :: FilePath -> FilePath -> FilePath -> IO (S.Set String)
listAllFiles root objectsDir packManifest = do
  allPaths <- go root
  pure (S.fromList [ rel | p <- allPaths, let rel = makeRelative root p, not (excluded rel) ])
  where
    excluded rel = rel == packManifest || take (length objectsDir) rel == objectsDir
    go d = do
      ents <- listDirectory d
      fmap concat $ mapM (\e -> do
        let p = d </> e
        isDir <- doesDirectoryExist p
        if isDir then go p else pure [p]
        ) ents

parseRecord :: BS.ByteString -> (String, M.Map String String)
parseRecord line =
  let ws = BSC.words line
      rt = if null ws then "" else BSC.unpack (head ws)
      kvs = mapMaybeKV (tail ws)
  in (rt, M.fromList kvs)

mapMaybeKV :: [BS.ByteString] -> [(String,String)]
mapMaybeKV = foldr step []
  where
    step t acc =
      case BSC.break (== '=') t of
        (k, v) | not (BS.null v) -> (BSC.unpack k, BSC.unpack (BS.drop 1 v)) : acc
        _ -> acc

parseDigest :: String -> IO (String,String)
parseDigest d =
  case break (== ':') d of
    (a, ':' : hex) -> if isLowerHex64 hex then pure (a, hex) else die "bad_digest"
    _ -> die "bad_digest"

isLowerHex64 :: String -> Bool
isLowerHex64 s = length s == 64 && all isLowerHex s

isLowerHex :: Char -> Bool
isLowerHex c = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')

parseManifestLine :: BS.ByteString -> Either () (String,String)
parseManifestLine line =
  let (h, rest) = BSC.break (== ' ') line
  in if BS.null rest then Left () else
       let hex = BSC.unpack h
           p = BSC.unpack (BSC.drop 1 rest)
       in if not (isLowerHex64 hex) then Left () else Right (hex, p)

hasDup :: Ord a => [a] -> Bool
hasDup xs = length (L.nub xs) /= length xs

stripCR :: BS.ByteString -> BS.ByteString
stripCR bs = case BS.unsnoc bs of
  Just (init', 13) -> init'
  _ -> bs

verifyObjectStoreBlob :: FilePath -> String -> BS.ByteString -> IO ()
verifyObjectStoreBlob objDirPath hex expected = do
  let obj = objDirPath </> hex
  bs <- readFileBS obj `orDie` "missing_object"
  if sha256Hex bs /= hex then die "object_hash_mismatch" else pure ()
  if bs /= expected then die "object_hash_mismatch" else pure ()

maybeVerifySig :: FilePath -> BS.ByteString -> Maybe String -> IO ()
maybeVerifySig root raBytes pubkeyB64 = do
  let sigPath = root </> "root_attestation.txt.sig"
  hasSig <- doesFileExist sigPath
  if not hasSig then pure () else
    case pubkeyB64 of
      Nothing -> pure ()
      Just pkB64 -> do
        sigTxt <- readFileBS sigPath `orDie` "bad_digest"
        let sigB64 = BSC.takeWhile (not . isSpace8) sigTxt
        case (B64.decode (BSC.pack pkB64), B64.decode sigB64) of
          (Left _, _) -> die "bad_digest"
          (_, Left _) -> die "bad_digest"
          (Right pkRaw, Right sigRaw) -> do
            let pkF = Ed.publicKey pkRaw
            let sigF = Ed.signature sigRaw
            case (pkF, sigF) of
              (CE.CryptoFailed _, _) -> die "bad_digest"
              (_, CE.CryptoFailed _) -> die "bad_digest"
              (CE.CryptoPassed pk, CE.CryptoPassed sig) ->
                if Ed.verify pk raBytes sig then pure () else die "bad_digest"

isSpace8 :: Char -> Bool
isSpace8 c = c == ' ' || c == '\n' || c == '\t' || c == '\r'

readFileBS :: FilePath -> IO BS.ByteString
readFileBS = BS.readFile

orDie :: IO a -> String -> IO a
orDie action tag = action `catch` (\(_ :: IOException) -> die tag)

die :: String -> IO a
die tag = do
  putStrLn tag
  exitFailure

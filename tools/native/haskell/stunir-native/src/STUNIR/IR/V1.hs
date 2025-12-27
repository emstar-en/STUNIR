{-# LANGUAGE OverloadedStrings #-}

module STUNIR.IR.V1
  ( validateIrV1
  ) where

import qualified Data.Aeson as A
import qualified Data.Aeson.Key as Key
import qualified Data.Aeson.KeyMap as KM
import qualified Data.Text as T

validateIrV1 :: A.Value -> Either String ()
validateIrV1 = A.withObject "IrV1" $ \o -> do
  requireOnlyKeys ["ir_version","module_name","docstring","types","functions"] o
  v <- o A..: "ir_version"
  if (v :: T.Text) /= "v1" then Left "bad_digest" else pure ()
  mn <- o A..: "module_name"
  if not (checkModuleName mn) then Left "unsafe_filename" else pure ()
  _ <- o A..:? "docstring" :: Either String (Maybe T.Text)
  ts <- o A..: "types"
  fs <- o A..: "functions"
  mapM_ validateType ts
  mapM_ validateFn fs

validateType :: A.Value -> Either String ()
validateType = A.withObject "TypeDecl" $ \o -> do
  requireOnlyKeys ["name","docstring","fields"] o
  _ <- o A..: "name" :: Either String T.Text
  _ <- o A..:? "docstring" :: Either String (Maybe T.Text)
  fields <- o A..: "fields"
  mapM_ validateField fields

validateField :: A.Value -> Either String ()
validateField = A.withObject "FieldDecl" $ \o -> do
  requireOnlyKeys ["name","type","optional"] o
  _ <- o A..: "name" :: Either String T.Text
  _ <- o A..: "type" :: Either String T.Text
  _ <- o A..:? "optional" :: Either String (Maybe Bool)
  pure ()

validateFn :: A.Value -> Either String ()
validateFn = A.withObject "FnDecl" $ \o -> do
  requireOnlyKeys ["name","docstring","args","return_type","steps"] o
  _ <- o A..: "name" :: Either String T.Text
  _ <- o A..:? "docstring" :: Either String (Maybe T.Text)
  args <- o A..: "args"
  mapM_ validateArg args
  _ <- o A..: "return_type" :: Either String T.Text
  steps <- o A..:? "steps" :: Either String (Maybe [A.Value])
  case steps of
    Nothing -> pure ()
    Just ss -> mapM_ validateStep ss

validateArg :: A.Value -> Either String ()
validateArg = A.withObject "ArgDecl" $ \o -> do
  requireOnlyKeys ["name","type"] o
  _ <- o A..: "name" :: Either String T.Text
  _ <- o A..: "type" :: Either String T.Text
  pure ()

validateStep :: A.Value -> Either String ()
validateStep = A.withObject "Step" $ \o -> do
  requireOnlyKeys ["op","target","value"] o
  op <- o A..: "op" :: Either String T.Text
  case T.toLower op of
    "return" -> pure ()
    "call" -> pure ()
    "assign" -> pure ()
    "error" -> pure ()
    _ -> Left "bad_digest"
  _ <- o A..:? "target" :: Either String (Maybe T.Text)
  mv <- o A..:? "value" :: Either String (Maybe A.Value)
  case mv of
    Nothing -> pure ()
    Just v -> case v of
      A.String _ -> pure ()
      A.Number _ -> pure ()
      A.Bool _ -> pure ()
      A.Object _ -> pure ()
      _ -> Left "bad_digest"

requireOnlyKeys :: [T.Text] -> KM.KeyMap A.Value -> Either String ()
requireOnlyKeys allowed o =
  let allowedKeys = map Key.fromText allowed
      ks = KM.keys o
  in if all (`elem` allowedKeys) ks then Right () else Left "bad_digest"

checkModuleName :: T.Text -> Bool
checkModuleName t =
  case T.uncons t of
    Nothing -> False
    Just (c, rest) -> isAsciiAlpha c && T.all isAllowed rest
  where
    isAsciiAlpha c = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
    isAllowed c = isAsciiAlpha c || (c >= '0' && c <= '9') || c == '_'

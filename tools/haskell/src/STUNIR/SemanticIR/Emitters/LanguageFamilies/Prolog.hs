{-# LANGUAGE OverloadedStrings #-}

{-|
Module      : STUNIR.SemanticIR.Emitters.LanguageFamilies.Prolog
Description : Prolog family emitter (SWI-Prolog, GNU Prolog, SICStus, YAP, XSB, Ciao, B-Prolog, ECLiPSe)
Copyright   : (c) STUNIR Team, 2026
License     : MIT

Emits code for various Prolog implementations.
Supports SWI-Prolog, GNU Prolog, SICStus, YAP, XSB, Ciao, B-Prolog, and ECLiPSe.
Based on Ada SPARK prolog emitters.
-}

module STUNIR.SemanticIR.Emitters.LanguageFamilies.Prolog
  ( PrologEmitter
  , PrologConfig(..)
  , PrologSystem(..)
  , defaultPrologConfig
  , emitProlog
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import STUNIR.SemanticIR.Emitters.Base
import STUNIR.SemanticIR.Emitters.Types
import STUNIR.SemanticIR.Emitters.CodeGen

-- | Prolog system types
data PrologSystem
  = SystemSWIProlog
  | SystemGNUProlog
  | SystemSICStus
  | SystemYAP
  | SystemXSB
  | SystemCiao
  | SystemBProlog
  | SystemECLiPSe
  deriving (Eq, Show)

-- | Prolog emitter configuration
data PrologConfig = PrologConfig
  { prologBaseConfig :: !EmitterConfig
  , prologSystem     :: !PrologSystem
  } deriving (Show)

-- | Default Prolog configuration
defaultPrologConfig :: FilePath -> Text -> PrologSystem -> PrologConfig
defaultPrologConfig outputDir moduleName system = PrologConfig
  { prologBaseConfig = defaultEmitterConfig outputDir moduleName
  , prologSystem = system
  }

-- | Prolog emitter
data PrologEmitter = PrologEmitter PrologConfig

instance Emitter PrologEmitter where
  emit (PrologEmitter config) irModule
    | not (validateIR irModule) = Left "Invalid IR module structure"
    | otherwise = Right $ EmitterResult
        { erStatus = Success
        , erFiles = [mainFile]
        , erTotalSize = gfSize mainFile
        , erErrorMessage = Nothing
        }
    where
      mainFile = generatePrologFile config irModule

-- | Generate Prolog file
generatePrologFile :: PrologConfig -> IRModule -> GeneratedFile
generatePrologFile config irModule =
  let content = generatePrologCode config irModule
      fileName = imModuleName irModule <> ".pl"
  in GeneratedFile
       { gfPath = T.unpack fileName
       , gfHash = computeFileHash content
       , gfSize = T.length content
       }

-- | Generate Prolog code
generatePrologCode :: PrologConfig -> IRModule -> Text
generatePrologCode config irModule = T.unlines $
  [ "% STUNIR Generated Prolog Code"
  , "% System: " <> systemName (prologSystem config)
  , "% Module: " <> imModuleName irModule
  , ""
  ] ++
  generatePrologDirectives config ++
  [""] ++
  concatMap (generatePrologPredicate config) (imFunctions irModule)

-- | Generate Prolog directives
generatePrologDirectives :: PrologConfig -> [Text]
generatePrologDirectives config =
  case prologSystem config of
    SystemSWIProlog ->
      [ ":- module(" <> T.toLower (ecModuleName (prologBaseConfig config)) <> ", [])."
      , ":- use_module(library(lists))."
      ]
    SystemGNUProlog ->
      [ "% GNU Prolog specific directives" ]
    SystemSICStus ->
      [ ":- module(" <> T.toLower (ecModuleName (prologBaseConfig config)) <> ", [])."
      , ":- use_module(library(lists))."
      ]
    SystemYAP ->
      [ ":- module(" <> T.toLower (ecModuleName (prologBaseConfig config)) <> ", [])."
      ]
    SystemXSB ->
      [ "% XSB specific directives" ]
    SystemCiao ->
      [ ":- module(" <> T.toLower (ecModuleName (prologBaseConfig config)) <> ", [])."
      ]
    SystemBProlog ->
      [ "% B-Prolog specific directives" ]
    SystemECLiPSe ->
      [ ":- module(" <> T.toLower (ecModuleName (prologBaseConfig config)) <> ")."
      ]

-- | Generate Prolog predicate
generatePrologPredicate :: PrologConfig -> IRFunction -> [Text]
generatePrologPredicate config func =
  let params = map (\p -> T.toUpper (T.take 1 (ipName p)) <> T.drop 1 (ipName p))
                   (ifParameters func)
      paramList = if null params then "" else T.intercalate ", " params
  in [ "% Predicate: " <> ifName func
     , ifName func <> "(" <> paramList <> ") :-"
     , "    % Predicate body"
     , "    true."
     , ""
     ]

-- | Get system name
systemName :: PrologSystem -> Text
systemName SystemSWIProlog = "SWI-Prolog"
systemName SystemGNUProlog = "GNU Prolog"
systemName SystemSICStus = "SICStus Prolog"
systemName SystemYAP = "YAP Prolog"
systemName SystemXSB = "XSB Prolog"
systemName SystemCiao = "Ciao Prolog"
systemName SystemBProlog = "B-Prolog"
systemName SystemECLiPSe = "ECLiPSe Prolog"

-- | Convenience function for direct usage
emitProlog
  :: IRModule
  -> FilePath
  -> PrologSystem
  -> Either Text EmitterResult
emitProlog irModule outputDir system =
  let config = defaultPrologConfig outputDir (imModuleName irModule) system
      emitter = PrologEmitter config
  in emit emitter irModule

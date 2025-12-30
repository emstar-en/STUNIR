{-# LANGUAGE OverloadedStrings #-}
module Stunir.Import (scanDirectory) where

import System.Directory
import System.FilePath
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import Control.Monad (forM)
import Stunir.Spec

-- | Map extensions to languages
extToLang :: String -> Maybe Text
extToLang ".py" = Just "python"
extToLang ".js" = Just "javascript"
extToLang ".ts" = Just "typescript"
extToLang ".go" = Just "go"
extToLang ".rs" = Just "rust"
extToLang ".c"  = Just "c"
extToLang ".cpp" = Just "cpp"
extToLang ".java" = Just "java"
extToLang ".rb" = Just "ruby"
extToLang ".php" = Just "php"
extToLang ".sh" = Just "bash"
extToLang _     = Nothing

scanDirectory :: FilePath -> IO [SpecModule]
scanDirectory root = do
    exists <- doesDirectoryExist root
    if not exists then return [] else do
        files <- listDirectoryRecursive root
        modules <- forM files $ \f -> do
            let ext = takeExtension f
            case extToLang ext of
                Nothing -> return Nothing
                Just lang -> do
                    content <- TIO.readFile f
                    let name = T.pack $ takeBaseName f -- Simplified name
                    return $ Just $ SpecModule name content lang
        return [m | Just m <- modules]

listDirectoryRecursive :: FilePath -> IO [FilePath]
listDirectoryRecursive path = do
    isFile <- doesFileExist path
    if isFile then return [path] else do
        isDir <- doesDirectoryExist path
        if not isDir then return [] else do
            entries <- listDirectory path
            let fullEntries = map (path </>) entries
            concat <$> mapM listDirectoryRecursive fullEntries

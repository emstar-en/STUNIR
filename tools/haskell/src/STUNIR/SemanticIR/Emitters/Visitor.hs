{-# LANGUAGE OverloadedStrings #-}

{-|
Module      : STUNIR.SemanticIR.Emitters.Visitor
Description : Visitor pattern for IR traversal
Copyright   : (c) STUNIR Team, 2026
License     : MIT
Maintainer  : stunir@example.com

Visitor pattern for traversing and processing IR structures.
Based on Ada SPARK visitor patterns.
-}

module STUNIR.SemanticIR.Emitters.Visitor
  ( -- * IR Visitor Typeclass
    IRVisitor(..)
  , -- * Code Generation Visitor
    CodeGenVisitor
  , CodeGenState(..)
  , initialCodeGenState
  , runCodeGenVisitor
  , emitLine
  , increaseIndent
  , decreaseIndent
  , getGeneratedCode
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import Control.Monad.State
import STUNIR.SemanticIR.Emitters.Types

-- | IR Visitor typeclass
-- Implements the Visitor pattern for walking the IR tree
class Monad m => IRVisitor m where
  -- | Visit an IR module (entry point)
  visitModule :: IRModule -> m ()
  
  -- | Visit a type definition
  visitType :: IRType -> m ()
  
  -- | Visit a type field
  visitField :: IRTypeField -> m ()
  
  -- | Visit a function definition
  visitFunction :: IRFunction -> m ()
  
  -- | Visit a function parameter
  visitParameter :: IRParameter -> m ()
  
  -- | Visit a statement
  visitStatement :: IRStatement -> m ()
  
  -- Default implementations that traverse the structure
  visitModule irModule = do
    mapM_ visitType (imTypes irModule)
    mapM_ visitFunction (imFunctions irModule)
  
  visitType irType = do
    mapM_ visitField (itFields irType)
  
  visitField _ = return ()
  
  visitFunction irFunc = do
    mapM_ visitParameter (ifParameters irFunc)
    mapM_ visitStatement (ifStatements irFunc)
  
  visitParameter _ = return ()
  
  visitStatement _ = return ()

-- | Code generation visitor state
data CodeGenState = CodeGenState
  { cgsCode        :: ![Text]    -- ^ Accumulated code lines
  , cgsIndentLevel :: !Int       -- ^ Current indentation level
  , cgsIndentSize  :: !Int       -- ^ Spaces per indent level
  } deriving (Eq, Show)

-- | Initial code generation state
initialCodeGenState :: Int -> CodeGenState
initialCodeGenState indentSize = CodeGenState
  { cgsCode = []
  , cgsIndentLevel = 0
  , cgsIndentSize = indentSize
  }

-- | Code generation visitor monad
type CodeGenVisitor a = State CodeGenState a

-- | Run code generation visitor and return generated code
runCodeGenVisitor :: CodeGenVisitor a -> Int -> (a, Text)
runCodeGenVisitor visitor indentSize =
  let (result, finalState) = runState visitor (initialCodeGenState indentSize)
  in (result, T.unlines $ reverse $ cgsCode finalState)

-- | Emit a line of code with current indentation
emitLine :: Text -> CodeGenVisitor ()
emitLine line = modify $ \s ->
  let indent = T.replicate (cgsIndentLevel s * cgsIndentSize s) " "
      codeLine = if T.null line then "" else indent <> line
  in s { cgsCode = codeLine : cgsCode s }

-- | Increase indentation level
increaseIndent :: CodeGenVisitor ()
increaseIndent = modify $ \s -> s { cgsIndentLevel = cgsIndentLevel s + 1 }

-- | Decrease indentation level
decreaseIndent :: CodeGenVisitor ()
decreaseIndent = modify $ \s -> s { cgsIndentLevel = max 0 (cgsIndentLevel s - 1) }

-- | Get generated code from current state
getGeneratedCode :: CodeGenVisitor Text
getGeneratedCode = do
  s <- get
  return $ T.unlines $ reverse $ cgsCode s

-- | Default IRVisitor instance for CodeGenVisitor
instance IRVisitor CodeGenVisitor where
  visitModule irModule = do
    emitLine $ "-- Module: " <> imModuleName irModule
    mapM_ visitType (imTypes irModule)
    mapM_ visitFunction (imFunctions irModule)
  
  visitType irType = do
    emitLine $ "-- Type: " <> itName irType
    increaseIndent
    mapM_ visitField (itFields irType)
    decreaseIndent
  
  visitField field = do
    emitLine $ "-- Field: " <> itfName field <> " : " <> itfType field
  
  visitFunction irFunc = do
    emitLine $ "-- Function: " <> ifName irFunc
    increaseIndent
    mapM_ visitParameter (ifParameters irFunc)
    mapM_ visitStatement (ifStatements irFunc)
    decreaseIndent
  
  visitParameter param = do
    emitLine $ "-- Param: " <> ipName param
  
  visitStatement stmt = do
    emitLine $ "-- Statement: " <> (T.pack . show $ isType stmt)

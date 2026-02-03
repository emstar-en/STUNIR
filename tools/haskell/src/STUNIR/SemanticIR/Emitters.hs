{-# LANGUAGE OverloadedStrings #-}

{-|
Module      : STUNIR.SemanticIR.Emitters
Description : STUNIR Semantic IR Emitters - All 24 emitters
Copyright   : (c) STUNIR Team, 2026
License     : MIT
Maintainer  : stunir@example.com

Complete suite of 24 Semantic IR emitters for STUNIR.
Provides confluence with Ada SPARK, Python, and Rust implementations.

This module re-exports all emitters for convenient access.
-}

module STUNIR.SemanticIR.Emitters
  ( -- * Base Infrastructure
    module STUNIR.SemanticIR.Emitters.Base
  , module STUNIR.SemanticIR.Emitters.Types
  , module STUNIR.SemanticIR.Emitters.Visitor
  , module STUNIR.SemanticIR.Emitters.CodeGen
    
    -- * Core Category Emitters (5)
  , module STUNIR.SemanticIR.Emitters.Core.Embedded
  , module STUNIR.SemanticIR.Emitters.Core.GPU
  , module STUNIR.SemanticIR.Emitters.Core.WASM
  , module STUNIR.SemanticIR.Emitters.Core.Assembly
  , module STUNIR.SemanticIR.Emitters.Core.Polyglot
    
    -- * Language Family Emitters (2)
  , module STUNIR.SemanticIR.Emitters.LanguageFamilies.Lisp
  , module STUNIR.SemanticIR.Emitters.LanguageFamilies.Prolog
    
    -- * Specialized Category Emitters (17)
  , module STUNIR.SemanticIR.Emitters.Specialized.Business
  , module STUNIR.SemanticIR.Emitters.Specialized.FPGA
  , module STUNIR.SemanticIR.Emitters.Specialized.Grammar
  , module STUNIR.SemanticIR.Emitters.Specialized.Lexer
  , module STUNIR.SemanticIR.Emitters.Specialized.Parser
  , module STUNIR.SemanticIR.Emitters.Specialized.Expert
  , module STUNIR.SemanticIR.Emitters.Specialized.Constraints
  , module STUNIR.SemanticIR.Emitters.Specialized.Functional
  , module STUNIR.SemanticIR.Emitters.Specialized.OOP
  , module STUNIR.SemanticIR.Emitters.Specialized.Mobile
  , module STUNIR.SemanticIR.Emitters.Specialized.Scientific
  , module STUNIR.SemanticIR.Emitters.Specialized.Bytecode
  , module STUNIR.SemanticIR.Emitters.Specialized.Systems
  , module STUNIR.SemanticIR.Emitters.Specialized.Planning
  , module STUNIR.SemanticIR.Emitters.Specialized.AsmIR
  , module STUNIR.SemanticIR.Emitters.Specialized.BEAM
  , module STUNIR.SemanticIR.Emitters.Specialized.ASP
  ) where

-- Base Infrastructure
import STUNIR.SemanticIR.Emitters.Base
import STUNIR.SemanticIR.Emitters.Types
import STUNIR.SemanticIR.Emitters.Visitor
import STUNIR.SemanticIR.Emitters.CodeGen

-- Core Category Emitters (5)
import STUNIR.SemanticIR.Emitters.Core.Embedded
import STUNIR.SemanticIR.Emitters.Core.GPU
import STUNIR.SemanticIR.Emitters.Core.WASM
import STUNIR.SemanticIR.Emitters.Core.Assembly
import STUNIR.SemanticIR.Emitters.Core.Polyglot

-- Language Family Emitters (2)
import STUNIR.SemanticIR.Emitters.LanguageFamilies.Lisp
import STUNIR.SemanticIR.Emitters.LanguageFamilies.Prolog

-- Specialized Category Emitters (17)
import STUNIR.SemanticIR.Emitters.Specialized.Business
import STUNIR.SemanticIR.Emitters.Specialized.FPGA
import STUNIR.SemanticIR.Emitters.Specialized.Grammar
import STUNIR.SemanticIR.Emitters.Specialized.Lexer
import STUNIR.SemanticIR.Emitters.Specialized.Parser
import STUNIR.SemanticIR.Emitters.Specialized.Expert
import STUNIR.SemanticIR.Emitters.Specialized.Constraints
import STUNIR.SemanticIR.Emitters.Specialized.Functional
import STUNIR.SemanticIR.Emitters.Specialized.OOP
import STUNIR.SemanticIR.Emitters.Specialized.Mobile
import STUNIR.SemanticIR.Emitters.Specialized.Scientific
import STUNIR.SemanticIR.Emitters.Specialized.Bytecode
import STUNIR.SemanticIR.Emitters.Specialized.Systems
import STUNIR.SemanticIR.Emitters.Specialized.Planning
import STUNIR.SemanticIR.Emitters.Specialized.AsmIR
import STUNIR.SemanticIR.Emitters.Specialized.BEAM
import STUNIR.SemanticIR.Emitters.Specialized.ASP

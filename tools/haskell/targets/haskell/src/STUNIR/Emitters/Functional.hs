{-# LANGUAGE OverloadedStrings #-}

-- | Functional language emitters
module STUNIR.Emitters.Functional
  ( emitHaskell
  , emitScala
  , emitOCaml
  , FunctionalLanguage(..)
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import STUNIR.Emitters.Types

-- | Functional language
data FunctionalLanguage
  = Haskell
  | Scala
  | OCaml
  | FSharp
  | Erlang
  | Elixir
  deriving (Show, Eq)

-- | Emit Haskell code
emitHaskell :: Text -> EmitterResult Text
emitHaskell moduleName = Right $ T.unlines
  [ "-- STUNIR Generated Haskell"
  , "-- Module: " <> moduleName
  , "-- Generator: Haskell Pipeline"
  , ""
  , "module " <> moduleName <> " where"
  , ""
  , "-- Factorial function"
  , "factorial :: Integer -> Integer"
  , "factorial 0 = 1"
  , "factorial n = n * factorial (n - 1)"
  , ""
  , "-- Map doubling"
  , "doubleList :: [Int] -> [Int]"
  , "doubleList = map (*2)"
  ]

-- | Emit Scala code
emitScala :: Text -> EmitterResult Text
emitScala moduleName = Right $ T.unlines
  [ "// STUNIR Generated Scala"
  , "// Module: " <> moduleName
  , "// Generator: Haskell Pipeline"
  , ""
  , "object " <> moduleName <> " {"
  , "  def factorial(n: Int): Int = n match {"
  , "    case 0 => 1"
  , "    case _ => n * factorial(n - 1)"
  , "  }"
  , "  "
  , "  def doubleList(xs: List[Int]): List[Int] ="
  , "    xs.map(_ * 2)"
  , "}"
  ]

-- | Emit OCaml code
emitOCaml :: Text -> EmitterResult Text
emitOCaml moduleName = Right $ T.unlines
  [ "(* STUNIR Generated OCaml *)"
  , "(* Module: " <> moduleName <> " *)"
  , "(* Generator: Haskell Pipeline *)"
  , ""
  , "let rec factorial n ="
  , "  match n with"
  , "  | 0 -> 1"
  , "  | _ -> n * factorial (n - 1)"
  , ""
  , "let double_list xs ="
  , "  List.map (fun x -> x * 2) xs"
  ]

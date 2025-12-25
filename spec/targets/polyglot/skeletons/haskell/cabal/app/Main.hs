module Main where

import STUNIR.Runtime (program)
import System.Exit (exitWith, ExitCode(..))

main :: IO ()
main = do
  code <- program
  if code == 0 then exitWith ExitSuccess else exitWith (ExitFailure code)

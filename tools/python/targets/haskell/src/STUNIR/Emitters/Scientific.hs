{-# LANGUAGE OverloadedStrings #-}

-- | Scientific computing emitters
module STUNIR.Emitters.Scientific
  ( emitMATLAB
  , emitJulia
  , emitNumPy
  , ScientificLanguage(..)
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import STUNIR.Emitters.Types

-- | Scientific language
data ScientificLanguage
  = MATLAB
  | Julia
  | R
  | NumPy
  deriving (Show, Eq)

-- | Emit MATLAB code
emitMATLAB :: Text -> EmitterResult Text
emitMATLAB moduleName = Right $ T.unlines
  [ "% STUNIR Generated MATLAB"
  , "% Module: " <> moduleName
  , "% Generator: Haskell Pipeline"
  , ""
  , "function result = " <> moduleName <> "(x)"
  , "    % Matrix operations"
  , "    A = [1, 2, 3; 4, 5, 6; 7, 8, 9];"
  , "    b = [1; 2; 3];"
  , "    "
  , "    % Solve linear system"
  , "    result = A \\ b;"
  , "    "
  , "    % Eigenvalues"
  , "    eigenvalues = eig(A);"
  , "end"
  ]

-- | Emit Julia code
emitJulia :: Text -> EmitterResult Text
emitJulia moduleName = Right $ T.unlines
  [ "# STUNIR Generated Julia"
  , "# Module: " <> moduleName
  , "# Generator: Haskell Pipeline"
  , ""
  , "module " <> moduleName
  , ""
  , "using LinearAlgebra"
  , ""
  , "function compute(x::Vector{Float64})"
  , "    A = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]"
  , "    b = [1.0, 2.0, 3.0]"
  , "    "
  , "    # Solve linear system"
  , "    result = A \\ b"
  , "    "
  , "    return result"
  , "end"
  , ""
  , "end # module"
  ]

-- | Emit NumPy/SciPy code
emitNumPy :: Text -> EmitterResult Text
emitNumPy moduleName = Right $ T.unlines
  [ "# STUNIR Generated NumPy/SciPy"
  , "# Module: " <> moduleName
  , "# Generator: Haskell Pipeline"
  , ""
  , "import numpy as np"
  , "from scipy import linalg"
  , ""
  , "def compute(x):"
  , "    \"\"\"Compute with matrices.\"\"\""
  , "    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])"
  , "    b = np.array([1, 2, 3])"
  , "    "
  , "    # Solve linear system"
  , "    result = linalg.solve(A, b)"
  , "    "
  , "    return result"
  ]

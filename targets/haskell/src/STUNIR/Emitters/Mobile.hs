{-# LANGUAGE OverloadedStrings #-}

-- | Mobile platform code emitters
module STUNIR.Emitters.Mobile
  ( emitIOS
  , emitAndroid
  , MobilePlatform(..)
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import STUNIR.Emitters.Types

-- | Mobile platform
data MobilePlatform
  = IOS
  | Android
  | ReactNative
  | Flutter
  deriving (Show, Eq)

-- | Emit iOS Swift code
emitIOS :: Text -> EmitterResult Text
emitIOS moduleName = Right $ T.unlines
  [ "// STUNIR Generated iOS Swift"
  , "// Module: " <> moduleName
  , "// Generator: Haskell Pipeline"
  , ""
  , "import Foundation"
  , "import UIKit"
  , ""
  , "class " <> moduleName <> " {"
  , "    var value: Int"
  , "    "
  , "    init(value: Int) {"
  , "        self.value = value"
  , "    }"
  , "    "
  , "    func process() -> Int {"
  , "        return value * 2"
  , "    }"
  , "}"
  ]

-- | Emit Android Kotlin code
emitAndroid :: Text -> EmitterResult Text
emitAndroid moduleName = Right $ T.unlines
  [ "// STUNIR Generated Android Kotlin"
  , "// Module: " <> moduleName
  , "// Generator: Haskell Pipeline"
  , ""
  , "package com.stunir." <> T.toLower moduleName
  , ""
  , "import android.app.Activity"
  , "import android.os.Bundle"
  , ""
  , "class " <> moduleName <> "Activity : Activity() {"
  , "    override fun onCreate(savedInstanceState: Bundle?) {"
  , "        super.onCreate(savedInstanceState)"
  , "        // Implementation"
  , "    }"
  , "}"
  ]

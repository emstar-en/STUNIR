{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

{-|
Module      : STUNIR.Types
Description : Core type definitions for STUNIR
Copyright   : (c) STUNIR Team, 2026
License     : MIT

Production-ready type definitions with strong correctness guarantees.
-}

module STUNIR.Types
  ( IRDataType(..)
  , IRParameter(..)
  , IRStatement(..)
  , IRExpression(..)
  , IRFunction(..)
  , IRModule(..)
  , IRManifest(..)
  , toCType
  , toRustType
  , toPythonType
  ) where

import Data.Aeson
import Data.Text (Text)
import qualified Data.Text as T
import GHC.Generics

-- | IR data types
data IRDataType
  = TypeI8
  | TypeI16
  | TypeI32
  | TypeI64
  | TypeU8
  | TypeU16
  | TypeU32
  | TypeU64
  | TypeF32
  | TypeF64
  | TypeBool
  | TypeString
  | TypeVoid
  deriving (Eq, Show, Generic)

instance ToJSON IRDataType where
  toJSON TypeI8     = String "type_i8"
  toJSON TypeI16    = String "type_i16"
  toJSON TypeI32    = String "type_i32"
  toJSON TypeI64    = String "type_i64"
  toJSON TypeU8     = String "type_u8"
  toJSON TypeU16    = String "type_u16"
  toJSON TypeU32    = String "type_u32"
  toJSON TypeU64    = String "type_u64"
  toJSON TypeF32    = String "type_f32"
  toJSON TypeF64    = String "type_f64"
  toJSON TypeBool   = String "type_bool"
  toJSON TypeString = String "type_string"
  toJSON TypeVoid   = String "type_void"

instance FromJSON IRDataType where
  parseJSON = withText "IRDataType" $ \case
    "type_i8"     -> pure TypeI8
    "type_i16"    -> pure TypeI16
    "type_i32"    -> pure TypeI32
    "type_i64"    -> pure TypeI64
    "type_u8"     -> pure TypeU8
    "type_u16"    -> pure TypeU16
    "type_u32"    -> pure TypeU32
    "type_u64"    -> pure TypeU64
    "type_f32"    -> pure TypeF32
    "type_f64"    -> pure TypeF64
    "type_bool"   -> pure TypeBool
    "type_string" -> pure TypeString
    "type_void"   -> pure TypeVoid
    _             -> fail "Unknown type"

-- | Convert IR type to C type
toCType :: IRDataType -> Text
toCType TypeI8     = "int8_t"
toCType TypeI16    = "int16_t"
toCType TypeI32    = "int32_t"
toCType TypeI64    = "int64_t"
toCType TypeU8     = "uint8_t"
toCType TypeU16    = "uint16_t"
toCType TypeU32    = "uint32_t"
toCType TypeU64    = "uint64_t"
toCType TypeF32    = "float"
toCType TypeF64    = "double"
toCType TypeBool   = "bool"
toCType TypeString = "char*"
toCType TypeVoid   = "void"

-- | Convert IR type to Rust type
toRustType :: IRDataType -> Text
toRustType TypeI8     = "i8"
toRustType TypeI16    = "i16"
toRustType TypeI32    = "i32"
toRustType TypeI64    = "i64"
toRustType TypeU8     = "u8"
toRustType TypeU16    = "u16"
toRustType TypeU32    = "u32"
toRustType TypeU64    = "u64"
toRustType TypeF32    = "f32"
toRustType TypeF64    = "f64"
toRustType TypeBool   = "bool"
toRustType TypeString = "String"
toRustType TypeVoid   = "()"

-- | Convert IR type to Python type hint
toPythonType :: IRDataType -> Text
toPythonType TypeI8     = "int"
toPythonType TypeI16    = "int"
toPythonType TypeI32    = "int"
toPythonType TypeI64    = "int"
toPythonType TypeU8     = "int"
toPythonType TypeU16    = "int"
toPythonType TypeU32    = "int"
toPythonType TypeU64    = "int"
toPythonType TypeF32    = "float"
toPythonType TypeF64    = "float"
toPythonType TypeBool   = "bool"
toPythonType TypeString = "str"
toPythonType TypeVoid   = "None"

-- | IR parameter
data IRParameter = IRParameter
  { paramName :: Text
  , paramType :: IRDataType
  } deriving (Eq, Show, Generic)

instance ToJSON IRParameter where
  toJSON (IRParameter n t) = object ["name" .= n, "param_type" .= t]

instance FromJSON IRParameter where
  parseJSON = withObject "IRParameter" $ \v ->
    IRParameter <$> v .: "name" <*> v .: "param_type"

-- | IR expression
data IRExpression
  = Literal Value
  | Variable Text
  | BinaryOp Text IRExpression IRExpression
  deriving (Eq, Show, Generic)

instance ToJSON IRExpression
instance FromJSON IRExpression

-- | IR statement
data IRStatement
  = Return (Maybe IRExpression)
  | Assignment Text IRExpression
  | Call Text [IRExpression]
  deriving (Eq, Show, Generic)

instance ToJSON IRStatement
instance FromJSON IRStatement

-- | IR function
data IRFunction = IRFunction
  { funcName       :: Text
  , funcReturnType :: IRDataType
  , funcParameters :: [IRParameter]
  , funcBody       :: [IRStatement]
  } deriving (Eq, Show, Generic)

instance ToJSON IRFunction where
  toJSON (IRFunction n r p b) = object
    [ "name"        .= n
    , "return_type" .= r
    , "parameters"  .= p
    , "body"        .= b
    ]

instance FromJSON IRFunction where
  parseJSON = withObject "IRFunction" $ \v ->
    IRFunction <$> v .: "name"
               <*> v .: "return_type"
               <*> v .: "parameters"
               <*> v .: "body"

-- | IR module
data IRModule = IRModule
  { moduleName      :: Text
  , moduleVersion   :: Text
  , moduleFunctions :: [IRFunction]
  } deriving (Eq, Show, Generic)

instance ToJSON IRModule where
  toJSON (IRModule n v f) = object
    [ "name"      .= n
    , "version"   .= v
    , "functions" .= f
    ]

instance FromJSON IRModule where
  parseJSON = withObject "IRModule" $ \v ->
    IRModule <$> v .: "name"
             <*> v .: "version"
             <*> v .: "functions"

-- | IR manifest
data IRManifest = IRManifest
  { manifestSchema :: Text
  , manifestIRHash :: Text
  , manifestModule :: IRModule
  } deriving (Eq, Show, Generic)

instance ToJSON IRManifest where
  toJSON (IRManifest s h m) = object
    [ "schema"  .= s
    , "ir_hash" .= h
    , "module"  .= m
    ]

instance FromJSON IRManifest where
  parseJSON = withObject "IRManifest" $ \v ->
    IRManifest <$> v .: "schema"
               <*> v .: "ir_hash"
               <*> v .: "module"

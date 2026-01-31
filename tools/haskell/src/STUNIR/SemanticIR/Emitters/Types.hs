{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

{-|
Module      : STUNIR.SemanticIR.Emitters.Types
Description : Emitter-specific type definitions
Copyright   : (c) STUNIR Team, 2026
License     : MIT
Maintainer  : stunir@example.com

Core types and enumerations for Semantic IR emitters.
Based on Ada SPARK Emitter_Types package.
-}

module STUNIR.SemanticIR.Emitters.Types
  ( -- * IR Data Types
    IRDataType(..)
  , IRStatementType(..)
  , -- * Architecture Types
    Architecture(..)
  , Endianness(..)
  , ArchConfig(..)
  , archConfigs
  , -- * IR Types
    IRStatement(..)
  , IRParameter(..)
  , IRFunction(..)
  , IRTypeField(..)
  , IRType(..)
  , IRModule(..)
  , -- * Type Mappings
    mapIRTypeToC
  , mapIRTypeToRust
  , mapIRTypeToPython
  , getArchConfig
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Map.Strict as Map
import GHC.Generics

-- | IR data types (matching SPARK IR_Data_Type enumeration)
data IRDataType
  = TypeVoid
  | TypeBool
  | TypeI8
  | TypeI16
  | TypeI32
  | TypeI64
  | TypeU8
  | TypeU16
  | TypeU32
  | TypeU64
  | TypeF32
  | TypeF64
  | TypeChar
  | TypeString
  | TypePointer
  | TypeArray
  | TypeStruct
  deriving (Eq, Show, Read, Generic)

-- | IR statement types (matching SPARK IR_Statement_Type enumeration)
data IRStatementType
  = StmtNop
  | StmtVarDecl
  | StmtAssign
  | StmtReturn
  | StmtAdd
  | StmtSub
  | StmtMul
  | StmtDiv
  | StmtCall
  | StmtIf
  | StmtLoop
  | StmtBreak
  | StmtContinue
  | StmtBlock
  deriving (Eq, Show, Read, Generic)

-- | Architecture types (matching SPARK Architecture_Type enumeration)
data Architecture
  = ArchARM
  | ArchARM64
  | ArchAVR
  | ArchMIPS
  | ArchRISCV
  | ArchX86
  | ArchX86_64
  | ArchPowerPC
  | ArchGeneric
  deriving (Eq, Show, Read, Ord, Generic)

-- | Endianness types (matching SPARK Endianness_Type enumeration)
data Endianness
  = LittleEndian
  | BigEndian
  deriving (Eq, Show, Read, Generic)

-- | Architecture configuration (matching SPARK Arch_Config_Type record)
data ArchConfig = ArchConfig
  { acWordSize       :: !Int       -- ^ 8 to 64 bits
  , acEndianness     :: !Endianness
  , acAlignment      :: !Int       -- ^ 1 to 16 bytes
  , acStackGrowsDown :: !Bool
  } deriving (Eq, Show, Generic)

-- | IR statement representation
data IRStatement = IRStatement
  { isType     :: !IRStatementType
  , isDataType :: !(Maybe IRDataType)
  , isTarget   :: !(Maybe Text)
  , isValue    :: !(Maybe Text)
  , isLeftOp   :: !(Maybe Text)
  , isRightOp  :: !(Maybe Text)
  } deriving (Eq, Show, Generic)

-- | Function parameter representation
data IRParameter = IRParameter
  { ipName :: !Text
  , ipType :: !IRDataType
  } deriving (Eq, Show, Generic)

-- | Function representation
data IRFunction = IRFunction
  { ifName       :: !Text
  , ifReturnType :: !IRDataType
  , ifParameters :: ![IRParameter]
  , ifStatements :: ![IRStatement]
  , ifDocstring  :: !(Maybe Text)
  } deriving (Eq, Show, Generic)

-- | Type field representation
data IRTypeField = IRTypeField
  { itfName     :: !Text
  , itfType     :: !Text
  , itfOptional :: !Bool
  } deriving (Eq, Show, Generic)

-- | Custom type/struct definition
data IRType = IRType
  { itName      :: !Text
  , itFields    :: ![IRTypeField]
  , itDocstring :: !(Maybe Text)
  } deriving (Eq, Show, Generic)

-- | Complete IR module representation
data IRModule = IRModule
  { imIRVersion  :: !Text
  , imModuleName :: !Text
  , imTypes      :: ![IRType]
  , imFunctions  :: ![IRFunction]
  , imDocstring  :: !(Maybe Text)
  } deriving (Eq, Show, Generic)

-- | Map IR data type to C type name (matching SPARK Map_IR_Type_To_C)
mapIRTypeToC :: IRDataType -> Text
mapIRTypeToC TypeVoid    = "void"
mapIRTypeToC TypeBool    = "bool"
mapIRTypeToC TypeI8      = "int8_t"
mapIRTypeToC TypeI16     = "int16_t"
mapIRTypeToC TypeI32     = "int32_t"
mapIRTypeToC TypeI64     = "int64_t"
mapIRTypeToC TypeU8      = "uint8_t"
mapIRTypeToC TypeU16     = "uint16_t"
mapIRTypeToC TypeU32     = "uint32_t"
mapIRTypeToC TypeU64     = "uint64_t"
mapIRTypeToC TypeF32     = "float"
mapIRTypeToC TypeF64     = "double"
mapIRTypeToC TypeChar    = "char"
mapIRTypeToC TypeString  = "char*"
mapIRTypeToC TypePointer = "void*"
mapIRTypeToC TypeArray   = "array"
mapIRTypeToC TypeStruct  = "struct"

-- | Map IR data type to Rust type name
mapIRTypeToRust :: IRDataType -> Text
mapIRTypeToRust TypeVoid    = "()"
mapIRTypeToRust TypeBool    = "bool"
mapIRTypeToRust TypeI8      = "i8"
mapIRTypeToRust TypeI16     = "i16"
mapIRTypeToRust TypeI32     = "i32"
mapIRTypeToRust TypeI64     = "i64"
mapIRTypeToRust TypeU8      = "u8"
mapIRTypeToRust TypeU16     = "u16"
mapIRTypeToRust TypeU32     = "u32"
mapIRTypeToRust TypeU64     = "u64"
mapIRTypeToRust TypeF32     = "f32"
mapIRTypeToRust TypeF64     = "f64"
mapIRTypeToRust TypeChar    = "char"
mapIRTypeToRust TypeString  = "String"
mapIRTypeToRust TypePointer = "*const c_void"
mapIRTypeToRust TypeArray   = "Vec"
mapIRTypeToRust TypeStruct  = "struct"

-- | Map IR data type to Python type hint
mapIRTypeToPython :: IRDataType -> Text
mapIRTypeToPython TypeVoid    = "None"
mapIRTypeToPython TypeBool    = "bool"
mapIRTypeToPython TypeI8      = "int"
mapIRTypeToPython TypeI16     = "int"
mapIRTypeToPython TypeI32     = "int"
mapIRTypeToPython TypeI64     = "int"
mapIRTypeToPython TypeU8      = "int"
mapIRTypeToPython TypeU16     = "int"
mapIRTypeToPython TypeU32     = "int"
mapIRTypeToPython TypeU64     = "int"
mapIRTypeToPython TypeF32     = "float"
mapIRTypeToPython TypeF64     = "float"
mapIRTypeToPython TypeChar    = "str"
mapIRTypeToPython TypeString  = "str"
mapIRTypeToPython TypePointer = "Any"
mapIRTypeToPython TypeArray   = "List"
mapIRTypeToPython TypeStruct  = "Dict"

-- | Architecture configuration presets
archConfigs :: Map.Map Architecture ArchConfig
archConfigs = Map.fromList
  [ (ArchARM, ArchConfig 32 LittleEndian 4 True)
  , (ArchARM64, ArchConfig 64 LittleEndian 8 True)
  , (ArchAVR, ArchConfig 8 LittleEndian 1 True)
  , (ArchMIPS, ArchConfig 32 BigEndian 4 True)
  , (ArchRISCV, ArchConfig 32 LittleEndian 4 True)
  , (ArchX86, ArchConfig 32 LittleEndian 4 True)
  , (ArchX86_64, ArchConfig 64 LittleEndian 8 True)
  , (ArchPowerPC, ArchConfig 32 BigEndian 4 True)
  , (ArchGeneric, ArchConfig 32 LittleEndian 4 True)
  ]

-- | Get architecture configuration (matching SPARK Get_Arch_Config)
getArchConfig :: Architecture -> ArchConfig
getArchConfig arch =
  Map.findWithDefault (archConfigs Map.! ArchGeneric) arch archConfigs

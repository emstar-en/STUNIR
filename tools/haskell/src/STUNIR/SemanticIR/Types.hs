{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

-- | STUNIR Semantic IR Core Types
--   DO-178C Level A Compliant
--   Type definitions with JSON serialization

module STUNIR.SemanticIR.Types
    ( IRName
    , IRHash
    , IRPath
    , NodeID
    , IRPrimitiveType(..)
    , IRNodeKind(..)
    , BinaryOperator(..)
    , UnaryOperator(..)
    , StorageClass(..)
    , VisibilityKind(..)
    , MutabilityKind(..)
    , InlineHint(..)
    , TargetCategory(..)
    , SafetyLevel(..)
    , SourceLocation(..)
    ) where

import Data.Aeson
import qualified Data.Text as T
import GHC.Generics

-- Type aliases
type IRName = T.Text
type IRHash = T.Text
type IRPath = T.Text
type NodeID = T.Text

-- Primitive types
data IRPrimitiveType
    = TypeVoid
    | TypeBool
    | TypeI8 | TypeI16 | TypeI32 | TypeI64
    | TypeU8 | TypeU16 | TypeU32 | TypeU64
    | TypeF32 | TypeF64
    | TypeString
    | TypeChar
    deriving (Show, Eq, Generic)

instance ToJSON IRPrimitiveType
instance FromJSON IRPrimitiveType

-- Node kinds
data IRNodeKind
    = KindModule
    | KindFunctionDecl
    | KindTypeDecl
    | KindConstDecl
    | KindVarDecl
    | KindBlockStmt
    | KindExprStmt
    | KindIfStmt
    | KindWhileStmt
    | KindForStmt
    | KindReturnStmt
    | KindBreakStmt
    | KindContinueStmt
    | KindVarDeclStmt
    | KindAssignStmt
    | KindIntegerLiteral
    | KindFloatLiteral
    | KindStringLiteral
    | KindBoolLiteral
    | KindVarRef
    | KindBinaryExpr
    | KindUnaryExpr
    | KindFunctionCall
    | KindMemberExpr
    | KindArrayAccess
    | KindCastExpr
    | KindTernaryExpr
    deriving (Show, Eq, Generic)

instance ToJSON IRNodeKind
instance FromJSON IRNodeKind

-- Binary operators
data BinaryOperator
    = OpAdd | OpSub | OpMul | OpDiv | OpMod
    | OpEq | OpNeq | OpLt | OpLeq | OpGt | OpGeq
    | OpAnd | OpOr
    | OpBitAnd | OpBitOr | OpBitXor | OpShl | OpShr
    | OpAssign
    deriving (Show, Eq, Generic)

instance ToJSON BinaryOperator
instance FromJSON BinaryOperator

-- Unary operators
data UnaryOperator
    = OpNeg | OpNot | OpBitNot
    | OpPreInc | OpPreDec | OpPostInc | OpPostDec
    | OpDeref | OpAddrOf
    deriving (Show, Eq, Generic)

instance ToJSON UnaryOperator
instance FromJSON UnaryOperator

-- Storage class
data StorageClass
    = StorageAuto | StorageStatic | StorageExtern
    | StorageRegister | StorageStack | StorageHeap | StorageGlobal
    deriving (Show, Eq, Generic)

instance ToJSON StorageClass
instance FromJSON StorageClass

-- Visibility
data VisibilityKind
    = VisPublic | VisPrivate | VisProtected | VisInternal
    deriving (Show, Eq, Generic)

instance ToJSON VisibilityKind
instance FromJSON VisibilityKind

-- Mutability
data MutabilityKind
    = MutMutable | MutImmutable | MutConst
    deriving (Show, Eq, Generic)

instance ToJSON MutabilityKind
instance FromJSON MutabilityKind

-- Inline hint
data InlineHint
    = InlineAlways | InlineNever | InlineHint | InlineNone
    deriving (Show, Eq, Generic)

instance ToJSON InlineHint
instance FromJSON InlineHint

-- Target categories
data TargetCategory
    = TargetEmbedded
    | TargetRealtime
    | TargetSafetyCritical
    | TargetGpu
    | TargetWasm
    | TargetNative
    deriving (Show, Eq, Generic)

instance ToJSON TargetCategory
instance FromJSON TargetCategory

-- Safety level
data SafetyLevel
    = LevelNone
    | LevelDO178C_D
    | LevelDO178C_C
    | LevelDO178C_B
    | LevelDO178C_A
    deriving (Show, Eq, Generic)

instance ToJSON SafetyLevel
instance FromJSON SafetyLevel

-- Source location
data SourceLocation = SourceLocation
    { locFile   :: IRPath
    , locLine   :: Int
    , locColumn :: Int
    , locLength :: Int
    } deriving (Show, Eq, Generic)

instance ToJSON SourceLocation where
    toJSON loc = object
        [ "file" .= locFile loc
        , "line" .= locLine loc
        , "column" .= locColumn loc
        , "length" .= locLength loc
        ]

instance FromJSON SourceLocation where
    parseJSON = withObject "SourceLocation" $ \v -> SourceLocation
        <$> v .: "file"
        <*> v .: "line"
        <*> v .: "column"
        <*> v .:? "length" .!= 0

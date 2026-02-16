-- Core type definitions for reference (CONDENSED VERSION)

-- Statement types
type IR_Statement_Type is (
   Stmt_Nop, Stmt_Var_Decl, Stmt_Assign, Stmt_Return,
   Stmt_Add, Stmt_Sub, Stmt_Mul, Stmt_Div,
   Stmt_Call, Stmt_If, Stmt_Loop, Stmt_Break, Stmt_Continue, Stmt_Block
);

-- Data types
type IR_Data_Type is (
   Type_Void, Type_Bool,
   Type_I8, Type_I16, Type_I32, Type_I64,
   Type_U8, Type_U16, Type_U32, Type_U64,
   Type_F32, Type_F64,
   Type_Char, Type_String, Type_Pointer, Type_Array, Type_Struct
);

-- Status codes
type Emitter_Status is (
   Success,
   Error_Invalid_IR,
   Error_Write_Failed,
   Error_Unsupported_Type,
   Error_Buffer_Overflow,
   Error_Invalid_Architecture
);

-- Bounded strings (max lengths)
Max_Identifier_Length : constant := 128;
Max_Content_Length    : constant := 4096;

subtype Identifier_String is Identifier_Strings.Bounded_String;
subtype Content_String is Content_Strings.Bounded_String;

-- Embedded statement record
type Embedded_Statement is record
   Stmt_Type : IR_Statement_Type;
   Data_Type : IR_Data_Type;
   Target    : Identifier_String;  -- Variable name receiving result
   Value     : Content_String;      -- Literal value or expression
   Left_Op   : Identifier_String;  -- Left operand for binary ops
   Right_Op  : Identifier_String;  -- Right operand for binary ops
end record;

-- Helper functions available
procedure Append_To_Content (
   Content : in out Content_String;
   Text    : in String;
   Status  : out Emitter_Status);

function Get_C_Type (
   Data_Type : IR_Data_Type;
   Config    : Embedded_Config) return Type_Name_String;

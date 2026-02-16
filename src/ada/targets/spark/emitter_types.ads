--  STUNIR Emitter Types - Ada SPARK
--  Core types for all target emitters
--  DO-178C Level A compliant implementation
--
--  SPARK_Mode: On - Enables formal verification

pragma SPARK_Mode (On);

with Ada.Strings.Bounded;

package Emitter_Types is
   
   --  Maximum sizes for bounded strings (safety-critical bounds)
   Max_Path_Length      : constant := 1024;
   Max_Content_Length   : constant := 4096;
   Max_Type_Name_Length : constant := 64;
   Max_Identifier_Length: constant := 128;
   Max_Hash_Length      : constant := 64;
   Max_Files_Count      : constant := 1000;
   
   --  Bounded string packages for memory-safe operations
   package Path_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length
      (Max => Max_Path_Length);
   package Content_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length
      (Max => Max_Content_Length);
   package Type_Name_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length
      (Max => Max_Type_Name_Length);
   package Identifier_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length
      (Max => Max_Identifier_Length);
   package Hash_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length
      (Max => Max_Hash_Length);
   
   --  Subtype aliases for convenience
   subtype Path_String is Path_Strings.Bounded_String;
   subtype Content_String is Content_Strings.Bounded_String;
   subtype Type_Name_String is Type_Name_Strings.Bounded_String;
   subtype Identifier_String is Identifier_Strings.Bounded_String;
   subtype Hash_String is Hash_Strings.Bounded_String;
   
   --  Architecture enumeration for embedded/assembly targets
   type Architecture_Type is (
      Arch_ARM,
      Arch_ARM64,
      Arch_AVR,
      Arch_MIPS,
      Arch_RISCV,
      Arch_X86,
      Arch_X86_64,
      Arch_PowerPC,
      Arch_Generic
   );
   
   --  Endianness type
   type Endianness_Type is (Little_Endian, Big_Endian);
   
   --  Architecture configuration record
   type Arch_Config_Type is record
      Word_Size   : Positive range 8 .. 64;
      Endianness  : Endianness_Type;
      Alignment   : Positive range 1 .. 16;
      Stack_Grows_Down : Boolean;
   end record;
   
   --  IR statement types
   type IR_Statement_Type is (
      Stmt_Nop,
      Stmt_Var_Decl,
      Stmt_Assign,
      Stmt_Return,
      Stmt_Add,
      Stmt_Sub,
      Stmt_Mul,
      Stmt_Div,
      Stmt_Call,
      Stmt_If,
      Stmt_Loop,
      Stmt_Break,
      Stmt_Continue,
      Stmt_Block
   );
   
   --  IR data types
   type IR_Data_Type is (
      Type_Void,
      Type_Bool,
      Type_I8,
      Type_I16,
      Type_I32,
      Type_I64,
      Type_U8,
      Type_U16,
      Type_U32,
      Type_U64,
      Type_F32,
      Type_F64,
      Type_Char,
      Type_String,
      Type_Pointer,
      Type_Array,
      Type_Struct
   );
   
   --  Generated file record
   type Generated_File_Record is record
      Path   : Path_String;
      Hash   : Hash_String;
      Size   : Natural;
   end record;
   
   --  Array of generated files
   type Generated_Files_Array is array (Positive range <>) of Generated_File_Record;
   
   --  Emitter result status
   type Emitter_Status is (
      Success,
      Error_Invalid_IR,
      Error_Write_Failed,
      Error_Unsupported_Type,
      Error_Buffer_Overflow,
      Error_Invalid_Architecture
   );
   
   --  Emitter result record
   type Emitter_Result is record
      Status : Emitter_Status;
      Files_Count : Natural;
      Total_Size  : Natural;
   end record;
   
   --  Helper functions for type mapping
   function Map_IR_Type_To_C (IR_Type : IR_Data_Type) return Type_Name_String
      with Post => Type_Name_Strings.Length (Map_IR_Type_To_C'Result) > 0;
   
   function Get_Arch_Config (Arch : Architecture_Type) return Arch_Config_Type;
   
   --  Hash computation (placeholder - actual impl in body)
   function Compute_SHA256 (Content : String) return Hash_String
      with Pre => Content'Length <= Max_Content_Length,
           Post => Hash_Strings.Length (Compute_SHA256'Result) = 64;

end Emitter_Types;

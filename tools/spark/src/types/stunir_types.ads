--  STUNIR Common Types Package
--  DO-333 / SPARK 2014 compliant
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  This package defines common types used across all STUNIR SPARK components.
--  All types are bounded and suitable for formal verification.

pragma SPARK_Mode (On);

with Ada.Strings.Bounded;

package STUNIR_Types is

   --  ========================================================================
   --  String Type Definitions
   --  ========================================================================

   --  Maximum sizes for bounded strings
   Max_JSON_Length       : constant := 100_000;  --  100KB for JSON content
   Max_Identifier_Length : constant := 256;        --  Function/variable names
   Max_Type_Length       : constant := 128;        --  Type names
   Max_Type_Name_Length  : constant := 128;        --  Type names (alias)
   Max_Path_Length       : constant := 4096;       --  File paths
   Max_Error_Length      : constant := 512;        --  Error messages
   Max_Code_Length       : constant := 100_000;    --  100KB for code output

   --  Bounded string packages
   package JSON_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length
     (Max => Max_JSON_Length);
   subtype JSON_String is JSON_Strings.Bounded_String;

   package Identifier_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length
     (Max => Max_Identifier_Length);
   subtype Identifier_String is Identifier_Strings.Bounded_String;

   package Type_Name_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length
     (Max => Max_Type_Length);
   subtype Type_Name_String is Type_Name_Strings.Bounded_String;

   package Path_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length
     (Max => Max_Path_Length);
   subtype Path_String is Path_Strings.Bounded_String;

   package Error_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length
     (Max => Max_Error_Length);
   subtype Error_String is Error_Strings.Bounded_String;

   package Code_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length
     (Max => Max_Code_Length);
   subtype Code_String is Code_Strings.Bounded_String;

   --  ========================================================================
   --  Status and Error Types
   --  ========================================================================

   type Status_Code is (
      Success,
      Error_File_Not_Found,
      Error_File_Read,
      Error_File_Write,
      Error_Invalid_JSON,
      Error_Invalid_Schema,
      Error_Invalid_Syntax,
      Error_Buffer_Overflow,
      Error_Unsupported_Type,
      Error_Parse_Error,
      Error_Validation_Failed,
      Error_Conversion_Failed,
      Error_Emission_Failed,
      Error_Not_Implemented,
      Error_Invalid_Format,
      Error_Empty_Extraction,
      Error_Too_Large,
      Error_Parse,
      Error_File_IO,
      Error_Invalid_Input
   );

   --  Status code helper functions
   function Status_Code_Image (Status : Status_Code) return String
     with Global => null;

   function Is_Success (Status : Status_Code) return Boolean
     with Global => null,
          Post => Is_Success'Result = (Status = Success);

   function Is_Error (Status : Status_Code) return Boolean
     with Global => null,
          Post => Is_Error'Result = (Status /= Success);

   --  ========================================================================
   --  Schema Version Types
   --  ========================================================================

   type Schema_Kind is (
      Schema_Extraction_V1,
      Schema_Spec_V1,
      Schema_IR_V1,
      Schema_Unknown
   );

   --  ========================================================================
   --  Function Parameter Types
   --  ========================================================================

   Max_Parameters : constant := 4;

   type Parameter_Index is range 0 .. Max_Parameters;

   type Parameter is record
      Name       : Identifier_String;
      Param_Type : Type_Name_String;
   end record;

   type Parameter_Array is array (Parameter_Index range <>) of Parameter;

   --  Full parameter list with count
   type Parameter_List is record
      Params : Parameter_Array (1 .. Max_Parameters);
      Count  : Parameter_Index;
   end record
     with Dynamic_Predicate =>
       (for all I in 1 .. Parameter_List.Count =>
          Identifier_Strings.Length (Parameter_List.Params (I).Name) > 0);

   --  ========================================================================
   --  Function Signature Types
   --  ========================================================================

   type Function_Signature is record
      Name        : Identifier_String;
      Return_Type : Type_Name_String;
      Parameters  : Parameter_List;
   end record
     with Dynamic_Predicate =>
       Identifier_Strings.Length (Function_Signature.Name) > 0;

   --  ========================================================================
   --  Function Collection Types
   --  ========================================================================

   Max_Functions : constant := 16;  --  Reduced to prevent stack overflow

   type Function_Index is range 0 .. Max_Functions;

   type Function_Array is array (Function_Index range <>) of Function_Signature;

   type Function_Collection is record
      Functions : Function_Array (1 .. Max_Functions);
      Count     : Function_Index;
   end record;

   --  ========================================================================
   --  Import/Export Types
   --  ========================================================================

   Max_Imports : constant := 16;
   Max_Exports : constant := 16;

   type Import_Index is range 0 .. Max_Imports;
   type Export_Index is range 0 .. Max_Exports;

   type Import_Entry is record
      Name        : Identifier_String;  --  Import name (module or symbol)
      From_Module : Identifier_String;  --  Source module (optional)
   end record;

   type Export_Entry is record
      Name        : Identifier_String;  --  Exported symbol name
      Export_Type : Type_Name_String;   --  Type (function, type, constant)
   end record;

   type Import_Array is array (Import_Index range <>) of Import_Entry;
   type Export_Array is array (Export_Index range <>) of Export_Entry;

   type Import_Collection is record
      Imports : Import_Array (1 .. Max_Imports);
      Count   : Import_Index;
   end record;

   type Export_Collection is record
      Exports : Export_Array (1 .. Max_Exports);
      Count   : Export_Index;
   end record;

   --  ========================================================================
   --  Type Definition Types
   --  ========================================================================

   Max_Type_Defs : constant := 8;
   Max_Type_Fields : constant := 8;

   type Type_Def_Index is range 0 .. Max_Type_Defs;
   type Type_Field_Index is range 0 .. Max_Type_Fields;

   type Type_Field is record
      Name       : Identifier_String;
      Field_Type : Type_Name_String;
   end record;

   type Type_Field_Array is array (Type_Field_Index range <>) of Type_Field;

   type Type_Field_Collection is record
      Fields : Type_Field_Array (1 .. Max_Type_Fields);
      Count  : Type_Field_Index;
   end record;

   type Type_Kind is (Type_Struct, Type_Enum, Type_Alias, Type_Generic);

   type Type_Definition is record
      Name       : Identifier_String;
      Kind       : Type_Kind;
      Fields     : Type_Field_Collection;  --  For struct types
      Base_Type  : Type_Name_String;        --  For alias/generic types
   end record;

   type Type_Def_Array is array (Type_Def_Index range <>) of Type_Definition;

   type Type_Def_Collection is record
      Type_Defs : Type_Def_Array (1 .. Max_Type_Defs);
      Count     : Type_Def_Index;
   end record;

   --  ========================================================================
   --  Constant Definition Types
   --  ========================================================================

   Max_Constants : constant := 8;

   type Constant_Index is range 0 .. Max_Constants;

   type Constant_Definition is record
      Name         : Identifier_String;
      Const_Type   : Type_Name_String;
      Value_Str    : Identifier_String;  --  String representation of value
   end record;

   type Constant_Array is array (Constant_Index range <>) of Constant_Definition;

   type Constant_Collection is record
      Constants : Constant_Array (1 .. Max_Constants);
      Count     : Constant_Index;
   end record;

   --  ========================================================================
   --  Dependency Types
   --  ========================================================================

   Max_Dependencies : constant := 8;

   type Dependency_Index is range 0 .. Max_Dependencies;

   type Dependency_Entry is record
      Name    : Identifier_String;
      Version : Identifier_String;  --  Version string (optional)
   end record;

   type Dependency_Array is array (Dependency_Index range <>) of Dependency_Entry;

   type Dependency_Collection is record
      Dependencies : Dependency_Array (1 .. Max_Dependencies);
      Count        : Dependency_Index;
   end record;

   --  ========================================================================
   --  Module Types (Extended)
   --  ========================================================================

   type Module is record
      Name         : Identifier_String;
      Imports      : Import_Collection;
      Exports      : Export_Collection;
      Types        : Type_Def_Collection;
      Constants    : Constant_Collection;
      Functions    : Function_Collection;
   end record;

   --  ========================================================================
   --  IR Step Types (for IR representation)
   --  ========================================================================

   --  Extended step types aligned with stunir_ir_v1.schema.json
   type Step_Type_Enum is (
      --  Basic ops
      Step_Nop,
      Step_Assign,
      Step_Call,
      Step_Return,
      Step_Error,
      --  Control flow
      Step_If,
      Step_While,
      Step_For,
      Step_Break,
      Step_Continue,
      Step_Switch,
      --  Exceptions
      Step_Try,
      Step_Throw,
      --  Array operations
      Step_Array_New,
      Step_Array_Get,
      Step_Array_Set,
      Step_Array_Push,
      Step_Array_Pop,
      Step_Array_Len,
      --  Map operations
      Step_Map_New,
      Step_Map_Get,
      Step_Map_Set,
      Step_Map_Delete,
      Step_Map_Has,
      Step_Map_Keys,
      --  Set operations
      Step_Set_New,
      Step_Set_Add,
      Step_Set_Remove,
      Step_Set_Has,
      Step_Set_Union,
      Step_Set_Intersect,
      --  Struct operations
      Step_Struct_New,
      Step_Struct_Get,
      Step_Struct_Set,
      --  Advanced
      Step_Generic_Call,
      Step_Type_Cast
   );

   Max_Steps : constant := 16;  --  Reduced to prevent stack overflow

   type Step_Index is range 0 .. Max_Steps;

   --  Maximum nesting depth for blocks
   Max_Block_Depth : constant := 8;
   Max_Cases       : constant := 4;  --  Reduced to prevent stack overflow
   Max_Catch_Blocks : constant := 4;

   --  Case statement entry
   type Case_Entry is record
      Case_Value : Identifier_String;  --  String or integer value
      Body_Start : Step_Index;          --  Index into step array for body start
      Body_Count : Step_Index;          --  Number of steps in case body
   end record;

   type Case_Array is array (1 .. Max_Cases) of Case_Entry;

   --  Catch block entry
   type Catch_Entry is record
      Exception_Type : Identifier_String;  --  Exception type to catch
      Var_Name       : Identifier_String;  --  Variable name for exception
      Body_Start     : Step_Index;          --  Index into step array
      Body_Count     : Step_Index;          --  Number of steps in catch body
   end record;

   type Catch_Array is array (1 .. Max_Catch_Blocks) of Catch_Entry;

   --  Extended IR step with all payload fields
   type IR_Step is record
      Step_Type   : Step_Type_Enum;
      --  Common fields
      Target      : Identifier_String;  --  Target variable (assign, call, etc.)
      Value       : Identifier_String;  --  Value expression (assign, return, call)
      --  Control flow fields
      Condition   : Identifier_String;  --  Condition expression (if, while)
      Init        : Identifier_String;  --  Init expression (for)
      Increment   : Identifier_String;  --  Increment expression (for)
      --  Block indices (referencing global step array)
      Then_Start  : Step_Index;          --  Then block start (if)
      Then_Count  : Step_Index;          --  Then block count
      Else_Start  : Step_Index;          --  Else block start (if)
      Else_Count  : Step_Index;          --  Else block count
      Body_Start  : Step_Index;          --  Body start (while, for, try)
      Body_Count  : Step_Index;          --  Body count
      --  Switch/case fields
      Expr        : Identifier_String;   --  Switch expression
      Cases       : Case_Array;          --  Case entries
      Case_Count  : Step_Index;          --  Number of cases
      Default_Start : Step_Index;        --  Default case start
      Default_Count : Step_Index;        --  Default case count
      --  Exception handling fields
      Try_Start     : Step_Index;        --  Try block start
      Try_Count     : Step_Index;        --  Try block count
      Catch_Blocks  : Catch_Array;       --  Catch block entries
      Catch_Count   : Step_Index;        --  Number of catch blocks
      Finally_Start : Step_Index;        --  Finally block start
      Finally_Count : Step_Index;        --  Finally block count
      --  Data structure fields
      Index       : Identifier_String;   --  Array index
      Key         : Identifier_String;   --  Map key
      Field       : Identifier_String;   --  Struct field
      --  Call fields
      Args        : Identifier_String;   --  Arguments as string (comma-separated)
      Type_Args   : Identifier_String;   --  Type arguments (generic_call)
      --  Error message
      Error_Msg   : Identifier_String;   --  Error message (error op)
   end record;

   --  Function to create a default step (avoids stack overflow with large constant)
   function Make_Default_Step return IR_Step;

   type Step_Array is array (Step_Index range <>) of IR_Step;

   type Step_Collection is record
      Steps : Step_Array (1 .. Max_Steps);
      Count : Step_Index;
   end record;

   --  Procedure to initialize a step collection (avoids stack overflow)
   procedure Init_Step_Collection (Steps : out Step_Collection);

   --  ========================================================================
   --  IR Function Types
   --  ========================================================================

   type IR_Function is record
      Name        : Identifier_String;
      Return_Type : Type_Name_String;
      Parameters  : Parameter_List;
      Steps       : Step_Collection;
   end record;

   type IR_Function_Array is array (Function_Index range <>) of IR_Function;

   type IR_Function_Collection is record
      Functions : IR_Function_Array (1 .. Max_Functions);
      Count     : Function_Index;
   end record;

   --  ========================================================================
   --  IR Data Types (Top-level IR structure)
   --  ========================================================================

   type IR_Data is record
      Schema_Version : Identifier_String;
      IR_Version     : Identifier_String;
      Module_Name    : Identifier_String;
      Imports        : Import_Collection;
      Exports        : Export_Collection;
      Types          : Type_Def_Collection;
      Constants      : Constant_Collection;
      Dependencies   : Dependency_Collection;
      Functions      : IR_Function_Collection;
   end record;

   --  ========================================================================
   --  Target Language Types
   --  ========================================================================

   type Target_Language is (
      --  Mainstream languages
      Target_CPP,
      Target_C,
      Target_Python,
      Target_Rust,
      Target_Go,
      Target_JavaScript,
      Target_Java,
      Target_CSharp,
      Target_Swift,
      Target_Kotlin,
      Target_SPARK,
      Target_Ada,
      --  Lisp family
      Target_Common_Lisp,
      Target_Scheme,
      Target_Racket,
      Target_Emacs_Lisp,
      Target_Guile,
      Target_Hy,
      Target_Janet,
      Target_Clojure,
      Target_ClojureScript,
      --  Prolog family
      Target_SWI_Prolog,
      Target_GNU_Prolog,
      Target_Mercury,
      Target_Prolog,  --  Generic Prolog (deprecated, use specific variant)
      --  Functional/Formal languages
      Target_Futhark,
      Target_Lean4,
      Target_Haskell
   );

   subtype All_Targets is Target_Language range Target_CPP .. Target_Haskell;

   --  ========================================================================
   --  Constants
   --  ========================================================================

   Null_Identifier : constant Identifier_String :=
     Identifier_Strings.Null_Bounded_String;

   Null_Type_Name : constant Type_Name_String :=
     Type_Name_Strings.Null_Bounded_String;

   Null_Error_String : constant Error_String :=
     Error_Strings.Null_Bounded_String;

   --  ========================================================================
end STUNIR_Types;

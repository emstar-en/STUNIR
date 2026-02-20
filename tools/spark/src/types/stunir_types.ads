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
   Max_JSON_Length       : constant := 1_000_000;  --  1MB for JSON content
   Max_Identifier_Length : constant := 256;        --  Function/variable names
   Max_Type_Length       : constant := 128;        --  Type names
   Max_Type_Name_Length  : constant := 128;        --  Type names (alias)
   Max_Path_Length       : constant := 4096;       --  File paths
   Max_Error_Length      : constant := 512;        --  Error messages

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

   Max_Parameters : constant := 32;

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

   Max_Functions : constant := 1000;

   type Function_Index is range 0 .. Max_Functions;

   type Function_Array is array (Function_Index range <>) of Function_Signature;

   type Function_Collection is record
      Functions : Function_Array (1 .. Max_Functions);
      Count     : Function_Index;
   end record;

   --  ========================================================================
   --  Module Types
   --  ========================================================================

   type Module is record
      Name      : Identifier_String;
      Functions : Function_Collection;
   end record;

   --  ========================================================================
   --  IR Step Types (for IR representation)
   --  ========================================================================

   type Step_Type_Enum is (
      Step_Noop,
      Step_Assign,
      Step_Call,
      Step_Return
   );

   Max_Steps : constant := 10000;

   type Step_Index is range 0 .. Max_Steps;

   type IR_Step is record
      Step_Type : Step_Type_Enum;
      Target    : Identifier_String;
      Source    : Identifier_String;
      Value     : Identifier_String;
   end record;

   type Step_Array is array (Step_Index range <>) of IR_Step;

   type Step_Collection is record
      Steps : Step_Array (1 .. Max_Steps);
      Count : Step_Index;
   end record;

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
   --  Target Language Types
   --  ========================================================================

   type Target_Language is (
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
      Target_Clojure,
      Target_ClojureScript,
      Target_Prolog,
      Target_Futhark,
      Target_Lean4
   );

   subtype All_Targets is Target_Language range Target_CPP .. Target_Lean4;

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

-------------------------------------------------------------------------------
--  STUNIR Spec Assemble - Ada SPARK Specification
--  PRIMARY IMPLEMENTATION (Ada SPARK is the default language for STUNIR tools)
--
--  Assembles STUNIR spec JSON from AI extraction elements.
--
--  Copyright (c) 2026 STUNIR Project
--  License: MIT
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with Ada.Strings.Bounded;

package STUNIR_Spec_Assemble is

   Max_Path_Length : constant := 512;
   Max_Name_Length : constant := 128;
   Max_Hash_Length : constant := 64;
   Max_Params      : constant := 16;
   Max_Functions   : constant := 256;
   Max_Types       : constant := 256;
   Max_Consts      : constant := 256;

   package Path_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length
     (Max => Max_Path_Length);
   subtype Path_String is Path_Strings.Bounded_String;

   package Name_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length
     (Max => Max_Name_Length);
   subtype Name_String is Name_Strings.Bounded_String;

   package Hash_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length
     (Max => Max_Hash_Length);
   subtype Hash_String is Hash_Strings.Bounded_String;

   type Type_Kind is (TYPE_STRUCT, TYPE_ENUM, TYPE_ALIAS, TYPE_UNKNOWN);
   type Param_Direction is (PARAM_IN, PARAM_OUT, PARAM_INOUT);

   type Function_Param is record
      Name : Name_String;
      Type_Ref : Name_String;
      Direction : Param_Direction := PARAM_IN;
   end record;

   type Param_Array is array (Positive range <>) of Function_Param;

   type Function_Def is record
      Name : Name_String;
      Return_Type : Name_String;
      Param_Count : Natural := 0;
      Params : Param_Array (1 .. Max_Params);
   end record;

   type Function_Array is array (Positive range <>) of Function_Def;

   type Type_Def is record
      Name : Name_String;
      Kind : Type_Kind := TYPE_UNKNOWN;
      Definition : Name_String;
   end record;

   type Type_Array is array (Positive range <>) of Type_Def;

   type Constant_Def is record
      Name : Name_String;
      Value : Name_String;
   end record;

   type Constant_Array is array (Positive range <>) of Constant_Def;

   type Spec_Module is record
      Name : Name_String;
      Function_Count : Natural := 0;
      Type_Count : Natural := 0;
      Constant_Count : Natural := 0;
      Functions : Function_Array (1 .. Max_Functions);
      Types : Type_Array (1 .. Max_Types);
      Constants : Constant_Array (1 .. Max_Consts);
   end record;

   type Assemble_Status is
     (Success,
      Error_Input_Not_Found,
      Error_Output_Failed,
      Error_Parse_Failed,
      Error_Validation_Failed);

   type Assemble_Result is record
      Status : Assemble_Status := Success;
      Module : Spec_Module;
      Spec_Hash : Hash_String;
   end record;

   type Assemble_Config is record
      Input_Dir  : Path_String;
      Output_Path : Path_String;
      Index_Path  : Path_String;
      Module_Name : Name_String;
   end record;

   procedure Run_Spec_Assemble;

   procedure Assemble_Spec
     (Config : Assemble_Config;
      Result : in out Assemble_Result);

end STUNIR_Spec_Assemble;

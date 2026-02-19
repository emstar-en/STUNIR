--  STUNIR IR Converter
--  Converts spec.json to ir.json format
--  DO-333 / SPARK 2014 compliant
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  SPARK equivalent of bridge_spec_to_ir.py
--  Phase 2 of the STUNIR pipeline

pragma SPARK_Mode (On);

with STUNIR_Types;
use STUNIR_Types;
use type Identifier_Strings.Bounded_String;
use type Type_Name_Strings.Bounded_String;

package IR_Converter is

   --  ========================================================================
   --  Spec Types (Input)
   --  ========================================================================

   type Spec_Argument is record
      Name : Identifier_String;
      Arg_Type : Type_Name_String;
   end record;

   type Spec_Signature is record
      Return_Type : Type_Name_String;
      Args        : Parameter_List;
   end record;

   type Spec_Function_Detail is record
      Name      : Identifier_String;
      Signature : Spec_Signature;
   end record;

   type Spec_Module_Input is record
      Name      : Identifier_String;
      Functions : Function_Collection;
   end record;

   type Spec_Input_Data is record
      Schema_Version : Identifier_String;
      Module         : Spec_Module_Input;
   end record;

   --  ========================================================================
   --  IR Types (Output)
   --  ========================================================================

   type IR_Argument is record
      Name     : Identifier_String;
      Arg_Type : Type_Name_String;
   end record;

   type IR_Function_Detail is record
      Name        : Identifier_String;
      Return_Type : Type_Name_String;
      Args        : Parameter_List;
      Step_Count  : Step_Index;
   end record;

   type IR_Module is record
      Schema_Version : Identifier_String;
      IR_Version     : Identifier_String;
      Module_Name    : Identifier_String;
      Functions      : IR_Function_Collection;
   end record;

   --  ========================================================================
   --  Core Operations
   --  ========================================================================

   procedure Parse_Spec_JSON
     (JSON_Content : in     JSON_String;
      Spec         :    out Spec_Input_Data;
      Status       :    out Status_Code)
   with
      Pre  => JSON_Strings.Length (JSON_Content) > 0,
      Post => (if Status = Success then
                 Spec.Module.Functions.Count >= 0);

   procedure Validate_Spec
     (Spec   : in     Spec_Input_Data;
      Status :    out Status_Code)
   with
      Post => (if Status = Success then
                 (for all I in 1 .. Spec.Module.Functions.Count =>
                    Identifier_Strings.Length (Spec.Module.Functions.Functions (I).Name) > 0));

   procedure Convert_Spec_To_IR
     (Spec        : in     Spec_Input_Data;
      Module_Name : in     Identifier_String;
      IR          :    out IR_Module;
      Status      :    out Status_Code)
   with
      Pre  => Identifier_Strings.Length (Module_Name) > 0,
      Post => (if Status = Success then
                 IR.Functions.Count = Spec.Module.Functions.Count);

   procedure Generate_IR_JSON
     (IR          : in     IR_Module;
      JSON_Output :    out JSON_String;
      Status      :    out Status_Code)
   with
      Post => (if Status = Success then
                 JSON_Strings.Length (JSON_Output) > 0);

   --  ========================================================================
   --  Conversion Helpers
   --  ========================================================================

   procedure Convert_Argument
     (Spec_Arg : in     Parameter;
      IR_Arg   :    out Parameter;
      Status   :    out Status_Code)
   with
      Post => (if Status = Success then
                 IR_Arg.Name = Spec_Arg.Name and
                 IR_Arg.Param_Type = Spec_Arg.Param_Type);

   procedure Convert_Function
     (Spec_Func : in     Function_Signature;
      IR_Func   :    out IR_Function;
      Status    :    out Status_Code)
   with
      Post => (if Status = Success then
                 IR_Func.Name = Spec_Func.Name and
                 IR_Func.Return_Type = Spec_Func.Return_Type);

   procedure Add_Noop_Step
     (IR_Func : in out IR_Function;
      Status  :    out Status_Code)
   with
      Post => (if Status = Success then
                 IR_Func.Steps.Count = IR_Func.Steps.Count'Old + 1);

   --  ========================================================================
   --  Main Entry Point
   --  ========================================================================

   procedure Process_Spec_File
     (Input_Path  : in     Path_String;
      Output_Path : in     Path_String;
      Module_Name : in     Identifier_String;
      Status      :    out Status_Code)
   with
      Pre  => Path_Strings.Length (Input_Path) > 0 and
              Path_Strings.Length (Output_Path) > 0,
      Post => Status in Status_Code'Range;

end IR_Converter;

--  STUNIR Code Emitter
--  Converts ir.json to target language code
--  DO-333 / SPARK 2014 compliant
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  SPARK equivalent of bridge_ir_to_code.py
--  Phase 3 of the STUNIR pipeline

pragma SPARK_Mode (On);

with Ada.Strings.Bounded;
with STUNIR_Types;
use STUNIR_Types;

package Code_Emitter is

   --  ========================================================================
   --  Code Buffer Type
   --  ========================================================================

   Max_Code_Length : constant := 100_000;  --  100KB max code output

   package Code_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length
     (Max => Max_Code_Length);
   subtype Code_String is Code_Strings.Bounded_String;

   --  ========================================================================
   --  IR Types (Input)
   --  ========================================================================

   type IR_Step_Detail is record
      Step_Type : Step_Type_Enum;
   end record;

   type IR_Argument_Detail is record
      Name     : Identifier_String;
      Arg_Type : Type_Name_String;
   end record;

   type IR_Func_Detail is record
      Name        : Identifier_String;
      Return_Type : Type_Name_String;
      Parameters  : Parameter_List;
      Steps       : Step_Collection;
   end record;

   type IR_Data is record
      Schema_Version : Identifier_String;
      IR_Version     : Identifier_String;
      Module_Name    : Identifier_String;
      Functions      : IR_Function_Collection;
   end record;

   --  ========================================================================
   --  Type Mapping
   --  ========================================================================

   procedure Map_Type_To_Target
     (IR_Type : in     Type_Name_String;
      Target  : in     Target_Language;
      Mapped  :    out Type_Name_String;
      Status  :    out Status_Code)
   with
      Post => (if Status = Success then
                 Type_Name_Strings.Length (Mapped) > 0);

   --  ========================================================================
   --  Code Generation
   --  ========================================================================

   procedure Generate_Function_Code
     (Func      : in     IR_Function;
      Target    : in     Target_Language;
      Code      :    out Code_String;
      Status    :    out Status_Code)
   with
      Post => (if Status = Success then
                 Code_Strings.Length (Code) > 0);

   procedure Generate_Header
     (Target : in     Target_Language;
      Header :    out Code_String;
      Status :    out Status_Code)
   with
      Post => (if Status = Success then
                 Code_Strings.Length (Header) > 0);

   procedure Generate_Footer
     (Target : in     Target_Language;
      Footer :    out Code_String;
      Status :    out Status_Code);

   procedure Generate_All_Code
     (IR       : in     IR_Data;
      Target   : in     Target_Language;
      Complete :    out Code_String;
      Status   :    out Status_Code)
   with
      Post => (if Status = Success then
                 Code_Strings.Length (Complete) > 0);

   --  ========================================================================
   --  Target-Specific Generation
   --  ========================================================================

   procedure Generate_CPP_Code
     (IR     : in     IR_Data;
      Code   :    out Code_String;
      Status :    out Status_Code);

   procedure Generate_C_Code
     (IR     : in     IR_Data;
      Code   :    out Code_String;
      Status :    out Status_Code);

   procedure Generate_Python_Code
     (IR     : in     IR_Data;
      Code   :    out Code_String;
      Status :    out Status_Code);

   procedure Generate_Rust_Code
     (IR     : in     IR_Data;
      Code   :    out Code_String;
      Status :    out Status_Code);

   procedure Generate_Go_Code
     (IR     : in     IR_Data;
      Code   :    out Code_String;
      Status :    out Status_Code);

   --  ========================================================================
   --  File Operations
   --  ========================================================================

   function Get_File_Extension
     (Target : Target_Language) return Identifier_String
   with
      Post => Identifier_Strings.Length (Get_File_Extension'Result) > 0;

   --  ========================================================================
   --  Main Entry Point
   --  ========================================================================

   procedure Process_IR_File
     (Input_Path   : in     Path_String;
      Output_Dir   : in     Path_String;
      Target       : in     Target_Language;
      Status       :    out Status_Code)
   with
      Pre  => Path_Strings.Length (Input_Path) > 0 and
              Path_Strings.Length (Output_Dir) > 0,
      Post => Status in Status_Code'Range;

   procedure Process_IR_File_All_Targets
     (Input_Path : in     Path_String;
      Output_Dir : in     Path_String;
      Status     :    out Status_Code)
   with
      Pre  => Path_Strings.Length (Input_Path) > 0 and
              Path_Strings.Length (Output_Dir) > 0,
      Post => Status in Status_Code'Range;

end Code_Emitter;

-------------------------------------------------------------------------------
--  STUNIR IR to Code Emitter - Ada SPARK Specification
--  PRIMARY IMPLEMENTATION (Ada SPARK is the default language for STUNIR tools)
--
--  This package provides deterministic code generation from STUNIR IR.
--
--  Design principles:
--  - Determinism: output depends ONLY on IR + template pack files
--  - Hermeticity: no network access, stdlib only
--  - Verification: all operations are SPARK-provable
--
--  Copyright (c) 2026 STUNIR Project
--  License: MIT
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with Ada.Strings.Bounded;

package STUNIR_IR_To_Code is

   --  Version information
   Version : constant String := "0.2.0";
   Tool_ID : constant String := "stunir_ir_to_code_spark";

   --  Maximum sizes
   Max_Path_Length    : constant := 4096;
   Max_Name_Length    : constant := 256;
   Max_Template_Size  : constant := 65536;
   Max_Output_Size    : constant := 1048576;
   Max_Functions      : constant := 1000;
   Max_Types          : constant := 500;

   --  Bounded string types
   package Path_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length
     (Max => Max_Path_Length);
   subtype Path_String is Path_Strings.Bounded_String;

   package Name_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length
     (Max => Max_Name_Length);
   subtype Name_String is Name_Strings.Bounded_String;

   --  Supported target languages
   type Target_Language is
     (Target_Python,
      Target_Rust,
      Target_C,
      Target_Cpp,
      Target_Go,
      Target_JavaScript,
      Target_TypeScript,
      Target_Java,
      Target_CSharp,
      Target_WASM,
      Target_Assembly_X86,
      Target_Assembly_ARM);

   --  Function parameter record
   type Function_Param is record
      Name      : Name_String;
      Type_Name : Name_String;
   end record;

   --  Array of parameters
   Max_Params : constant := 20;
   type Param_Array is array (1 .. Max_Params) of Function_Param;

   --  Function definition record
   type Function_Definition is record
      Name        : Name_String;
      Params      : Param_Array;
      Param_Count : Natural := 0;
      Return_Type : Name_String;
      Is_Public   : Boolean := True;
   end record;

   --  IR module record
   type IR_Module is record
      Schema       : Name_String;
      Module_Name  : Name_String;
      Description  : Path_String;
      Functions    : array (1 .. Max_Functions) of Function_Definition;
      Func_Count   : Natural := 0;
   end record;

   --  Emission result status
   type Emission_Status is
     (Success,
      Error_IR_Not_Found,
      Error_IR_Parse_Failed,
      Error_Template_Not_Found,
      Error_Template_Invalid,
      Error_Output_Write_Failed,
      Error_Unsupported_Target);

   --  Emission result record
   type Emission_Result is record
      Status       : Emission_Status := Success;
      Output_Path  : Path_String;
      Output_Size  : Natural := 0;
      Functions_Emitted : Natural := 0;
   end record;

   --  Emission configuration
   type Emission_Config is record
      IR_Path        : Path_String;
      Template_Path  : Path_String;
      Output_Path    : Path_String;
      Target         : Target_Language := Target_Python;
      Emit_Comments  : Boolean := True;
      Emit_Metadata  : Boolean := True;
   end record;

   --  Initialize emission configuration
   procedure Initialize_Config
     (Config        : out Emission_Config;
      IR_Path       : String;
      Template_Path : String;
      Output_Path   : String;
      Target        : Target_Language := Target_Python)
   with
     Pre => IR_Path'Length <= Max_Path_Length and
            Template_Path'Length <= Max_Path_Length and
            Output_Path'Length <= Max_Path_Length;

   --  Parse IR from JSON file
   procedure Parse_IR
     (IR_Path   : Path_String;
      Module    : out IR_Module;
      Success   : out Boolean);

   --  Load template for target language
   procedure Load_Template
     (Template_Path : Path_String;
      Target        : Target_Language;
      Template      : out Path_String;
      Success       : out Boolean);

   --  Get file extension for target language
   function Get_File_Extension (Target : Target_Language) return String;

   --  Emit code for a single function
   procedure Emit_Function
     (Func     : Function_Definition;
      Target   : Target_Language;
      Output   : out Path_String;
      Success  : out Boolean);

   --  Main emission procedure
   procedure Emit_Code
     (Config : Emission_Config;
      Result : out Emission_Result)
   with
     Post => (if Result.Status = Success
              then Result.Functions_Emitted >= 0);

   --  Entry point for command-line execution
   procedure Run_IR_To_Code;

end STUNIR_IR_To_Code;

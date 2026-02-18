--  STUNIR Pipeline Driver
--  Orchestrates the complete STUNIR pipeline
--  DO-333 / SPARK 2014 compliant
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  SPARK equivalent of stunir_pipeline.py
--  Master orchestrator for all pipeline phases

pragma SPARK_Mode (On);

with STUNIR_Types;
use STUNIR_Types;

package Pipeline_Driver is

   --  ========================================================================
   --  Pipeline Configuration
   --  ========================================================================

   type Pipeline_Phase is (
      Phase_Extraction,
      Phase_Spec_Assembly,
      Phase_IR_Conversion,
      Phase_Code_Emission
   );

   type Phase_Enabled is array (Pipeline_Phase) of Boolean;

   type Pipeline_Config is record
      Input_Path       : Path_String;
      Output_Dir       : Path_String;
      Module_Name      : Identifier_String;
      Enabled_Phases   : Phase_Enabled;
      Targets          : Target_Language;
      Generate_All     : Boolean;
      Verbose          : Boolean;
   end record;

   --  ========================================================================
   --  Pipeline Results
   --  ========================================================================

   type Phase_Result is record
      Success      : Boolean;
      Functions_In : Function_Index;
      Functions_Out: Function_Index;
      Message      : Error_String;
   end record;

   type Pipeline_Results is record
      Phase_1_Result : Phase_Result;
      Phase_2_Result : Phase_Result;
      Phase_3_Result : Phase_Result;
      Phase_4_Result : Phase_Result;
      Overall_Success: Boolean;
   end record;

   --  ========================================================================
   --  Phase Execution
   --  ========================================================================

   procedure Run_Phase_1_Spec_Assembly
     (Config : in     Pipeline_Config;
      Result :    out Phase_Result;
      Status :    out Status_Code)
   with
      Pre  => Path_Strings.Length (Config.Input_Path) > 0 and
              Path_Strings.Length (Config.Output_Dir) > 0,
      Post => (if Status = Success then Result.Success = True);

   procedure Run_Phase_2_IR_Conversion
     (Config : in     Pipeline_Config;
      Result :    out Phase_Result;
      Status :    out Status_Code)
   with
      Pre  => Path_Strings.Length (Config.Output_Dir) > 0,
      Post => (if Status = Success then Result.Success = True);

   procedure Run_Phase_3_Code_Emission
     (Config : in     Pipeline_Config;
      Result :    out Phase_Result;
      Status :    out Status_Code)
   with
      Pre  => Path_Strings.Length (Config.Output_Dir) > 0,
      Post => (if Status = Success then Result.Success = True);

   --  ========================================================================
   --  Full Pipeline Execution
   --  ========================================================================

   procedure Run_Full_Pipeline
     (Config  : in     Pipeline_Config;
      Results :    out Pipeline_Results;
      Status  :    out Status_Code)
   with
      Pre  => Path_Strings.Length (Config.Input_Path) > 0 and
              Path_Strings.Length (Config.Output_Dir) > 0,
      Post => Status in Status_Code'Range;

   --  ========================================================================
   --  Utility Functions
   --  ========================================================================

   procedure Print_Results
     (Results : in Pipeline_Results;
      Verbose : in Boolean);

   function Get_Phase_Name
     (Phase : Pipeline_Phase) return Identifier_String;

   --  ========================================================================
   --  Main Entry Point
   --  ========================================================================

   procedure Main
     (Arg_Count : in Natural;
      Args      : in String;
      Status    : out Status_Code)
   with
      Post => Status in Status_Code'Range;

end Pipeline_Driver;

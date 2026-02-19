--  STUNIR Pipeline Driver Package Body
--  Orchestrates the complete STUNIR pipeline
--  DO-333 / SPARK 2014 compliant
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

with Spec_Assembler;
with IR_Converter;
with Code_Emitter;

with Ada.Text_IO;
with Ada.Strings.Fixed;

package body Pipeline_Driver is

   use Ada.Strings;
   use Ada.Strings.Fixed;

   procedure Run_Phase_1_Spec_Assembly
     (Config : in     Pipeline_Config;
      Result :    out Phase_Result;
      Status :    out Status_Code)
   is
      Spec_Status : Status_Code;
   begin
      Result := Phase_Result'(
         Success       => False,
         Functions_In  => 0,
         Functions_Out => 0,
         Message       => Null_Error_String);

      if not Config.Enabled_Phases (Phase_Spec_Assembly) then
         Result.Success := True;
         Result.Message := Error_Strings.To_Bounded_String ("Phase skipped");
         Status := Success;
         return;
      end if;

      --  Call Spec_Assembler to process extraction file
      Spec_Assembler.Process_Extraction_File
        (Input_Path  => Config.Input_Path,
         Output_Path => Config.Output_Dir,
         Module_Name => Config.Module_Name,
         Status      => Spec_Status);

      if Spec_Status = Success then
         Result.Success := True;
         Result.Functions_Out := 1;  --  Placeholder
      else
         Result.Success := False;
         Result.Message := Error_Strings.To_Bounded_String ("Spec assembly failed");
      end if;

      Status := Spec_Status;
   end Run_Phase_1_Spec_Assembly;

   procedure Run_Phase_2_IR_Conversion
     (Config : in     Pipeline_Config;
      Result :    out Phase_Result;
      Status :    out Status_Code)
   is
      IR_Status : Status_Code;
      Spec_Path : Path_String;
      IR_Path   : Path_String;
   begin
      Result := Phase_Result'(
         Success       => False,
         Functions_In  => 0,
         Functions_Out => 0,
         Message       => Null_Error_String);

      if not Config.Enabled_Phases (Phase_IR_Conversion) then
         Result.Success := True;
         Result.Message := Error_Strings.To_Bounded_String ("Phase skipped");
         Status := Success;
         return;
      end if;

      --  Construct paths
      Spec_Path := Config.Output_Dir;
      IR_Path := Config.Output_Dir;

      --  Call IR_Converter to process spec file
      IR_Converter.Process_Spec_File
        (Input_Path  => Spec_Path,
         Output_Path => IR_Path,
         Module_Name => Config.Module_Name,
         Status      => IR_Status);

      if IR_Status = Success then
         Result.Success := True;
         Result.Functions_Out := 1;  --  Placeholder
      else
         Result.Success := False;
         Result.Message := Error_Strings.To_Bounded_String ("IR conversion failed");
      end if;

      Status := IR_Status;
   end Run_Phase_2_IR_Conversion;

   procedure Run_Phase_3_Code_Emission
     (Config : in     Pipeline_Config;
      Result :    out Phase_Result;
      Status :    out Status_Code)
   is
      Emit_Status : Status_Code;
      IR_Path     : Path_String;
   begin
      Result := Phase_Result'(
         Success       => False,
         Functions_In  => 0,
         Functions_Out => 0,
         Message       => Null_Error_String);

      if not Config.Enabled_Phases (Phase_Code_Emission) then
         Result.Success := True;
         Result.Message := Error_Strings.To_Bounded_String ("Phase skipped");
         Status := Success;
         return;
      end if;

      --  Construct IR path
      IR_Path := Config.Output_Dir;

      --  Call Code_Emitter to generate target code
      if Config.Generate_All then
         Code_Emitter.Process_IR_File_All_Targets
           (Input_Path  => IR_Path,
            Output_Dir  => Config.Output_Dir,
            Status      => Emit_Status);
      else
         Code_Emitter.Process_IR_File
           (Input_Path  => IR_Path,
            Output_Dir  => Config.Output_Dir,
            Target      => Config.Targets,
            Status      => Emit_Status);
      end if;

      if Emit_Status = Success then
         Result.Success := True;
         Result.Functions_Out := 1;  --  Placeholder
      else
         Result.Success := False;
         Result.Message := Error_Strings.To_Bounded_String ("Code emission failed");
      end if;

      Status := Emit_Status;
   end Run_Phase_3_Code_Emission;

   procedure Run_Full_Pipeline
     (Config  : in     Pipeline_Config;
      Results :    out Pipeline_Results;
      Status  :    out Status_Code)
   is
      Phase_Status : Status_Code;
   begin
      Results := Pipeline_Results'(
         Phase_1_Result  => Phase_Result'(
            Success => False, Functions_In => 0, Functions_Out => 0,
            Message => Null_Error_String),
         Phase_2_Result  => Phase_Result'(
            Success => False, Functions_In => 0, Functions_Out => 0,
            Message => Null_Error_String),
         Phase_3_Result  => Phase_Result'(
            Success => False, Functions_In => 0, Functions_Out => 0,
            Message => Null_Error_String),
         Phase_4_Result  => Phase_Result'(
            Success => False, Functions_In => 0, Functions_Out => 0,
            Message => Null_Error_String),
         Overall_Success => False);

      --  Phase 1: Spec Assembly
      Run_Phase_1_Spec_Assembly (Config, Results.Phase_1_Result, Phase_Status);
      if Phase_Status /= Success then
         Results.Overall_Success := False;
         Status := Phase_Status;
         return;
      end if;

      --  Phase 2: IR Conversion
      Run_Phase_2_IR_Conversion (Config, Results.Phase_2_Result, Phase_Status);
      if Phase_Status /= Success then
         Results.Overall_Success := False;
         Status := Phase_Status;
         return;
      end if;

      --  Phase 3: Code Emission
      Run_Phase_3_Code_Emission (Config, Results.Phase_3_Result, Phase_Status);
      if Phase_Status /= Success then
         Results.Overall_Success := False;
         Status := Phase_Status;
         return;
      end if;

      Results.Overall_Success := True;
      Status := Success;
   end Run_Full_Pipeline;

   procedure Print_Results
     (Results : in Pipeline_Results;
      Verbose : in Boolean)
   is
      pragma Unreferenced (Verbose);
   begin
      Ada.Text_IO.Put_Line ("STUNIR Pipeline Results:");
      Ada.Text_IO.Put_Line ("  Phase 1 (Spec Assembly): " &
         (if Results.Phase_1_Result.Success then "SUCCESS" else "FAILED"));
      Ada.Text_IO.Put_Line ("  Phase 2 (IR Conversion): " &
         (if Results.Phase_2_Result.Success then "SUCCESS" else "FAILED"));
      Ada.Text_IO.Put_Line ("  Phase 3 (Code Emission): " &
         (if Results.Phase_3_Result.Success then "SUCCESS" else "FAILED"));
      Ada.Text_IO.Put_Line ("  Overall: " &
         (if Results.Overall_Success then "SUCCESS" else "FAILED"));
   end Print_Results;

   function Get_Phase_Name
     (Phase : Pipeline_Phase) return Identifier_String
   is
   begin
      case Phase is
         when Phase_Extraction    =>
            return Identifier_Strings.To_Bounded_String ("Extraction");
         when Phase_Spec_Assembly =>
            return Identifier_Strings.To_Bounded_String ("Spec Assembly");
         when Phase_IR_Conversion =>
            return Identifier_Strings.To_Bounded_String ("IR Conversion");
         when Phase_Code_Emission =>
            return Identifier_Strings.To_Bounded_String ("Code Emission");
      end case;
   end Get_Phase_Name;

   procedure Main
     (Arg_Count : in Natural;
      Args      : in String;
      Status    : out Status_Code)
   is
      pragma Unreferenced (Arg_Count, Args);
   begin
      --  Main entry point for command-line execution
      --  Would parse arguments and run pipeline
      --  For now, return not implemented
      Status := Error_Not_Implemented;
   end Main;

end Pipeline_Driver;
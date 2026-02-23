--  STUNIR Pipeline Driver Package Body
--  Orchestrates the complete STUNIR pipeline
--  DO-333 / SPARK 2014 compliant
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

--  STUNIR_Types is already use-visible via pipeline_driver.ads (use STUNIR_Types)
with Assemble_Spec;
with Spec_To_IR;
with Emit_Target;

with Ada.Text_IO;

package body Pipeline_Driver is

   --  Helper function to build file paths
   function Build_Path (Dir : Path_String; File_Name : String) return Path_String is
      Dir_Str : constant String := Path_Strings.To_String (Dir);
      Full    : constant String := Dir_Str & "/" & File_Name;
   begin
      if Full'Length <= Max_Path_Length then
         return Path_Strings.To_Bounded_String (Full);
      else
         return Path_Strings.Null_Bounded_String;
      end if;
   end Build_Path;

   procedure Run_Phase_1_Spec_Assembly
     (Config : in     Pipeline_Config;
      Result :    out Phase_Result;
      Status :    out Status_Code)
   is
      Spec_Status : Status_Code;
      Spec_Path   : Path_String;
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

      --  Build spec.json output path
      Spec_Path := Build_Path (Config.Output_Dir, "spec.json");
      if Path_Strings.Length (Spec_Path) = 0 then
         Result.Success := False;
         Result.Message := Error_Strings.To_Bounded_String ("Path too long");
         Status := Error_Too_Large;
         return;
      end if;

      --  Call Assemble_Spec to process extraction file
      Assemble_Spec.Assemble_Spec_File
        (Input_Path  => Config.Input_Path,
         Output_Path => Spec_Path,
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

      --  Construct input/output paths
      Spec_Path := Build_Path (Config.Output_Dir, "spec.json");
      IR_Path   := Build_Path (Config.Output_Dir, "ir.json");
      
      if Path_Strings.Length (Spec_Path) = 0 or else Path_Strings.Length (IR_Path) = 0 then
         Result.Success := False;
         Result.Message := Error_Strings.To_Bounded_String ("Path too long");
         Status := Error_Too_Large;
         return;
      end if;

      --  Call Spec_To_IR to process spec file
      Spec_To_IR.Convert_Spec_File
        (Input_Path  => Spec_Path,
         Output_Path => IR_Path,
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
      Output_Path : Path_String;
      
      --  Helper to emit for a single target
      procedure Emit_Single_Target (Target : Target_Language; Success : out Boolean) is
         Target_Ext : constant String := Emit_Target.Get_Target_Extension (Target);
         Target_File : constant String := "output" & Target_Ext;
      begin
         Success := False;
         
         Output_Path := Build_Path (Config.Output_Dir, Target_File);
         if Path_Strings.Length (Output_Path) = 0 then
            return;
         end if;
         
         Emit_Target.Emit_Target_File
           (Input_Path  => IR_Path,
            Target      => Target,
            Output_Path => Output_Path,
            Status      => Emit_Status);
         
         Success := Is_Success (Emit_Status);
      end Emit_Single_Target;
      
      --  Track emission success
      All_Success : Boolean := True;
      Target_OK   : Boolean;
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
      IR_Path := Build_Path (Config.Output_Dir, "ir.json");
      if Path_Strings.Length (IR_Path) = 0 then
         Result.Success := False;
         Result.Message := Error_Strings.To_Bounded_String ("Path too long");
         Status := Error_Too_Large;
         return;
      end if;

      --  Emit code for target(s)
      if Config.Generate_All then
         --  Emit for all target languages
         Emit_Single_Target (Target_CPP, Target_OK);
         All_Success := All_Success and Target_OK;
         
         Emit_Single_Target (Target_C, Target_OK);
         All_Success := All_Success and Target_OK;
         
         Emit_Single_Target (Target_Python, Target_OK);
         All_Success := All_Success and Target_OK;
         
         Emit_Single_Target (Target_Rust, Target_OK);
         All_Success := All_Success and Target_OK;
         
         Emit_Single_Target (Target_Go, Target_OK);
         All_Success := All_Success and Target_OK;
         
         Emit_Single_Target (Target_JavaScript, Target_OK);
         All_Success := All_Success and Target_OK;
         
         Emit_Single_Target (Target_Java, Target_OK);
         All_Success := All_Success and Target_OK;
         
         Emit_Single_Target (Target_CSharp, Target_OK);
         All_Success := All_Success and Target_OK;
         
         Emit_Single_Target (Target_Swift, Target_OK);
         All_Success := All_Success and Target_OK;
         
         Emit_Single_Target (Target_Kotlin, Target_OK);
         All_Success := All_Success and Target_OK;
         
         Emit_Single_Target (Target_SPARK, Target_OK);
         All_Success := All_Success and Target_OK;
         
         Emit_Single_Target (Target_Clojure, Target_OK);
         All_Success := All_Success and Target_OK;
         
         Emit_Single_Target (Target_ClojureScript, Target_OK);
         All_Success := All_Success and Target_OK;
         
         Emit_Single_Target (Target_Prolog, Target_OK);
         All_Success := All_Success and Target_OK;
         
         Emit_Single_Target (Target_Futhark, Target_OK);
         All_Success := All_Success and Target_OK;
         
         Emit_Single_Target (Target_Lean4, Target_OK);
         All_Success := All_Success and Target_OK;
      else
         --  Emit for single target
         Emit_Single_Target (Config.Targets, Target_OK);
         All_Success := Target_OK;
      end if;

      if All_Success then
         Result.Success := True;
         Result.Functions_Out := 1;  --  Placeholder
         Status := Success;
      else
         Result.Success := False;
         Result.Message := Error_Strings.To_Bounded_String ("Code emission failed");
         Status := Error_Emission_Failed;
      end if;
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
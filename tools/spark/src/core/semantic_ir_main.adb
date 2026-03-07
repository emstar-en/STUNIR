--  Semantic IR Main Program
--  Command-line entry point for ir.json to semantic_ir.json conversion
--  DO-333 / SPARK 2014 compliant
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (Off);  --  Command-line parsing not in SPARK

with Ada.Command_Line;
with Ada.Text_IO;
with STUNIR_Types;
use STUNIR_Types;
with Semantic_IR.Modules;
use Semantic_IR.Modules;
with Semantic_IR.Parse;
use Semantic_IR.Parse;
with Semantic_IR.Normalizer;
use Semantic_IR.Normalizer;
with Semantic_IR.Emitter;
use Semantic_IR.Emitter;

procedure Semantic_IR_Main is
   package ACL renames Ada.Command_Line;
   use Ada.Text_IO;

   procedure Print_Usage is
   begin
      Put_Line ("Usage: semantic_ir_main <input_ir.json> <output_semantic_ir.json>");
      Put_Line ("  input_ir.json          Input flat IR file path");
      Put_Line ("  output_semantic_ir.json Output Semantic IR file path");
      Put_Line ("  -h, --help             Show this help message");
      Put_Line ("  -v, --verbose          Enable verbose output");
      Put_Line ("");
      Put_Line ("Description:");
      Put_Line ("  Converts flat IR to Semantic IR with:");
      Put_Line ("    - Explicit type bindings");
      Put_Line ("    - Control flow graph edges");
      Put_Line ("    - Semantic annotations (safety levels, target categories)");
      Put_Line ("    - Normalized form with confluence guarantees");
   end Print_Usage;

   Input_Path     : Path_String := Path_Strings.Null_Bounded_String;
   Output_Path    : Path_String := Path_Strings.Null_Bounded_String;
   Verbose        : Boolean := False;
   Status         : Status_Code;
   Sem_Module     : Semantic_Module;
   Sem_Config     : Normalizer_Config;
   Sem_Result     : Normalization_Result;
   Emit_Result    : Emitter_Result;

begin
   --  Parse command-line arguments
   if ACL.Argument_Count < 2 then
      Print_Usage;
      ACL.Set_Exit_Status (ACL.Failure);
      return;
   end if;

   --  Process arguments
   declare
      Arg_Count : constant Natural := ACL.Argument_Count;
      Arg_Index : Natural := 1;
   begin
      while Arg_Index <= Arg_Count loop
         declare
            Arg : constant String := ACL.Argument (Arg_Index);
         begin
            if Arg = "-h" or Arg = "--help" then
               Print_Usage;
               ACL.Set_Exit_Status (ACL.Success);
               return;
            elsif Arg = "-v" or Arg = "--verbose" then
               Verbose := True;
            elsif Arg_Index = 1 then
               Input_Path := Path_Strings.To_Bounded_String (Arg);
            elsif Arg_Index = 2 then
               Output_Path := Path_Strings.To_Bounded_String (Arg);
            end if;
         end;
         Arg_Index := Arg_Index + 1;
      end loop;
   end;

   --  Validate paths
   if Path_Strings.Length (Input_Path) = 0 then
      Put_Line (Standard_Error, "Error: Input path is required");
      ACL.Set_Exit_Status (ACL.Failure);
      return;
   end if;

   if Path_Strings.Length (Output_Path) = 0 then
      Put_Line (Standard_Error, "Error: Output path is required");
      ACL.Set_Exit_Status (ACL.Failure);
      return;
   end if;

   --  Process the IR file
   if Verbose then
      Put_Line ("Converting IR to Semantic IR...");
      Put_Line ("  Input:  " & Path_Strings.To_String (Input_Path));
      Put_Line ("  Output: " & Path_Strings.To_String (Output_Path));
   else
      Put_Line ("Converting IR to Semantic IR...");
   end if;

   --  Initialize Semantic IR module
   Init_Semantic_Module (Sem_Module);

   --  Parse flat IR and convert to Semantic IR
   Parse_Semantic_IR_File (Input_Path, Sem_Module, Status);

   if Status /= Success then
      Put_Line (Standard_Error, "Error: Failed to parse input IR");
      Put_Line (Standard_Error, "  Status: " & Status_Code'Image (Status));
      ACL.Set_Exit_Status (ACL.Failure);
      return;
   end if;

   if Verbose then
      Put_Line ("  Parsed " & Natural'Image (Sem_Module.Decl_Count) & " declarations");
   end if;

   --  Configure normalizer for Semantic IR
   Sem_Config := (Enabled_Passes => (others => True),
                  Max_Temps      => 64,
                  Verbose        => Verbose,
                  Enforce_Confluence => True);

   --  Run Semantic IR normalization
   Normalize_Module (Sem_Module, Sem_Config, Sem_Result);

   if not Sem_Result.Success then
      Put_Line (Standard_Error, "Error: Normalization failed");
      Put_Line (Standard_Error, "  " & Error_Strings.To_String (Sem_Result.Message));
      ACL.Set_Exit_Status (ACL.Failure);
      return;
   end if;

   if Verbose then
      Put_Line ("  Normalized: " &
                "fields=" & Natural'Image (Sem_Result.Stats.Fields_Ordered) &
                ", arrays=" & Natural'Image (Sem_Result.Stats.Arrays_Ordered) &
                ", temps=" & Natural'Image (Sem_Result.Stats.Temps_Renamed));
   end if;

   --  Emit Semantic IR to JSON
   Emit_Module_To_File (Sem_Module, Output_Path,
      (Pretty_Print    => True,
       Include_Hashes   => True,
       Include_Loc      => True,
       Include_Annots   => True,
       Enforce_Normal   => True),
      Emit_Result);

   if not Emit_Result.Success then
      Put_Line (Standard_Error, "Error: Failed to emit Semantic IR");
      Put_Line (Standard_Error, "  " & Error_Strings.To_String (Emit_Result.Message));
      ACL.Set_Exit_Status (ACL.Failure);
      return;
   end if;

   --  Success
   if Verbose then
      Put_Line ("  Module hash: " & Hash_Strings.To_String (Sem_Module.Module_Hash));
      Put_Line ("Semantic IR conversion complete (confluence guaranteed)");
   else
      Put_Line ("Semantic IR conversion successful");
   end if;

   ACL.Set_Exit_Status (ACL.Success);

end Semantic_IR_Main;
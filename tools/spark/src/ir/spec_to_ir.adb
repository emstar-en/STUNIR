--  Spec to IR Micro-Tool Body
--  Converts spec JSON to IR JSON
--  Phase: 2 (IR)
--  SPARK_Mode: On
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

with Spec_Parse;
with Ada.Text_IO;
use Ada.Text_IO;

package body Spec_To_IR is

   procedure Convert_Spec_To_IR
     (Spec   : in     Spec_Parse.Spec_Data;
      IR     :    out IR_Data;
      Status :    out Status_Code)
   is
   begin
      Status := Success;
      
      --  Copy schema version
      IR.Schema_Version := Spec.Schema_Version;
      IR.IR_Version := Identifier_Strings.To_Bounded_String ("v1");
      IR.Module_Name := Spec.Module_Name;
      
      --  Copy imports
      IR.Imports.Count := Spec.Imports.Count;
      for I in Import_Index range 1 .. Spec.Imports.Count loop
         IR.Imports.Imports (I) := Spec.Imports.Imports (I);
      end loop;
      for I in Import_Index range Spec.Imports.Count + 1 .. Max_Imports loop
         IR.Imports.Imports (I).Name := Identifier_Strings.Null_Bounded_String;
         IR.Imports.Imports (I).From_Module := Identifier_Strings.Null_Bounded_String;
      end loop;
      
      --  Copy exports
      IR.Exports.Count := Spec.Exports.Count;
      for I in Export_Index range 1 .. Spec.Exports.Count loop
         IR.Exports.Exports (I) := Spec.Exports.Exports (I);
      end loop;
      for I in Export_Index range Spec.Exports.Count + 1 .. Max_Exports loop
         IR.Exports.Exports (I).Name := Identifier_Strings.Null_Bounded_String;
         IR.Exports.Exports (I).Export_Type := Type_Name_Strings.Null_Bounded_String;
      end loop;
      
      --  Copy types
      IR.Types.Count := Spec.Types.Count;
      for I in Type_Def_Index range 1 .. Spec.Types.Count loop
         IR.Types.Type_Defs (I) := Spec.Types.Type_Defs (I);
      end loop;
      for I in Type_Def_Index range Spec.Types.Count + 1 .. Max_Type_Defs loop
         IR.Types.Type_Defs (I).Name := Identifier_Strings.Null_Bounded_String;
         IR.Types.Type_Defs (I).Base_Type := Type_Name_Strings.Null_Bounded_String;
         IR.Types.Type_Defs (I).Fields.Count := 0;
      end loop;
      
      --  Copy constants
      IR.Constants.Count := Spec.Constants.Count;
      for I in Constant_Index range 1 .. Spec.Constants.Count loop
         IR.Constants.Constants (I) := Spec.Constants.Constants (I);
      end loop;
      for I in Constant_Index range Spec.Constants.Count + 1 .. Max_Constants loop
         IR.Constants.Constants (I).Name := Identifier_Strings.Null_Bounded_String;
         IR.Constants.Constants (I).Const_Type := Type_Name_Strings.Null_Bounded_String;
         IR.Constants.Constants (I).Value_Str := Identifier_Strings.Null_Bounded_String;
      end loop;
      
      --  Copy dependencies
      IR.Dependencies.Count := Spec.Dependencies.Count;
      for I in Dependency_Index range 1 .. Spec.Dependencies.Count loop
         IR.Dependencies.Dependencies (I) := Spec.Dependencies.Dependencies (I);
      end loop;
      for I in Dependency_Index range Spec.Dependencies.Count + 1 .. Max_Dependencies loop
         IR.Dependencies.Dependencies (I).Name := Identifier_Strings.Null_Bounded_String;
         IR.Dependencies.Dependencies (I).Version := Identifier_Strings.Null_Bounded_String;
      end loop;
      
      --  Copy functions and convert body statements to IR steps
      IR.Functions.Count := Spec.Functions.Count;
      for I in Function_Index range 1 .. Spec.Functions.Count loop
         IR.Functions.Functions (I).Name := Spec.Functions.Functions (I).Name;
         IR.Functions.Functions (I).Return_Type := Spec.Functions.Functions (I).Return_Type;
         IR.Functions.Functions (I).Parameters := Spec.Functions.Functions (I).Parameters;
         IR.Functions.Functions (I).Body_Hint := Spec.Functions.Functions (I).Body_Hint;
         IR.Functions.Functions (I).Hint_Detail := Spec.Functions.Functions (I).Hint_Detail;
         
         --  Convert body statements to IR steps
         if Spec.Functions.Functions (I).Stmts.Count > 0 then
            --  Convert parsed statements to IR steps
            IR.Functions.Functions (I).Steps.Count := 0;
            for J in 1 .. Spec.Functions.Functions (I).Stmts.Count loop
               if IR.Functions.Functions (I).Steps.Count < Max_Steps then
                  declare
                     Stmt : constant Spec_Statement := 
                        Spec.Functions.Functions (I).Stmts.Statements (J);
                     Step_Idx : constant Step_Index := 
                        IR.Functions.Functions (I).Steps.Count + 1;
                  begin
                     IR.Functions.Functions (I).Steps.Count := Step_Idx;
                     
                     --  Convert statement type to step type
                     case Stmt.Stmt_Type is
                        when Stmt_Assign =>
                           IR.Functions.Functions (I).Steps.Steps (Step_Idx).Step_Type := Step_Assign;
                           IR.Functions.Functions (I).Steps.Steps (Step_Idx).Target := Stmt.Target;
                           IR.Functions.Functions (I).Steps.Steps (Step_Idx).Value := Stmt.Value;
                        when Stmt_Return =>
                           IR.Functions.Functions (I).Steps.Steps (Step_Idx).Step_Type := Step_Return;
                           IR.Functions.Functions (I).Steps.Steps (Step_Idx).Value := Stmt.Value;
                        when Stmt_Call =>
                           IR.Functions.Functions (I).Steps.Steps (Step_Idx).Step_Type := Step_Call;
                           IR.Functions.Functions (I).Steps.Steps (Step_Idx).Target := Stmt.Target;
                           IR.Functions.Functions (I).Steps.Steps (Step_Idx).Value := Stmt.Value;
                           IR.Functions.Functions (I).Steps.Steps (Step_Idx).Args := Stmt.Args;
                        when Stmt_If =>
                           IR.Functions.Functions (I).Steps.Steps (Step_Idx).Step_Type := Step_If;
                           IR.Functions.Functions (I).Steps.Steps (Step_Idx).Condition := Stmt.Condition;
                             --  If body counts were recorded during parsing, set Then/Else ranges
                             if Stmt.Then_Count > 0 then
                                IR.Functions.Functions (I).Steps.Steps (Step_Idx).Then_Start := Step_Idx + 1;
                                IR.Functions.Functions (I).Steps.Steps (Step_Idx).Then_Count := Step_Index (Stmt.Then_Count);
                             else
                                IR.Functions.Functions (I).Steps.Steps (Step_Idx).Then_Start := 0;
                                IR.Functions.Functions (I).Steps.Steps (Step_Idx).Then_Count := 0;
                             end if;
                             if Stmt.Else_Count > 0 then
                                IR.Functions.Functions (I).Steps.Steps (Step_Idx).Else_Start := Step_Idx + 1 + Step_Index (Stmt.Then_Count);
                                IR.Functions.Functions (I).Steps.Steps (Step_Idx).Else_Count := Step_Index (Stmt.Else_Count);
                             else
                                IR.Functions.Functions (I).Steps.Steps (Step_Idx).Else_Start := 0;
                                IR.Functions.Functions (I).Steps.Steps (Step_Idx).Else_Count := 0;
                             end if;
                        when Stmt_While =>
                           IR.Functions.Functions (I).Steps.Steps (Step_Idx).Step_Type := Step_While;
                           IR.Functions.Functions (I).Steps.Steps (Step_Idx).Condition := Stmt.Condition;
                              IR.Functions.Functions (I).Steps.Steps (Step_Idx).Body_Start := Step_Idx + 1;
                              IR.Functions.Functions (I).Steps.Steps (Step_Idx).Body_Count := Step_Index (Stmt.Body_Count);
                        when Stmt_For =>
                           IR.Functions.Functions (I).Steps.Steps (Step_Idx).Step_Type := Step_For;
                           IR.Functions.Functions (I).Steps.Steps (Step_Idx).Condition := Stmt.Condition;
                           IR.Functions.Functions (I).Steps.Steps (Step_Idx).Init := Stmt.Init;
                           IR.Functions.Functions (I).Steps.Steps (Step_Idx).Increment := Stmt.Increment;
                              IR.Functions.Functions (I).Steps.Steps (Step_Idx).Body_Start := Step_Idx + 1;
                              IR.Functions.Functions (I).Steps.Steps (Step_Idx).Body_Count := Step_Index (Stmt.Body_Count);
                        when Stmt_Break =>
                           IR.Functions.Functions (I).Steps.Steps (Step_Idx).Step_Type := Step_Break;
                        when Stmt_Continue =>
                           IR.Functions.Functions (I).Steps.Steps (Step_Idx).Step_Type := Step_Continue;
                        when Stmt_Switch =>
                           IR.Functions.Functions (I).Steps.Steps (Step_Idx).Step_Type := Step_Switch;
                           IR.Functions.Functions (I).Steps.Steps (Step_Idx).Expr := Stmt.Expr;
                           IR.Functions.Functions (I).Steps.Steps (Step_Idx).Case_Count := Step_Index (Stmt.Case_Count);
                           --  Cases' bodies are flattened into the function's statement list after this switch; compute starts
                           declare
                              Accum : Natural := 0;
                           begin
                              for C in 1 .. Stmt.Case_Count loop
                                 IR.Functions.Functions (I).Steps.Steps (Step_Idx).Cases (C).Case_Value := Stmt.Cases (C).Case_Value;
                                 IR.Functions.Functions (I).Steps.Steps (Step_Idx).Cases (C).Body_Start := Step_Idx + Step_Index (Accum) + 1;
                                 IR.Functions.Functions (I).Steps.Steps (Step_Idx).Cases (C).Body_Count := Step_Index (Stmt.Cases (C).Body_Count);
                                 Accum := Accum + Stmt.Cases (C).Body_Count;
                              end loop;
                              if Stmt.Has_Default then
                                 IR.Functions.Functions (I).Steps.Steps (Step_Idx).Default_Start := Step_Idx + Step_Index (Accum) + 1;
                                 IR.Functions.Functions (I).Steps.Steps (Step_Idx).Default_Count := Step_Index (0);
                              end if;
                           end;
                        when Stmt_Try =>
                           IR.Functions.Functions (I).Steps.Steps (Step_Idx).Step_Type := Step_Try;
                        when Stmt_Throw =>
                           IR.Functions.Functions (I).Steps.Steps (Step_Idx).Step_Type := Step_Throw;
                           IR.Functions.Functions (I).Steps.Steps (Step_Idx).Error_Msg := Stmt.Error_Msg;
                        when Stmt_Error =>
                           IR.Functions.Functions (I).Steps.Steps (Step_Idx).Step_Type := Step_Error;
                           IR.Functions.Functions (I).Steps.Steps (Step_Idx).Error_Msg := Stmt.Error_Msg;
                        when Stmt_Nop =>
                           IR.Functions.Functions (I).Steps.Steps (Step_Idx).Step_Type := Step_Nop;
                     end case;
                  end;
               end if;
            end loop;
         else
            --  Generate placeholder steps (assign + return)
            if IR.Functions.Functions (I).Steps.Count < Max_Steps then
               IR.Functions.Functions (I).Steps.Count := 1;
               IR.Functions.Functions (I).Steps.Steps (1).Step_Type := Step_Assign;
               IR.Functions.Functions (I).Steps.Steps (1).Target :=
                  Identifier_Strings.To_Bounded_String ("result");
            end if;
            
            --  Set default return value based on return type
            declare
               Ret_Type : constant String := Type_Name_Strings.To_String (Spec.Functions.Functions (I).Return_Type);
            begin
               if Ret_Type = "void" then
                  IR.Functions.Functions (I).Steps.Steps (1).Value := Identifier_Strings.Null_Bounded_String;
                  IR.Functions.Functions (I).Steps.Count := 0;  --  No steps for void
               elsif Ret_Type = "int" or Ret_Type = "i32" or Ret_Type = "i64" then
                  IR.Functions.Functions (I).Steps.Steps (1).Value := Identifier_Strings.To_Bounded_String ("0");
               elsif Ret_Type = "float" or Ret_Type = "f32" or Ret_Type = "f64" then
                  IR.Functions.Functions (I).Steps.Steps (1).Value := Identifier_Strings.To_Bounded_String ("0.0");
               elsif Ret_Type = "bool" or Ret_Type = "boolean" then
                  IR.Functions.Functions (I).Steps.Steps (1).Value := Identifier_Strings.To_Bounded_String ("false");
               elsif Ret_Type = "string" then
                  IR.Functions.Functions (I).Steps.Steps (1).Value := Identifier_Strings.To_Bounded_String ("");
               else
                  IR.Functions.Functions (I).Steps.Steps (1).Value := Identifier_Strings.To_Bounded_String ("null");
               end if;
            end;

            if IR.Functions.Functions (I).Steps.Count > 0 and then IR.Functions.Functions (I).Steps.Count < Max_Steps then
               IR.Functions.Functions (I).Steps.Count := IR.Functions.Functions (I).Steps.Count + 1;
               IR.Functions.Functions (I).Steps.Steps (2).Step_Type := Step_Return;
               IR.Functions.Functions (I).Steps.Steps (2).Value :=
                  IR.Functions.Functions (I).Steps.Steps (1).Target;
            end if;
         end if;
      end loop;
      
      --  Initialize remaining functions
      for I in Function_Index range Spec.Functions.Count + 1 .. Max_Functions loop
         IR.Functions.Functions (I).Name := Identifier_Strings.Null_Bounded_String;
         IR.Functions.Functions (I).Return_Type := Type_Name_Strings.Null_Bounded_String;
         IR.Functions.Functions (I).Parameters.Count := 0;
         IR.Functions.Functions (I).Steps.Count := 0;
      end loop;
      
      --  Copy artifacts (GPU binaries and microcode blobs)
      IR.Precompiled.GPU_Binaries.Count := Spec.Precompiled.GPU_Binaries.Count;
      for I in GPU_Binary_Index range 1 .. Spec.Precompiled.GPU_Binaries.Count loop
         IR.Precompiled.GPU_Binaries.Binaries (I) := Spec.Precompiled.GPU_Binaries.Binaries (I);
      end loop;
      for I in GPU_Binary_Index range Spec.Precompiled.GPU_Binaries.Count + 1 .. Max_GPU_Binaries loop
         IR.Precompiled.GPU_Binaries.Binaries (I).Format := Format_PTX;
         IR.Precompiled.GPU_Binaries.Binaries (I).Digest := Identifier_Strings.Null_Bounded_String;
         IR.Precompiled.GPU_Binaries.Binaries (I).Target_Arch := Identifier_Strings.Null_Bounded_String;
         IR.Precompiled.GPU_Binaries.Binaries (I).Entry_Count := 0;
         IR.Precompiled.GPU_Binaries.Binaries (I).Blob_Path := Path_Strings.Null_Bounded_String;
         IR.Precompiled.GPU_Binaries.Binaries (I).Kernel_Name := Identifier_Strings.Null_Bounded_String;
         IR.Precompiled.GPU_Binaries.Binaries (I).Policy := Prefer_Source;
      end loop;
      
      IR.Precompiled.Microcode_Blobs.Count := Spec.Precompiled.Microcode_Blobs.Count;
      for I in Microcode_Blob_Index range 1 .. Spec.Precompiled.Microcode_Blobs.Count loop
         IR.Precompiled.Microcode_Blobs.Blobs (I) := Spec.Precompiled.Microcode_Blobs.Blobs (I);
      end loop;
      for I in Microcode_Blob_Index range Spec.Precompiled.Microcode_Blobs.Count + 1 .. Max_Microcode_Blobs loop
         IR.Precompiled.Microcode_Blobs.Blobs (I).Format := Format_Microcode;
         IR.Precompiled.Microcode_Blobs.Blobs (I).Digest := Identifier_Strings.Null_Bounded_String;
         IR.Precompiled.Microcode_Blobs.Blobs (I).Target_Device := Identifier_Strings.Null_Bounded_String;
         IR.Precompiled.Microcode_Blobs.Blobs (I).Blob_Path := Path_Strings.Null_Bounded_String;
         IR.Precompiled.Microcode_Blobs.Blobs (I).Load_Address := Identifier_Strings.Null_Bounded_String;
      end loop;
   end Convert_Spec_To_IR;

   procedure Convert_Spec_File
     (Input_Path  : in     Path_String;
      Output_Path : in     Path_String;
      Status      :    out Status_Code)
   is
      pragma SPARK_Mode (Off);  --  File I/O not in SPARK
      
      Spec   : Spec_Parse.Spec_Data;
      IR     : IR_Data;
      Output : File_Type;
   begin
      --  Parse spec
      Spec_Parse.Parse_Spec_File (Input_Path, Spec, Status);
      if Status /= Success then
         return;
      end if;
      
      --  Convert to IR
      Convert_Spec_To_IR (Spec, IR, Status);
      if Status /= Success then
         return;
      end if;
      
      --  Write IR JSON
      Create (Output, Out_File, Path_Strings.To_String (Output_Path));
      Put_Line (Output, "{");
      Put_Line (Output, "  ""schema"": ""stunir_ir_v1"",");
      Put_Line (Output, "  ""ir_version"": ""v1"",");
      Put_Line (Output, "  ""module_name"": """ & Identifier_Strings.To_String (IR.Module_Name) & """,");
      
      --  Emit types array
      Put (Output, "  ""types"": [");
      for I in Type_Def_Index range 1 .. IR.Types.Count loop
         Put (Output, "{");
         Put (Output, """name"": """ & Identifier_Strings.To_String (IR.Types.Type_Defs (I).Name) & """");
         
         --  Emit kind
         Put (Output, ", ""kind"": """);
         case IR.Types.Type_Defs (I).Kind is
            when Type_Struct => Put (Output, "struct");
            when Type_Enum => Put (Output, "enum");
            when Type_Alias => Put (Output, "alias");
            when Type_Generic => Put (Output, "generic");
         end case;
         Put (Output, """");
         
         --  Emit base_type if present
         if Type_Name_Strings.Length (IR.Types.Type_Defs (I).Base_Type) > 0 then
            Put (Output, ", ""base_type"": """ & 
               Type_Name_Strings.To_String (IR.Types.Type_Defs (I).Base_Type) & """");
         end if;
         
         --  Emit fields for struct types
         if IR.Types.Type_Defs (I).Kind = Type_Struct and then IR.Types.Type_Defs (I).Fields.Count > 0 then
            Put (Output, ", ""fields"": [");
            for J in Type_Field_Index range 1 .. IR.Types.Type_Defs (I).Fields.Count loop
               Put (Output, "{");
               Put (Output, """name"": """ & 
                  Identifier_Strings.To_String (IR.Types.Type_Defs (I).Fields.Fields (J).Name) & """");
               Put (Output, ", ""type"": """ & 
                  Type_Name_Strings.To_String (IR.Types.Type_Defs (I).Fields.Fields (J).Field_Type) & """");
               Put (Output, "}");
               if J < IR.Types.Type_Defs (I).Fields.Count then
                  Put (Output, ", ");
               end if;
            end loop;
            Put (Output, "]");
         end if;
         
         Put (Output, "}");
         if I < IR.Types.Count then
            Put (Output, ", ");
         end if;
      end loop;
      Put_Line (Output, "],");
      
      --  Emit constants array
      Put (Output, "  ""constants"": [");
      for I in Constant_Index range 1 .. IR.Constants.Count loop
         Put (Output, "{");
         Put (Output, """name"": """ & Identifier_Strings.To_String (IR.Constants.Constants (I).Name) & """");
         if Type_Name_Strings.Length (IR.Constants.Constants (I).Const_Type) > 0 then
            Put (Output, ", ""type"": """ & 
               Type_Name_Strings.To_String (IR.Constants.Constants (I).Const_Type) & """");
         end if;
         if Identifier_Strings.Length (IR.Constants.Constants (I).Value_Str) > 0 then
            Put (Output, ", ""value"": """ & 
               Identifier_Strings.To_String (IR.Constants.Constants (I).Value_Str) & """");
         end if;
         Put (Output, "}");
         if I < IR.Constants.Count then
            Put (Output, ", ");
         end if;
      end loop;
      Put_Line (Output, "],");
      
      --  Emit functions array
      Put_Line (Output, "  ""functions"": [");
      for I in Function_Index range 1 .. IR.Functions.Count loop
         Put_Line (Output, "    {");
         Put_Line (Output, "      ""name"": """ & Identifier_Strings.To_String (IR.Functions.Functions (I).Name) & """,");
         Put_Line (Output, "      ""return_type"": """ & Type_Name_Strings.To_String (IR.Functions.Functions (I).Return_Type) & """,");
         
         --  Emit args array
         Put (Output, "      ""args"": [");
         for J in Parameter_Index range 1 .. IR.Functions.Functions (I).Parameters.Count loop
            Put (Output, "{""name"": """ & 
               Identifier_Strings.To_String (IR.Functions.Functions (I).Parameters.Params (J).Name) & 
               """, ""type"": """ & 
               Type_Name_Strings.To_String (IR.Functions.Functions (I).Parameters.Params (J).Param_Type) & """}");
            if J < IR.Functions.Functions (I).Parameters.Count then
               Put (Output, ", ");
            end if;
         end loop;
         Put_Line (Output, "],");
         
         --  Emit steps array
         Put (Output, "      ""steps"": [");
         for J in Step_Index range 1 .. IR.Functions.Functions (I).Steps.Count loop
            Put (Output, "{""op"": """); 
            case IR.Functions.Functions (I).Steps.Steps (J).Step_Type is
               when Step_Return => Put (Output, "return");
               when Step_Assign => Put (Output, "assign");
               when Step_Call => Put (Output, "call");
               when Step_If => Put (Output, "if");
               when Step_While => Put (Output, "while");
               when Step_For => Put (Output, "for");
               when Step_Break => Put (Output, "break");
               when Step_Continue => Put (Output, "continue");
               when Step_Switch => Put (Output, "switch");
               when Step_Try => Put (Output, "try");
               when Step_Throw => Put (Output, "throw");
               when Step_Error => Put (Output, "error");
               when Step_Nop => Put (Output, "nop");
               when others => Put (Output, "unknown");
            end case;
            Put (Output, """, ""value"": """ & 
               Identifier_Strings.To_String (IR.Functions.Functions (I).Steps.Steps (J).Value) & """"); 
            if Identifier_Strings.Length (IR.Functions.Functions (I).Steps.Steps (J).Target) > 0 then
               Put (Output, ", ""target"": """ &
                  Identifier_Strings.To_String (IR.Functions.Functions (I).Steps.Steps (J).Target) & """");
            end if;
            if Identifier_Strings.Length (IR.Functions.Functions (I).Steps.Steps (J).Condition) > 0 then
               Put (Output, ", ""condition"": """ &
                  Identifier_Strings.To_String (IR.Functions.Functions (I).Steps.Steps (J).Condition) & """");
            end if;
            if Identifier_Strings.Length (IR.Functions.Functions (I).Steps.Steps (J).Args) > 0 then
               Put (Output, ", ""args"": """ &
                  Identifier_Strings.To_String (IR.Functions.Functions (I).Steps.Steps (J).Args) & """");
            end if;
            if Identifier_Strings.Length (IR.Functions.Functions (I).Steps.Steps (J).Init) > 0 then
               Put (Output, ", ""init"": """ &
                  Identifier_Strings.To_String (IR.Functions.Functions (I).Steps.Steps (J).Init) & """");
            end if;
            if Identifier_Strings.Length (IR.Functions.Functions (I).Steps.Steps (J).Increment) > 0 then
               Put (Output, ", ""increment"": """ &
                  Identifier_Strings.To_String (IR.Functions.Functions (I).Steps.Steps (J).Increment) & """");
            end if;
            Put (Output, "}");
            if J < IR.Functions.Functions (I).Steps.Count then
               Put (Output, ", ");
            end if;
         end loop;
         Put_Line (Output, "],");
         
         --  Emit body_hint field
         Put (Output, "      ""body_hint"": """);
         case IR.Functions.Functions (I).Body_Hint is
            when Hint_None => Put (Output, "none");
            when Hint_Simple_Return => Put (Output, "simple_return");
            when Hint_Getter => Put (Output, "getter");
            when Hint_Setter => Put (Output, "setter");
            when Hint_Loop_Accum => Put (Output, "loop_accum");
            when Hint_Conditional => Put (Output, "conditional");
            when Hint_Switch => Put (Output, "switch");
            when Hint_Try_Catch => Put (Output, "try_catch");
            when Hint_Recursive => Put (Output, "recursive");
            when Hint_Callback => Put (Output, "callback");
            when Hint_Complex => Put (Output, "complex");
         end case;
         Put (Output, """");
         
         --  Emit hint_detail if present
         if Hint_Strings.Length (IR.Functions.Functions (I).Hint_Detail) > 0 then
            Put (Output, ", ""hint_detail"": """ & 
               Hint_Strings.To_String (IR.Functions.Functions (I).Hint_Detail) & """");
         end if;
         Put_Line (Output, "");
         
         if I < IR.Functions.Count then
            Put_Line (Output, "    },");
         else
            Put_Line (Output, "    }");
         end if;
      end loop;
      
      Put_Line (Output, "  ],");
      
      --  Emit artifacts section
      Put_Line (Output, "  ""artifacts"": {");
      
      --  GPU binaries
      Put (Output, "    ""gpu_binaries"": [");
      for I in GPU_Binary_Index range 1 .. IR.Precompiled.GPU_Binaries.Count loop
         Put (Output, "{");
         Put (Output, """format"": """);
         case IR.Precompiled.GPU_Binaries.Binaries (I).Format is
            when Format_PTX => Put (Output, "ptx");
            when Format_CUBIN => Put (Output, "cubin");
            when Format_HSACO => Put (Output, "hsaco");
            when Format_SPIRV => Put (Output, "spirv");
         end case;
         Put (Output, """, ");
         Put (Output, """digest"": """ & 
            Identifier_Strings.To_String (IR.Precompiled.GPU_Binaries.Binaries (I).Digest) & """, ");
         Put (Output, """target_arch"": """ & 
            Identifier_Strings.To_String (IR.Precompiled.GPU_Binaries.Binaries (I).Target_Arch) & """, ");
         Put (Output, """blob_path"": """ & 
            Path_Strings.To_String (IR.Precompiled.GPU_Binaries.Binaries (I).Blob_Path) & """, ");
         Put (Output, """kernel_name"": """ & 
            Identifier_Strings.To_String (IR.Precompiled.GPU_Binaries.Binaries (I).Kernel_Name) & """, ");
         Put (Output, """policy"": """);
         case IR.Precompiled.GPU_Binaries.Binaries (I).Policy is
            when Prefer_Source => Put (Output, "prefer_source");
            when Prefer_Binary => Put (Output, "prefer_binary");
            when Require_Binary => Put (Output, "require_binary");
         end case;
         Put (Output, """}");
         if I < IR.Precompiled.GPU_Binaries.Count then
            Put (Output, ", ");
         end if;
      end loop;
      Put_Line (Output, "],");
      
      --  Microcode blobs
      Put (Output, "    ""microcode_blobs"": [");
      for I in Microcode_Blob_Index range 1 .. IR.Precompiled.Microcode_Blobs.Count loop
         Put (Output, "{");
         Put (Output, """format"": """);
         case IR.Precompiled.Microcode_Blobs.Blobs (I).Format is
            when Format_Microcode => Put (Output, "microcode");
            when Format_ROM => Put (Output, "rom");
            when Format_UCode => Put (Output, "ucode");
         end case;
         Put (Output, """, ");
         Put (Output, """digest"": """ & 
            Identifier_Strings.To_String (IR.Precompiled.Microcode_Blobs.Blobs (I).Digest) & """, ");
         Put (Output, """target_device"": """ & 
            Identifier_Strings.To_String (IR.Precompiled.Microcode_Blobs.Blobs (I).Target_Device) & """, ");
         Put (Output, """blob_path"": """ & 
            Path_Strings.To_String (IR.Precompiled.Microcode_Blobs.Blobs (I).Blob_Path) & """, ");
         Put (Output, """load_address"": """ & 
            Identifier_Strings.To_String (IR.Precompiled.Microcode_Blobs.Blobs (I).Load_Address) & """");
         Put (Output, "}");
         if I < IR.Precompiled.Microcode_Blobs.Count then
            Put (Output, ", ");
         end if;
      end loop;
      Put_Line (Output, "]");
      
      Put_Line (Output, "  }");
      Put_Line (Output, "}");
      Close (Output);
      
      Status := Success;
   exception
      when others =>
         Status := Error_File_IO;
   end Convert_Spec_File;

end Spec_To_IR;

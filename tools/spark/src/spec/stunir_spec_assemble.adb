-------------------------------------------------------------------------------
--  STUNIR Spec Assemble - Ada SPARK Implementation
--  PRIMARY IMPLEMENTATION (Ada SPARK is the default language for STUNIR tools)
--
--  Assembles STUNIR spec JSON from AI extraction elements.
--
--  Copyright (c) 2026 STUNIR Project
--  License: MIT
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with Ada.Text_IO;
with Ada.Command_Line;
with Ada.Directories;
with Ada.Strings.Unbounded;
with GNAT.SHA256;
with STUNIR_String_Builder;
with STUNIR_JSON_Utils;

package body STUNIR_Spec_Assemble is

   use Ada.Text_IO;
   use Ada.Directories;

   function Extract_Field (JSON : String; Field : String) return String is
      Val : constant String := STUNIR_JSON_Utils.Extract_String_Value (JSON, Field);
   begin
      return Val;
   end Extract_Field;

   procedure Parse_Element_File
     (File_Path : String;
      Module    : in out Spec_Module)
   is
      File_In : Ada.Text_IO.File_Type;
      Content : String := "";
      Line    : String (1 .. 4096);
      Last    : Natural;
   begin
      Open (File_In, In_File, File_Path);
      while not End_Of_File (File_In) loop
         Get_Line (File_In, Line, Last);
         if Last > 0 then
            if Content'Length = 0 then
               Content := Line (1 .. Last);
            else
               Content := Content & ASCII.LF & Line (1 .. Last);
            end if;
         end if;
      end loop;
      Close (File_In);

      declare
         Elements_Pos : constant Natural := STUNIR_JSON_Utils.Find_Array (Content, "elements");
         Obj_Start : Natural;
         Obj_End : Natural;
         Pos : Natural := Elements_Pos + 1;
      begin
         if Elements_Pos = 0 then
            return;
         end if;

         loop
            STUNIR_JSON_Utils.Get_Next_Object (Content, Pos, Obj_Start, Obj_End);
            exit when Obj_Start = 0 or Obj_End = 0;

            declare
               Obj_JSON : constant String := Content (Obj_Start .. Obj_End);
               Elem_Name : constant String := Extract_Field (Obj_JSON, "name");
               Elem_Type : constant String := Extract_Field (Obj_JSON, "type");
               Signature : constant String := Extract_Field (Obj_JSON, "signature");
            begin
               if Elem_Name'Length > 0 then
                  if Elem_Type = "function" then
                     if Module.Function_Count < Max_Functions then
                        Module.Function_Count := Module.Function_Count + 1;
                        declare
                           Idx : constant Positive := Module.Function_Count;
                        begin
                           Module.Functions (Idx).Name := Name_Strings.To_Bounded_String (Elem_Name);
                           Module.Functions (Idx).Return_Type := Name_Strings.To_Bounded_String ("void");
                           Module.Functions (Idx).Param_Count := 0;
                           if Signature'Length > 0 then
                              Module.Functions (Idx).Return_Type := Name_Strings.To_Bounded_String ("auto");
                           end if;
                        end;
                     end if;
                  elsif Elem_Type = "type" then
                     if Module.Type_Count < Max_Types then
                        Module.Type_Count := Module.Type_Count + 1;
                        Module.Types (Module.Type_Count).Name := Name_Strings.To_Bounded_String (Elem_Name);
                        Module.Types (Module.Type_Count).Kind := TYPE_UNKNOWN;
                        Module.Types (Module.Type_Count).Definition := Name_Strings.To_Bounded_String (Signature);
                     end if;
                  elsif Elem_Type = "constant" then
                     if Module.Constant_Count < Max_Consts then
                        Module.Constant_Count := Module.Constant_Count + 1;
                        Module.Constants (Module.Constant_Count).Name := Name_Strings.To_Bounded_String (Elem_Name);
                        Module.Constants (Module.Constant_Count).Value := Name_Strings.To_Bounded_String (Signature);
                     end if;
                  end if;
               end if;
            end;

            Pos := Obj_End + 1;
         end loop;
      end;

   exception
      when others =>
         if Is_Open (File_In) then
            Close (File_In);
         end if;
   end Parse_Element_File;

   procedure Compute_Spec_Hash
     (JSON_Text : String;
      Hash      : out Hash_String)
   is
      Context : GNAT.SHA256.Context := GNAT.SHA256.Initial_Context;
   begin
      GNAT.SHA256.Update (Context, JSON_Text);
      Hash := Hash_Strings.To_Bounded_String (GNAT.SHA256.Digest (Context));
   end Compute_Spec_Hash;

   procedure Write_Spec_JSON
     (Module : Spec_Module;
      Output_Path : String;
      Spec_Hash : out Hash_String;
      Success : out Boolean)
   is
      use STUNIR_String_Builder;
      Builder : String_Builder;
      File_Out : Ada.Text_IO.File_Type;
   begin
      Initialize (Builder);
      Append_Line (Builder, "{");
      Append_Line (Builder, "  ""kind"": ""stunir.spec.v1"",");
      Append_Line (Builder, "  ""meta"": {");
      Append_Line (Builder, "    ""origin"": ""stunir_spec_assemble"",");
      Append_Line (Builder, "    ""spec_hash"": """",");
      Append_Line (Builder, "    ""source_index"": """"");
      Append_Line (Builder, "  },");
      Append_Line (Builder, "  ""modules"": [");
      Append_Line (Builder, "    {");
      Append_Line (Builder, "      ""name"": """ & Name_Strings.To_String (Module.Name) & """,");
      Append_Line (Builder, "      ""functions"": [");
      for I in 1 .. Module.Function_Count loop
         if I > 1 then
            Append_Line (Builder, "        ,{");
         else
            Append_Line (Builder, "        {");
         end if;
         Append_Line (Builder, "          ""name"": """ & Name_Strings.To_String (Module.Functions (I).Name) & """,");
         Append_Line (Builder, "          ""return_type"": """ & Name_Strings.To_String (Module.Functions (I).Return_Type) & """,");
         Append_Line (Builder, "          ""parameters"": []");
         Append_Line (Builder, "        }");
      end loop;
      Append_Line (Builder, "      ],");
      Append_Line (Builder, "      ""types"": [");
      for I in 1 .. Module.Type_Count loop
         if I > 1 then
            Append_Line (Builder, "        ,{");
         else
            Append_Line (Builder, "        {");
         end if;
         Append_Line (Builder, "          ""name"": """ & Name_Strings.To_String (Module.Types (I).Name) & """,");
         Append_Line (Builder, "          ""kind"": """ & Name_Strings.To_String (Module.Types (I).Definition) & """");
         Append_Line (Builder, "        }");
      end loop;
      Append_Line (Builder, "      ]");
      Append_Line (Builder, "    }");
      Append_Line (Builder, "  ]");
      Append_Line (Builder, "}");

      declare
         JSON_Text : constant String := To_String (Builder);
      begin
         Compute_Spec_Hash (JSON_Text, Spec_Hash);
      end;

      Create (File_Out, Out_File, Output_Path);
      Put (File_Out, To_String (Builder));
      Close (File_Out);
      Success := True;

   exception
      when others =>
         if Is_Open (File_Out) then
            Close (File_Out);
         end if;
         Spec_Hash := Hash_Strings.Null_Bounded_String;
         Success := False;
   end Write_Spec_JSON;

   procedure Assemble_Spec
     (Config : Assemble_Config;
      Result : in out Assemble_Result)
   is
      Search  : Search_Type;
      Dir_Ent : Directory_Entry_Type;
      Root    : constant String := Path_Strings.To_String (Config.Input_Dir);
      Output_OK : Boolean;
   begin
      Result.Status := Success;
      Result.Module.Name := Config.Module_Name;
      Result.Module.Function_Count := 0;
      Result.Module.Type_Count := 0;
      Result.Module.Constant_Count := 0;

      if not Exists (Root) then
         Result.Status := Error_Input_Not_Found;
         return;
      end if;

      Start_Search (Search, Root, "*.json", (Directory => False, others => True));
      while More_Entries (Search) loop
         Get_Next_Entry (Search, Dir_Ent);
         Parse_Element_File (Full_Name (Dir_Ent), Result.Module);
      end loop;
      End_Search (Search);

      Write_Spec_JSON (Result.Module, Path_Strings.To_String (Config.Output_Path), Result.Spec_Hash, Output_OK);
      if not Output_OK then
         Result.Status := Error_Output_Failed;
      end if;

   exception
      when others =>
         Result.Status := Error_Parse_Failed;
   end Assemble_Spec;

   procedure Run_Spec_Assemble is
      Config : Assemble_Config;
      Result : Assemble_Result;
      Arg_Count : constant Natural := Ada.Command_Line.Argument_Count;
      Input_Path : Ada.Strings.Unbounded.Unbounded_String := Ada.Strings.Unbounded.Null_Unbounded_String;
      Output_Path : Ada.Strings.Unbounded.Unbounded_String := Ada.Strings.Unbounded.Null_Unbounded_String;
      Index_Path : Ada.Strings.Unbounded.Unbounded_String := Ada.Strings.Unbounded.Null_Unbounded_String;
      Module_Name : Ada.Strings.Unbounded.Unbounded_String := Ada.Strings.Unbounded.Null_Unbounded_String;
   begin
      for I in 1 .. Arg_Count loop
         declare
            Arg : constant String := Ada.Command_Line.Argument (I);
         begin
            if Arg = "--input" and then I < Arg_Count then
               Input_Path := Ada.Strings.Unbounded.To_Unbounded_String (Ada.Command_Line.Argument (I + 1));
            elsif Arg = "--output" and then I < Arg_Count then
               Output_Path := Ada.Strings.Unbounded.To_Unbounded_String (Ada.Command_Line.Argument (I + 1));
            elsif Arg = "--index" and then I < Arg_Count then
               Index_Path := Ada.Strings.Unbounded.To_Unbounded_String (Ada.Command_Line.Argument (I + 1));
            elsif Arg = "--name" and then I < Arg_Count then
               Module_Name := Ada.Strings.Unbounded.To_Unbounded_String (Ada.Command_Line.Argument (I + 1));
            end if;
         end;
      end loop;

      if Ada.Strings.Unbounded.Length (Input_Path) = 0 or else Ada.Strings.Unbounded.Length (Output_Path) = 0 or else Ada.Strings.Unbounded.Length (Module_Name) = 0 then
         Put_Line ("Usage: stunir_spec_assemble --input <extractions_dir> --output <spec.json> --index <index.json> --name <module_name>");
         return;
      end if;

      Config.Input_Dir := Path_Strings.To_Bounded_String (Ada.Strings.Unbounded.To_String (Input_Path));
      Config.Output_Path := Path_Strings.To_Bounded_String (Ada.Strings.Unbounded.To_String (Output_Path));
      Config.Index_Path := Path_Strings.To_Bounded_String (Ada.Strings.Unbounded.To_String (Index_Path));
      Config.Module_Name := Name_Strings.To_Bounded_String (Ada.Strings.Unbounded.To_String (Module_Name));

      Assemble_Spec (Config, Result);
      if Result.Status /= Success then
         Put_Line ("[ERROR] Spec assemble failed");
      end if;
   end Run_Spec_Assemble;

end STUNIR_Spec_Assemble;

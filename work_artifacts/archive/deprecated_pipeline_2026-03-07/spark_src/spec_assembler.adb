--  STUNIR Spec Assembler Package Body
--  Converts extraction.json to spec.json format
--  DO-333 / SPARK 2014 compliant
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

--  STUNIR_Types is already use-visible via spec_assembler.ads (use STUNIR_Types)
with STUNIR_JSON_Parser;
use STUNIR_JSON_Parser;

with Ada.Strings.Unbounded;
with Ada.Text_IO;

package body Spec_Assembler is

   use Ada.Strings.Unbounded;
   use Ada.Text_IO;

   --  Forward declarations for helper procedures
   procedure Parse_Parameter_Array_Internal
     (Parser     : in out Parser_State;
      Param_List :    out Parameter_List;
      Status     :    out Status_Code);

   procedure Parse_Function_Object_Internal
     (Parser       : in out Parser_State;
      Function_Out :    out Extraction_Function;
      Source_File  :    out Path_String;
      Status       :    out Status_Code);

   procedure Parse_File_Object_Internal
     (Parser   : in out Parser_State;
      File_Out :    out Extraction_File;
      Status   :    out Status_Code);

   procedure Parse_Parameter_Array_Internal
     (Parser     : in out Parser_State;
      Param_List :    out Parameter_List;
      Status     :    out Status_Code)
   is
      Param_Idx : Parameter_Index := 0;
      Temp_Status : Status_Code;
   begin
      Param_List := (Count => 0, Params => (others => (Name => Null_Identifier, Param_Type => Null_Type_Name)));
      Status := Success;

      if Current_Token (Parser) /= Token_Array_Start then
         Status := Error_Parse;
         return;
      end if;

      Next_Token (Parser, Temp_Status);
      while Temp_Status = Success and then Current_Token (Parser) /= Token_Array_End loop
         if Current_Token (Parser) /= Token_Object_Start then
            Status := Error_Parse;
            return;
         end if;
         if Param_Idx >= Max_Parameters then
            Status := Error_Too_Large;
            return;
         end if;
         Param_Idx := Param_Idx + 1;
         Next_Token (Parser, Temp_Status);
         while Temp_Status = Success and then Current_Token (Parser) /= Token_Object_End loop
            declare
               Member_Name  : Identifier_String;
               Member_Value : JSON_String;
            begin
               Parse_String_Member (Parser, Member_Name, Member_Value, Temp_Status);
               exit when Temp_Status /= Success;
               if Identifier_Strings.To_String (Member_Name) = "name" then
                  Param_List.Params (Param_Idx).Name :=
                    Identifier_Strings.To_Bounded_String (JSON_Strings.To_String (Member_Value));
                  Next_Token (Parser, Temp_Status);
               elsif Identifier_Strings.To_String (Member_Name) = "type" then
                  Param_List.Params (Param_Idx).Param_Type :=
                    Type_Name_Strings.To_Bounded_String (JSON_Strings.To_String (Member_Value));
                  Next_Token (Parser, Temp_Status);
               else
                  if Current_Token (Parser) = Token_Array_Start
                     or Current_Token (Parser) = Token_Object_Start
                  then
                     Skip_Value (Parser, Temp_Status);
                  else
                     Next_Token (Parser, Temp_Status);
                  end if;
               end if;
               if Current_Token (Parser) = Token_Comma then
                  Next_Token (Parser, Temp_Status);
               end if;
            end;
         end loop;
         if Current_Token (Parser) = Token_Object_End then
            Next_Token (Parser, Temp_Status);
         end if;
         if Current_Token (Parser) = Token_Comma then
            Next_Token (Parser, Temp_Status);
         end if;
      end loop;
      if Current_Token (Parser) = Token_Array_End then
         Next_Token (Parser, Temp_Status);
      end if;
      Param_List.Count := Param_Idx;
      Status := Temp_Status;
   end Parse_Parameter_Array_Internal;

   procedure Parse_Function_Object_Internal
     (Parser       : in out Parser_State;
      Function_Out :    out Extraction_Function;
      Source_File  :    out Path_String;
      Status       :    out Status_Code)
   is
      Temp_Status : Status_Code;
   begin
      Function_Out.Name := Null_Identifier;
      Function_Out.Return_Type := Type_Name_Strings.To_Bounded_String ("void");
      Function_Out.Parameters.Count := 0;
      Source_File := Path_Strings.Null_Bounded_String;

      if Current_Token (Parser) /= Token_Object_Start then
         Status := Error_Parse;
         return;
      end if;

      Next_Token (Parser, Temp_Status);
      while Temp_Status = Success and then Current_Token (Parser) /= Token_Object_End loop
         declare
            Member_Name  : Identifier_String;
            Member_Value : JSON_String;
         begin
            Parse_String_Member (Parser, Member_Name, Member_Value, Temp_Status);
            exit when Temp_Status /= Success;
            if Identifier_Strings.To_String (Member_Name) = "name" then
               Function_Out.Name := Identifier_Strings.To_Bounded_String (JSON_Strings.To_String (Member_Value));
               Next_Token (Parser, Temp_Status);
            elsif Identifier_Strings.To_String (Member_Name) = "return_type" then
               Function_Out.Return_Type := Type_Name_Strings.To_Bounded_String (JSON_Strings.To_String (Member_Value));
               Next_Token (Parser, Temp_Status);
            elsif Identifier_Strings.To_String (Member_Name) = "parameters" and then Current_Token (Parser) = Token_Array_Start then
               Parse_Parameter_Array_Internal (Parser, Function_Out.Parameters, Temp_Status);
            elsif Identifier_Strings.To_String (Member_Name) = "source_file" then
               Source_File := Path_Strings.To_Bounded_String (JSON_Strings.To_String (Member_Value));
               Next_Token (Parser, Temp_Status);
            else
               if Current_Token (Parser) = Token_Array_Start
                  or Current_Token (Parser) = Token_Object_Start
               then
                  Skip_Value (Parser, Temp_Status);
               else
                  Next_Token (Parser, Temp_Status);
               end if;
            end if;
            if Current_Token (Parser) = Token_Comma then
               Next_Token (Parser, Temp_Status);
            end if;
         end;
      end loop;

      if Current_Token (Parser) = Token_Object_End then
         Next_Token (Parser, Temp_Status);
      end if;
      Status := Temp_Status;
   end Parse_Function_Object_Internal;

   procedure Parse_File_Object_Internal
     (Parser   : in out Parser_State;
      File_Out :    out Extraction_File;
      Status   :    out Status_Code)
   is
      Temp_Status : Status_Code;
   begin
      File_Out.Source_File := Path_Strings.Null_Bounded_String;
      File_Out.Count := 0;

      if Current_Token (Parser) /= Token_Object_Start then
         Status := Error_Parse;
         return;
      end if;

      Next_Token (Parser, Temp_Status);
      while Temp_Status = Success and then Current_Token (Parser) /= Token_Object_End loop
         declare
            Member_Name  : Identifier_String;
            Member_Value : JSON_String;
         begin
            Parse_String_Member (Parser, Member_Name, Member_Value, Temp_Status);
            exit when Temp_Status /= Success;
            if Identifier_Strings.To_String (Member_Name) = "path" then
               File_Out.Source_File := Path_Strings.To_Bounded_String (JSON_Strings.To_String (Member_Value));
            elsif Identifier_Strings.To_String (Member_Name) = "functions" and then Current_Token (Parser) = Token_Array_Start then
               Next_Token (Parser, Temp_Status);
               while Temp_Status = Success and then Current_Token (Parser) /= Token_Array_End loop
                  declare
                     Func : Extraction_Function;
                     Src  : Path_String;
                  begin
                     Parse_Function_Object_Internal (Parser, Func, Src, Temp_Status);
                     if Temp_Status /= Success then
                        Status := Temp_Status;
                        return;
                     end if;
                     if File_Out.Count < Max_Extractions_Per_File then
                        File_Out.Count := File_Out.Count + 1;
                        File_Out.Functions (File_Out.Count) := Func;
                     else
                        Status := Error_Too_Large;
                        return;
                     end if;
                  end;
                  if Current_Token (Parser) = Token_Comma then
                     Next_Token (Parser, Temp_Status);
                  end if;
               end loop;
               if Current_Token (Parser) = Token_Array_End then
                  Next_Token (Parser, Temp_Status);
               end if;
            else
               Skip_Value (Parser, Temp_Status);
            end if;
            if Current_Token (Parser) = Token_Comma then
               Next_Token (Parser, Temp_Status);
            end if;
         end;
      end loop;
      if Current_Token (Parser) = Token_Object_End then
         Next_Token (Parser, Temp_Status);
      end if;
      Status := Temp_Status;
   end Parse_File_Object_Internal;

   procedure Parse_Extraction_JSON
     (JSON_Content : in     JSON_String;
      Extraction   :    out Extraction_Data;
      Status       :    out Status_Code)
   is
      Parser      : Parser_State;
      Temp_Status : Status_Code;
   begin
      --  Initialize extraction data field-by-field to avoid aggregate
      --  constraint issues with unconstrained array types.
      Extraction.Schema_Version := Null_Identifier;
      Extraction.Source_Index   := Path_Strings.Null_Bounded_String;
      Extraction.File_Count     := 0;
      --  Files array elements are accessed only up to File_Count, so
      --  leaving them uninitialized is safe (SPARK: no reads before writes).

      --  Initialize parser
      Initialize_Parser (Parser, JSON_Content, Temp_Status);
      if Temp_Status /= Success then
         Status := Temp_Status;
         return;
      end if;

      --  Expect object start
      Expect_Token (Parser, Token_Object_Start, Temp_Status);
      if Temp_Status /= Success then
         Status := Error_Invalid_JSON;
         return;
      end if;

      --  Advance to first member (or object end)
      Next_Token (Parser, Temp_Status);
      if Temp_Status /= Success then
         Status := Error_Invalid_JSON;
         return;
      end if;

      --  Parse top-level members
      loop
         exit when Current_Token (Parser) = Token_Object_End;

         declare
            Member_Name  : Identifier_String;
            Member_Value : JSON_String;
         begin
            Parse_String_Member (Parser, Member_Name, Member_Value, Temp_Status);
            if Temp_Status /= Success then
               Status := Error_Invalid_JSON;
               return;
            end if;

            --  Handle known members
            declare
               Name_Str : constant String := Identifier_Strings.To_String (Member_Name);
            begin
               if Name_Str = "schema_version" then
                  Extraction.Schema_Version := Identifier_Strings.To_Bounded_String (
                     JSON_Strings.To_String (Member_Value));
                  --  Advance past the string value
                  Next_Token (Parser, Temp_Status);
               elsif Name_Str = "source_index" then
                  Extraction.Source_Index := Path_Strings.To_Bounded_String (
                     JSON_Strings.To_String (Member_Value));
                  --  Advance past the string value
                  Next_Token (Parser, Temp_Status);
               elsif Name_Str = "source_files" and then Current_Token (Parser) = Token_Array_Start then
                  --  source_files array - skip it (not used in spec assembly)
                  Skip_Value (Parser, Temp_Status);
               elsif Name_Str = "total_functions" then
                  --  total_functions number - already parsed, advance past it
                  Next_Token (Parser, Temp_Status);
               elsif Name_Str = "files" and then Current_Token (Parser) = Token_Array_Start then
                  --  Format B: files array
                  Next_Token (Parser, Temp_Status);
                  while Temp_Status = Success and then Current_Token (Parser) /= Token_Array_End loop
                     declare
                        File_Item : Extraction_File;
                     begin
                        Parse_File_Object_Internal (Parser, File_Item, Temp_Status);
                        if Temp_Status /= Success then
                           Status := Temp_Status;
                           return;
                        end if;
                        if Extraction.File_Count < Max_Extraction_Files then
                           Extraction.File_Count := Extraction.File_Count + 1;
                           Extraction.Files (Extraction.File_Count) := File_Item;
                        else
                           Status := Error_Too_Large;
                           return;
                        end if;
                     end;
                     if Current_Token (Parser) = Token_Comma then
                        Next_Token (Parser, Temp_Status);
                     end if;
                  end loop;
                  if Current_Token (Parser) = Token_Array_End then
                     Next_Token (Parser, Temp_Status);
                  end if;
               elsif Name_Str = "module_name" then
                  --  Single module extraction format
                  null;  --  Will be handled differently
               elsif Name_Str = "functions" and then Current_Token (Parser) = Token_Array_Start then
                  --  Format A: direct functions array
                  declare
                     File_Item : Extraction_File;
                     Src_Path  : Path_String := Path_Strings.Null_Bounded_String;
                  begin
                     File_Item.Source_File := Path_Strings.Null_Bounded_String;
                     File_Item.Count := 0;
                     Next_Token (Parser, Temp_Status);
                     while Temp_Status = Success and then Current_Token (Parser) /= Token_Array_End loop
                        declare
                           Func : Extraction_Function;
                        begin
                           Parse_Function_Object_Internal (Parser, Func, Src_Path, Temp_Status);
                           if Temp_Status /= Success then
                              Status := Temp_Status;
                              return;
                           end if;
                           if File_Item.Count < Max_Extractions_Per_File then
                              File_Item.Count := File_Item.Count + 1;
                              File_Item.Functions (File_Item.Count) := Func;
                              if Path_Strings.Length (File_Item.Source_File) = 0 then
                                 File_Item.Source_File := Src_Path;
                              end if;
                           else
                              Status := Error_Too_Large;
                              return;
                           end if;
                        end;
                        if Current_Token (Parser) = Token_Comma then
                           Next_Token (Parser, Temp_Status);
                        end if;
                     end loop;
                     if Current_Token (Parser) = Token_Array_End then
                        Next_Token (Parser, Temp_Status);
                     end if;
                     --  Only add file item if it has functions or a source file
                     if File_Item.Count > 0
                        or Path_Strings.Length (File_Item.Source_File) > 0
                     then
                        if Extraction.File_Count < Max_Extraction_Files then
                           Extraction.File_Count := Extraction.File_Count + 1;
                           Extraction.Files (Extraction.File_Count) := File_Item;
                        else
                           Status := Error_Too_Large;
                           return;
                        end if;
                     end if;
                  end;
               else
                  --  Unknown member - skip its value
                  if Current_Token (Parser) = Token_Array_Start
                     or Current_Token (Parser) = Token_Object_Start
                  then
                     Skip_Value (Parser, Temp_Status);
                  else
                     --  Simple value - just advance past it
                     Next_Token (Parser, Temp_Status);
                  end if;
               end if;
            end;
         end;

         --  Check for comma or object end
         if Current_Token (Parser) = Token_Comma then
            Next_Token (Parser, Temp_Status);
            if Temp_Status /= Success then
               Status := Temp_Status;
               return;
            end if;
         elsif Current_Token (Parser) /= Token_Object_End then
            Status := Error_Invalid_JSON;
            return;
         end if;
      end loop;

      Status := Success;
   end Parse_Extraction_JSON;

   procedure Validate_Extraction
     (Extraction : in     Extraction_Data;
      Status     :    out Status_Code)
   is
   begin
      --  Check schema version is present
      if Identifier_Strings.Length (Extraction.Schema_Version) = 0 then
         Status := Error_Invalid_Schema;
         return;
      end if;

      --  Validate all files have valid source paths
      for I in 1 .. Extraction.File_Count loop
         if Path_Strings.Length (Extraction.Files (I).Source_File) = 0 then
            Status := Error_Invalid_Schema;
            return;
         end if;
      end loop;

      Status := Success;
   end Validate_Extraction;

   procedure Assemble_Spec
     (Extraction  : in     Extraction_Data;
      Module_Name : in     Identifier_String;
      Spec        :    out Spec_Data;
      Status      :    out Status_Code)
   is
      Total_Functions : Function_Index := 0;
   begin
      --  Initialize spec data field-by-field to avoid Dynamic_Predicate
      --  violation on Function_Signature (Name must be non-empty at runtime).
      --  Count=0 ensures no element is accessed before being populated.
      Spec.Schema_Version          := Identifier_Strings.To_Bounded_String ("1.0");
      Spec.Origin                  := Identifier_Strings.To_Bounded_String ("stunir");
      Spec.Spec_Hash               := Null_Identifier;
      Spec.Source_Index            := Extraction.Source_Index;
      Spec.Module.Name             := Module_Name;
      Spec.Module.Functions.Count  := 0;

      --  Collect all functions from all extraction files
      for File_Idx in 1 .. Extraction.File_Count loop
         declare
            File : Extraction_File renames Extraction.Files (File_Idx);
         begin
            for Func_Idx in 1 .. File.Count loop
               if Total_Functions < Max_Functions then
                  Total_Functions := Total_Functions + 1;
                  Spec.Module.Functions.Functions (Total_Functions) := Function_Signature'(
                     Name        => File.Functions (Func_Idx).Name,
                     Return_Type => File.Functions (Func_Idx).Return_Type,
                     Parameters  => File.Functions (Func_Idx).Parameters);
               end if;
            end loop;
         end;
      end loop;

      Spec.Module.Functions.Count := Total_Functions;

      --  Generate spec hash (simplified)
      Spec.Spec_Hash := Generate_Simple_Hash (Spec);

      Status := Success;
   end Assemble_Spec;

   function Generate_Simple_Hash (Spec : Spec_Data) return Identifier_String is
      Hash_Value : constant String := "spec_" &
        Function_Index'Image (Spec.Module.Functions.Count);
   begin
      return Identifier_Strings.To_Bounded_String (Hash_Value);
   end Generate_Simple_Hash;

   procedure Generate_Spec_JSON
     (Spec        : in     Spec_Data;
      JSON_Output :    out JSON_String;
      Status      :    out Status_Code)
   is
      use JSON_Strings;
      Output : JSON_String := Null_Bounded_String;
   begin
      --  Build JSON output string
      Append (Output, "{");
      Append (Output, """schema_version"":""");
      Append (Output, Identifier_Strings.To_String (Spec.Schema_Version));
      Append (Output, """,");

      Append (Output, """origin"":""");
      Append (Output, Identifier_Strings.To_String (Spec.Origin));
      Append (Output, """,");

      Append (Output, """spec_hash"":""");
      Append (Output, Identifier_Strings.To_String (Spec.Spec_Hash));
      Append (Output, """,");

      Append (Output, """source_index"":""");
      Append (Output, Path_Strings.To_String (Spec.Source_Index));
      Append (Output, """,");

      Append (Output, """module"":{");
      Append (Output, """name"":""");
      Append (Output, Identifier_Strings.To_String (Spec.Module.Name));
      Append (Output, """,");

      Append (Output, """functions"":[");

      --  Add functions
      for I in 1 .. Spec.Module.Functions.Count loop
         if I > 1 then
            Append (Output, ",");
         end if;

         Append (Output, "{");
         Append (Output, """name"":""");
         Append (Output, Identifier_Strings.To_String (Spec.Module.Functions.Functions (I).Name));
         Append (Output, """,");

         Append (Output, """return_type"":""");
         Append (Output, Type_Name_Strings.To_String (Spec.Module.Functions.Functions (I).Return_Type));
         Append (Output, """,");

         Append (Output, """parameters"":[");

         declare
            Params : Parameter_List renames
               Spec.Module.Functions.Functions (I).Parameters;
         begin
            for P in 1 .. Params.Count loop
               if P > 1 then
                  Append (Output, ",");
               end if;

               Append (Output, "{");
               Append (Output, """name"":""");
               Append (Output, Identifier_Strings.To_String (Params.Params (P).Name));
               Append (Output, """,");
               Append (Output, """type"":""");
               Append (Output, Type_Name_Strings.To_String (Params.Params (P).Param_Type));
               Append (Output, """}");
            end loop;
         end;

         Append (Output, "]}");
      end loop;

      Append (Output, "]}}");

      JSON_Output := Output;
      Status := Success;
   end Generate_Spec_JSON;

   procedure Process_Extraction_File
     (Input_Path  : in     Path_String;
      Output_Path : in     Path_String;
      Module_Name : in     Identifier_String;
      Status      :    out Status_Code)
   is
      pragma SPARK_Mode (Off);  --  File I/O not in SPARK

      Input_File  : File_Type;
      Output_File : File_Type;
      Content_Buffer : Unbounded_String := Null_Unbounded_String;
      JSON_Content : JSON_String;
      Extraction   : Extraction_Data;
      Spec         : Spec_Data;
      JSON_Output  : JSON_String;
   begin
      Status := Success;

      --  Read extraction JSON file
      begin
         Open (Input_File, In_File, Path_Strings.To_String (Input_Path));
         while not End_Of_File (Input_File) loop
            declare
               Line : constant String := Get_Line (Input_File);
            begin
               if Length (Content_Buffer) + Line'Length <= Max_JSON_Length then
                  Append (Content_Buffer, Line);
               else
                  Close (Input_File);
                  Status := Error_Too_Large;
                  return;
               end if;
            end;
         end loop;
         Close (Input_File);
      exception
         when others =>
            Status := Error_File_IO;
            return;
      end;

      if Length (Content_Buffer) = 0 then
         Status := Error_Invalid_Input;
         return;
      end if;

      JSON_Content := JSON_Strings.To_Bounded_String (To_String (Content_Buffer));

      Parse_Extraction_JSON (JSON_Content, Extraction, Status);
      if Status /= Success then
         return;
      end if;

      Validate_Extraction (Extraction, Status);
      if Status /= Success then
         return;
      end if;

      Assemble_Spec (Extraction, Module_Name, Spec, Status);
      if Status /= Success then
         return;
      end if;

      Generate_Spec_JSON (Spec, JSON_Output, Status);
      if Status /= Success then
         return;
      end if;

      --  Write spec JSON
      begin
         Create (Output_File, Out_File, Path_Strings.To_String (Output_Path));
         Put (Output_File, JSON_Strings.To_String (JSON_Output));
         Close (Output_File);
      exception
         when others =>
            Status := Error_File_IO;
            return;
      end;

      Status := Success;
   end Process_Extraction_File;

end Spec_Assembler;

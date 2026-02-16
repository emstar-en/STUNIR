--  STUNIR Spec Assembler Package Body
--  Converts extraction.json to spec.json format
--  DO-333 / SPARK 2014 compliant
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

with STUNIR_JSON_Parser;
use STUNIR_JSON_Parser;

with Ada.Strings.Bounded;

package body Spec_Assembler is

   procedure Parse_Extraction_JSON
     (JSON_Content : in     JSON_String;
      Extraction   :    out Extraction_Data;
      Status       :    out Status_Code)
   is
      Parser     : Parser_State;
      Temp_Status : Status_Code;
   begin
      --  Initialize extraction data
      Extraction := Extraction_Data'(
         Schema_Version => Null_Identifier,
         Source_Index   => Path_Strings.Null_Bounded_String,
         Files          => (others => Extraction_File'(
            Source_File => Path_Strings.Null_Bounded_String,
            Functions   => (others => Extraction_Function'(
               Name        => Null_Identifier,
               Return_Type => Null_Type_Name,
               Parameters  => Parameter_List'(Params => (others => Parameter'(
                  Name       => Null_Identifier,
                  Param_Type => Null_Type_Name)),
               Count => 0)),
            Count => 0)),
         File_Count => 0);

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
                  Extraction.Schema_Version := Member_Name;
               elsif Name_Str = "source_index" then
                  Extraction.Source_Index := Path_Strings.To_Bounded_String (
                     JSON_Strings.To_String (Member_Value));
               elsif Name_Str = "files" then
                  --  Parse files array (simplified - would need full array parsing)
                  --  For now, skip the array value
                  Skip_Value (Parser, Temp_Status);
                  if Temp_Status /= Success then
                     Status := Error_Invalid_JSON;
                     return;
                  end if;
               elsif Name_Str = "module_name" then
                  --  Single module extraction format
                  null;  --  Will be handled differently
               elsif Name_Str = "functions" then
                  --  Parse functions directly (alternative format)
                  Skip_Value (Parser, Temp_Status);
                  if Temp_Status /= Success then
                     Status := Error_Invalid_JSON;
                     return;
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
      --  Initialize spec data
      Spec := Spec_Data'(
         Schema_Version => Identifier_Strings.To_Bounded_String ("1.0"),
         Origin         => Identifier_Strings.To_Bounded_String ("stunir"),
         Spec_Hash      => Null_Identifier,
         Source_Index   => Extraction.Source_Index,
         Module         => Spec_Module'(
            Name      => Module_Name,
            Functions => Function_Collection'(
               Functions => (others => Function_Signature'(
                  Name        => Null_Identifier,
                  Return_Type => Null_Type_Name,
                  Parameters  => Parameter_List'(Params => (others => Parameter'(
                     Name       => Null_Identifier,
                     Param_Type => Null_Type_Name)),
                  Count => 0)),
               Count => 0)));

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
      Append (Output, """schema_version":"");
      Append (Output, Identifier_Strings.To_String (Spec.Schema_Version));
      Append (Output, """,");

      Append (Output, """origin":"");
      Append (Output, Identifier_Strings.To_String (Spec.Origin));
      Append (Output, """,");

      Append (Output, """spec_hash":"");
      Append (Output, Identifier_Strings.To_String (Spec.Spec_Hash));
      Append (Output, """,");

      Append (Output, """source_index":"");
      Append (Output, Path_Strings.To_String (Spec.Source_Index));
      Append (Output, """,");

      Append (Output, """module":{");
      Append (Output, """name":"");
      Append (Output, Identifier_Strings.To_String (Spec.Module.Name));
      Append (Output, """,");

      Append (Output, """functions":[");

      --  Add functions
      for I in 1 .. Spec.Module.Functions.Count loop
         if I > 1 then
            Append (Output, ",");
         end if;

         Append (Output, "{");
         Append (Output, """name":"");
         Append (Output, Identifier_Strings.To_String (Spec.Module.Functions.Functions (I).Name));
         Append (Output, """,");

         Append (Output, """return_type":"");
         Append (Output, Type_Name_Strings.To_String (Spec.Module.Functions.Functions (I).Return_Type));
         Append (Output, """,");

         Append (Output, """parameters":[");

         declare
            Params : Parameter_List renames
               Spec.Module.Functions.Functions (I).Parameters;
         begin
            for P in 1 .. Params.Count loop
               if P > 1 then
                  Append (Output, ",");
               end if;

               Append (Output, "{");
               Append (Output, """name":"");
               Append (Output, Identifier_Strings.To_String (Params.Params (P).Name));
               Append (Output, """,");
               Append (Output, """type":"");
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
      pragma Unreferenced (Input_Path, Output_Path, Module_Name);
   begin
      --  This is the main entry point that would:
      --  1. Read the input file
      --  2. Parse JSON
      --  3. Assemble spec
      --  4. Write output file
      --
      --  For now, return not implemented
      --  Full implementation requires file I/O wrappers
      Status := Error_Not_Implemented;
   end Process_Extraction_File;

   --  Helper procedure to parse a function object from JSON
   procedure Parse_Function_Object
     (Parser   : in out Parser_State;
      Function_Out : out Extraction_Function;
      Status   : out Status_Code)
   with
      Pre => Current_Token (Parser) = Token_Object_Start
   is
      Temp_Status : Status_Code;
   begin
      Function_Out := Extraction_Function'(
         Name        => Null_Identifier,
         Return_Type => Null_Type_Name,
         Parameters  => Parameter_List'(Params => (others => Parameter'(
            Name       => Null_Identifier,
            Param_Type => Null_Type_Name)),
         Count => 0));

      --  Expect object start (already checked in precondition)
      Expect_Token (Parser, Token_Object_Start, Temp_Status);
      if Temp_Status /= Success then
         Status := Temp_Status;
         return;
      end if;

      --  Parse function members
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

            declare
               Name_Str : constant String := Identifier_Strings.To_String (Member_Name);
            begin
               if Name_Str = "name" then
                  Function_Out.Name := Identifier_Strings.To_Bounded_String (
                     JSON_Strings.To_String (Member_Value));
               elsif Name_Str = "return_type" then
                  Function_Out.Return_Type := Type_Name_Strings.To_Bounded_String (
                     JSON_Strings.To_String (Member_Value));
               elsif Name_Str = "parameters" then
                  --  Parse parameters array
                  Skip_Value (Parser, Temp_Status);
                  if Temp_Status /= Success then
                     Status := Error_Invalid_JSON;
                     return;
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
   end Parse_Function_Object;

end Spec_Assembler;

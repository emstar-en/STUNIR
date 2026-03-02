--  source_extract - Minimal source -> extraction.json (C/C++)
--  Phase 0: Source extraction
--  SPARK_Mode: Off (file I/O + parsing)
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (Off);

with Ada.Text_IO;
with Ada.Strings.Fixed;
with Ada.Strings.Unbounded;
with Ada.Exceptions;
with STUNIR_Types;

package body Source_Extract is
   use Ada.Text_IO;
   use Ada.Strings.Fixed;
   use Ada.Strings.Unbounded;
   use STUNIR_Types;

   function Trim_Spaces (S : String) return String is
   begin
      return Trim (S, Ada.Strings.Both);
   end Trim_Spaces;

   function Is_Identifier_Char (C : Character) return Boolean is
   begin
      return (C in 'a' .. 'z') or else (C in 'A' .. 'Z') or else (C in '0' .. '9') or else C = '_';
   end Is_Identifier_Char;

   function Extract_Name (S : String) return String is
      I : Integer := S'Last;
   begin
      while I >= S'First and then not Is_Identifier_Char (S (I)) loop
         I := I - 1;
      end loop;
      if I < S'First then
         return "";
      end if;
      declare
         J : Integer := I;
      begin
         while J >= S'First and then Is_Identifier_Char (S (J)) loop
            J := J - 1;
         end loop;
         return S (J + 1 .. I);
      end;
   end Extract_Name;

   procedure Emit_Function
     (Output : in File_Type;
      Name   : in String;
      Ret    : in String;
      Args   : in Unbounded_String;
      First  : in out Boolean)
   is
   begin
      if not First then
         Put_Line (Output, ",");
      end if;
      First := False;
      Put_Line (Output, "    {");
      Put_Line (Output, "      ""name"": """ & Name & """,");
      Put_Line (Output, "      ""return_type"": """ & Ret & """,");
      Put_Line (Output, "      ""parameters"": [" & To_String (Args) & "]");
      Put (Output, "    }");
   end Emit_Function;

   procedure Parse_Params
     (Param_Str : in String;
      Args_JSON : out Unbounded_String)
   is
      P    : constant String := Trim_Spaces (Param_Str);
      Start : Positive := 1;
      Depth : Integer := 0;
      First : Boolean := True;
   begin
      Args_JSON := Null_Unbounded_String;
      if P'Length = 0 or else P = "void" then
         return;
      end if;

      for I in P'Range loop
         if P (I) = '(' or else P (I) = '<' then
            Depth := Depth + 1;
         elsif P (I) = ')' or else P (I) = '>' then
            Depth := Depth - 1;
         elsif P (I) = ',' and then Depth = 0 then
            declare
               Part : constant String := Trim_Spaces (P (Start .. I - 1));
               Name : constant String := Extract_Name (Part);
               Type_Str : Unbounded_String := Null_Unbounded_String;
            begin
               if Name'Length > 0 and then Name'Length < Part'Length then
                  Type_Str := To_Unbounded_String (Trim_Spaces (Part (Part'First .. Part'Last - Name'Length)));
               elsif Name'Length = 0 then
                  Type_Str := To_Unbounded_String (Part);
               end if;
               if not First then
                  Append (Args_JSON, ", ");
               end if;
               First := False;
               if Name'Length = 0 then
                  Append (Args_JSON, "{""name"": ""arg"", ""type"": """ & To_String (Type_Str) & """}");
               else
                  Append (Args_JSON, "{""name"": """ & Name & """, ""type"": """ & To_String (Type_Str) & """}");
               end if;
            end;
            Start := I + 1;
         end if;
      end loop;

      if Start <= P'Last then
         declare
            Part : constant String := Trim_Spaces (P (Start .. P'Last));
            Name : constant String := Extract_Name (Part);
            Type_Str : Unbounded_String := Null_Unbounded_String;
         begin
            if Name'Length > 0 and then Name'Length < Part'Length then
               Type_Str := To_Unbounded_String (Trim_Spaces (Part (Part'First .. Part'Last - Name'Length)));
            elsif Name'Length = 0 then
               Type_Str := To_Unbounded_String (Part);
            end if;
            if not First then
               Append (Args_JSON, ", ");
            end if;
            if Name'Length = 0 then
               Append (Args_JSON, "{""name"": ""arg"", ""type"": """ & To_String (Type_Str) & """}");
            else
               Append (Args_JSON, "{""name"": """ & Name & """, ""type"": """ & To_String (Type_Str) & """}");
            end if;
         end;
      end if;
   end Parse_Params;

   procedure Extract_File
     (Input_Path  : in     Path_String;
      Output_Path : in     Path_String;
      Module_Name : in     Identifier_String;
      Language    : in     Identifier_String;
      Status      :    out Status_Code)
   is
      Input  : File_Type;
      Output : File_Type;
      Line   : String (1 .. 4096);
      Last   : Natural;
      First_Function : Boolean := True;
   begin
      Status := Success;
      Open (Input, In_File, Path_Strings.To_String (Input_Path));
      Create (Output, Out_File, Path_Strings.To_String (Output_Path));

      Put_Line (Output, "{");
      Put_Line (Output, "  ""schema_version"": ""extraction.v2"",");
      Put_Line (Output, "  ""module_name"": """ & Identifier_Strings.To_String (Module_Name) & """,");
      Put_Line (Output, "  ""language"": """ & Identifier_Strings.To_String (Language) & """,");
      Put_Line (Output, "  ""functions"": [");

      while not End_Of_File (Input) loop
         Get_Line (Input, Line, Last);
         if Last = 0 then
            goto Continue_Loop;
         end if;
         declare
            L : constant String := Line (1 .. Last);
            Open_Paren : constant Natural := Index (L, "(");
            Close_Paren : constant Natural := Index (L, ")");
            Ret_End     : Natural;
            Name_Start  : Natural;
            Name_End    : Natural;
         begin
            if Open_Paren = 0 or else Close_Paren = 0 then
               goto Continue_Loop;
            end if;
            if Index (L, "=") > 0 and then Index (L, "{") = 0 then
               goto Continue_Loop;
            end if;
            if L'Length >= 1 and then (L (1) = ' ' or else L (1) = ASCII.HT) then
               goto Continue_Loop;
            end if;

            --  Extract return type and name by scanning before '(' from right
            Name_End := Open_Paren - 1;
            while Name_End >= L'First and then L (Name_End) = ' ' loop
               Name_End := Name_End - 1;
            end loop;
            Name_Start := Name_End;
            while Name_Start > L'First and then Is_Identifier_Char (L (Name_Start - 1)) loop
               Name_Start := Name_Start - 1;
            end loop;
            Ret_End := Name_Start - 1;
            while Ret_End >= L'First and then L (Ret_End) = ' ' loop
               Ret_End := Ret_End - 1;
            end loop;

            if Name_Start < L'First or else Name_End < Name_Start or else Ret_End < L'First then
               goto Continue_Loop;
            end if;

            declare
               Ret_Type  : constant String := Trim_Spaces (L (L'First .. Ret_End));
               Func_Name : constant String := L (Name_Start .. Name_End);
               Params    : constant String := L (Open_Paren + 1 .. Close_Paren - 1);
               Args_JSON : Unbounded_String;
            begin
               if Ret_Type'Length = 0 or else Func_Name'Length = 0 then
                  goto Continue_Loop;
               end if;
               Parse_Params (Params, Args_JSON);
               Emit_Function (Output, Func_Name, Ret_Type, Args_JSON, First_Function);
            end;
         end;
         <<Continue_Loop>> null;
      end loop;

      Put_Line (Output, "  ]");
      Put_Line (Output, "}");
      Close (Input);
      Close (Output);
   exception
      when E : others =>
         Status := Error_File_IO;
         Put_Line (Standard_Error, "Exception in Extract_File: " & Ada.Exceptions.Exception_Information (E));
         if Is_Open (Input) then
            Close (Input);
         end if;
         if Is_Open (Output) then
            Close (Output);
         end if;
   end Extract_File;

end Source_Extract;

--  spark_extract - Minimal SPARK source -> extraction.json
--  Phase 0: Source extraction (SPARK)
--  SPARK_Mode: Off (file I/O + parsing)
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (Off);

with Ada.Text_IO;
with Ada.Strings.Fixed;
with Ada.Strings.Unbounded;
with Ada.Directories;
with Ada.Exceptions;
with STUNIR_Types;

package body Spark_Extract is
   use Ada.Text_IO;
   use Ada.Strings.Fixed;
   use Ada.Strings.Unbounded;
   use STUNIR_Types;

   Last_Error_Text : Unbounded_String := Null_Unbounded_String;

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

   --  Extract first identifier (for Ada param names)
   function Extract_First_Name (S : String) return String is
      I : Integer := S'First;
   begin
      --  Skip leading non-identifier chars
      while I <= S'Last and then not Is_Identifier_Char (S (I)) loop
         I := I + 1;
      end loop;
      if I > S'Last then
         return "";
      end if;
      declare
         J : Integer := I;
      begin
         while J <= S'Last and then Is_Identifier_Char (S (J)) loop
            J := J + 1;
         end loop;
         return S (I .. J - 1);
      end;
   end Extract_First_Name;

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
      if P'Length = 0 then
         return;
      end if;

      --  Ada uses semicolon as parameter separator
      for I in P'Range loop
         if P (I) = '(' then
            Depth := Depth + 1;
         elsif P (I) = ')' then
            Depth := Depth - 1;
         elsif P (I) = ';' and then Depth = 0 then
            declare
               Part : constant String := Trim_Spaces (P (Start .. I - 1));
               Param_Name : constant String := Extract_First_Name (Part);
               Type_Name : constant String := Extract_Name (Part);
            begin
               if not First then
                  Append (Args_JSON, ", ");
               end if;
               First := False;
               if Param_Name'Length = 0 then
                  Append (Args_JSON, "{""name"": ""arg"", ""type"": """ & Type_Name & """}");
               else
                  Append (Args_JSON, "{""name"": """ & Param_Name & """, ""type"": """ & Type_Name & """}");
               end if;
            end;
            Start := I + 1;
         end if;
      end loop;

      if Start <= P'Last then
         declare
            Part : constant String := Trim_Spaces (P (Start .. P'Last));
            Param_Name : constant String := Extract_First_Name (Part);
            Type_Name : constant String := Extract_Name (Part);
         begin
            if not First then
               Append (Args_JSON, ", ");
            end if;
            if Param_Name'Length = 0 then
               Append (Args_JSON, "{""name"": ""arg"", ""type"": """ & Type_Name & """}");
            else
               Append (Args_JSON, "{""name"": """ & Param_Name & """, ""type"": """ & Type_Name & """}");
            end if;
         end;
      end if;
   end Parse_Params;

   function Looks_Like_Subprogram (L : String) return Boolean is
   begin
      return Index (L, "procedure ") > 0 or else Index (L, "function ") > 0;
   end Looks_Like_Subprogram;

   function Signature_End_Pos (L : String) return Natural is
      In_String : Boolean := False;
      Depth : Integer := 0;
   begin
      for I in L'Range loop
         if L (I) = '"' then
            In_String := not In_String;
         elsif not In_String then
            if L (I) = '(' then
               Depth := Depth + 1;
            elsif L (I) = ')' then
               if Depth > 0 then
                  Depth := Depth - 1;
               end if;
            elsif Depth = 0 then
               if L (I) = ';' then
                  return I;
               end if;
               if L (I) = 'i'
                 and then I > L'First
                 and then L (I - 1) = ' '
                 and then I < L'Last
                 and then L (I + 1) = 's'
               then
                  if I + 1 = L'Last or else L (I + 2) = ' ' then
                     return I - 1;
                  end if;
               end if;
            end if;
         end if;
      end loop;
      return 0;
   end Signature_End_Pos;

   function Has_Signature_End (L : String) return Boolean is
   begin
      return Signature_End_Pos (L) > 0;
   end Has_Signature_End;

   function Extract_Return_Type_From_Spec
     (Body_Path : String;
      Subprogram_Name : String) return String
   is
      Spec_Path : String := Body_Path;
      Name_Lower : constant String := Subprogram_Name;
      Spec_File  : File_Type;
      Line       : String (1 .. 4096);
      Last       : Natural;
   begin
      if Spec_Path'Length >= 4 and then Spec_Path (Spec_Path'Last - 3 .. Spec_Path'Last) = ".adb" then
         Spec_Path := Spec_Path (Spec_Path'First .. Spec_Path'Last - 1) & "s";
      else
         return "";
      end if;

      if not Ada.Directories.Exists (Spec_Path) then
         return "";
      end if;

      begin
         Open (Spec_File, In_File, Spec_Path);
      exception
         when others =>
            return "";
      end;

      declare
         Sig_Buffer : Unbounded_String := Null_Unbounded_String;
         In_Signature : Boolean := False;
      begin
         while not End_Of_File (Spec_File) loop
            Get_Line (Spec_File, Line, Last);
            if Last = 0 then
               goto Next_Spec_Line;
            end if;
            declare
               L : constant String := Trim_Spaces (Line (1 .. Last));
            begin
               if L'Length = 0 then
                  goto Next_Spec_Line;
               end if;

               if not In_Signature then
                  if Index (L, "function ") > 0 then
                     Sig_Buffer := To_Unbounded_String (L);
                     In_Signature := True;
                  else
                     goto Next_Spec_Line;
                  end if;
               else
                  Append (Sig_Buffer, " " & L);
               end if;

               if In_Signature and then Has_Signature_End (To_String (Sig_Buffer)) then
                  declare
                     Sig_Line : constant String := To_String (Sig_Buffer);
                     Func_Pos : constant Natural := Index (Sig_Line, "function ");
                     Ret_Pos  : constant Natural := Index (Sig_Line, " return ");
                     Semi_Pos : Natural := 0;
                  begin
                     Semi_Pos := Signature_End_Pos (Sig_Line);

                     if Func_Pos > 0 and then Ret_Pos > Func_Pos and then Semi_Pos > Ret_Pos then
                        declare
                           Name_Start : constant Natural := Func_Pos + 9;
                           Open_Paren : constant Natural := Index (Sig_Line, "(");
                           Name_End   : Natural := Ret_Pos - 1;
                           Name_Text  : String := "";
                        begin
                           if Open_Paren > 0 and then Open_Paren > Name_Start then
                              Name_End := Open_Paren - 1;
                           end if;
                           if Name_End >= Name_Start then
                              Name_Text := Trim_Spaces (Sig_Line (Name_Start .. Name_End));
                              if Name_Text = Name_Lower then
                                 Close (Spec_File);
                                 return Trim_Spaces (Sig_Line (Ret_Pos + 8 .. Semi_Pos - 1));
                              end if;
                           end if;
                        end;
                     end if;
                  end;

                  Sig_Buffer := Null_Unbounded_String;
                  In_Signature := False;
               end if;
            end;
            <<Next_Spec_Line>> null;
         end loop;
      end;

      Close (Spec_File);
      return "";
   end Extract_Return_Type_From_Spec;

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
      Output_Path_Str : constant String := Path_Strings.To_String (Output_Path);
      Output_Dir : constant String := Ada.Directories.Containing_Directory (Output_Path_Str);
      Error_Path : constant String := Output_Path_Str & ".error.txt";
      Line_Number : Natural := 0;
      Debug_Line  : Unbounded_String := Null_Unbounded_String;
      Input_Path_Str : constant String := Path_Strings.To_String (Input_Path);
   begin
      Status := Success;
      Last_Error_Text := Null_Unbounded_String;
      if Output_Dir'Length > 0 and then not Ada.Directories.Exists (Output_Dir) then
         Ada.Directories.Create_Path (Output_Dir);
      end if;

      if not Ada.Directories.Exists (Input_Path_Str) then
         Status := Error_File_Not_Found;
         Last_Error_Text := To_Unbounded_String ("Input not found: " & Input_Path_Str);
         return;
      end if;

      begin
         Open (Input, In_File, Input_Path_Str);
      exception
         when E : others =>
            Status := Error_File_Read;
            Last_Error_Text := To_Unbounded_String ("Open failed: " & Ada.Exceptions.Exception_Information (E));
            return;
      end;

      begin
         Create (Output, Out_File, Output_Path_Str);
      exception
         when E : others =>
            Status := Error_File_Write;
            Last_Error_Text := To_Unbounded_String ("Create failed: " & Ada.Exceptions.Exception_Information (E));
            if Is_Open (Input) then
               Close (Input);
            end if;
            return;
      end;

      Put_Line (Output, "{");
      Put_Line (Output, "  ""schema_version"": ""extraction.v2"",");
      Put_Line (Output, "  ""module_name"": """ & Identifier_Strings.To_String (Module_Name) & """,");
      Put_Line (Output, "  ""language"": """ & Identifier_Strings.To_String (Language) & """,");
      Put_Line (Output, "  ""functions"": [");

      declare
         Sig_Buffer : Unbounded_String := Null_Unbounded_String;
         In_Signature : Boolean := False;
      begin
         while not End_Of_File (Input) loop
            Get_Line (Input, Line, Last);
            Line_Number := Line_Number + 1;
            if Last > 0 then
               Debug_Line := To_Unbounded_String (Line (1 .. Last));
            else
               Debug_Line := To_Unbounded_String ("");
            end if;
            if Last = 0 then
               goto Continue_Loop;
            end if;
            declare
               L : constant String := Trim_Spaces (Line (1 .. Last));
            begin
               if L'Length = 0 then
                  goto Continue_Loop;
               end if;
               if L (1) = '-' then
                  goto Continue_Loop;
               end if;

               if not In_Signature then
                  if Looks_Like_Subprogram (L) then
                     Sig_Buffer := To_Unbounded_String (L);
                     In_Signature := True;
                  else
                     goto Continue_Loop;
                  end if;
               else
                  Append (Sig_Buffer, " " & L);
               end if;

               if In_Signature and then Has_Signature_End (To_String (Sig_Buffer)) then
                  declare
                     Sig_Line : constant String := To_String (Sig_Buffer);
                     Is_Function : constant Boolean := Index (Sig_Line, "function ") > 0;
                     Is_Procedure : constant Boolean := Index (Sig_Line, "procedure ") > 0;
                     Ret_Type : Unbounded_String := Null_Unbounded_String;
                     Name : Unbounded_String := Null_Unbounded_String;
                     Params : Unbounded_String := Null_Unbounded_String;
                     Open_Paren : Natural := Index (Sig_Line, "(");
                     Close_Paren : Natural := Index (Sig_Line, ")");
                     Has_Params : Boolean := Open_Paren > 0 and then Close_Paren > Open_Paren;
                     Semi_Pos : Natural := 0;
                     Skip_This : Boolean := False;
                  begin
                     Semi_Pos := Signature_End_Pos (Sig_Line);

                     if Semi_Pos = 0 then
                        Skip_This := True;
                     end if;

                     if not Skip_This and then Is_Function then
                        declare
                           Func_Pos : constant Natural := Index (Sig_Line, "function ") + 9;
                           Ret_Pos : constant Natural := Index (Sig_Line, " return ");
                        begin
                           if Ret_Pos = 0 or else Ret_Pos <= Func_Pos then
                              Skip_This := True;
                           elsif Ret_Pos + 8 > Sig_Line'Last then
                              Skip_This := True;
                           else
                              if Has_Params then
                                 if Open_Paren <= Func_Pos then
                                    Skip_This := True;
                                 else
                                    Name := To_Unbounded_String (Trim_Spaces (Sig_Line (Func_Pos .. Open_Paren - 1)));
                                 end if;
                              else
                                 Name := To_Unbounded_String (Trim_Spaces (Sig_Line (Func_Pos .. Ret_Pos - 1)));
                              end if;
                              if not Skip_This then
                                 Ret_Type := To_Unbounded_String (Trim_Spaces (Sig_Line (Ret_Pos + 8 .. Semi_Pos - 1)));
                              end if;
                           end if;
                        end;
                     elsif not Skip_This and then Is_Procedure then
                        declare
                           Proc_Pos : constant Natural := Index (Sig_Line, "procedure ") + 10;
                        begin
                           if Has_Params then
                              if Open_Paren <= Proc_Pos then
                                 Skip_This := True;
                              else
                                 Name := To_Unbounded_string (Trim_Spaces (Sig_Line (Proc_Pos .. Open_Paren - 1)));
                              end if;
                           else
                              Name := To_Unbounded_String (Trim_Spaces (Sig_Line (Proc_Pos .. Semi_Pos - 1)));
                           end if;
                        end;
                     else
                        Skip_This := True;
                     end if;
                     if not Skip_This and then Is_Procedure then
                        Ret_Type := To_Unbounded_String ("void");
                     end if;

                     if not Skip_This then
                        if Has_Params and then Close_Paren > Open_Paren + 1 then
                           Params := To_Unbounded_String (Sig_Line (Open_Paren + 1 .. Close_Paren - 1));
                        else
                           Params := Null_Unbounded_String;
                        end if;

                        if Length (Name) = 0 then
                           Skip_This := True;
                        end if;
                     end if;

                     if not Skip_This then
                        if Is_Function and then (Length (Ret_Type) = 0 or else To_String (Ret_Type) = "void") and then Input_Path_Str'Length > 0 then
                           declare
                              Lookup : constant String := Extract_Return_Type_From_Spec (Input_Path_Str, To_String (Name));
                           begin
                              if Lookup'Length > 0 then
                                 Ret_Type := To_Unbounded_String (Lookup);
                              end if;
                           end;
                        end if;
                        declare
                           Args_JSON : Unbounded_String;
                        begin
                           Parse_Params (To_String (Params), Args_JSON);
                           Emit_Function (Output, To_String (Name), To_String (Ret_Type), Args_JSON, First_Function);
                        end;
                     end if;
                  exception
                     when E : others =>
                        Put_Line (Standard_Error, "DEBUG: Exception in parsing: " & Ada.Exceptions.Exception_Information (E));
                  end;

                  Sig_Buffer := Null_Unbounded_String;
                  In_Signature := False;
               end if;
            end;
            <<Continue_Loop>> null;
         end loop;
      end;

      Put_Line (Output, "  ]");
      Put_Line (Output, "}");
      Close (Input);
      Close (Output);
   exception
      when E : others =>
         Status := Error_File_IO;
         Last_Error_Text := To_Unbounded_String (Ada.Exceptions.Exception_Information (E));
         Put_Line (Standard_Error, "Error: spark_extract failed - " & Ada.Exceptions.Exception_Information (E));
         Put_Line (Standard_Error, "Line: " & Natural'Image (Line_Number));
         Put_Line (Standard_Error, "Content: " & To_String (Debug_Line));

         if Is_Open (Output) then
            begin
               Put_Line (Output, "  ]");
               Put_Line (Output, "}");
            exception
               when others => null;
            end;
         end if;

         declare
            Err : File_Type;
         begin
            Create (Err, Out_File, Error_Path);
            Put_Line (Err, "Error: spark_extract failed");
            Put_Line (Err, Ada.Exceptions.Exception_Information (E));
            Put_Line (Err, "Line: " & Natural'Image (Line_Number));
            Put_Line (Err, "Content: " & To_String (Debug_Line));
            Close (Err);
         exception
            when others => null;
         end;

         declare
            Err2 : File_Type;
         begin
            Create (Err2, Out_File, "spark_extract_last_error.txt");
            Put_Line (Err2, "Error: spark_extract failed");
            Put_Line (Err2, Ada.Exceptions.Exception_Information (E));
            Put_Line (Err2, "Line: " & Natural'Image (Line_Number));
            Put_Line (Err2, "Content: " & To_String (Debug_Line));
            Close (Err2);
         exception
            when others => null;
         end;

         if Is_Open (Input) then
            Close (Input);
         end if;
         if Is_Open (Output) then
            Close (Output);
         end if;
   end Extract_File;

   function Last_Error return String is
   begin
      return To_String (Last_Error_Text);
   end Last_Error;

end Spark_Extract;

--  python_extract - Python source -> extraction.json
--  Phase 2: Improved parsing with multiline support, decorators, type hints
--  SPARK_Mode: Off (file I/O + parsing)
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (Off);

with Ada.Text_IO;
with Ada.Strings.Fixed;
with Ada.Strings.Unbounded;
with Ada.Characters.Latin_1;
with STUNIR_Types;

package body Python_Extract is
   use Ada.Text_IO;
   use Ada.Strings.Fixed;
   use Ada.Strings.Unbounded;
   use STUNIR_Types;
   use Ada.Characters.Latin_1;

   function Trim_Spaces (S : String) return String is
      Result : String := Trim (S, Ada.Strings.Both);
      Last   : Natural := Result'Last;
   begin
      --  Also trim CR characters (Windows line endings)
      while Last >= Result'First and then Result (Last) = ASCII.CR loop
         Last := Last - 1;
      end loop;
      if Last < Result'Last then
         return Result (Result'First .. Last);
      else
         return Result;
      end if;
   end Trim_Spaces;

   function Is_Identifier_Char (C : Character) return Boolean is
   begin
      return (C in 'a' .. 'z') or else (C in 'A' .. 'Z') or else (C in '0' .. '9') or else C = '_';
   end Is_Identifier_Char;

   --  Check if line is a decorator (starts with @)
   function Is_Decorator_Line (L : String) return Boolean is
   begin
      for I in L'Range loop
         if L (I) = ' ' or else L (I) = ASCII.HT then
            null;  --  Skip leading whitespace
         elsif L (I) = '@' then
            return True;
         else
            return False;
         end if;
      end loop;
      return False;
   end Is_Decorator_Line;

   --  Check if line is a comment (starts with #)
   function Is_Comment_Line (L : String) return Boolean is
   begin
      for I in L'Range loop
         if L (I) = ' ' or else L (I) = ASCII.HT then
            null;  --  Skip leading whitespace
         elsif L (I) = '#' then
            return True;
         else
            return False;
         end if;
      end loop;
      return False;
   end Is_Comment_Line;

   --  Count unbalanced parens to detect incomplete function signatures
   function Count_Paren_Balance (S : String) return Integer is
      Count : Integer := 0;
   begin
      for I in S'Range loop
         if S (I) = '(' then
            Count := Count + 1;
         elsif S (I) = ')' then
            Count := Count - 1;
         end if;
      end loop;
      return Count;
   end Count_Paren_Balance;

   procedure Emit_Function
     (Output : in File_Type;
      Name   : in String;
      Ret    : in String;
      Params : in Unbounded_String;
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
      Put_Line (Output, "      ""parameters"": [" & To_String (Params) & "]");
      Put (Output, "    }");
   end Emit_Function;

   --  Parse Python parameter with optional type hint and default value
   --  Formats: name, name: type, name=default, name: type = default
   procedure Parse_Single_Param
     (Part        : in     String;
      Param_Name  :    out Unbounded_String;
      Param_Type  :    out Unbounded_String)
   is
      Colon_Pos  : Natural := 0;
      Equal_Pos  : Natural := 0;
      Name_Part  : String := Trim_Spaces (Part);
      Type_Part  : String := "";
   begin
      Param_Name := To_Unbounded_String (Name_Part);
      Param_Type := To_Unbounded_String ("any");
      
      --  Find colon (type hint separator) - but not inside brackets
      declare
         Depth : Integer := 0;
      begin
         for I in Name_Part'Range loop
            if Name_Part (I) = '[' then
               Depth := Depth + 1;
            elsif Name_Part (I) = ']' then
               Depth := Depth - 1;
            elsif Name_Part (I) = ':' and then Depth = 0 then
               Colon_Pos := I;
               exit;
            end if;
         end loop;
      end;
      
      --  Find equal sign (default value separator)
      if Colon_Pos > 0 then
         declare
            After_Colon : String := Trim_Spaces (Name_Part (Colon_Pos + 1 .. Name_Part'Last));
         begin
            --  Look for = in the type part
            for I in After_Colon'Range loop
               if After_Colon (I) = '=' then
                  Equal_Pos := I;
                  exit;
               end if;
            end loop;
            
            if Equal_Pos > 0 then
               --  Has default value: name: type = default
               Param_Name := To_Unbounded_String (Trim_Spaces (Name_Part (Name_Part'First .. Colon_Pos - 1)));
               Param_Type := To_Unbounded_String (Trim_Spaces (After_Colon (After_Colon'First .. Equal_Pos - 1)));
            else
               --  No default: name: type
               Param_Name := To_Unbounded_String (Trim_Spaces (Name_Part (Name_Part'First .. Colon_Pos - 1)));
               Param_Type := To_Unbounded_String (After_Colon);
            end if;
         end;
      else
         --  No type hint, check for default value
         for I in Name_Part'Range loop
            if Name_Part (I) = '=' then
               Equal_Pos := I;
               exit;
            end if;
         end loop;
         
         if Equal_Pos > 0 then
            --  Has default: name = default
            Param_Name := To_Unbounded_String (Trim_Spaces (Name_Part (Name_Part'First .. Equal_Pos - 1)));
         end if;
      end if;
   end Parse_Single_Param;

   procedure Parse_Params
     (Param_Str   : in String;
      Params_JSON : out Unbounded_String)
   is
      P    : constant String := Trim_Spaces (Param_Str);
      Start : Positive := 1;
      Depth : Integer := 0;
      First : Boolean := True;
   begin
      Params_JSON := Null_Unbounded_String;
      if P'Length = 0 then
         return;
      end if;

      for I in P'Range loop
         if P (I) = '(' or else P (I) = '[' or else P (I) = '{' then
            Depth := Depth + 1;
         elsif P (I) = ')' or else P (I) = ']' or else P (I) = '}' then
            Depth := Depth - 1;
         elsif P (I) = ',' and then Depth = 0 then
            declare
               Part : constant String := Trim_Spaces (P (Start .. I - 1));
               Param_Name : Unbounded_String;
               Param_Type : Unbounded_String;
            begin
               --  Skip special Python params
               if Part = "self" or else Part = "cls" then
                  null;  --  Skip self/cls
               else
                  Parse_Single_Param (Part, Param_Name, Param_Type);
                  
                  if not First then
                     Append (Params_JSON, ", ");
                  end if;
                  First := False;
                  Append (Params_JSON, "{""name"": """ & To_String (Param_Name) & """, ""type"": """ & To_String (Param_Type) & """}");
               end if;
            end;
            Start := I + 1;
         end if;
      end loop;

      if Start <= P'Last then
         declare
            Part : constant String := Trim_Spaces (P (Start .. P'Last));
            Param_Name : Unbounded_String;
            Param_Type : Unbounded_String;
         begin
            --  Skip special Python params
            if Part = "self" or else Part = "cls" then
               null;  --  Skip self/cls
            elsif Part'Length > 0 then
               Parse_Single_Param (Part, Param_Name, Param_Type);
               
               if not First then
                  Append (Params_JSON, ", ");
               end if;
               Append (Params_JSON, "{""name"": """ & To_String (Param_Name) & """, ""type"": """ & To_String (Param_Type) & """}");
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
      
      --  Multiline accumulation
      Accumulated : Unbounded_String := Null_Unbounded_String;
      Accum_Balance : Integer := 0;
      
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
            L : constant String := Trim_Spaces (Line (1 .. Last));
         begin
            --  Skip empty lines
            if L'Length = 0 then
               goto Continue_Loop;
            end if;
            
            --  Skip decorator lines (@...)
            if Is_Decorator_Line (L) then
               goto Continue_Loop;
            end if;
            
            --  Skip comment lines
            if Is_Comment_Line (L) then
               goto Continue_Loop;
            end if;
            
            --  Accumulate multiline signatures
            if Length (Accumulated) > 0 then
               Append (Accumulated, " " & L);
               Accum_Balance := Accum_Balance + Count_Paren_Balance (L);
               
               --  Check if we have a complete signature
               if Accum_Balance <= 0 then
                  --  Process accumulated line
                  declare
                     Full_Line : constant String := To_String (Accumulated);
                  begin
                     Accumulated := Null_Unbounded_String;
                     Accum_Balance := 0;
                     
                     --  Parse the accumulated line as a function
                     declare
                        Def_Pos : Natural;
                        Open_Paren : Natural;
                        Close_Paren : Natural := 0;
                     begin
                        Def_Pos := Index (Full_Line, "def ");
                        Open_Paren := Index (Full_Line, "(");
                        
                        --  Find matching close paren
                        declare
                           Depth : Integer := 0;
                        begin
                           for I in Full_Line'Range loop
                              if Full_Line (I) = '(' then
                                 Depth := Depth + 1;
                              elsif Full_Line (I) = ')' then
                                 Depth := Depth - 1;
                                 if Depth = 0 then
                                    Close_Paren := I;
                                    exit;
                                 end if;
                              end if;
                           end loop;
                        end;
                        
                        if Def_Pos > 0 and then Open_Paren > 0 and then Close_Paren > 0 then
                           declare
                              Name_Start : constant Natural := Def_Pos + 4;
                              Name_End   : Natural := Open_Paren - 1;
                           begin
                              --  Skip spaces in function name
                              while Name_End > Name_Start and then Full_Line (Name_End) = ' ' loop
                                 Name_End := Name_End - 1;
                              end loop;
                              
                              if Name_End >= Name_Start then
                                 declare
                                    Func_Name : constant String := Full_Line (Name_Start .. Name_End);
                                    Params    : constant String := Full_Line (Open_Paren + 1 .. Close_Paren - 1);
                                    Params_JSON : Unbounded_String;
                                 begin
                                    Parse_Params (Params, Params_JSON);
                                    Emit_Function (Output, Func_Name, "any", Params_JSON, First_Function);
                                 end;
                              end if;
                           end;
                        end if;
                     end;
                  end;
               end if;
               goto Continue_Loop;
            end if;
            
            --  Check for def keyword
            declare
               Def_Pos : Natural;
               Open_Paren : Natural;
               Close_Paren : Natural;
            begin
               Def_Pos := Index (L, "def ");
               Open_Paren := Index (L, "(");
               Close_Paren := Index (L, ")");
               
               if Def_Pos = 0 then
                  goto Continue_Loop;
               end if;
               
               --  Check if this is a multiline signature
               if Open_Paren > 0 and then Close_Paren = 0 then
                  --  Start accumulating
                  Accumulated := To_Unbounded_String (L);
                  Accum_Balance := Count_Paren_Balance (L);
                  goto Continue_Loop;
               end if;
               
               if Open_Paren = 0 or else Close_Paren = 0 then
                  goto Continue_Loop;
               end if;

               declare
                  Name_Start : constant Natural := Def_Pos + 4;
                  Name_End   : Natural := Open_Paren - 1;
               begin
                  while Name_End > Name_Start and then L (Name_End) = ' ' loop
                     Name_End := Name_End - 1;
                  end loop;
                  if Name_End < Name_Start then
                     goto Continue_Loop;
                  end if;
                  declare
                     Func_Name : constant String := L (Name_Start .. Name_End);
                     Params    : constant String := L (Open_Paren + 1 .. Close_Paren - 1);
                     Params_JSON : Unbounded_String;
                  begin
                     Parse_Params (Params, Params_JSON);
                     Emit_Function (Output, Func_Name, "any", Params_JSON, First_Function);
                  end;
               end;
            end;
         exception
            when others =>
               null;  --  Skip lines that cause parsing errors
         end;
         <<Continue_Loop>> null;
      end loop;

      Put_Line (Output, "");
      Put_Line (Output, "  ]");
      Put_Line (Output, "}");
      Close (Input);
      Close (Output);
   exception
      when others =>
         Status := Error_File_IO;
         if Is_Open (Input) then
            Close (Input);
         end if;
         if Is_Open (Output) then
            Close (Output);
         end if;
   end Extract_File;

end Python_Extract;

--  rust_extract - Rust source -> extraction.json
--  Phase 2: Improved parsing with multiline support, attributes, generics
--  SPARK_Mode: Off (file I/O + parsing)
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (Off);

with Ada.Text_IO;
with Ada.Strings.Fixed;
with Ada.Strings.Unbounded;
with Ada.Characters.Latin_1;
with STUNIR_Types;

package body Rust_Extract is
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

   --  Check if line is a Rust attribute (starts with #)
   function Is_Attribute_Line (L : String) return Boolean is
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
   end Is_Attribute_Line;

   --  Check if line is a comment (starts with //)
   function Is_Comment_Line (L : String) return Boolean is
   begin
      for I in L'Range loop
         if L (I) = ' ' or else L (I) = ASCII.HT then
            null;  --  Skip leading whitespace
         elsif I < L'Last and then L (I) = '/' and then L (I + 1) = '/' then
            return True;
         else
            return False;
         end if;
      end loop;
      return False;
   end Is_Comment_Line;

   --  Count unbalanced parens to detect incomplete function signatures
   --  Only counts () not {} or [] or <> (those are type delimiters)
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

   --  Remove Rust generics from function name (e.g., "foo<T>" -> "foo")
   function Strip_Generics (Name : String) return String is
      Bracket_Pos : Natural := 0;
   begin
      for I in Name'Range loop
         if Name (I) = '<' then
            Bracket_Pos := I;
            exit;
         end if;
      end loop;
      if Bracket_Pos > 0 then
         return Name (Name'First .. Bracket_Pos - 1);
      else
         return Name;
      end if;
   end Strip_Generics;

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
      Put_Line (Output, "      ""name"": """ & Strip_Generics (Name) & """,");
      Put_Line (Output, "      ""return_type"": """ & Ret & """,");
      Put_Line (Output, "      ""parameters"": [" & To_String (Params) & "]");
      Put (Output, "    }");
   end Emit_Function;

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
         if P (I) = '(' or else P (I) = '[' then
            Depth := Depth + 1;
         elsif P (I) = ')' or else P (I) = ']' then
            Depth := Depth - 1;
         elsif P (I) = '<' then
            Depth := Depth + 1;  --  Track generics
         elsif P (I) = '>' then
            Depth := Depth - 1;
         elsif P (I) = ',' and then Depth = 0 then
            declare
               Part : constant String := Trim_Spaces (P (Start .. I - 1));
               Colon_Pos : Natural := 0;
            begin
               --  Rust syntax: name: type
               --  Find colon that's not inside generics
               declare
                  Colon_Depth : Integer := 0;
               begin
                  for J in Part'Range loop
                     if Part (J) = '<' then
                        Colon_Depth := Colon_Depth + 1;
                     elsif Part (J) = '>' then
                        Colon_Depth := Colon_Depth - 1;
                     elsif Part (J) = ':' and then Colon_Depth = 0 then
                        Colon_Pos := J;
                        exit;
                     end if;
                  end loop;
               end;
               
               if not First then
                  Append (Params_JSON, ", ");
               end if;
               First := False;
               
               if Colon_Pos > 0 then
                  declare
                     Name : constant String := Trim_Spaces (Part (Part'First .. Colon_Pos - 1));
                     Type_Str : constant String := Trim_Spaces (Part (Colon_Pos + 1 .. Part'Last));
                  begin
                     Append (Params_JSON, "{""name"": """ & Name & """, ""type"": """ & Type_Str & """}");
                  end;
               else
                  --  No colon, use Extract_Name
                  declare
                     Name : constant String := Extract_Name (Part);
                     Type_Str : constant String := 
                       (if Name'Length > 0 and then Part'Length > Name'Length
                        then Trim_Spaces (Part (Part'First .. Part'Last - Name'Length))
                        else Part);
                  begin
                     if Name'Length = 0 then
                        Append (Params_JSON, "{""name"": ""arg"", ""type"": """ & Type_Str & """}");
                     else
                        Append (Params_JSON, "{""name"": """ & Name & """, ""type"": """ & Type_Str & """}");
                     end if;
                  end;
               end if;
            end;
            Start := I + 1;
         end if;
      end loop;

      if Start <= P'Last then
         declare
            Part : constant String := Trim_Spaces (P (Start .. P'Last));
            Colon_Pos : Natural := 0;
         begin
            --  Find colon that's not inside generics
            declare
               Colon_Depth : Integer := 0;
            begin
               for J in Part'Range loop
                  if Part (J) = '<' then
                     Colon_Depth := Colon_Depth + 1;
                  elsif Part (J) = '>' then
                     Colon_Depth := Colon_Depth - 1;
                  elsif Part (J) = ':' and then Colon_Depth = 0 then
                     Colon_Pos := J;
                     exit;
                  end if;
               end loop;
            end;
            
            if not First then
               Append (Params_JSON, ", ");
            end if;
            
            if Colon_Pos > 0 then
               declare
                  Name : constant String := Trim_Spaces (Part (Part'First .. Colon_Pos - 1));
                  Type_Str : constant String := Trim_Spaces (Part (Colon_Pos + 1 .. Part'Last));
               begin
                  Append (Params_JSON, "{""name"": """ & Name & """, ""type"": """ & Type_Str & """}");
               end;
            else
               --  No colon, use Extract_Name
               declare
                  Name : constant String := Extract_Name (Part);
                  Type_Str : constant String := 
                    (if Name'Length > 0 and then Part'Length > Name'Length
                     then Trim_Spaces (Part (Part'First .. Part'Last - Name'Length))
                     else Part);
               begin
                  if Name'Length = 0 then
                     Append (Params_JSON, "{""name"": ""arg"", ""type"": """ & Type_Str & """}");
                  else
                     Append (Params_JSON, "{""name"": """ & Name & """, ""type"": """ & Type_Str & """}");
                  end if;
               end;
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
            
            --  Skip attribute lines (#[...])
            if Is_Attribute_Line (L) then
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
                        Fn_Pos : Natural;
                        Open_Paren : Natural;
                        Close_Paren : Natural;
                        Ret_Pos : Natural := 0;
                     begin
                        Fn_Pos := Index (Full_Line, "fn ");
                        Open_Paren := Index (Full_Line, "(");
                        Close_Paren := 0;
                        
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
                        
                        --  Find "->" manually
                        for I in Full_Line'First .. Full_Line'last - 1 loop
                           if Full_Line (I) = '-' and then Full_Line (I + 1) = '>' then
                              Ret_Pos := I;
                              exit;
                           end if;
                        end loop;
                        
                        if Fn_Pos > 0 and then Open_Paren > 0 and then Close_Paren > 0 then
                           declare
                              Name_Start : constant Natural := Fn_Pos + 3;
                              Name_End   : Natural := Open_Paren - 1;
                           begin
                              --  Skip spaces in function name
                              while Name_End > Name_Start and then Full_Line (Name_End) = ' ' loop
                                 Name_End := Name_End - 1;
                              end loop;
                              
                              --  Handle generic brackets in name
                              declare
                                 Generic_Pos : Natural := 0;
                              begin
                                 for I in Name_Start .. Name_End loop
                                    if Full_Line (I) = '<' then
                                       Generic_Pos := I;
                                       exit;
                                    end if;
                                 end loop;
                                 if Generic_Pos > 0 then
                                    Name_End := Generic_Pos - 1;
                                 end if;
                              end;
                              
                              if Name_End >= Name_Start then
                                 declare
                                    Func_Name : constant String := Full_Line (Name_Start .. Name_End);
                                    Params    : constant String := Full_Line (Open_Paren + 1 .. Close_Paren - 1);
                                    Params_JSON : Unbounded_String;
                                    Ret_Type_Str : Unbounded_String := To_Unbounded_String ("void");
                                 begin
                                    if Ret_Pos > 0 and then Ret_Pos + 2 <= Full_Line'Last then
                                       declare
                                          Ret_Part : constant String := Full_Line (Ret_Pos + 2 .. Full_Line'Last);
                                          Brace_Pos : Natural;
                                       begin
                                          Ret_Type_Str := To_Unbounded_String (Ret_Part);
                                          Brace_Pos := Index (To_String (Ret_Type_Str), "{");
                                          if Brace_Pos > 0 then
                                             Ret_Type_Str := To_Unbounded_String (Ret_Part (Ret_Part'First .. Ret_Part'First + Brace_Pos - 2));
                                          end if;
                                          Ret_Type_Str := To_Unbounded_String (Trim_Spaces (To_String (Ret_Type_Str)));
                                       end;
                                    end if;
                                    Parse_Params (Params, Params_JSON);
                                    Emit_Function (Output, Func_Name, To_String (Ret_Type_Str), Params_JSON, First_Function);
                                 end;
                              end if;
                           end;
                        end if;
                     end;
                  end;
               end if;
               goto Continue_Loop;
            end if;
            
            --  Check for fn keyword
            declare
               Fn_Pos : Natural;
               Open_Paren : Natural;
               Close_Paren : Natural;
               Ret_Pos : Natural := 0;
            begin
               Fn_Pos := Index (L, "fn ");
               Open_Paren := Index (L, "(");
               Close_Paren := Index (L, ")");
               
               --  Find "->" manually
               for I in L'first .. L'last - 1 loop
                  if L (I) = '-' and then L (I + 1) = '>' then
                     Ret_Pos := I;
                     exit;
                  end if;
               end loop;
               
               if Fn_Pos = 0 then
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
                  Name_Start : constant Natural := Fn_Pos + 3;
                  Name_End   : Natural := Open_Paren - 1;
               begin
                  while Name_End > Name_Start and then L (Name_End) = ' ' loop
                     Name_End := Name_End - 1;
                  end loop;
                  if Name_End < Name_Start then
                     goto Continue_Loop;
                  end if;
                  
                  --  Handle generic brackets in name
                  declare
                     Generic_Pos : Natural := 0;
                  begin
                     for I in Name_Start .. Name_End loop
                        if L (I) = '<' then
                           Generic_Pos := I;
                           exit;
                        end if;
                     end loop;
                     if Generic_Pos > 0 then
                        Name_End := Generic_Pos - 1;
                     end if;
                  end;
                  
                  declare
                     Func_Name : constant String := L (Name_Start .. Name_End);
                     Params    : constant String := L (Open_Paren + 1 .. Close_Paren - 1);
                     Params_JSON : Unbounded_String;
                     Ret_Type_Str : Unbounded_String := To_Unbounded_String ("void");
                  begin
                     if Ret_Pos > 0 and then Ret_Pos + 2 <= L'Last then
                        declare
                           Ret_Part : constant String := L (Ret_Pos + 2 .. L'Last);
                           Brace_Pos : Natural;
                        begin
                           Ret_Type_Str := To_Unbounded_String (Ret_Part);
                           Brace_Pos := Index (To_String (Ret_Type_Str), "{");
                           if Brace_Pos > 0 then
                              Ret_Type_Str := To_Unbounded_String (Ret_Part (Ret_Part'First .. Ret_Part'First + Brace_Pos - 2));
                           end if;
                           Ret_Type_Str := To_Unbounded_String (Trim_Spaces (To_String (Ret_Type_Str)));
                        end;
                     end if;
                     Parse_Params (Params, Params_JSON);
                     Emit_Function (Output, Func_Name, To_String (Ret_Type_Str), Params_JSON, First_Function);
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

end Rust_Extract;

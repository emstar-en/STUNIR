--  func_to_ir - Convert spec functions array to IR functions array
--  Input:  JSON array of function spec objects (stdin)
--  Output: JSON array of IR function objects (stdout)

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;

procedure Func_To_IR is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;

   Exit_Success : constant := 0;
   Exit_Error   : constant := 1;

   Show_Help     : Boolean := False;
   Show_Version  : Boolean := False;
   Show_Describe : Boolean := False;

   Version : constant String := "0.1.0-alpha";

   Describe_Output : constant String :=
     "{""tool"":""func_to_ir"",""version"":""0.1.0-alpha""," &
     """description"":""Convert spec functions array to IR functions array""," &
     """inputs"":[{""type"":""json_array"",""source"":""stdin"",""required"":true}]," &
     """outputs"":[{""type"":""json_array"",""source"":""stdout""}]}";

   function Read_Stdin return String is
      Result : Unbounded_String := Null_Unbounded_String;
      Line   : String (1 .. 4096);
      Last   : Natural;
   begin
      while not End_Of_File loop
         Get_Line (Line, Last);
         Append (Result, Line (1 .. Last));
         Append (Result, ASCII.LF);
      end loop;
      return To_String (Result);
   end Read_Stdin;

   function Find (S : String; P : String) return Natural is
   begin
      if P'Length = 0 or P'Length > S'Length then return 0; end if;
      for I in S'First .. S'Last - P'Length + 1 loop
         if S (I .. I + P'Length - 1) = P then return I; end if;
      end loop;
      return 0;
   end Find;

   function Get_String (JSON : String; Key : String) return String is
      Pat  : constant String := """" & Key & """";
      K    : constant Natural := Find (JSON, Pat);
      P, E : Natural;
   begin
      if K = 0 then return ""; end if;
      P := K + Pat'Length;
      while P <= JSON'Last and then
            (JSON (P) = ':' or JSON (P) = ' ' or
             JSON (P) = ASCII.HT or JSON (P) = ASCII.LF or
             JSON (P) = ASCII.CR) loop
         P := P + 1;
      end loop;
      if P > JSON'Last or else JSON (P) /= '"' then return ""; end if;
      P := P + 1; E := P;
      while E <= JSON'Last and then JSON (E) /= '"' loop E := E + 1; end loop;
      if E > JSON'Last then return JSON (P .. JSON'Last); end if;
      return JSON (P .. E - 1);
   end Get_String;

   function Get_Block (JSON : String; Key : String) return String is
      Pat   : constant String := """" & Key & """";
      K     : constant Natural := Find (JSON, Pat);
      P, E  : Natural;
      Depth : Integer := 0;
      InStr : Boolean := False;
   begin
      if K = 0 then return ""; end if;
      P := K + Pat'Length;
      while P <= JSON'Last and then
            (JSON (P) = ':' or JSON (P) = ' ' or
             JSON (P) = ASCII.HT or JSON (P) = ASCII.LF or
             JSON (P) = ASCII.CR) loop
         P := P + 1;
      end loop;
      if P > JSON'Last then return ""; end if;
      if JSON (P) /= '[' and then JSON (P) /= '{' then return ""; end if;
      E := P;
      while E <= JSON'Last loop
         if InStr then
            if JSON (E) = '"' and then (E = JSON'First or else JSON (E - 1) /= '\') then
               InStr := False;
            end if;
         else
            if    JSON (E) = '"'                  then InStr := True;
            elsif JSON (E) = '{' or JSON (E) = '[' then Depth := Depth + 1;
            elsif JSON (E) = '}' or JSON (E) = ']' then
               Depth := Depth - 1;
               if Depth = 0 then return JSON (P .. E); end if;
            end if;
         end if;
         E := E + 1;
      end loop;
      return "";
   end Get_Block;

   function Get_Element (Arr : String; N : Natural) return String is
      Pos   : Natural := Arr'First;
      Cnt   : Natural := 0;
      Depth : Integer := 0;
      InStr : Boolean := False;
      ElS   : Natural := 0;
   begin
      while Pos <= Arr'Last and then Arr (Pos) /= '[' loop Pos := Pos + 1; end loop;
      if Pos > Arr'Last then return ""; end if;
      Pos := Pos + 1;
      while Pos <= Arr'Last loop
         if InStr then
            if Arr (Pos) = '"' and then (Pos = Arr'First or else Arr (Pos - 1) /= '\') then
               InStr := False;
            end if;
         else
            case Arr (Pos) is
               when '"'       => InStr := True;
               when '{' | '[' => if Depth = 0 then ElS := Pos; end if; Depth := Depth + 1;
               when '}' | ']' =>
                  Depth := Depth - 1;
                  if Depth = 0 and ElS > 0 then
                     if Cnt = N then return Arr (ElS .. Pos); end if;
                     Cnt := Cnt + 1; ElS := 0;
                  end if;
               when others => null;
            end case;
         end if;
         Pos := Pos + 1;
      end loop;
      return "";
   end Get_Element;

   function Count_Elements (Arr : String) return Natural is
      Pos   : Natural := Arr'First;
      Cnt   : Natural := 0;
      Depth : Integer := 0;
      InStr : Boolean := False;
   begin
      while Pos <= Arr'Last and then Arr (Pos) /= '[' loop Pos := Pos + 1; end loop;
      if Pos > Arr'Last then return 0; end if;
      Pos := Pos + 1;
      while Pos <= Arr'Last loop
         if InStr then
            if Arr (Pos) = '"' and then (Pos = Arr'First or else Arr (Pos - 1) /= '\') then
               InStr := False;
            end if;
         else
            case Arr (Pos) is
               when '"'       => InStr := True;
               when '{' | '[' => if Depth = 0 then Cnt := Cnt + 1; end if; Depth := Depth + 1;
               when '}' | ']' => Depth := Depth - 1;
               when others    => null;
            end case;
         end if;
         Pos := Pos + 1;
      end loop;
      return Cnt;
   end Count_Elements;

   function Norm_Type (T : String) return String is
   begin
      if    T = "int"  or T = "integer" or T = "Int" or T = "Integer" then return "i32";
      elsif T = "uint" or T = "unsigned int"                           then return "u32";
      elsif T = "long"                                                 then return "i64";
      elsif T = "ulong"                                                then return "u64";
      elsif T = "short"                                                then return "i16";
      elsif T = "byte"                                                 then return "i8";
      elsif T = "ubyte" or T = "char"                                  then return "u8";
      elsif T = "float"                                                then return "f32";
      elsif T = "double"                                               then return "f64";
      elsif T = "bool" or T = "boolean" or T = "Bool"                  then return "bool";
      elsif T = "string" or T = "String" or T = "str"                  then return "str";
      elsif T = "void"                                                 then return "void";
      else return T;
      end if;
   end Norm_Type;

   function Conv_Function (Func_Spec : String) return String is
      Name   : constant String := Get_String (Func_Spec, "name");
      Sig    : constant String := Get_Block (Func_Spec, "signature");
      Ret    : constant String := Norm_Type (Get_String (Sig, "return_type"));
      Args   : constant String := Get_Block (Sig, "args");
      N_Args : constant Natural := Count_Elements (Args);
      Result : Unbounded_String;
   begin
      if Name'Length = 0 then return ""; end if;
      Append (Result, "{""name"":""" & Name & """");
      Append (Result, ",""args"":[");
      for I in 0 .. N_Args - 1 loop
         declare
            Arg    : constant String := Get_Element (Args, I);
            A_Name : constant String := Get_String (Arg, "name");
            A_Type : constant String := Norm_Type (Get_String (Arg, "type"));
         begin
            if I > 0 then Append (Result, ","); end if;
            Append (Result, "{""name"":""" & A_Name & """,""type"":""" & A_Type & """}");
         end;
      end loop;
      Append (Result, "],""return_type"":""" & Ret & """,""steps"":[],""is_public"":true}");
      return To_String (Result);
   end Conv_Function;

begin
   for I in 1 .. Argument_Count loop
      declare
         Arg : constant String := Argument (I);
      begin
         if    Arg = "--help" or Arg = "-h"    then Show_Help    := True;
         elsif Arg = "--version" or Arg = "-v" then Show_Version := True;
         elsif Arg = "--describe"              then Show_Describe := True;
         end if;
      end;
   end loop;

   if Show_Help then
      Put_Line ("func_to_ir - Convert spec functions array to IR functions array");
      Put_Line ("Version: " & Version);
      New_Line;
      Put_Line ("Usage: spec_extract_funcs < spec.json | func_to_ir");
      Put_Line ("  Reads JSON array of function spec objects, outputs IR functions array");
      Set_Exit_Status (Exit_Success); return;
   end if;
   if Show_Version then
      Put_Line ("func_to_ir " & Version);
      Set_Exit_Status (Exit_Success); return;
   end if;
   if Show_Describe then
      Put_Line (Describe_Output);
      Set_Exit_Status (Exit_Success); return;
   end if;

   declare
      Input   : constant String  := Read_Stdin;
      N_Funcs : constant Natural := Count_Elements (Input);
      Result  : Unbounded_String;
   begin
      if Input'Length = 0 then
         Put_Line (Standard_Error, "ERROR: No input on stdin");
         Set_Exit_Status (Exit_Error); return;
      end if;
      Append (Result, "[");
      for I in 0 .. N_Funcs - 1 loop
         declare
            Func    : constant String := Get_Element (Input, I);
            IR_Func : constant String := Conv_Function (Func);
         begin
            if I > 0 then Append (Result, ","); end if;
            if IR_Func'Length > 0 then
               Append (Result, IR_Func);
            end if;
         end;
      end loop;
      Append (Result, "]");
      Put_Line (To_String (Result));
      Set_Exit_Status (Exit_Success);
   end;

exception
   when others =>
      Put_Line (Standard_Error, "ERROR: Processing failed");
      Set_Exit_Status (Exit_Error);
end Func_To_IR;

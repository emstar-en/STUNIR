--  code_gen_func_sig - Generate function signature from IR function JSON
--  Input:  IR function JSON object (stdin)
--  Output: Function signature in target language (stdout)

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;
with GNAT.Strings;
with GNAT.Command_Line;

procedure Code_Gen_Func_Sig is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;

   Exit_Success : constant := 0;
   Exit_Error   : constant := 1;

   Target_Lang   : aliased GNAT.Strings.String_Access := new String'("");
   Show_Version  : aliased Boolean := False;
   Show_Help     : aliased Boolean := False;
   Show_Describe : aliased Boolean := False;

   Version : constant String := "0.1.0-alpha";

   Describe_Output : constant String :=
     "{\"tool\":\"code_gen_func_sig\",\"version\":\"0.1.0-alpha\"," &
     "\"description\":\"Generate function signature from IR function JSON\"," &
     "\"inputs\":[{\"type\":\"json\",\"source\":\"stdin\",\"required\":true}]," &
     "\"outputs\":[{\"type\":\"code\",\"source\":\"stdout\"}]," &
     "\"options\":[\"--target LANG\"]," &
     "\"targets\":[\"c\",\"cpp\",\"rust\",\"python\",\"js\",\"go\"]}";

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
      Pat  : constant String := "\"" & Key & "\"";
      K    : constant Natural := Find (JSON, Pat);
      P, E : Natural;
   begin
      if K = 0 then return ""; end if;
      P := K + Pat'Length;
      while P <= JSON'Last and then
            (JSON (P) = ':' or JSON (P) = ' ' or
             JSON (P) = ASCII.HT or JSON (P) = ASCII.LF or
             JSON (P) = ASCII.CR) loop P := P + 1; end loop;
      if P > JSON'Last or else JSON (P) /= '"' then return ""; end if;
      P := P + 1; E := P;
      while E <= JSON'Last and then JSON (E) /= '"' loop E := E + 1; end loop;
      if E > JSON'Last then return JSON (P .. JSON'Last); end if;
      return JSON (P .. E - 1);
   end Get_String;

   function Get_Block (JSON : String; Key : String) return String is
      Pat   : constant String := "\"" & Key & "\"";
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
             JSON (P) = ASCII.CR) loop P := P + 1; end loop;
      if P > JSON'Last then return ""; end if;
      if JSON (P) /= '[' and then JSON (P) /= '{' then return ""; end if;
      E := P;
      while E <= JSON'Last loop
         if InStr then
            if JSON (E) = '"' and then (E = JSON'First or else JSON (E - 1) /= '\\') then
               InStr := False; end if;
         else
            if    JSON (E) = '"'                   then InStr := True;
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
            if Arr (Pos) = '"' and then (Pos = Arr'First or else Arr (Pos - 1) /= '\\') then
               InStr := False; end if;
         else
            case Arr (Pos) is
               when '"'       => InStr := True;
               when '{' | '[' => if Depth = 0 then ElS := Pos; end if; Depth := Depth + 1;
               when '}' | ']' =>
                  Depth := Depth - 1;
                  if Depth = 0 and ElS > 0 then
                     if Cnt = N then return Arr (ElS .. Pos); end if;
                     Cnt := Cnt + 1; ElS := 0; end if;
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
            if Arr (Pos) = '"' and then (Pos = Arr'First or else Arr (Pos - 1) /= '\\') then
               InStr := False; end if;
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

   function Map_Type (IR_Type : String; Lang : String) return String is
   begin
      if Lang = "c" then
         if    IR_Type = "i32"  then return "int32_t";
         elsif IR_Type = "i64"  then return "int64_t";
         elsif IR_Type = "i16"  then return "int16_t";
         elsif IR_Type = "i8"   then return "int8_t";
         elsif IR_Type = "u32"  then return "uint32_t";
         elsif IR_Type = "u64"  then return "uint64_t";
         elsif IR_Type = "u16"  then return "uint16_t";
         elsif IR_Type = "u8"   then return "uint8_t";
         elsif IR_Type = "f32"  then return "float";
         elsif IR_Type = "f64"  then return "double";
         elsif IR_Type = "bool" then return "bool";
         elsif IR_Type = "str"  then return "const char*";
         elsif IR_Type = "void" then return "void";
         else return IR_Type; end if;
      elsif Lang = "cpp" then
         if    IR_Type = "i32"  then return "int32_t";
         elsif IR_Type = "i64"  then return "int64_t";
         elsif IR_Type = "i16"  then return "int16_t";
         elsif IR_Type = "i8"   then return "int8_t";
         elsif IR_Type = "u32"  then return "uint32_t";
         elsif IR_Type = "u64"  then return "uint64_t";
         elsif IR_Type = "u16"  then return "uint16_t";
         elsif IR_Type = "u8"   then return "uint8_t";
         elsif IR_Type = "f32"  then return "float";
         elsif IR_Type = "f64"  then return "double";
         elsif IR_Type = "bool" then return "bool";
         elsif IR_Type = "str"  then return "std::string";
         elsif IR_Type = "void" then return "void";
         else return IR_Type; end if;
      elsif Lang = "rust" then
         if    IR_Type = "str"  then return "String";
         elsif IR_Type = "void" then return "()";
         else return IR_Type; end if;
      elsif Lang = "python" then
         if    IR_Type = "i32" or IR_Type = "i64" or IR_Type = "i16" or IR_Type = "i8"
            or IR_Type = "u32" or IR_Type = "u64" or IR_Type = "u16" or IR_Type = "u8"
                                                                        then return "int";
         elsif IR_Type = "f32" or IR_Type = "f64"                      then return "float";
         elsif IR_Type = "bool"                                         then return "bool";
         elsif IR_Type = "str"                                          then return "str";
         elsif IR_Type = "void"                                         then return "None";
         else return IR_Type; end if;
      elsif Lang = "js" or Lang = "javascript" then
         if    IR_Type = "u64" or IR_Type = "i64"                      then return "bigint";
         elsif IR_Type = "bool"                                         then return "boolean";
         elsif IR_Type = "str"                                          then return "string";
         elsif IR_Type = "void"                                         then return "void";
         else return "number"; end if;
      elsif Lang = "go" then
         if    IR_Type = "i32"  then return "int32";
         elsif IR_Type = "i64"  then return "int64";
         elsif IR_Type = "i16"  then return "int16";
         elsif IR_Type = "i8"   then return "int8";
         elsif IR_Type = "u32"  then return "uint32";
         elsif IR_Type = "u64"  then return "uint64";
         elsif IR_Type = "u16"  then return "uint16";
         elsif IR_Type = "u8"   then return "uint8";
         elsif IR_Type = "f32"  then return "float32";
         elsif IR_Type = "f64"  then return "float64";
         elsif IR_Type = "bool" then return "bool";
         elsif IR_Type = "str"  then return "string";
         elsif IR_Type = "void" then return "";
         else return IR_Type; end if;
      else
         return IR_Type;
      end if;
   end Map_Type;

   function Build_Args_C (Args : String; Lang : String; N : Natural) return String is
      Result : Unbounded_String;
   begin
      for I in 0 .. N - 1 loop
         declare
            Arg    : constant String := Get_Element (Args, I);
            A_Name : constant String := Get_String (Arg, "name");
            A_Type : constant String := Map_Type (Get_String (Arg, "type"), Lang);
         begin
            if I > 0 then Append (Result, ", "); end if;
            if Lang = "rust" then
               Append (Result, A_Name & ": " & A_Type);
            elsif Lang = "python" then
               Append (Result, A_Name & ": " & A_Type);
            elsif Lang = "go" then
               Append (Result, A_Name & " " & A_Type);
            elsif Lang = "js" or Lang = "javascript" then
               Append (Result, A_Name);
            else
               Append (Result, A_Type & " " & A_Name);
            end if;
         end;
      end loop;
      return To_String (Result);
   end Build_Args_C;

   function Gen_Sig (JSON : String; Lang : String) return String is
      Name   : constant String  := Get_String (JSON, "name");
      Args   : constant String  := Get_Block (JSON, "args");
      Ret    : constant String  := Get_String (JSON, "return_type");
      N_Args : constant Natural := Count_Elements (Args);
      Ret_L  : constant String  := Map_Type (Ret, Lang);
      Args_L : constant String  := Build_Args_C (Args, Lang, N_Args);
      Result : Unbounded_String;
   begin
      if Name'Length = 0 then return ""; end if;
      if Lang = "c" or Lang = "cpp" then
         Append (Result, Ret_L & " " & Name & "(" & Args_L & ")");
      elsif Lang = "rust" then
         Append (Result, "pub fn " & Name & "(" & Args_L & ")");
         if Ret /= "void" then
            Append (Result, " -> " & Ret_L);
         end if;
         Append (Result, " {");
      elsif Lang = "python" then
         Append (Result, "def " & Name & "(" & Args_L & ")");
         if Ret /= "void" then
            Append (Result, " -> " & Ret_L);
         end if;
         Append (Result, ":");
      elsif Lang = "js" or Lang = "javascript" then
         Append (Result, "function " & Name & "(" & Args_L & ") {");
      elsif Lang = "go" then
         Append (Result, "func " & Name & "(" & Args_L & ")");
         if Ret_L'Length > 0 then
            Append (Result, " " & Ret_L);
         end if;
         Append (Result, " {");
      else
         return "";
      end if;
      return To_String (Result);
   end Gen_Sig;

   Config : GNAT.Command_Line.Command_Line_Configuration;

begin
   GNAT.Command_Line.Define_Switch (Config, Show_Help'Access,    "-h", "--help");
   GNAT.Command_Line.Define_Switch (Config, Show_Version'Access, "-v", "--version");
   GNAT.Command_Line.Define_Switch (Config, Show_Describe'Access, "", "--describe");
   GNAT.Command_Line.Define_Switch (Config, Target_Lang'Access,  "-t:", "--target=");

   begin
      GNAT.Command_Line.Getopt (Config);
   exception
      when others =>
         Put_Line (Standard_Error, "ERROR: Invalid arguments");
         Set_Exit_Status (Exit_Error); return;
   end;

   if Show_Help then
      Put_Line ("code_gen_func_sig - Generate function signature from IR function JSON");
      Put_Line ("Version: " & Version);
      New_Line;
      Put_Line ("Usage: echo IR_FUNC_JSON | code_gen_func_sig --target LANG");
      Put_Line ("  --target LANG   Target: c cpp rust python js go");
      Set_Exit_Status (Exit_Success); return;
   end if;
   if Show_Version then
      Put_Line ("code_gen_func_sig " & Version);
      Set_Exit_Status (Exit_Success); return;
   end if;
   if Show_Describe then
      Put_Line (Describe_Output);
      Set_Exit_Status (Exit_Success); return;
   end if;

   if Target_Lang.all = "" then
      Put_Line (Standard_Error, "ERROR: --target LANG required");
      Set_Exit_Status (Exit_Error); return;
   end if;

   declare
      Input : constant String := Read_Stdin;
      Sig   : constant String := Gen_Sig (Input, Target_Lang.all);
   begin
      if Input'Length = 0 then
         Put_Line (Standard_Error, "ERROR: No input on stdin");
         Set_Exit_Status (Exit_Error); return;
      end if;
      if Sig'Length = 0 then
         Put_Line (Standard_Error, "ERROR: Unsupported target or invalid IR: " & Target_Lang.all);
         Set_Exit_Status (Exit_Error); return;
      end if;
      Put_Line (Sig);
      Set_Exit_Status (Exit_Success);
   end;

exception
   when others =>
      Put_Line (Standard_Error, "ERROR: Processing failed");
      Set_Exit_Status (Exit_Error);
end Code_Gen_Func_Sig;

--  code_gen_func_body - Generate function body stub from IR function JSON
--  Input:  IR function JSON object (stdin)
--  Output: Function body stub in target language (stdout)

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;
with GNAT.Strings;
with GNAT.Command_Line;

procedure Code_Gen_Func_Body is
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
     "{""tool"":""code_gen_func_body"",""version"":""0.1.0-alpha""," &
     """description"":""Generate function body stub from IR function JSON""," &
     """inputs"":[{""type"":""json"",""source"":""stdin"",""required"":true}]," &
     """outputs"":[{""type"":""code"",""source"":""stdout""}]," &
     """options"":[""--target LANG""]," &
     """targets"":[""c"",""cpp"",""rust"",""python"",""js"",""go""]}";

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
             JSON (P) = ASCII.CR) loop P := P + 1; end loop;
      if P > JSON'Last or else JSON (P) /= '"' then return ""; end if;
      P := P + 1; E := P;
      while E <= JSON'Last and then JSON (E) /= '"' loop E := E + 1; end loop;
      if E > JSON'Last then return JSON (P .. JSON'Last); end if;
      return JSON (P .. E - 1);
   end Get_String;

   function Default_Value (IR_Type : String; Lang : String) return String is
   begin
      if Lang = "c" or Lang = "cpp" then
         if    IR_Type = "bool"              then return "false";
         elsif IR_Type = "str"               then return "NULL";
         elsif IR_Type = "void"              then return "";
         elsif IR_Type = "f32" or IR_Type = "f64" then return "0.0";
         else return "0"; end if;
      elsif Lang = "rust" then
         if    IR_Type = "bool"              then return "false";
         elsif IR_Type = "str"               then return "String::new()";
         elsif IR_Type = "void"              then return "";
         elsif IR_Type = "f32"               then return "0.0_f32";
         elsif IR_Type = "f64"               then return "0.0_f64";
         else return "0"; end if;
      elsif Lang = "python" then
         if    IR_Type = "bool"              then return "False";
         elsif IR_Type = "str"               then return """";
         elsif IR_Type = "void"              then return "";
         elsif IR_Type = "f32" or IR_Type = "f64" then return "0.0";
         else return "0"; end if;
      elsif Lang = "js" or Lang = "javascript" then
         if    IR_Type = "bool"              then return "false";
         elsif IR_Type = "str"               then return """""";
         elsif IR_Type = "void"              then return "";
         else return "0"; end if;
      elsif Lang = "go" then
         if    IR_Type = "bool"              then return "false";
         elsif IR_Type = "str"               then return """""";
         elsif IR_Type = "void"              then return "";
         elsif IR_Type = "f32" or IR_Type = "f64" then return "0.0";
         else return "0"; end if;
      else
         return "nil";
      end if;
   end Default_Value;

   function Gen_Body (JSON : String; Lang : String) return String is
      Name    : constant String := Get_String (JSON, "name");
      Ret     : constant String := Get_String (JSON, "return_type");
      Def_Val : constant String := Default_Value (Ret, Lang);
      Result  : Unbounded_String;
   begin
      if Name'Length = 0 then return ""; end if;
      if Lang = "c" or Lang = "cpp" then
         Append (Result, "{" & ASCII.LF);
         Append (Result, "    /* TODO: implement " & Name & " */" & ASCII.LF);
         if Ret /= "void" then
            Append (Result, "    return " & Def_Val & ";" & ASCII.LF);
         end if;
         Append (Result, "}");
      elsif Lang = "rust" then
         Append (Result, "    // TODO: implement " & Name & ASCII.LF);
         if Ret /= "void" then
            Append (Result, "    " & Def_Val);
         end if;
         Append (Result, ASCII.LF & "}");
      elsif Lang = "python" then
         Append (Result, "    # TODO: implement " & Name & ASCII.LF);
         if Ret /= "void" and Def_Val'Length > 0 then
            Append (Result, "    return " & Def_Val);
         else
            Append (Result, "    pass");
         end if;
      elsif Lang = "js" or Lang = "javascript" then
         Append (Result, "    // TODO: implement " & Name & ASCII.LF);
         if Ret /= "void" then
            Append (Result, "    return " & Def_Val & ";" & ASCII.LF);
         end if;
         Append (Result, "}");
      elsif Lang = "go" then
         Append (Result, "    // TODO: implement " & Name & ASCII.LF);
         if Ret /= "void" and Def_Val'Length > 0 then
            Append (Result, "    return " & Def_Val & ASCII.LF);
         end if;
         Append (Result, "}");
      else
         return "";
      end if;
      return To_String (Result);
   end Gen_Body;

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
      Put_Line ("code_gen_func_body - Generate function body stub from IR function JSON");
      Put_Line ("Version: " & Version);
      New_Line;
      Put_Line ("Usage: echo IR_FUNC_JSON | code_gen_func_body --target LANG");
      Put_Line ("  --target LANG   Target: c cpp rust python js go");
      Set_Exit_Status (Exit_Success); return;
   end if;
   if Show_Version then
      Put_Line ("code_gen_func_body " & Version);
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
      Body_Str : constant String := Gen_Body (Input, Target_Lang.all);
   begin
      if Input'Length = 0 then
         Put_Line (Standard_Error, "ERROR: No input on stdin");
         Set_Exit_Status (Exit_Error); return;
      end if;
      if Body_Str'Length = 0 then
         Put_Line (Standard_Error, "ERROR: Unsupported target or invalid IR: " & Target_Lang.all);
         Set_Exit_Status (Exit_Error); return;
      end if;
      Put_Line (Body_Str);
      Set_Exit_Status (Exit_Success);
   end;

exception
   when others =>
      Put_Line (Standard_Error, "ERROR: Processing failed");
      Set_Exit_Status (Exit_Error);
end Code_Gen_Func_Body;

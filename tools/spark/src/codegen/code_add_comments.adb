--  code_add_comments - Add comments to generated code
--  Phase 1 Powertool for STUNIR

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;
with Ada.Exceptions;
with GNAT.Strings;
with GNAT.Command_Line;

procedure Code_Add_Comments is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;

   --  Exit codes
   Exit_Success : constant := 0;
   Exit_Error   : constant := 1;

   --  Configuration
   Comment_Style : aliased GNAT.Strings.String_Access := new String'("");
   Show_Version  : aliased Boolean := False;
   Show_Help     : aliased Boolean := False;
   Show_Describe : aliased Boolean := False;

   Version : constant String := "0.1.0-alpha";

   Describe_Output : constant String :=
     "{" & ASCII.LF &
     "  ""tool"": ""code_add_comments""," & ASCII.LF &
     "  ""version"": ""0.1.0-alpha""," & ASCII.LF &
     "  ""description"": ""Add comments to generated code""," & ASCII.LF &
     "  ""inputs"": [{" & ASCII.LF &
     "    ""name"": ""code""," & ASCII.LF &
     "    ""type"": ""string""," & ASCII.LF &
     "    ""source"": [""stdin""]," & ASCII.LF &
     "    ""required"": true" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""outputs"": [{" & ASCII.LF &
     "    ""name"": ""commented_code""," & ASCII.LF &
     "    ""type"": ""string""," & ASCII.LF &
     "    ""source"": ""stdout""" & ASCII.LF &
     "  }]" & ASCII.LF &
     "}";

   procedure Print_Usage is
   begin
      Put_Line ("code_add_comments - Add comments to generated code");
      Put_Line ("Version: " & Version);
      Put_Line ("");
      Put_Line ("Usage: code_add_comments [OPTIONS] --style STYLE < code.txt");
      Put_Line ("");
      Put_Line ("Options:");
      Put_Line ("  --help, -h        Show this help message");
      Put_Line ("  --version, -v     Show version information");
      Put_Line ("  --describe        Show tool description (JSON)");
      Put_Line ("  --style STYLE     Comment style (cpp, rust, python)");
   end Print_Usage;

   procedure Print_Error (Msg : String) is
   begin
      Put_Line (Standard_Error, "ERROR: " & Msg);
   end Print_Error;

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

   function Add_Comments (Code : String; Style : String) return String is
   begin
      if Style = "cpp" then
         return "// Auto-generated code" & ASCII.LF & Code;
      elsif Style = "rust" then
         return "// Auto-generated code" & ASCII.LF & Code;
      elsif Style = "python" then
         return "# Auto-generated code" & ASCII.LF & Code;
      else
         return Code;
      end if;
   end Add_Comments;

   Config : GNAT.Command_Line.Command_Line_Configuration;

begin
   GNAT.Command_Line.Define_Switch (Config, Show_Help'Access, "-h", "--help");
   GNAT.Command_Line.Define_Switch (Config, Show_Version'Access, "-v", "--version");
   GNAT.Command_Line.Define_Switch (Config, Show_Describe'Access, "", "--describe");
   GNAT.Command_Line.Define_Switch (Config, Comment_Style'Access, "-s:", "--style=");

   begin
      GNAT.Command_Line.Getopt (Config);
   exception
      when others =>
         Print_Error ("Invalid arguments");
         Set_Exit_Status (Exit_Error);
         return;
   end;

   if Show_Help then
      Print_Usage;
      Set_Exit_Status (Exit_Success);
      return;
   end if;

   if Show_Version then
      Put_Line (Version);
      Set_Exit_Status (Exit_Success);
      return;
   end if;

   if Show_Describe then
      Put_Line (Describe_Output);
      Set_Exit_Status (Exit_Success);
      return;
   end if;

   if Comment_Style.all = "" then
      Comment_Style.all := "cpp";
   end if;

   declare
      Code : constant String := Read_Stdin;
   begin
      if Code'Length = 0 then
         Print_Error ("Empty input");
         Set_Exit_Status (Exit_Error);
         return;
      end if;

      Put (Add_Comments (Code, Comment_Style.all));
      Set_Exit_Status (Exit_Success);
   end;

exception
   when others =>
      Print_Error ("Unexpected error");
      Set_Exit_Status (Exit_Error);
end Code_Add_Comments;

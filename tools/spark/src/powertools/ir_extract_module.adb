--  ir_extract_module - Extract module information from IR JSON
--  Phase 1 Powertool for STUNIR

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;
with Ada.Strings.Fixed;
with Ada.Exceptions;

procedure IR_Extract_Module is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;
   use Ada.Strings.Fixed;

   --  Exit codes
   Exit_Success : constant := 0;
   Exit_Error   : constant := 1;

   --  Configuration
   Show_Version  : Boolean := False;
   Show_Help     : Boolean := False;
   Show_Describe : Boolean := False;

   Version : constant String := "0.1.0-alpha";

   Describe_Output : constant String :=
     "{" & ASCII.LF &
     "  ""tool"": ""ir_extract_module""," & ASCII.LF &
     "  ""version"": ""0.1.0-alpha""," & ASCII.LF &
     "  ""description"": ""Extract module information from IR JSON""," & ASCII.LF &
     "  ""inputs"": [{" & ASCII.LF &
     "    ""name"": ""ir_json""," & ASCII.LF &
     "    ""type"": ""string""," & ASCII.LF &
     "    ""source"": [""stdin""]," & ASCII.LF &
     "    ""required"": true" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""outputs"": [{" & ASCII.LF &
     "    ""name"": ""module_info""," & ASCII.LF &
     "    ""type"": ""object""," & ASCII.LF &
     "    ""source"": ""stdout""" & ASCII.LF &
     "  }]" & ASCII.LF &
     "}";

   procedure Print_Usage is
   begin
      Put_Line ("ir_extract_module - Extract module info from IR");
      Put_Line ("Version: " & Version);
      Put_Line ("");
      Put_Line ("Usage: ir_extract_module [OPTIONS] < ir.json");
      Put_Line ("");
      Put_Line ("Options:");
      Put_Line ("  --help, -h        Show this help message");
      Put_Line ("  --version, -v     Show version information");
      Put_Line ("  --describe        Show tool description (JSON)");
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
      end loop;
      return To_String (Result);
   end Read_Stdin;

   function Extract_Module_Name (Content : String) return String is
      Pattern : constant String := """module_name""";
      Idx     : constant Natural := Index (Content, Pattern);
      Start_Idx : Natural;
      End_Idx   : Natural;
   begin
      if Idx = 0 then
         return "";
      end if;
      Start_Idx := Index (Content (Idx + Pattern'Length .. Content'Last), """");
      if Start_Idx = 0 then
         return "";
      end if;
      Start_Idx := Start_Idx + Idx + Pattern'Length;
      End_Idx := Index (Content (Start_Idx .. Content'Last), """");
      if End_Idx = 0 then
         return "";
      end if;
      return Content (Start_Idx .. Start_Idx + End_Idx - 2);
   end Extract_Module_Name;

   function Extract_IR_Version (Content : String) return String is
      Pattern : constant String := """ir_version""";
      Idx     : constant Natural := Index (Content, Pattern);
      Start_Idx : Natural;
      End_Idx   : Natural;
   begin
      if Idx = 0 then
         return "";
      end if;
      Start_Idx := Index (Content (Idx + Pattern'Length .. Content'Last), """");
      if Start_Idx = 0 then
         return "";
      end if;
      Start_Idx := Start_Idx + Idx + Pattern'Length;
      End_Idx := Index (Content (Start_Idx .. Content'Last), """");
      if End_Idx = 0 then
         return "";
      end if;
      return Content (Start_Idx .. Start_Idx + End_Idx - 2);
   end Extract_IR_Version;

begin
   --  Parse arguments
   for I in 1 .. Argument_Count loop
      declare
         Arg : constant String := Argument (I);
      begin
         if Arg = "--help" or Arg = "-h" then
            Show_Help := True;
         elsif Arg = "--version" or Arg = "-v" then
            Show_Version := True;
         elsif Arg = "--describe" then
            Show_Describe := True;
         end if;
      end;
   end loop;

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

   --  Read and extract module info
   declare
      Content      : constant String := Read_Stdin;
      Module_Name  : constant String := Extract_Module_Name (Content);
      IR_Version   : constant String := Extract_IR_Version (Content);
   begin
      if Content'Length = 0 then
         Print_Error ("Empty input");
         Set_Exit_Status (Exit_Error);
         return;
      end if;

      Put_Line ("{");
      Put_Line ("  ""module_name"": """ & Module_Name & """,");
      Put_Line ("  ""ir_version"": """ & IR_Version & """");
      Put_Line ("}");
      Set_Exit_Status (Exit_Success);
   end;

exception
   when others =>
      Print_Error ("Unexpected error");
      Set_Exit_Status (Exit_Error);
end IR_Extract_Module;
--  code_gen_preamble - Generate code preamble (headers, includes)
--  Phase 1 Powertool for STUNIR

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;
with Ada.Exceptions;
with GNAT.Strings;
with GNAT.Command_Line;

procedure Code_Gen_Preamble is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;

   --  Exit codes
   Exit_Success : constant := 0;
   Exit_Error   : constant := 1;

   --  Configuration
   Target_Lang   : aliased GNAT.Strings.String_Access := new String'("");
   Show_Version  : Boolean := False;
   Show_Help     : Boolean := False;
   Show_Describe : Boolean := False;

   Version : constant String := "0.1.0-alpha";

   Describe_Output : constant String :=
     "{" & ASCII.LF &
     "  ""tool"": ""code_gen_preamble""," & ASCII.LF &
     "  ""version"": ""0.1.0-alpha""," & ASCII.LF &
     "  ""description"": ""Generate code preamble (headers, includes)""," & ASCII.LF &
     "  ""inputs"": [{" & ASCII.LF &
     "    ""name"": ""target""," & ASCII.LF &
     "    ""type"": ""string""," & ASCII.LF &
     "    ""source"": [""--target""]," & ASCII.LF &
     "    ""required"": true" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""outputs"": [{" & ASCII.LF &
     "    ""name"": ""preamble""," & ASCII.LF &
     "    ""type"": ""string""," & ASCII.LF &
     "    ""source"": ""stdout""" & ASCII.LF &
     "  }]" & ASCII.LF &
     "}";

   procedure Print_Usage is
   begin
      Put_Line ("code_gen_preamble - Generate code preamble");
      Put_Line ("Version: " & Version);
      Put_Line ("");
      Put_Line ("Usage: code_gen_preamble [OPTIONS] --target LANG");
      Put_Line ("");
      Put_Line ("Options:");
      Put_Line ("  --help, -h        Show this help message");
      Put_Line ("  --version, -v     Show version information");
      Put_Line ("  --describe        Show tool description (JSON)");
      Put_Line ("  --target LANG     Target language (cpp, rust, python)");
   end Print_Usage;

   procedure Print_Error (Msg : String) is
   begin
      Put_Line (Standard_Error, "ERROR: " & Msg);
   end Print_Error;

   function Gen_Preamble (Lang : String) return String is
   begin
      if Lang = "cpp" then
         return "#include <iostream>" & ASCII.LF &
                "#include <string>" & ASCII.LF &
                "#include <vector>";
      elsif Lang = "rust" then
         return "use std::io;";
      elsif Lang = "python" then
         return "import sys" & ASCII.LF &
                "import json";
      else
         return "";
      end if;
   end Gen_Preamble;

   Config : GNAT.Command_Line.Command_Line_Configuration;

begin
   GNAT.Command_Line.Define_Switch (Config, Show_Help'Access, "-h", "--help");
   GNAT.Command_Line.Define_Switch (Config, Show_Version'Access, "-v", "--version");
   GNAT.Command_Line.Define_Switch (Config, Show_Describe'Access, "", "--describe");
   GNAT.Command_Line.Define_Switch (Config, Target_Lang'Access, "-t:", "--target=");

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

   if Target_Lang.all = "" then
      Print_Error ("Target language required (--target)");
      Set_Exit_Status (Exit_Error);
      return;
   end if;

   declare
      Preamble : constant String := Gen_Preamble (Target_Lang.all);
   begin
      if Preamble'Length = 0 then
         Print_Error ("Unsupported target language: " & Target_Lang.all);
         Set_Exit_Status (Exit_Error);
         return;
      end if;

      Put_Line (Preamble);
      Set_Exit_Status (Exit_Success);
   end;

exception
   when others =>
      Print_Error ("Unexpected error");
      Set_Exit_Status (Exit_Error);
end Code_Gen_Preamble;
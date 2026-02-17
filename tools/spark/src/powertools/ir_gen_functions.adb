--  ir_gen_functions - Generate IR function representations from spec
--  Uses json_extract, json_formatter, json_merge_objects utilities
--  Phase 3 Powertool for STUNIR

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;
with GNAT.Command_Line;
with GNAT.OS_Lib;
with GNAT.Strings;
with Command_Utils;

procedure IR_Gen_Functions is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;

   --  Exit codes
   Exit_Success         : constant := 0;
   Exit_Generation_Error : constant := 1;

   --  Configuration
   Show_Version  : aliased Boolean := False;
   Show_Help     : aliased Boolean := False;
   Show_Describe : aliased Boolean := False;
   Module_Name   : aliased GNAT.Strings.String_Access := new String'("");

   Version : constant String := "0.1.0-alpha";

   Describe_Output : constant String :=
     "{" & ASCII.LF &
     "  ""tool"": ""ir_gen_functions""," & ASCII.LF &
     "  ""version"": ""0.1.0-alpha""," & ASCII.LF &
     "  ""description"": ""Generate IR function representations from spec""," & ASCII.LF &
     "  ""inputs"": [{" & ASCII.LF &
     "    ""name"": ""spec_json""," & ASCII.LF &
     "    ""type"": ""json""," & ASCII.LF &
     "    ""source"": [""stdin""]," & ASCII.LF &
     "    ""required"": true" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""outputs"": [{" & ASCII.LF &
     "    ""name"": ""ir_functions""," & ASCII.LF &
     "    ""type"": ""json""," & ASCII.LF &
     "    ""source"": ""stdout""" & ASCII.LF &
     "  }]" & ASCII.LF &
     "}";

   procedure Print_Usage is
   begin
      Put_Line ("ir_gen_functions - Generate IR function representations");
      Put_Line ("Version: " & Version);
      Put_Line ("");
      Put_Line ("Usage: ir_gen_functions [OPTIONS] < spec.json");
      Put_Line ("");
      Put_Line ("Options:");
      Put_Line ("  --help, -h        Show this help message");
      Put_Line ("  --version, -v     Show version information");
      Put_Line ("  --describe        Show tool description (JSON)");
      Put_Line ("  --module NAME     Override module name");
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

   function Run_Command (Cmd : String; Input : String := "") return String is
      Success : aliased Boolean;
      Result  : constant String :=
        Command_Utils.Get_Command_Output (Cmd, Input, Success'Access);
   begin
      return Result;
   end Run_Command;

   Config : GNAT.Command_Line.Command_Line_Configuration;

begin
   GNAT.Command_Line.Define_Switch (Config, Show_Help'Access, "-h", "--help");
   GNAT.Command_Line.Define_Switch (Config, Show_Version'Access, "-v", "--version");
   GNAT.Command_Line.Define_Switch (Config, Show_Describe'Access, "", "--describe");
   GNAT.Command_Line.Define_Switch (Config, Module_Name'Access, "", "--module=");

   begin
      GNAT.Command_Line.Getopt (Config);
   exception
      when others =>
         Print_Error ("Invalid arguments");
         Set_Exit_Status (Exit_Generation_Error);
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

   declare
      Spec_JSON : constant String := Read_Stdin;
   begin
      if Spec_JSON'Length = 0 then
         Print_Error ("Empty input");
         Set_Exit_Status (Exit_Generation_Error);
         return;
      end if;

      --  Extract functions array from spec
      declare
         Functions_JSON : constant String :=
           Run_Command ("json_extract --path module.functions", Spec_JSON);
      begin
         if Functions_JSON'Length = 0 then
            Print_Error ("No functions found in spec");
            Set_Exit_Status (Exit_Generation_Error);
            return;
         end if;

         --  Get module name (or use override)
         declare
            Mod_Name : Unbounded_String := To_Unbounded_String (Module_Name.all);
         begin
            if Mod_Name = Null_Unbounded_String then
               declare
                  Extracted_Name : constant String :=
                    Run_Command ("json_extract --raw --path module.name", Spec_JSON);
               begin
                  if Extracted_Name'Length > 0 then
                     Mod_Name := To_Unbounded_String (Extracted_Name);
                  else
                     Mod_Name := To_Unbounded_String ("unnamed");
                  end if;
               end;
            end if;

            --  Build IR structure
            declare
               IR_Base : constant String :=
                 "{" & ASCII.LF &
                 "  ""module_name"": """ & To_String (Mod_Name) & """," & ASCII.LF &
                 "  ""functions"": " & Functions_JSON & ASCII.LF &
                 "}";
               Formatted : constant String :=
                 Run_Command ("json_formatter", IR_Base);
            begin
               if Formatted'Length > 0 then
                  Put_Line (Formatted);
               else
                  Put_Line (IR_Base);
               end if;
            end;
         end;
      end;
   end;

   Set_Exit_Status (Exit_Success);

exception
   when others =>
      Print_Error ("Unexpected error");
      Set_Exit_Status (Exit_Generation_Error);
end IR_Gen_Functions;
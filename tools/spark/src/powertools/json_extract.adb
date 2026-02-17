--  json_extract - Extract values from JSON by path (Refactored)
--  Orchestrates json_path_parser, json_path_eval, json_value_format
--  Phase 1 Powertool for STUNIR - Refactored for Unix philosophy

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;
with GNAT.Command_Line;
with GNAT.OS_Lib;
with GNAT.Strings;

procedure JSON_Extract is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;

   --  Exit codes per powertools spec
   Exit_Success          : constant := 0;
   Exit_Path_Not_Found   : constant := 1;
   Exit_Invalid_JSON     : constant := 2;
   Exit_Invalid_Path     : constant := 3;

   --  Configuration
   Extract_Path  : aliased GNAT.Strings.String_Access := new String'("");
   Default_Value : aliased GNAT.Strings.String_Access := new String'("");
   Show_Version  : aliased Boolean := False;
   Show_Help     : aliased Boolean := False;
   Show_Describe : aliased Boolean := False;
   Raw_Output    : aliased Boolean := False;

   Version : constant String := "0.1.0-alpha";

   Describe_Output : constant String :=
     "{" & ASCII.LF &
     "  ""tool"": ""json_extract""," & ASCII.LF &
     "  ""version"": ""0.1.0-alpha""," & ASCII.LF &
     "  ""description"": ""Extract values from JSON by path""," & ASCII.LF &
     "  ""inputs"": [{" & ASCII.LF &
     "    ""name"": ""json_input""," & ASCII.LF &
     "    ""type"": ""json""," & ASCII.LF &
     "    ""source"": [""stdin""]," & ASCII.LF &
     "    ""required"": true" & ASCII.LF &
     "  }, {" & ASCII.LF &
     "    ""name"": ""path""," & ASCII.LF &
     "    ""type"": ""string""," & ASCII.LF &
     "    ""required"": true" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""outputs"": [{" & ASCII.LF &
     "    ""name"": ""extracted_value""," & ASCII.LF &
     "    ""type"": ""json""," & ASCII.LF &
     "    ""source"": ""stdout""" & ASCII.LF &
     "  }]" & ASCII.LF &
     "}";

   procedure Print_Usage is
   begin
      Put_Line ("json_extract - Extract values from JSON by path");
      Put_Line ("Version: " & Version);
      Put_Line ("");
      Put_Line ("Usage: json_extract [OPTIONS] --path PATH < input.json");
      Put_Line ("");
      Put_Line ("Options:");
      Put_Line ("  --help, -h        Show this help message");
      Put_Line ("  --version, -v     Show version information");
      Put_Line ("  --describe        Show tool description (JSON)");
      Put_Line ("  --path PATH       Path to extract (e.g., 'functions.0.name')");
      Put_Line ("  --default VALUE   Default value if path not found");
      Put_Line ("  --raw             Output raw strings without quotes");
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

   function Run_Command (Cmd : String; Input : String) return String is
      use GNAT.OS_Lib;
      Args : Argument_List_Access;
      Success : aliased Boolean;
      Output : Unbounded_String := Null_Unbounded_String;
   begin
      Args := Argument_String_To_List (Cmd);
      declare
         Result : constant String :=
           Get_Command_Output ("sh", Args.all, Input, Success'Access);
      begin
         Free (Args);
         return Result;
      end;
   exception
      when others =>
         if Args /= null then
            Free (Args);
         end if;
         return "";
   end Run_Command;

   Config : GNAT.Command_Line.Command_Line_Configuration;

begin
   GNAT.Command_Line.Define_Switch (Config, Show_Help'Access, "-h", "--help");
   GNAT.Command_Line.Define_Switch (Config, Show_Version'Access, "-v", "--version");
   GNAT.Command_Line.Define_Switch (Config, Show_Describe'Access, "", "--describe");
   GNAT.Command_Line.Define_Switch (Config, Raw_Output'Access, "", "--raw");
   GNAT.Command_Line.Define_Switch (Config, Extract_Path'Access, "", "--path=");
   GNAT.Command_Line.Define_Switch (Config, Default_Value'Access, "", "--default=");

   begin
      GNAT.Command_Line.Getopt (Config);
   exception
      when others =>
         Print_Error ("Invalid arguments");
         Set_Exit_Status (Exit_Invalid_Path);
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

   if Extract_Path.all = "" then
      Print_Error ("Path required (--path)");
      Set_Exit_Status (Exit_Invalid_Path);
      return;
   end if;

   declare
      JSON_Input : constant String := Read_Stdin;
   begin
      if JSON_Input'Length = 0 then
         Print_Error ("Empty input");
         Set_Exit_Status (Exit_Invalid_JSON);
         return;
      end if;

      --  Step 1: Parse the path
      declare
         Parsed_Path : constant String :=
           Run_Command ("json_path_parser --path " & Extract_Path.all, "");
      begin
         if Parsed_Path'Length = 0 then
            Print_Error ("Invalid path syntax");
            Set_Exit_Status (Exit_Invalid_Path);
            return;
         end if;

         --  Step 2: Evaluate the path on JSON
         declare
            Raw_Value : constant String :=
              Run_Command ("json_path_eval --path '" & Parsed_Path & "'", JSON_Input);
         begin
            if Raw_Value'Length = 0 then
               if Default_Value.all /= "" then
                  Put_Line (Default_Value.all);
                  Set_Exit_Status (Exit_Success);
               else
                  Print_Error ("Path not found: " & Extract_Path.all);
                  Set_Exit_Status (Exit_Path_Not_Found);
               end if;
               return;
            end if;

            --  Step 3: Format the value
            declare
               Format_Flag : constant String := (if Raw_Output then " --raw" else "");
               Formatted : constant String :=
                 Run_Command ("json_value_format" & Format_Flag, Raw_Value);
            begin
               Put_Line (Formatted);
               Set_Exit_Status (Exit_Success);
            end;
         end;
      end;
   end;

exception
   when others =>
      Print_Error ("Unexpected error");
      Set_Exit_Status (Exit_Invalid_JSON);
end JSON_Extract;
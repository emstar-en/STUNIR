--  type_resolver - Resolve type references with expansion and dependency ordering
--  Orchestrates type_lookup, type_expand, type_dependency
--  Phase 1 Powertool for STUNIR

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;
with GNAT.Command_Line;
with GNAT.OS_Lib;
with GNAT.Strings;

procedure Type_Resolver is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;

   --  Exit codes
   Exit_Success         : constant := 0;
   Exit_Resolution_Error : constant := 1;

   --  Configuration
   Type_Name     : aliased GNAT.Strings.String_Access := new String'("");
   Registry_File : aliased GNAT.Strings.String_Access := new String'("");
   Recursive     : aliased Boolean := False;
   Show_Version  : aliased Boolean := False;
   Show_Help     : aliased Boolean := False;
   Show_Describe : aliased Boolean := False;

   Version : constant String := "0.1.0-alpha";

   Describe_Output : constant String :=
     "{" & ASCII.LF &
     "  ""tool"": ""type_resolver""," & ASCII.LF &
     "  ""version"": ""0.1.0-alpha""," & ASCII.LF &
     "  ""description"": ""Resolve type references with expansion and dependency ordering""," & ASCII.LF &
     "  ""inputs"": [{" & ASCII.LF &
     "    ""name"": ""type_name""," & ASCII.LF &
     "    ""type"": ""string""," & ASCII.LF &
     "    ""source"": [""argument""]," & ASCII.LF &
     "    ""required"": true" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""outputs"": [{" & ASCII.LF &
     "    ""name"": ""resolved_type""," & ASCII.LF &
     "    ""type"": ""json""," & ASCII.LF &
     "    ""source"": ""stdout""" & ASCII.LF &
     "  }]" & ASCII.LF &
     "}";

   procedure Print_Usage is
   begin
      Put_Line ("type_resolver - Resolve type references");
      Put_Line ("Version: " & Version);
      Put_Line ("");
      Put_Line ("Usage: type_resolver [OPTIONS] <type_name>");
      Put_Line ("");
      Put_Line ("Options:");
      Put_Line ("  --help, -h          Show this help message");
      Put_Line ("  --version, -v       Show version information");
      Put_Line ("  --describe          Show tool description (JSON)");
      Put_Line ("  --registry FILE     Type registry file");
      Put_Line ("  --recursive         Recursively resolve dependencies");
   end Print_Usage;

   procedure Print_Error (Msg : String) is
   begin
      Put_Line (Standard_Error, "ERROR: " & Msg);
   end Print_Error;

   function Run_Command (Cmd : String; Input : String := "") return String is
      use GNAT.OS_Lib;
      Args    : Argument_List_Access;
      Success : aliased Boolean;
   begin
      Args := Argument_String_To_List (Cmd);
      declare
         Result : constant String :=
           Get_Command_Output ("sh", Args.all, Input, Success'Access);
      begin
         Free (Args);
         if Success then
            return Result;
         else
            return "";
         end if;
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
   GNAT.Command_Line.Define_Switch (Config, Recursive'Access, "-r", "--recursive");
   GNAT.Command_Line.Define_Switch (Config, Registry_File'Access, "", "--registry=");

   begin
      GNAT.Command_Line.Getopt (Config);
   exception
      when others =>
         Print_Error ("Invalid arguments");
         Set_Exit_Status (Exit_Resolution_Error);
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

   --  Get type name from remaining argument
   declare
      Arg : String := GNAT.Command_Line.Get_Argument;
   begin
      if Arg'Length > 0 then
         Type_Name.all := Arg;
      end if;
   end;

   if Type_Name.all = "" then
      Print_Error ("Type name required");
      Set_Exit_Status (Exit_Resolution_Error);
      return;
   end if;

   --  Step 1: Look up type definition
   declare
      Registry_Flag : constant String :=
        (if Registry_File.all = "" then "" else " --registry " & Registry_File.all);
      Lookup_Result : constant String :=
        Run_Command ("type_lookup" & Registry_Flag & " " & Type_Name.all);
   begin
      if Lookup_Result'Length = 0 then
         Print_Error ("Type not found: " & Type_Name.all);
         Set_Exit_Status (Exit_Resolution_Error);
         return;
      end if;

      --  Step 2: Expand type aliases
      declare
         Expand_Result : constant String := Run_Command ("type_expand", Lookup_Result);
      begin
         if Expand_Result'Length = 0 then
            Print_Error ("Failed to expand type");
            Set_Exit_Status (Exit_Resolution_Error);
            return;
         end if;

         --  Step 3: Resolve dependencies if recursive
         if Recursive then
            declare
               Dep_Result : constant String :=
                 Run_Command ("type_dependency" & Registry_Flag, Expand_Result);
            begin
               if Dep_Result'Length = 0 then
                  Print_Error ("Failed to resolve dependencies");
                  Set_Exit_Status (Exit_Resolution_Error);
                  return;
               end if;
               Put_Line (Dep_Result);
            end;
         else
            Put_Line (Expand_Result);
         end if;
      end;
   end;

   Set_Exit_Status (Exit_Success);

exception
   when others =>
      Print_Error ("Unexpected error");
      Set_Exit_Status (Exit_Resolution_Error);
end Type_Resolver;
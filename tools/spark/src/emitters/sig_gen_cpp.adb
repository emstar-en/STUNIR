--  sig_gen_cpp - Generate C++ function signatures from STUNIR spec (Refactored)
--  Orchestrates type_map_cpp, cpp_header_gen, cpp_impl_gen
--  Phase 3 Powertool for STUNIR - Refactored for Unix philosophy

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;
with GNAT.Command_Line;
with GNAT.OS_Lib;
with GNAT.Strings;
with Command_Utils;

procedure Sig_Gen_CPP is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;

   --  Exit codes
   Exit_Success          : constant := 0;
   Exit_Invalid_IR       : constant := 1;
   Exit_Generation_Error : constant := 2;

   --  Configuration
   Output_File   : aliased GNAT.Strings.String_Access := new String'("");
   Namespace     : aliased GNAT.Strings.String_Access := new String'("");
   Show_Version  : aliased Boolean := False;
   Show_Help     : aliased Boolean := False;
   Show_Describe : aliased Boolean := False;
   Generate_Impl : aliased Boolean := False;

   Version : constant String := "0.1.0-alpha";

   Describe_Output : constant String :=
     "{" & ASCII.LF &
     "  \"tool\": \"sig_gen_cpp\"," & ASCII.LF &
     "  \"version\": \"0.1.0-alpha\"," & ASCII.LF &
     "  \"description\": \"Generate C++ function signatures from STUNIR IR\"," & ASCII.LF &
     "  \"inputs\": [{" & ASCII.LF &
     "    \"name\": \"ir_json\"," & ASCII.LF &
     "    \"type\": \"json\"," & ASCII.LF &
     "    \"source\": [\"stdin\"]," & ASCII.LF &
     "    \"required\": true" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  \"outputs\": [{" & ASCII.LF &
     "    \"name\": \"cpp_header\"," & ASCII.LF &
     "    \"type\": \"string\"," & ASCII.LF &
     "    \"source\": \"stdout\"" & ASCII.LF &
     "  }]" & ASCII.LF &
     "}";

   procedure Print_Usage is
   begin
      Put_Line ("sig_gen_cpp - Generate C++ function signatures");
      Put_Line ("Version: " & Version);
      Put_Line ("");
      Put_Line ("Usage: sig_gen_cpp [OPTIONS] < ir.json");
      Put_Line ("");
      Put_Line ("Options:");
      Put_Line ("  --help, -h        Show this help message");
      Put_Line ("  --version, -v     Show version information");
      Put_Line ("  --describe        Show tool description (JSON)");
      Put_Line ("  --namespace NS    Wrap in namespace");
      Put_Line ("  --output FILE     Output file (default: stdout)");
      Put_Line ("  --impl            Generate implementation stubs");
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
      Success : aliased Boolean;
      Result  : constant String :=
        Command_Utils.Get_Command_Output (Cmd, Input, Success'Access);
   begin
      if Success then
         return Result;
      else
         return "";
      end if;
   end Run_Command;

   Config : GNAT.Command_Line.Command_Line_Configuration;

begin
   GNAT.Command_Line.Define_Switch (Config, Show_Help'Access, "-h", "--help");
   GNAT.Command_Line.Define_Switch (Config, Show_Version'Access, "-v", "--version");
   GNAT.Command_Line.Define_Switch (Config, Show_Describe'Access, "", "--describe");
   GNAT.Command_Line.Define_Switch (Config, Generate_Impl'Access, "", "--impl");
   GNAT.Command_Line.Define_Switch (Config, Namespace'Access, "", "--namespace=");
   GNAT.Command_Line.Define_Switch (Config, Output_File'Access, "-o:", "--output=");

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
      IR_Input : constant String := Read_Stdin;
   begin
      if IR_Input'Length = 0 then
         Print_Error ("Empty input");
         Set_Exit_Status (Exit_Invalid_IR);
         return;
      end if;

      --  Generate header or implementation using appropriate utility
      declare
         NS_Flag : constant String :=
           (if Namespace.all = "" then "" else " --namespace " & Namespace.all);
         Cmd : constant String :=
           (if Generate_Impl then "cpp_impl_gen" else "cpp_header_gen") & NS_Flag;
         Result : constant String := Run_Command (Cmd, IR_Input);
      begin
         if Result'Length = 0 then
            Print_Error ("Generation failed");
            Set_Exit_Status (Exit_Generation_Error);
            return;
         end if;

         --  Write to file or stdout
         if Output_File.all /= "" then
            declare
               use Ada.Text_IO;
               File : File_Type;
            begin
               Create (File, Out_File, Output_File.all);
               Put (File, Result);
               Close (File);
            exception
               when others =>
                  Print_Error ("Cannot write: " & Output_File.all);
                  Set_Exit_Status (Exit_Generation_Error);
                  return;
            end;
         else
            Put_Line (Result);
         end if;

         Set_Exit_Status (Exit_Success);
      end;
   end;

exception
   when others =>
      Print_Error ("Unexpected error");
      Set_Exit_Status (Exit_Generation_Error);
end Sig_Gen_CPP;

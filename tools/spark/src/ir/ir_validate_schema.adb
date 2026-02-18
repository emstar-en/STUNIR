--  ir_validate_schema - Validate IR JSON against schema requirements
--  Phase 1 Powertool for STUNIR

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;
with Ada.Strings.Fixed;
with Ada.Exceptions;

procedure IR_Validate_Schema is
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
     "  ""tool"": ""ir_validate_schema""," & ASCII.LF &
     "  ""version"": ""0.1.0-alpha""," & ASCII.LF &
     "  ""description"": ""Validate IR JSON against schema requirements""," & ASCII.LF &
     "  ""inputs"": [{" & ASCII.LF &
     "    ""name"": ""ir_json""," & ASCII.LF &
     "    ""type"": ""string""," & ASCII.LF &
     "    ""source"": [""stdin""]," & ASCII.LF &
     "    ""required"": true" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""outputs"": [{" & ASCII.LF &
     "    ""name"": ""valid""," & ASCII.LF &
     "    ""type"": ""boolean""," & ASCII.LF &
     "    ""source"": ""stdout""" & ASCII.LF &
     "  }]" & ASCII.LF &
     "}";

   procedure Print_Usage is
   begin
      Put_Line ("ir_validate_schema - Validate IR JSON against schema");
      Put_Line ("Version: " & Version);
      Put_Line ("");
      Put_Line ("Usage: ir_validate_schema [OPTIONS] < ir.json");
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

   function Contains (S : String; Pattern : String) return Boolean is
   begin
      return Index (S, Pattern) > 0;
   end Contains;

   function Is_Valid_IR (Content : String) return Boolean is
   begin
      --  Check for required fields
      if not Contains (Content, """schema""") then
         return False;
      end if;

      if not Contains (Content, """ir_version""") then
         return False;
      end if;

      if not Contains (Content, """module_name""") then
         return False;
      end if;

      if not Contains (Content, """functions""") then
         return False;
      end if;

      return True;
   end Is_Valid_IR;

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

   --  Read and validate IR
   declare
      Content : constant String := Read_Stdin;
   begin
      if Content'Length = 0 then
         Print_Error ("Empty input");
         Set_Exit_Status (Exit_Error);
         return;
      end if;

      if Is_Valid_IR (Content) then
         Put_Line ("true");
         Set_Exit_Status (Exit_Success);
      else
         Put_Line ("false");
         Set_Exit_Status (Exit_Error);
      end if;
   end;

exception
   when others =>
      Print_Error ("Unexpected error");
      Set_Exit_Status (Exit_Error);
end IR_Validate_Schema;
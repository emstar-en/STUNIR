--  ir_check_required - Check required IR fields
--  IR field validation utility for STUNIR powertools
--  Phase 3 Utility for STUNIR

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;

procedure IR_Check_Required is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;

   Exit_Success          : constant := 0;
   Exit_Validation_Error : constant := 1;
   Exit_Processing_Error : constant := 2;

   Show_Version  : Boolean := False;
   Show_Help     : Boolean := False;
   Show_Describe : Boolean := False;

   Version : constant String := "0.1.0-alpha";

   Describe_Output : constant String :=
     "{" & ASCII.LF &
     "  ""tool"": ""ir_check_required""," & ASCII.LF &
     "  ""version"": ""0.1.0-alpha""," & ASCII.LF &
     "  ""description"": ""Check required IR fields""," & ASCII.LF &
     "  ""inputs"": [{" & ASCII.LF &
     "    ""name"": ""ir_json""," & ASCII.LF &
     "    ""type"": ""json""," & ASCII.LF &
     "    ""source"": ""stdin""," & ASCII.LF &
     "    ""required"": true" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""outputs"": [{" & ASCII.LF &
     "    ""name"": ""missing_fields""," & ASCII.LF &
     "    ""type"": ""json""," & ASCII.LF &
     "    ""source"": ""stdout""" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  ""complexity"": ""O(n)""," & ASCII.LF &
     "  ""options"": [" & ASCII.LF &
     "    ""--help"", ""--version"", ""--describe""" & ASCII.LF &
     "  ]" & ASCII.LF &
     "}";

   type String_Array is array (Positive range <>) of Unbounded_String;
   Required_Fields : constant String_Array (1 .. 4) := (
      To_Unbounded_String ("ir_version"),
      To_Unbounded_String ("module_name"),
      To_Unbounded_String ("functions"),
      To_Unbounded_String ("types")
   );

   procedure Print_Usage;
   procedure Print_Error (Msg : String);
   function Read_Stdin return String;
   function Check_Required (IR : String) return Boolean;

   procedure Print_Usage is
   begin
      Put_Line ("ir_check_required - Check required IR fields");
      Put_Line ("Version: " & Version);
      Put_Line ("");
      Put_Line ("Usage: ir_check_required [OPTIONS]");
      Put_Line ("");
      Put_Line ("Options:");
      Put_Line ("  --help, -h        Show this help message");
      Put_Line ("  --version, -v     Show version information");
      Put_Line ("  --describe        Show tool description (JSON)");
      Put_Line ("");
      Put_Line ("Required Fields:");
      Put_Line ("  - ir_version");
      Put_Line ("  - module_name");
      Put_Line ("  - functions");
      Put_Line ("  - types");
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
      while not End_Of_File (Standard_Input) loop
         Get_Line (Standard_Input, Line, Last);
         Append (Result, Line (1 .. Last));
      end loop;
      return To_String (Result);
   end Read_Stdin;

   function Check_Required (IR : String) return Boolean is
      All_Present : Boolean := True;
   begin
      for I in Required_Fields'Range loop
         declare
            Field : constant String := To_String (Required_Fields (I));
            Pattern : constant String := '"' & Field & '"';
            Found : Boolean := False;
         begin
            for J in IR'First .. IR'Last - Pattern'Length + 1 loop
               if IR (J .. J + Pattern'Length - 1) = Pattern then
                  Found := True;
                  exit;
               end if;
            end loop;

            if not Found then
               Put_Line ("Missing required field: " & Field);
               All_Present := False;
            end if;
         end;
      end loop;

      return All_Present;
   end Check_Required;

begin
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
      return;
   end if;

   if Show_Version then
      Put_Line ("ir_check_required " & Version);
      return;
   end if;

   if Show_Describe then
      Put_Line (Describe_Output);
      return;
   end if;

   declare
      Input : constant String := Read_Stdin;
   begin
      if Input'Length = 0 then
         Print_Error ("No IR input provided");
         Set_Exit_Status (Exit_Validation_Error);
         return;
      end if;

      if Check_Required (Input) then
         Set_Exit_Status (Exit_Success);
      else
         Set_Exit_Status (Exit_Validation_Error);
      end if;
   exception
      when others =>
         Print_Error ("Processing error");
         Set_Exit_Status (Exit_Processing_Error);
   end;

end IR_Check_Required;

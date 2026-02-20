--  cpp_header_gen - Generate C++ header files from IR
--  C++ header generation utility for STUNIR powertools
--  Phase 2 Utility for STUNIR

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;

procedure Cpp_Header_Gen is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;

   --  Exit codes per powertools spec
   Exit_Success          : constant := 0;
   Exit_Validation_Error : constant := 1;
   Exit_Processing_Error : constant := 2;

   --  Configuration
   Guard_Name    : Unbounded_String := Null_Unbounded_String;
   Show_Version  : Boolean := False;
   Show_Help     : Boolean := False;
   Show_Describe : Boolean := False;

   Version : constant String := "0.1.0-alpha";

   --  Description output for --describe
   Describe_Output : constant String :=
     "{" & ASCII.LF &
     "  \"tool\": \"cpp_header_gen\"," & ASCII.LF &
     "  \"version\": \"0.1.0-alpha\"," & ASCII.LF &
     "  \"description\": \"Generate C++ header files from IR\"," & ASCII.LF &
     "  \"inputs\": [{" & ASCII.LF &
     "    \"name\": \"ir_json\"," & ASCII.LF &
     "    \"type\": \"json\"," & ASCII.LF &
     "    \"source\": \"stdin\"," & ASCII.LF &
     "    \"required\": true" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  \"outputs\": [{" & ASCII.LF &
     "    \"name\": \"header_content\"," & ASCII.LF &
     "    \"type\": \"cpp\"," & ASCII.LF &
     "    \"source\": \"stdout\"" & ASCII.LF &
     "  }]," & ASCII.LF &
     "  \"complexity\": \"O(n)\"," & ASCII.LF &
     "  \"options\": [" & ASCII.LF &
     "    \"--help\", \"--version\", \"--describe\"," & ASCII.LF &
     "    \"--guard\"" & ASCII.LF &
     "  ]" & ASCII.LF &
     "}";

   procedure Print_Usage;
   procedure Print_Error (Msg : String);
   function Read_Stdin return String;
   function Generate_Header (IR : String; Guard : String) return String;

   procedure Print_Usage is
   begin
      Put_Line ("cpp_header_gen - Generate C++ header files from IR");
      Put_Line ("Version: " & Version);
      Put_Line ("");
      Put_Line ("Usage: cpp_header_gen [OPTIONS]");
      Put_Line ("");
      Put_Line ("Options:");
      Put_Line ("  --help, -h        Show this help message");
      Put_Line ("  --version, -v     Show version information");
      Put_Line ("  --describe        Show tool description (JSON)");
      Put_Line ("  --guard NAME      Include guard name");
      Put_Line ("");
      Put_Line ("Exit Codes:");
      Put_Line ("  0                 Success");
      Put_Line ("  1                 Invalid IR input");
      Put_Line ("  2                 Processing error");
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
         Append (Result, ASCII.LF);
      end loop;
      return To_String (Result);
   end Read_Stdin;

   function Generate_Header (IR : String; Guard : String) return String is
      Result : Unbounded_String := Null_Unbounded_String;
      Use_Guard : constant Boolean := Guard'Length > 0;
   begin
      --  Add include guard if specified
      if Use_Guard then
         Append (Result, "#ifndef " & Guard & ASCII.LF);
         Append (Result, "#define " & Guard & ASCII.LF);
         Append (Result, ASCII.LF);
      end if;

      --  Add standard includes
      Append (Result, "#include <cstdint>" & ASCII.LF);
      Append (Result, "#include <string>" & ASCII.LF);
      Append (Result, ASCII.LF);

      --  Extract function signatures from IR and generate declarations
      Append (Result, "// Generated from STUNIR IR" & ASCII.LF);
      Append (Result, "// TODO: Parse IR and generate actual function declarations" & ASCII.LF);
      Append (Result, ASCII.LF);
      Append (Result, "namespace stunir {" & ASCII.LF);
      Append (Result, ASCII.LF);
      Append (Result, "  // Function declarations will be generated here" & ASCII.LF);
      Append (Result, ASCII.LF);
      Append (Result, "} // namespace stunir" & ASCII.LF);

      --  Close include guard
      if Use_Guard then
         Append (Result, ASCII.LF);
         Append (Result, "#endif // " & Guard & ASCII.LF);
      end if;

      return To_String (Result);
   end Generate_Header;

begin
   --  Parse command line arguments
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
         elsif Arg'Length > 8 and then Arg (1 .. 8) = "--guard=" then
            Guard_Name := To_Unbounded_String (Arg (9 .. Arg'Last));
         elsif Arg = "--guard" then
            if I < Argument_Count then
               Guard_Name := To_Unbounded_String (Argument (I + 1));
            end if;
         end if;
      end;
   end loop;

   --  Handle flags
   if Show_Help then
      Print_Usage;
      return;
   end if;

   if Show_Version then
      Put_Line ("cpp_header_gen " & Version);
      return;
   end if;

   if Show_Describe then
      Put_Line (Describe_Output);
      return;
   end if;

   --  Read IR and generate header
   declare
      IR_Input : constant String := Read_Stdin;
   begin
      if IR_Input'Length = 0 then
         Print_Error ("No IR input provided");
         Set_Exit_Status (Exit_Validation_Error);
         return;
      end if;

      declare
         Guard : constant String := To_String (Guard_Name);
         Header : constant String := Generate_Header (IR_Input, Guard);
      begin
         Put_Line (Header);
         Set_Exit_Status (Exit_Success);
      end;
   exception
      when others =>
         Print_Error ("Processing error");
         Set_Exit_Status (Exit_Processing_Error);
   end;

end Cpp_Header_Gen;

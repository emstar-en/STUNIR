with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;
with Ada.IO_Exceptions;

with STUNIR_JSON_Parser;
with STUNIR_Types;

procedure IR_Validate is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;
   use STUNIR_Types;

   Input_File    : Unbounded_String := Null_Unbounded_String;
   Output_Json   : Boolean := False;
   Show_Help     : Boolean := False;
   Show_Describe : Boolean := False;

   procedure Print_Usage is
   begin
      Put_Line (Standard_Error, "Usage: ir_validate [options] [file]");
      Put_Line (Standard_Error, "Options:");
      Put_Line (Standard_Error, "  --json           Output result as JSON");
      Put_Line (Standard_Error, "  --describe       Show tool description");
      Put_Line (Standard_Error, "  --help           Show this help");
   end Print_Usage;

   procedure Print_Describe is
   begin
      Put_Line ("{");
      Put_Line ("  ""name"": ""ir_validate"",");
      Put_Line ("  ""description"": ""Validate IR structure and syntax"",");
      Put_Line ("  ""version"": ""1.0.0"",");
      Put_Line ("  ""inputs"": [{""name"": ""ir"", ""type"": ""json"", ""source"": [""stdin"", ""file""]}],");
      Put_Line ("  ""outputs"": [{""name"": ""validation_report"", ""type"": ""json"", ""source"": ""stdout""}]");
      Put_Line ("}");
   end Print_Describe;

   function Read_All (Path : String) return String is
      File   : File_Type;
      Result : Unbounded_String := Null_Unbounded_String;
      Line   : String (1 .. 4096);
      Last   : Natural;
   begin
      if Path = "" then
         while not End_Of_File loop
            Get_Line (Line, Last);
            Append (Result, Line (1 .. Last));
            Append (Result, ASCII.LF);
         end loop;
         return To_String (Result);
      end if;

      Open (File, In_File, Path);
      while not End_Of_File (File) loop
         Get_Line (File, Line, Last);
         Append (Result, Line (1 .. Last));
         Append (Result, ASCII.LF);
      end loop;
      Close (File);
      return To_String (Result);
   exception
      when Ada.IO_Exceptions.Name_Error =>
         return "";
      when others =>
         return "";
   end Read_All;

   function Validate_JSON (Content : String) return Boolean is
      use STUNIR_JSON_Parser;
      State  : Parser_State;
      Status : Status_Code;
      Input_Str : JSON_String;
   begin
      if Content'Length = 0 then
         return False;
      end if;

      if Content'Length > Max_JSON_Length then
         return False;
      end if;

      Input_Str := JSON_Strings.To_Bounded_String (Content);
      Initialize_Parser (State, Input_Str, Status);
      if Status /= Success then
         return False;
      end if;

      loop
         Next_Token (State, Status);
         exit when Status /= Success or else State.Current_Token = Token_EOF;
      end loop;

      return Status = Success;
   end Validate_JSON;

begin
   for I in 1 .. Argument_Count loop
      declare
         Arg : constant String := Argument (I);
      begin
         if Arg = "--help" then
            Show_Help := True;
         elsif Arg = "--describe" then
            Show_Describe := True;
         elsif Arg = "--json" then
            Output_Json := True;
         elsif Arg (1) /= '-' then
            Input_File := To_Unbounded_String (Arg);
         end if;
      end;
   end loop;

   if Show_Help then
      Print_Usage;
      Set_Exit_Status (Success);
      return;
   end if;

   if Show_Describe then
      Print_Describe;
      Set_Exit_Status (Success);
      return;
   end if;

   declare
      Content : constant String := Read_All (To_String (Input_File));
      Valid   : constant Boolean := Validate_JSON (Content);
   begin
      if Valid then
         if Output_Json then
            Put_Line ("{""status"": ""valid""}");
         else
            Put_Line ("IR valid.");
         end if;
         Set_Exit_Status (Success);
      else
         if Output_Json then
            Put_Line ("{""status"": ""invalid""}");
         else
            Put_Line (Standard_Error, "Invalid IR.");
         end if;
         Set_Exit_Status (Failure);
      end if;
   end;
end IR_Validate;
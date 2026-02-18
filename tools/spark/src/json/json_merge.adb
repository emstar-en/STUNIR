--  json_merge - Merge multiple JSON objects or arrays (Refactored)
--  Orchestrates json_merge_objects, json_merge_arrays
--  Phase 1 Powertool for STUNIR - Refactored for Unix philosophy

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;
with GNAT.OS_Lib;
with Command_Utils;

procedure JSON_Merge is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;

   --  Exit codes
   Exit_Success : constant := 0;
   Exit_Error   : constant := 1;

   Show_Help     : Boolean := False;
   Show_Describe : Boolean := False;
   Strategy      : Unbounded_String := To_Unbounded_String ("last");

   Version : constant String := "0.1.0-alpha";

   procedure Print_Usage is
   begin
      Put_Line (Standard_Error, "json_merge - Merge multiple JSON objects or arrays");
      Put_Line (Standard_Error, "Version: " & Version);
      Put_Line (Standard_Error, "Usage: json_merge [options] < input.json");
      Put_Line (Standard_Error, "Options:");
      Put_Line (Standard_Error, "  --strategy STRAT  Conflict strategy: last|first|error (default: last)");
      Put_Line (Standard_Error, "  --describe        Show tool description");
      Put_Line (Standard_Error, "  --help            Show this help");
   end Print_Usage;

   procedure Print_Describe is
   begin
      Put_Line ("{");
      Put_Line ("  ""tool"": ""json_merge"",");
      Put_Line ("  ""description"": ""Merge multiple JSON objects or arrays"",");
      Put_Line ("  ""version"": ""0.1.0-alpha"",");
      Put_Line ("  ""inputs"": [{""name"": ""json_inputs"", ""type"": ""json"", ""source"": [""stdin""]}]");
      Put_Line ("}");
   end Print_Describe;

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
      return Result;
   end Run_Command;

   function Is_Object (S : String) return Boolean is
      Trimmed : constant String := S;
   begin
      return Trimmed'Length >= 2 and then
             Trimmed (Trimmed'First) = '{' and then
             Trimmed (Trimmed'Last) = '}';
   end Is_Object;

   function Is_Array (S : String) return Boolean is
      Trimmed : constant String := S;
   begin
      return Trimmed'Length >= 2 and then
             Trimmed (Trimmed'First) = '[' and then
             Trimmed (Trimmed'Last) = ']';
   end Is_Array;

begin
   for I in 1 .. Argument_Count loop
      declare
         Arg : constant String := Argument (I);
      begin
         if Arg = "--help" then
            Show_Help := True;
         elsif Arg = "--describe" then
            Show_Describe := True;
         elsif Arg'Length > 11 and then Arg (1 .. 11) = "--strategy=" then
            Strategy := To_Unbounded_String (Arg (12 .. Arg'Last));
         end if;
      end;
   end loop;

   if Show_Help then
      Print_Usage;
      Set_Exit_Status (Exit_Success);
      return;
   end if;

   if Show_Describe then
      Print_Describe;
      Set_Exit_Status (Exit_Success);
      return;
   end if;

   declare
      Content : constant String := Read_Stdin;
   begin
      if Content'Length = 0 then
         Put_Line (Standard_Error, "ERROR: Empty input");
         Set_Exit_Status (Exit_Error);
         return;
      end if;

      --  Determine type and use appropriate merge utility
      declare
         Strategy_Flag : constant String :=
           " --strategy " & To_String (Strategy);
         Result : constant String :=
           (if Is_Object (Content) then
               Run_Command ("json_merge_objects" & Strategy_Flag, Content)
            elsif Is_Array (Content) then
               Run_Command ("json_merge_arrays" & Strategy_Flag, Content)
            else
               "");
      begin
         if Result'Length = 0 then
            Put_Line (Standard_Error, "ERROR: Invalid JSON or merge failed");
            Set_Exit_Status (Exit_Error);
            return;
         end if;

         Put_Line (Result);
         Set_Exit_Status (Exit_Success);
      end;
   end;
end JSON_Merge;
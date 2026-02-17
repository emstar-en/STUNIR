with Ada.Text_IO;
with Ada.Command_Line;
with Ada.Strings.Unbounded;
with STUNIR_JSON_Parser;
with STUNIR_Types;

procedure Toolchain_Verify is
   use Ada.Text_IO;
   use Ada.Command_Line;
   use Ada.Strings.Unbounded;
   use STUNIR_JSON_Parser;
   use STUNIR_Types;

   procedure Print_Usage is
   begin
      Put_Line (Standard_Error, "Usage: toolchain_verify [options] [lockfile]");
      Put_Line (Standard_Error, "");
      Put_Line (Standard_Error, "Options:");
      Put_Line (Standard_Error, "  --check-versions Compare with installed tools (not implemented)");
      Put_Line (Standard_Error, "  --strict         Fail on warnings");
      Put_Line (Standard_Error, "  --json           Output status as JSON");
      Put_Line (Standard_Error, "  --describe       Show AI introspection data");
      Put_Line (Standard_Error, "  --help           Show this help");
   end Print_Usage;

   procedure Print_Describe is
   begin
      Put_Line ("{");
      Put_Line ("  ""name"": ""toolchain_verify"",");
      Put_Line ("  ""description"": ""Verify toolchain.lock file integrity"",");
      Put_Line ("  ""version"": ""0.1.0-alpha"",");
      Put_Line ("  ""inputs"": [");
      Put_Line ("    {""name"": ""lockfile"", ""type"": ""file"", ""description"": ""Lock file (default: toolchain.lock)""}");
      Put_Line ("  ],");
      Put_Line ("  ""outputs"": [");
      Put_Line ("    {""type"": ""report"", ""description"": ""Verification status""}");
      Put_Line ("  ],");
      Put_Line ("  ""options"": [");
      Put_Line ("    {""name"": ""--check-versions"", ""type"": ""boolean""},");
      Put_Line ("    {""name"": ""--json"", ""type"": ""boolean""}");
      Put_Line ("  ]");
      Put_Line ("}");
   end Print_Describe;

   File_Path   : Unbounded_String := To_Unbounded_String ("toolchain.lock");
   Output_Json : Boolean := False;

   function Read_File (Path : String) return String is
      File : Ada.Text_IO.File_Type;
      Len  : Long_Integer;
   begin
      begin
         Open (File, In_File, Path);
         Len := Long_Integer (Size (File));
         declare
            Content : String (1 .. Integer (Len));
            Last    : Natural;
         begin
            Get (File, Content, Last);
            Close (File);
            --  Strip trailing NULLs or generic cleanup if needed
            return Content (1 .. Last);
         end;
      exception
         when others => 
            if Is_Open (File) then Close (File); end if;
            return "";
      end;
   end Read_File;

   procedure Validate_Lock (Content : String; Valid : out Boolean; Msg : out Unbounded_String) is
      Parser : Parser_State;
      Status : Status_Code;
   begin
      Valid := False;
      Msg := Null_Unbounded_String;
      Initialize_Parser (Parser, JSON_Strings.To_Bounded_String (Content));
      
      if Next_Token (Parser, Status) /= Token_Object_Start then
         Msg := To_Unbounded_String ("Root must be object"); return;
      end if;

      Next_Token (Parser, Status);
      while Current_Token (Parser) /= Token_Object_End and Status = Success loop
         if Current_Token (Parser) /= Token_String then
            Msg := To_Unbounded_String ("Expected string key"); Status := Error_Parse; return;
         end if;
         Skip_Value (Parser, Status); -- Key
         if Current_Token (Parser) = Token_Colon then Next_Token (Parser, Status); end if;
         Skip_Value (Parser, Status); -- Value
         
         if Current_Token (Parser) = Token_Comma then Next_Token (Parser, Status); end if;
      end loop;

      if Status = Success then
         Valid := True;
      else
         Msg := To_Unbounded_String ("Parse error or incomplete");
      end if;
   exception
      when others => Msg := To_Unbounded_String ("Exception"); Valid := False;
   end Validate_Lock;

begin
   for I in 1 .. Argument_Count loop
      declare
         Arg : constant String := Argument (I);
      begin
         if Arg = "--help" then Print_Usage; return;
         elsif Arg = "--describe" then Print_Describe; return;
         elsif Arg = "--json" then Output_Json := True;
         elsif Arg (1) /= '-' then File_Path := To_Unbounded_String (Arg);
         end if;
      end;
   end loop;

   declare
      Content : constant String := Read_File (To_String (File_Path));
      Valid   : Boolean;
      Msg     : Unbounded_String;
   begin
      if Content = "" then
         Put_Line (Standard_Error, "Error: Lock file not found: " & To_String (File_Path));
         Set_Exit_Status (Failure);
         return;
      end if;

      Validate_Lock (Content, Valid, Msg);
      
      if Valid then
         if Output_Json then Put_Line ("{""status"": ""valid""}"); else Put_Line ("Lock file valid."); end if;
         Set_Exit_Status (Success);
      else
         if Output_Json then Put_Line ("{""status"": ""invalid"", ""error"": """ & To_String (Msg) & """}");
         else Put_Line (Standard_Error, "Invalid: " & To_String (Msg)); end if;
         Set_Exit_Status (Failure);
      end if;
   end;
end Toolchain_Verify;


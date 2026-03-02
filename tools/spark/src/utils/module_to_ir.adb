-------------------------------------------------------------------------------
--  module_to_ir.adb — Module Spec to IR Metadata Extractor
--
--  PURPOSE: Extract module metadata from a spec module object and produce
--           module metadata JSON for use with ir_add_metadata.
--
--  PIPELINE POSITION: Phase 2 — IR Pipeline
--    spec_extract_module → [module_to_ir] → ir_add_metadata → ir_converter
--
--  INPUTS:  stdin  — module spec JSON object (from spec_extract_module)
--  OUTPUTS: stdout — module metadata JSON for ir_add_metadata
--
--  EXIT CODES (per ARCHITECTURE.core.json):
--    0 = success
--    1 = validation error (bad input format or invalid identifier)
--    2 = processing error
--    3 = resource error (input too large)
--
--  REGEX_IR_REF: tools/spark/schema/stunir_regex_ir_v1.dcbor.json
--               group: validation.identifier (module_name pattern)
--               pattern_id: identifier_start
--               regex: ^[A-Za-z_][A-Za-z0-9_]*$
--
--  GOVERNANCE: Do NOT add new source directories without updating
--              stunir_tools.gpr. See CONTRIBUTING.md.
--
--  See: stunir_tools.gpr, tools/spark/ARCHITECTURE.md (Phase 2)
-------------------------------------------------------------------------------

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;
with Ada.Characters.Handling;

with STUNIR_JSON_Parser;
with STUNIR_Types;

procedure Module_To_IR is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;
   use STUNIR_JSON_Parser;
   use STUNIR_Types;

   --  Exit codes per ARCHITECTURE.core.json
   Exit_Success          : constant := 0;
   Exit_Validation_Error : constant := 1;
   Exit_Processing_Error : constant := 2;
   Exit_Resource_Error   : constant := 3;

   Show_Help     : Boolean := False;
   Show_Version  : Boolean := False;
   Show_Describe : Boolean := False;

   Version : constant String := "0.2.0";

   Describe_Output : constant String :=
     "{""tool"":""module_to_ir"",""version"":""0.2.0""," &
     """description"":""Extract module metadata from spec module object""," &
     """inputs"":[{""type"":""json"",""source"":""stdin"",""required"":true}]," &
     """outputs"":[{""type"":""json"",""source"":""stdout""}]," &
     """exit_codes"":{""0"":""success"",""1"":""validation_error""," &
     """2"":""processing_error"",""3"":""resource_error""}}";

   Max_Fields : constant := 16;

   type Field_Entry is record
      Key   : Unbounded_String;
      Value : Unbounded_String;
   end record;

   type Field_Array is array (1 .. Max_Fields) of Field_Entry;

   type Parsed_Object is record
      Fields    : Field_Array;
      Count     : Natural := 0;
      Valid     : Boolean := False;
   end record;

   --  Identifier validation per regex: ^[A-Za-z_][A-Za-z0-9_]*$
   function Is_Valid_Identifier (S : String) return Boolean is
   begin
      if S'Length = 0 then
         return False;
      end if;
      --  First char must be letter or underscore
      if not (Ada.Characters.Handling.Is_Letter (S (S'First)) or else
              S (S'First) = '_') then
         return False;
      end if;
      --  Remaining chars must be letter, digit, or underscore
      for I in S'First + 1 .. S'Last loop
         if not (Ada.Characters.Handling.Is_Letter (S (I)) or else
                 Ada.Characters.Handling.Is_Digit (S (I)) or else
                 S (I) = '_') then
            return False;
         end if;
      end loop;
      return True;
   end Is_Valid_Identifier;

   function Read_Stdin return String is
      Result : Unbounded_String := Null_Unbounded_String;
      Line   : String (1 .. 4096);
      Last   : Natural;
   begin
      while not End_Of_File loop
         Get_Line (Line, Last);
         if Last > 0 then
            Append (Result, Line (1 .. Last));
         end if;
         if not End_Of_File then
            Append (Result, ASCII.LF);
         end if;
      end loop;
      return To_String (Result);
   end Read_Stdin;

   function Escape_JSON (S : String) return String is
      Result : Unbounded_String;
   begin
      for I in S'Range loop
         if    S (I) = '"'      then Append (Result, "\""");
         elsif S (I) = '\'      then Append (Result, "\\");
         elsif S (I) = ASCII.LF then Append (Result, "\n");
         elsif S (I) = ASCII.CR then Append (Result, "\r");
         elsif S (I) = ASCII.HT then Append (Result, "\t");
         else Append (Result, S (I));
         end if;
      end loop;
      return To_String (Result);
   end Escape_JSON;

   --  Parse a JSON object and extract string fields
   procedure Parse_Object
     (JSON   : String;
      Obj    : out Parsed_Object;
      Status : out Status_Code)
   is
      State   : Parser_State;
      Tok     : Token_Kind;
      Key     : Unbounded_String;
      Value   : Unbounded_String;
      In_Obj  : Boolean := False;
      Expect_Key : Boolean := True;
   begin
      Obj.Valid := False;
      Obj.Count := 0;
      Status := Error_Parse_Error;

      if JSON'Length = 0 then
         return;
      end if;

      if JSON'Length > Max_JSON_Length then
         Status := Error_Too_Large;
         return;
      end if;

      declare
         Input_Str : JSON_String;
      begin
         Input_Str := JSON_Strings.To_Bounded_String (JSON);
         Initialize_Parser (State, Input_Str, Status);
         if Status /= STUNIR_Types.Success then
            return;
         end if;
      end;

      --  Expect object start
      Next_Token (State, Status);
      if Status /= STUNIR_Types.Success then
         return;
      end if;

      Tok := Current_Token (State);
      if Tok /= Token_Object_Start then
         Status := STUNIR_Types.Error_Invalid_JSON;
         return;
      end if;

      In_Obj := True;

      while In_Obj loop
         Next_Token (State, Status);
         if Status /= STUNIR_Types.Success then
            return;
         end if;

         Tok := Current_Token (State);

         case Tok is
            when Token_Object_End =>
               In_Obj := False;
               Status := STUNIR_Types.Success;
               Obj.Valid := True;

            when Token_String =>
               if Expect_Key then
                  --  This is a key
                  Key := To_Unbounded_String (JSON_Strings.To_String (Token_String_Value (State)));
                  Expect_Key := False;

                  --  Expect colon
                  Next_Token (State, Status);
                  if Status /= STUNIR_Types.Success then
                     return;
                  end if;
                  Tok := Current_Token (State);
                  if Tok /= Token_Colon then
                     Status := STUNIR_Types.Error_Invalid_JSON;
                     return;
                  end if;

                  --  Get value
                  Next_Token (State, Status);
                  if Status /= STUNIR_Types.Success then
                     return;
                  end if;
                  Tok := Current_Token (State);

                  case Tok is
                     when Token_String =>
                        Value := To_Unbounded_String (JSON_Strings.To_String (Token_String_Value (State)));
                     when Token_Number =>
                        Value := To_Unbounded_String (JSON_Strings.To_String (Token_String_Value (State)));
                     when Token_True =>
                        Value := To_Unbounded_String ("true");
                     when Token_False =>
                        Value := To_Unbounded_String ("false");
                     when Token_Null =>
                        Value := To_Unbounded_String ("null");
                     when Token_Array_Start | Token_Object_Start =>
                        --  Skip complex values for now
                        Value := To_Unbounded_String ("");
                        Skip_Value (State, Status);
                        if Status /= STUNIR_Types.Success then
                           return;
                        end if;
                     when others =>
                        Status := STUNIR_Types.Error_Invalid_JSON;
                        return;
                  end case;

                  --  Store field
                  if Obj.Count < Max_Fields then
                     Obj.Count := Obj.Count + 1;
                     Obj.Fields (Obj.Count).Key := Key;
                     Obj.Fields (Obj.Count).Value := Value;
                  end if;

                  Expect_Key := True;

                  --  Check for comma
                  Next_Token (State, Status);
                  if Status /= STUNIR_Types.Success then
                     return;
                  end if;
                  Tok := Current_Token (State);
                  if Tok = Token_Comma then
                     Expect_Key := True;
                  elsif Tok = Token_Object_End then
                     In_Obj := False;
                     Status := STUNIR_Types.Success;
                     Obj.Valid := True;
                  end if;
               end if;

            when others =>
               Status := STUNIR_Types.Error_Invalid_JSON;
               return;
         end case;
      end loop;
   end Parse_Object;

   function Get_Field
     (Obj : Parsed_Object;
      Key : String) return String
   is
   begin
      for I in 1 .. Obj.Count loop
         if To_String (Obj.Fields (I).Key) = Key then
            return To_String (Obj.Fields (I).Value);
         end if;
      end loop;
      return "";
   end Get_Field;

begin
   for I in 1 .. Argument_Count loop
      declare
         Arg : constant String := Argument (I);
      begin
         if    Arg = "--help" or Arg = "-h"    then Show_Help    := True;
         elsif Arg = "--version" or Arg = "-v" then Show_Version := True;
         elsif Arg = "--describe"              then Show_Describe := True;
         end if;
      end;
   end loop;

   if Show_Help then
      Put_Line ("module_to_ir - Extract module metadata from spec module object");
      Put_Line ("Version: " & Version);
      New_Line;
      Put_Line ("Usage: spec_extract_module < spec.json | module_to_ir");
      Put_Line ("  Outputs: {""module_name"": ""..."", ""description"": ""...""}");
      New_Line;
      Put_Line ("Exit Codes:");
      Put_Line ("  0 = success");
      Put_Line ("  1 = validation error (bad input format or invalid identifier)");
      Put_Line ("  2 = processing error");
      Put_Line ("  3 = resource error (input too large)");
      Set_Exit_Status (Exit_Success);
      return;
   end if;

   if Show_Version then
      Put_Line ("module_to_ir " & Version);
      Set_Exit_Status (Exit_Success);
      return;
   end if;

   if Show_Describe then
      Put_Line (Describe_Output);
      Set_Exit_Status (Exit_Success);
      return;
   end if;

   declare
      Input  : constant String := Read_Stdin;
      Obj    : Parsed_Object;
      Status : Status_Code;
      Name   : Unbounded_String;
      Desc   : Unbounded_String;
      Result : Unbounded_String;
   begin
      if Input'Length = 0 then
         Put_Line (Standard_Error, "ERROR: No input on stdin");
         Set_Exit_Status (Exit_Validation_Error);
         return;
      end if;

      --  Parse JSON object
      Parse_Object (Input, Obj, Status);

      if not Obj.Valid then
         Put_Line (Standard_Error, "ERROR: Invalid JSON input");
         Set_Exit_Status (Exit_Validation_Error);
         return;
      end if;

      --  Extract module name
      Name := To_Unbounded_String (Get_Field (Obj, "name"));
      if Length (Name) = 0 then
         Name := To_Unbounded_String (Get_Field (Obj, "module_name"));
      end if;

      --  Validate module name identifier
      if Length (Name) = 0 then
         Put_Line (Standard_Error, "ERROR: Missing module name field");
         Set_Exit_Status (Exit_Validation_Error);
         return;
      end if;

      if not Is_Valid_Identifier (To_String (Name)) then
         Put_Line (Standard_Error, "ERROR: Invalid module name identifier: " & To_String (Name));
         Put_Line (Standard_Error, "       Must match pattern: ^[A-Za-z_][A-Za-z0-9_]*$");
         Set_Exit_Status (Exit_Validation_Error);
         return;
      end if;

      --  Extract description
      Desc := To_Unbounded_String (Get_Field (Obj, "description"));
      if Length (Desc) = 0 then
         Desc := To_Unbounded_String (Get_Field (Obj, "docstring"));
      end if;

      --  Build output JSON with canonical field order
      Append (Result, "{");
      Append (Result, """module_name"":""" & Escape_JSON (To_String (Name)) & """");

      if Length (Desc) > 0 then
         Append (Result, ",""description"":""" & Escape_JSON (To_String (Desc)) & """");
      end if;

      --  Add schema version for downstream tools
      Append (Result, ",""schema_version"":""stunir_module_v1""");

      Append (Result, "}");

      Put_Line (To_String (Result));
      Set_Exit_Status (Exit_Success);
   end;

exception
   when others =>
      Put_Line (Standard_Error, "ERROR: Processing failed");
      Set_Exit_Status (Exit_Processing_Error);
end Module_To_IR;

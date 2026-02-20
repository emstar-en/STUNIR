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
--    1 = validation error (bad input format)
--    2 = processing error
--
--  REGEX_IR_REF: tools/spark/schema/stunir_regex_ir_v1.dcbor.json
--               group: validation.identifier (module_name pattern)
--               pattern_id: identifier_start
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

procedure Module_To_IR is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;

   Exit_Success : constant := 0;
   Exit_Error   : constant := 1;

   Show_Help     : Boolean := False;
   Show_Version  : Boolean := False;
   Show_Describe : Boolean := False;

   Version : constant String := "0.1.0-alpha";

   Describe_Output : constant String :=
     "{""tool"":""module_to_ir"",""version"":""0.1.0-alpha""," &
     """description"":""Extract module metadata from spec module object""," &
     """inputs"":[{""type"":""json"",""source"":""stdin"",""required"":true}]," &
     """outputs"":[{""type"":""json"",""source"":""stdout""}]}";

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

   function Find (S : String; P : String) return Natural is
   begin
      if P'Length = 0 or P'Length > S'Length then return 0; end if;
      for I in S'First .. S'Last - P'Length + 1 loop
         if S (I .. I + P'Length - 1) = P then return I; end if;
      end loop;
      return 0;
   end Find;

   function Get_String (JSON : String; Key : String) return String is
      Pat  : constant String := """" & Key & """";
      K    : constant Natural := Find (JSON, Pat);
      P, E : Natural;
   begin
      if K = 0 then return ""; end if;
      P := K + Pat'Length;
      while P <= JSON'Last and then
            (JSON (P) = ':' or JSON (P) = ' ' or
             JSON (P) = ASCII.HT or JSON (P) = ASCII.LF or
             JSON (P) = ASCII.CR) loop
         P := P + 1;
      end loop;
      if P > JSON'Last or else JSON (P) /= '"' then return ""; end if;
      P := P + 1; E := P;
      while E <= JSON'Last and then JSON (E) /= '"' loop E := E + 1; end loop;
      if E > JSON'Last then return JSON (P .. JSON'Last); end if;
      return JSON (P .. E - 1);
   end Get_String;

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
      Put_Line ("  Outputs: {""name"": ""..."", ""description"": ""...""}");
      Set_Exit_Status (Exit_Success); return;
   end if;
   if Show_Version then
      Put_Line ("module_to_ir " & Version);
      Set_Exit_Status (Exit_Success); return;
   end if;
   if Show_Describe then
      Put_Line (Describe_Output);
      Set_Exit_Status (Exit_Success); return;
   end if;

   declare
      Input : constant String := Read_Stdin;
      Name  : constant String := Get_String (Input, "name");
      Desc  : constant String := Get_String (Input, "description");
      Result : Unbounded_String;
   begin
      if Input'Length = 0 then
         Put_Line (Standard_Error, "ERROR: No input on stdin");
         Set_Exit_Status (Exit_Error); return;
      end if;
      Append (Result, "{""name"":""" & Escape_JSON (Name) & """");
      if Desc'Length > 0 then
         Append (Result, ",""description"":""" & Escape_JSON (Desc) & """");
      end if;
      Append (Result, "}");
      Put_Line (To_String (Result));
      Set_Exit_Status (Exit_Success);
   end;

exception
   when others =>
      Put_Line (Standard_Error, "ERROR: Processing failed");
      Set_Exit_Status (Exit_Error);
end Module_To_IR;

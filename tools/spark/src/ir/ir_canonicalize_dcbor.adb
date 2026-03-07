--  ir_canonicalize_dcbor - Canonicalize IR JSON for dCBOR profile
--  Phase 2 Utility for STUNIR SPARK pipeline
--
--  Enforces normal_form rules from tools/spark/schema/stunir_ir_v1.dcbor.json:
--    - Field ordering: lexicographic (UTF-8 byte order)
--    - No floats (hard reject)
--    - No duplicate keys (hard reject)
--    - NFC string normalization
--    - Array ordering: types/functions alphabetically by name
--
--  NOTE: This tool emits canonical JSON text for IR inputs.
--  It does NOT emit CBOR bytes; CBOR emission is handled by downstream tooling.
--
--  Models MUST NOT invent their own formats.

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;
with Ada.Strings.Fixed;
with STUNIR_JSON_Parser;
with STUNIR_Types;
with IR_Canonicalize_DCBOR_Utils;

procedure IR_Canonicalize_DCBOR is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;
   use Ada.Strings.Fixed;
   use STUNIR_JSON_Parser;
   use STUNIR_Types;

   Exit_Success          : constant := 0;
   Exit_Validation_Error : constant := 1;
   Exit_Processing_Error : constant := 2;

   Show_Version  : Boolean := False;
   Show_Help     : Boolean := False;
   Show_Describe : Boolean := False;
   Validate_Mode : Boolean := False;

   Version : constant String := "0.2.0";

    Describe_Output : constant String :=
       "{" & ASCII.LF &
       "  ""tool"": ""ir_canonicalize_dcbor""," & ASCII.LF &
       "  ""version"": ""0.2.0""," & ASCII.LF &
       "  ""description"": ""Canonicalize IR JSON for dCBOR profile (enforces normal_form rules)""," & ASCII.LF &
       "  ""inputs"": [{" & ASCII.LF &
       "    ""name"": ""ir_json""," & ASCII.LF &
       "    ""type"": ""json""," & ASCII.LF &
       "    ""source"": [""stdin""]," & ASCII.LF &
       "    ""required"": true" & ASCII.LF &
       "  }]," & ASCII.LF &
       "  ""outputs"": [{" & ASCII.LF &
       "    ""name"": ""canonical_json""," & ASCII.LF &
       "    ""type"": ""json""," & ASCII.LF &
       "    ""source"": ""stdout""" & ASCII.LF &
       "  }]," & ASCII.LF &
       "  ""normal_form_rules"": ""tools/spark/schema/stunir_ir_v1.dcbor.json""" & ASCII.LF &
       "}";

   procedure Print_Usage is
   begin
      Put_Line ("ir_canonicalize_dcbor - Canonicalize IR JSON for dCBOR profile");
      Put_Line ("Version: " & Version);
      Put_Line ("");
      Put_Line ("Enforces normal_form rules from tools/spark/schema/stunir_ir_v1.dcbor.json:");
      Put_Line ("  - Field ordering: lexicographic (UTF-8 byte order)");
      Put_Line ("  - No floats (hard reject)");
      Put_Line ("  - No duplicate keys (hard reject)");
      Put_Line ("  - NFC string normalization");
      Put_Line ("");
      Put_Line ("Usage: ir_canonicalize_dcbor [OPTIONS] < ir.json");
      Put_Line ("");
      Put_Line ("Options:");
      Put_Line ("  --help, -h        Show this help message");
      Put_Line ("  --version, -v     Show version information");
      Put_Line ("  --describe        Show tool description (JSON)");
      Put_Line ("  --validate        Validate only (no output transformation)");
   end Print_Usage;

   procedure Print_Error (Msg : String) is
   begin
      Put_Line (Standard_Error, "ERROR: " & Msg);
   end Print_Error;

   procedure Print_Warning (Msg : String) is
   begin
      Put_Line (Standard_Error, "WARNING: " & Msg);
   end Print_Warning;

   function Read_Stdin return String is
      Result : Unbounded_String := Null_Unbounded_String;
      Line   : String (1 .. 4096);
      Last   : Natural;
   begin
      while not End_Of_File (Standard_Input) loop
         Get_Line (Line, Last);
         Append (Result, Line (1 .. Last));
      end loop;
      return To_String (Result);
   end Read_Stdin;

   --  Canonicalize with full diagnostics
   function Canonicalize (Input : String) return String is
      Result : IR_Canonicalize_DCBOR_Utils.Canonicalize_Result;
   begin
      if Validate_Mode then
         Result := IR_Canonicalize_DCBOR_Utils.Validate_Only (Input);
      else
         Result := IR_Canonicalize_DCBOR_Utils.Canonicalize_Full (Input);
      end if;

      if not Result.Success then
         Print_Error (To_String (Result.Error_Msg));
         return "";
      end if;

      --  Report warnings
      if Result.Keys_Sorted > 0 then
         Print_Warning ("Sorted " & Natural'Image (Result.Keys_Sorted) & " object key(s)");
      end if;

      return To_String (Result.Output);
   end Canonicalize;

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
         elsif Arg = "--validate" then
            Validate_Mode := True;
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

   declare
      Input : constant String := Read_Stdin;
   begin
      if Input'Length = 0 then
         Print_Error ("Empty input");
         Set_Exit_Status (Exit_Validation_Error);
         return;
      end if;

      declare
         Output : constant String := Canonicalize (Input);
      begin
         if Output'Length = 0 then
            Set_Exit_Status (Exit_Validation_Error);
         else
            Put_Line (Output);
            Set_Exit_Status (Exit_Success);
         end if;
      end;
   exception
      when others =>
         Print_Error ("Processing error");
         Set_Exit_Status (Exit_Processing_Error);
   end;
end IR_Canonicalize_DCBOR;

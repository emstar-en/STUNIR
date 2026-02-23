--  ir_canonicalize_dcbor - Canonicalize IR JSON for dCBOR profile
--  Phase 2 Utility for STUNIR SPARK pipeline
--
--  Enforces:
--    - NFC normalization
--    - No floats
--    - No duplicate keys
--    - dCBOR canonical map ordering (deterministic JSON output)
--
--  NOTE: This tool emits canonical JSON text for IR inputs.
--  It does NOT emit CBOR bytes; CBOR emission is handled by downstream tooling.

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

   Version : constant String := "0.1.0-alpha";

    Describe_Output : constant String :=
       "{" & ASCII.LF &
       "  ""tool"": ""ir_canonicalize_dcbor""," & ASCII.LF &
       "  ""version"": ""0.1.0-alpha""," & ASCII.LF &
       "  ""description"": ""Canonicalize IR JSON for dCBOR profile""," & ASCII.LF &
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
       "  }]" & ASCII.LF &
       "}";

   procedure Print_Usage is
   begin
      Put_Line ("ir_canonicalize_dcbor - Canonicalize IR JSON for dCBOR profile");
      Put_Line ("Version: " & Version);
      Put_Line ("");
      Put_Line ("Usage: ir_canonicalize_dcbor [OPTIONS] < ir.json");
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
      while not End_Of_File (Standard_Input) loop
         Get_Line (Line, Last);
         Append (Result, Line (1 .. Last));
      end loop;
      return To_String (Result);
   end Read_Stdin;

   --  NOTE: Placeholder canonicalization. A full canonicalizer should:
   --   - parse JSON into an object model
   --   - sort all object keys lexicographically
   --   - reject duplicate keys
   --   - reject floats
   --   - normalize all strings to NFC
   --  For now, we just pass through the input.
   function Canonicalize (Input : String) return String is
   begin
      return IR_Canonicalize_DCBOR_Utils.Canonicalize (Input);
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

      Put_Line (Canonicalize (Input));
      Set_Exit_Status (Exit_Success);
   exception
      when others =>
         Print_Error ("Processing error");
         Set_Exit_Status (Exit_Processing_Error);
   end;
end IR_Canonicalize_DCBOR;

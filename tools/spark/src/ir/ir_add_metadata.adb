--  ir_add_metadata - Wrap IR functions array into a complete IR document
--  Input:  JSON array of IR functions (stdin)
--  Output: Complete IR JSON (stdout)
--  Usage:  func_to_ir < funcs.json | ir_add_metadata --module NAME > ir.json

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;

procedure IR_Add_Metadata is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;

   Exit_Success : constant := 0;
   Exit_Error   : constant := 1;

   Mod_Name   : Unbounded_String := Null_Unbounded_String;
   Docstring  : Unbounded_String := Null_Unbounded_String;
   Schema     : Unbounded_String := To_Unbounded_String ("stunir_flat_ir_v1");
   IR_Ver     : Unbounded_String := To_Unbounded_String ("0.1.0");
   Show_Help     : Boolean := False;
   Show_Version  : Boolean := False;
   Show_Describe : Boolean := False;

   Version : constant String := "0.1.0-alpha";

   Describe_Output : constant String :=
     "{""tool"":""ir_add_metadata"",""version"":""0.1.0-alpha""," &
     """description"":""Wrap IR functions array into a complete IR document""," &
     """inputs"":[{""type"":""json_array"",""source"":""stdin"",""required"":true}]," &
     """outputs"":[{""type"":""json"",""source"":""stdout""}]," &
     """options"":[""--module NAME"",""--docstring STR""," &
     """--schema NAME"",""--ir-version VER""]}";

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

   function Starts_With (S : String; Prefix : String) return Boolean is
   begin
      return S'Length >= Prefix'Length and then
             S (S'First .. S'First + Prefix'Length - 1) = Prefix;
   end Starts_With;

   function After (S : String; Prefix : String) return String is
   begin
      return S (S'First + Prefix'Length .. S'Last);
   end After;

begin
   declare
      Skip_Next : Boolean := False;
   begin
      for I in 1 .. Argument_Count loop
         if Skip_Next then
            Skip_Next := False;
         else
            declare
               Arg : constant String := Argument (I);
            begin
               if    Arg = "--help" or Arg = "-h"    then Show_Help    := True;
               elsif Arg = "--version" or Arg = "-v" then Show_Version := True;
               elsif Arg = "--describe"              then Show_Describe := True;
               elsif Starts_With (Arg, "--module=")     then Mod_Name  := To_Unbounded_String (After (Arg, "--module="));
               elsif Starts_With (Arg, "--docstring=")  then Docstring := To_Unbounded_String (After (Arg, "--docstring="));
               elsif Starts_With (Arg, "--schema=")     then Schema    := To_Unbounded_String (After (Arg, "--schema="));
               elsif Starts_With (Arg, "--ir-version=") then IR_Ver    := To_Unbounded_String (After (Arg, "--ir-version="));
               elsif Arg = "--module" and I < Argument_Count then
                  Mod_Name := To_Unbounded_String (Argument (I + 1)); Skip_Next := True;
               elsif Arg = "--docstring" and I < Argument_Count then
                  Docstring := To_Unbounded_String (Argument (I + 1)); Skip_Next := True;
               elsif Arg = "--schema" and I < Argument_Count then
                  Schema := To_Unbounded_String (Argument (I + 1)); Skip_Next := True;
               elsif Arg = "--ir-version" and I < Argument_Count then
                  IR_Ver := To_Unbounded_String (Argument (I + 1)); Skip_Next := True;
               end if;
            end;
         end if;
      end loop;
   end;

   if Show_Help then
      Put_Line ("ir_add_metadata - Wrap IR functions array into a complete IR document");
      Put_Line ("Version: " & Version);
      New_Line;
      Put_Line ("Usage: func_to_ir < funcs.json | ir_add_metadata --module NAME > ir.json");
      Put_Line ("  --module NAME       Module name (required)");
      Put_Line ("  --docstring STR     Module description");
      Put_Line ("  --schema NAME       IR schema (default: stunir_flat_ir_v1)");
      Put_Line ("  --ir-version VER    IR version (default: 0.1.0)");
      Set_Exit_Status (Exit_Success); return;
   end if;
   if Show_Version then
      Put_Line ("ir_add_metadata " & Version);
      Set_Exit_Status (Exit_Success); return;
   end if;
   if Show_Describe then
      Put_Line (Describe_Output);
      Set_Exit_Status (Exit_Success); return;
   end if;

   if Length (Mod_Name) = 0 then
      Put_Line (Standard_Error, "ERROR: --module NAME is required");
      Set_Exit_Status (Exit_Error); return;
   end if;

   declare
      Funcs  : constant String := Read_Stdin;
      Result : Unbounded_String;
   begin
      Append (Result, "{");
      Append (Result, """schema"":""" & To_String (Schema) & """");
      Append (Result, ",""ir_version"":""" & To_String (IR_Ver) & """");
      Append (Result, ",""module_name"":""" & To_String (Mod_Name) & """");
      if Length (Docstring) > 0 then
         Append (Result, ",""docstring"":""" & To_String (Docstring) & """");
      end if;
      Append (Result, ",""types"":[]");
      if Funcs'Length > 0 then
         Append (Result, ",""functions"":" & Funcs);
      else
         Append (Result, ",""functions"":[]");
      end if;
      Append (Result, "}");
      Put_Line (To_String (Result));
      Set_Exit_Status (Exit_Success);
   end;

exception
   when others =>
      Put_Line (Standard_Error, "ERROR: Processing failed");
      Set_Exit_Status (Exit_Error);
end IR_Add_Metadata;

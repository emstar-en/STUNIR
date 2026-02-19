-------------------------------------------------------------------------------
--  STUNIR Receipt Link - Ada SPARK Implementation
--  PRIMARY IMPLEMENTATION (Ada SPARK is the default language for STUNIR tools)
--
--  Links assembled specs to source indexes and AI-provided links.
--
--  Copyright (c) 2026 STUNIR Project
--  License: MIT
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with Ada.Text_IO;
with Ada.Command_Line;
with Ada.Directories;
with Ada.Strings.Fixed;
with Ada.Streams;
with Ada.Streams.Stream_IO;
with GNAT.SHA256;
with STUNIR_String_Builder;
with STUNIR_JSON_Utils;

package body STUNIR_Receipt_Link is

   use Ada.Text_IO;

   procedure Compute_Receipt_Hash
     (JSON_Text : String;
      Hash      : out Hash_String)
   is
      Context : GNAT.SHA256.Context := GNAT.SHA256.Initial_Context;
   begin
      GNAT.SHA256.Update (Context, JSON_Text);
      Hash := Hash_Strings.To_Bounded_String (GNAT.SHA256.Digest (Context));
   end Compute_Receipt_Hash;

   procedure Parse_Links
     (Links_Path : String;
      Result     : in out Receipt_Result)
   is
      File_In : Ada.Text_IO.File_Type;
      Content : String := "";
      Line    : String (1 .. 4096);
      Last    : Natural;
   begin
      Open (File_In, In_File, Links_Path);
      while not End_Of_File (File_In) loop
         Get_Line (File_In, Line, Last);
         if Last > 0 then
            if Content'Length = 0 then
               Content := Line (1 .. Last);
            else
               Content := Content & ASCII.LF & Line (1 .. Last);
            end if;
         end if;
      end loop;
      Close (File_In);

      declare
         Links_Pos : constant Natural := STUNIR_JSON_Utils.Find_Array (Content, "links");
         Obj_Start : Natural;
         Obj_End   : Natural;
         Pos       : Natural := Links_Pos + 1;
      begin
         if Links_Pos = 0 then
            return;
         end if;

         loop
            STUNIR_JSON_Utils.Get_Next_Object (Content, Pos, Obj_Start, Obj_End);
            exit when Obj_Start = 0 or Obj_End = 0;

            if Result.Receipt.Link_Count < Max_Links then
               Result.Receipt.Link_Count := Result.Receipt.Link_Count + 1;
               declare
                  Idx : constant Positive := Result.Receipt.Link_Count;
                  Obj_JSON : constant String := Content (Obj_Start .. Obj_End);
                  Spec_Element : constant String := STUNIR_JSON_Utils.Extract_String_Value (Obj_JSON, "spec_element");
                  Source_File  : constant String := STUNIR_JSON_Utils.Extract_String_Value (Obj_JSON, "source_file");
                  Source_Hash  : constant String := STUNIR_JSON_Utils.Extract_String_Value (Obj_JSON, "source_hash");
                  Source_Lines : constant String := STUNIR_JSON_Utils.Extract_String_Value (Obj_JSON, "source_lines");
                  Confidence   : constant Natural := STUNIR_JSON_Utils.Extract_Integer_Value (Obj_JSON, "confidence");
               begin
                  Result.Receipt.Links (Idx).Spec_Element := Name_Strings.To_Bounded_String (Spec_Element);
                  Result.Receipt.Links (Idx).Source_File := Path_Strings.To_Bounded_String (Source_File);
                  Result.Receipt.Links (Idx).Source_Hash := Hash_Strings.To_Bounded_String (Source_Hash);
                  Result.Receipt.Links (Idx).Source_Lines := Name_Strings.To_Bounded_String (Source_Lines);
                  Result.Receipt.Links (Idx).Confidence := Confidence;
               end;
            end if;

            Pos := Obj_End + 1;
         end loop;
      end;

   exception
      when others =>
         if Is_Open (File_In) then
            Close (File_In);
         end if;
   end Parse_Links;

   procedure Write_Receipt_JSON
     (Result : in out Receipt_Result;
      Success : out Boolean)
   is
      use STUNIR_String_Builder;
      Builder : String_Builder;
      File_Out : Ada.Text_IO.File_Type;
   begin
      Initialize (Builder);
      Append_Line (Builder, "{");
      Append_Line (Builder, "  ""kind"": ""stunir.receipt.v1"",");
      Append_Line (Builder, "  ""spec"": """ & Path_Strings.To_String (Result.Receipt.Spec_Path) & """,");
      Append_Line (Builder, "  ""index"": """ & Path_Strings.To_String (Result.Receipt.Index_Path) & """,");
      Append_Line (Builder, "  ""links"": [");

      for I in 1 .. Result.Receipt.Link_Count loop
         if I > 1 then
            Append_Line (Builder, "    ,{");
         else
            Append_Line (Builder, "    {");
         end if;
         Append_Line (Builder, "      ""spec_element"": """ & Name_Strings.To_String (Result.Receipt.Links (I).Spec_Element) & """,");
         Append_Line (Builder, "      ""source_file"": """ & Path_Strings.To_String (Result.Receipt.Links (I).Source_File) & """,");
         Append_Line (Builder, "      ""source_hash"": """ & Hash_Strings.To_String (Result.Receipt.Links (I).Source_Hash) & """,");
         Append_Line (Builder, "      ""source_lines"": """ & Name_Strings.To_String (Result.Receipt.Links (I).Source_Lines) & """,");
         Append_Line (Builder, "      ""confidence"": " & Natural'Image (Result.Receipt.Links (I).Confidence));
         Append_Line (Builder, "    }");
      end loop;

      Append_Line (Builder, "  ]");
      Append_Line (Builder, "}");

      declare
         JSON_Text : constant String := To_String (Builder);
      begin
         Compute_Receipt_Hash (JSON_Text, Result.Receipt.Receipt_Hash);
      end;

      Create (File_Out, Out_File, Path_Strings.To_String (Result.Receipt.Output_Path));
      Put (File_Out, To_String (Builder));
      Close (File_Out);
      Success := True;

   exception
      when others =>
         if Is_Open (File_Out) then
            Close (File_Out);
         end if;
         Success := False;
   end Write_Receipt_JSON;

   procedure Link_Receipt
     (Spec_Path  : String;
      Index_Path : String;
      Links_Path : String;
      Output_Path : String;
      Result : in out Receipt_Result)
   is
      Output_OK : Boolean;
   begin
      Result.Status := Success;
      Result.Receipt.Spec_Path := Path_Strings.To_Bounded_String (Spec_Path);
      Result.Receipt.Index_Path := Path_Strings.To_Bounded_String (Index_Path);
      Result.Receipt.Links_Path := Path_Strings.To_Bounded_String (Links_Path);
      Result.Receipt.Output_Path := Path_Strings.To_Bounded_String (Output_Path);
      Result.Receipt.Link_Count := 0;

      if not Ada.Directories.Exists (Spec_Path) or else not Ada.Directories.Exists (Index_Path)
        or else not Ada.Directories.Exists (Links_Path) then
         Result.Status := Error_Input_Not_Found;
         return;
      end if;

      Parse_Links (Links_Path, Result);
      Write_Receipt_JSON (Result, Output_OK);
      if not Output_OK then
         Result.Status := Error_Output_Failed;
      end if;
   end Link_Receipt;

   procedure Run_Receipt_Link is
      Result : Receipt_Result;
      Arg_Count : constant Natural := Ada.Command_Line.Argument_Count;
      Spec_Path : String := "";
      Index_Path : String := "";
      Links_Path : String := "";
      Output_Path : String := "";
   begin
      for I in 1 .. Arg_Count loop
         declare
            Arg : constant String := Ada.Command_Line.Argument (I);
         begin
            if Arg = "--spec" and then I < Arg_Count then
               Spec_Path := Ada.Command_Line.Argument (I + 1);
            elsif Arg = "--index" and then I < Arg_Count then
               Index_Path := Ada.Command_Line.Argument (I + 1);
            elsif Arg = "--links" and then I < Arg_Count then
               Links_Path := Ada.Command_Line.Argument (I + 1);
            elsif Arg = "--output" and then I < Arg_Count then
               Output_Path := Ada.Command_Line.Argument (I + 1);
            end if;
         end;
      end loop;

      if Spec_Path'Length = 0 or else Index_Path'Length = 0 or else Links_Path'Length = 0 or else Output_Path'Length = 0 then
         Put_Line ("Usage: stunir_receipt_link --spec <spec.json> --index <index.json> --links <links.json> --output <receipt.json>");
         return;
      end if;

      Link_Receipt (Spec_Path, Index_Path, Links_Path, Output_Path, Result);
      if Result.Status /= Success then
         Put_Line ("[ERROR] Receipt link failed");
      end if;
   end Run_Receipt_Link;

end STUNIR_Receipt_Link;

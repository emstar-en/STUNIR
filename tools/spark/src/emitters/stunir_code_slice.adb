-------------------------------------------------------------------------------
--  STUNIR Code Slice - Ada SPARK Implementation
--  PRIMARY IMPLEMENTATION (Ada SPARK is the default language for STUNIR tools)
--
--  Slices a source file into detected code regions for AI extraction.
--
--  Copyright (c) 2026 STUNIR Project
--  License: MIT
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with Ada.Text_IO;
with Ada.Command_Line;
with Ada.Directories;
with Ada.Strings.Fixed;
with Ada.Strings.Unbounded;
with GNAT.SHA256;
with STUNIR_String_Builder;
with STUNIR_JSON;

package body STUNIR_Code_Slice is

   use Ada.Text_IO;

   function To_Lower (S : String) return String is
      R : String := S;
   begin
      for I in R'Range loop
         if R (I) in 'A' .. 'Z' then
            R (I) := Character'Val (Character'Pos (R (I)) + 32);
         end if;
      end loop;
      return R;
   end To_Lower;

   function Is_Function_Start (Line : String) return Boolean is
      L : constant String := To_Lower (Line);
   begin
      return (Ada.Strings.Fixed.Index (L, "(") > 0 and then Ada.Strings.Fixed.Index (L, ")") > 0)
        and then (Ada.Strings.Fixed.Index (L, "{") > 0)
        and then (Ada.Strings.Fixed.Index (L, ";") = 0);
   end Is_Function_Start;

   function Is_Type_Start (Line : String) return Boolean is
      L : constant String := To_Lower (Line);
   begin
      return Ada.Strings.Fixed.Index (L, "struct") > 0 or else
             Ada.Strings.Fixed.Index (L, "enum") > 0 or else
             Ada.Strings.Fixed.Index (L, "typedef") > 0;
   end Is_Type_Start;

   function Is_Constant_Start (Line : String) return Boolean is
      L : constant String := To_Lower (Line);
   begin
      return Ada.Strings.Fixed.Index (L, "#define") > 0 or else
             Ada.Strings.Fixed.Index (L, "const ") > 0;
   end Is_Constant_Start;

   procedure Compute_Region_Hash
     (Lines : String;
      Hash  : out Hash_String)
   is
      Context : GNAT.SHA256.Context := GNAT.SHA256.Initial_Context;
   begin
      GNAT.SHA256.Update (Context, Lines);
      Hash := Hash_Strings.To_Bounded_String (GNAT.SHA256.Digest (Context));
   end Compute_Region_Hash;

   procedure Write_Slice_JSON
     (Slice : Sliced_File;
      Output_Path : String;
      Success : out Boolean)
   is
      use STUNIR_String_Builder;
      Builder : String_Builder;
      File_Out : Ada.Text_IO.File_Type;
      function Region_Type_To_String (Kind : Region_Type) return String is
      begin
         case Kind is
            when REGION_FUNCTION => return "REGION_FUNCTION";
            when REGION_TYPE_DEF => return "REGION_TYPE";
            when REGION_CONSTANT => return "REGION_CONSTANT";
            when others => return "REGION_UNKNOWN";
         end case;
      end Region_Type_To_String;
   begin
      Initialize (Builder);
      Append_Line (Builder, "{");
      Append_Line (Builder, "  \"kind\": \"stunir.code_slice.v1\",");
      Append_Line (Builder, "  \"file_path\": \"" & Path_Strings.To_String (Slice.File_Path) & "\",");
      Append_Line (Builder, "  \"file_hash\": \"" & Hash_Strings.To_String (Slice.File_Hash) & "\",");
      Append_Line (Builder, "  \"language\": \"" & Lang_Strings.To_String (Slice.Language) & "\",");
      Append_Line (Builder, "  \"region_count\": " & Natural'Image (Slice.Region_Count) & ",");
      Append_Line (Builder, "  \"regions\": [");

      for I in 1 .. Slice.Region_Count loop
         if I > 1 then
            Append_Line (Builder, "    ,{");
         else
            Append_Line (Builder, "    {");
         end if;

         Append_Line (Builder, "      \"start_line\": " & Natural'Image (Slice.Regions (I).Start_Line) & ",");
         Append_Line (Builder, "      \"end_line\": " & Natural'Image (Slice.Regions (I).End_Line) & ",");
         Append_Line (Builder, "      \"region_type\": \"" & Region_Type_To_String (Slice.Regions (I).Region_Kind) & "\",");
         Append_Line (Builder, "      \"content_hash\": \"" & Hash_Strings.To_String (Slice.Regions (I).Content_Hash) & "\"");
         Append_Line (Builder, "    }");
      end loop;

      Append_Line (Builder, "  ]");
      Append_Line (Builder, "}");

      Create (File_Out, Out_File, Output_Path);
      Put (File_Out, To_String (Builder));
      Close (File_Out);
      Success := True;

   exception
      when others =>
         if Is_Open (File_Out) then
            Close (File_Out);
         end if;
         Success := False;
   end Write_Slice_JSON;

   procedure Slice_File
     (Config : Slice_Config;
      Result : in out Slice_Result)
   is
      File_In : Ada.Text_IO.File_Type;
      Line    : String (1 .. 4096);
      Last    : Natural;
      Line_No : Natural := 0;
      Current_Start : Natural := 0;
      Current_Kind : Region_Type := REGION_UNKNOWN;
      Region_Lines : Ada.Strings.Unbounded.Unbounded_String := Ada.Strings.Unbounded.Null_Unbounded_String;
      Hash_OK : Boolean;
      File_Hash : STUNIR_JSON.Hash_String;
   begin
      Result.Status := Success;
      Result.Slice.Region_Count := 0;

      if not Ada.Directories.Exists (Path_Strings.To_String (Config.Input_File)) then
         Result.Status := Error_Input_Not_Found;
         return;
      end if;

      STUNIR_JSON.Compute_File_Hash (Path_Strings.To_String (Config.Input_File), File_Hash, Hash_OK);
      if Hash_OK then
         Result.Slice.File_Hash := Hash_Strings.To_Bounded_String (STUNIR_JSON.Hash_Strings.To_String (File_Hash));
      else
         Result.Slice.File_Hash := Hash_Strings.Null_Bounded_String;
      end if;

      Result.Slice.File_Path := Config.Input_File;

      declare
         Path : constant String := Path_Strings.To_String (Config.Input_File);
         Lower : constant String := To_Lower (Path);
         function Detect_Language (Ext_Path : String) return String is
         begin
            if Ext_Path'Length >= 2 and then Ext_Path (Ext_Path'Last - 1 .. Ext_Path'Last) = ".c" then
               return "LANG_C";
            elsif Ext_Path'Length >= 4 and then Ext_Path (Ext_Path'Last - 3 .. Ext_Path'Last) = ".cpp" then
               return "LANG_CPP";
            elsif Ext_Path'Length >= 4 and then Ext_Path (Ext_Path'Last - 3 .. Ext_Path'Last) = ".adb" then
               return "LANG_ADA";
            elsif Ext_Path'Length >= 3 and then Ext_Path (Ext_Path'Last - 2 .. Ext_Path'Last) = ".py" then
               return "LANG_PYTHON";
            else
               return "LANG_UNKNOWN";
            end if;
         end Detect_Language;
      begin
         Result.Slice.Language := Lang_Strings.To_Bounded_String (Detect_Language (Lower));
      end;

      Open (File_In, In_File, Path_Strings.To_String (Config.Input_File));
      while not End_Of_File (File_In) loop
         Get_Line (File_In, Line, Last);
         Line_No := Line_No + 1;

         declare
            L : constant String := (if Last > 0 then Line (1 .. Last) else "");
         begin
            if Is_Function_Start (L) then
               if Result.Slice.Region_Count < Max_Regions then
                  Current_Start := Line_No;
                  Current_Kind := REGION_FUNCTION;
                  Region_Lines := Ada.Strings.Unbounded.To_Unbounded_String (L & ASCII.LF);
               end if;
            elsif Is_Type_Start (L) then
               if Result.Slice.Region_Count < Max_Regions then
                  Current_Start := Line_No;
                  Current_Kind := REGION_TYPE_DEF;
                  Region_Lines := Ada.Strings.Unbounded.To_Unbounded_String (L & ASCII.LF);
               end if;
            elsif Is_Constant_Start (L) then
               if Result.Slice.Region_Count < Max_Regions then
                  Current_Start := Line_No;
                  Current_Kind := REGION_CONSTANT;
                  Region_Lines := Ada.Strings.Unbounded.To_Unbounded_String (L & ASCII.LF);
               end if;
            end if;

            if Current_Start > 0 and then (Ada.Strings.Fixed.Index (L, "}") > 0 or else Current_Kind = REGION_CONSTANT) then
               Result.Slice.Region_Count := Result.Slice.Region_Count + 1;
               declare
                  Idx : constant Positive := Result.Slice.Region_Count;
               begin
                  Result.Slice.Regions (Idx).Start_Line := Current_Start;
                  Result.Slice.Regions (Idx).End_Line := Line_No;
                  Result.Slice.Regions (Idx).Region_Kind := Current_Kind;
                  Compute_Region_Hash (Ada.Strings.Unbounded.To_String (Region_Lines), Result.Slice.Regions (Idx).Content_Hash);
               end;
               Current_Start := 0;
               Current_Kind := REGION_UNKNOWN;
               Region_Lines := Ada.Strings.Unbounded.Null_Unbounded_String;
            elsif Current_Start > 0 then
               Ada.Strings.Unbounded.Append (Region_Lines, L & ASCII.LF);
            end if;
         end;
      end loop;
      Close (File_In);

   exception
      when others =>
         if Is_Open (File_In) then
            Close (File_In);
         end if;
         Result.Status := Error_Parse_Failed;
   end Slice_File;

   procedure Run_Code_Slice is
      Config : Slice_Config;
      Result : Slice_Result;
      Output_OK : Boolean;
      Arg_Count : constant Natural := Ada.Command_Line.Argument_Count;
      Input_Path : Ada.Strings.Unbounded.Unbounded_String := Ada.Strings.Unbounded.Null_Unbounded_String;
      Output_Path : Ada.Strings.Unbounded.Unbounded_String := Ada.Strings.Unbounded.Null_Unbounded_String;
      Index_Path : Ada.Strings.Unbounded.Unbounded_String := Ada.Strings.Unbounded.Null_Unbounded_String;
   begin
      for I in 1 .. Arg_Count loop
         declare
            Arg : constant String := Ada.Command_Line.Argument (I);
         begin
            if Arg = "--input" and then I < Arg_Count then
               Input_Path := Ada.Strings.Unbounded.To_Unbounded_String (Ada.Command_Line.Argument (I + 1));
            elsif Arg = "--output" and then I < Arg_Count then
               Output_Path := Ada.Strings.Unbounded.To_Unbounded_String (Ada.Command_Line.Argument (I + 1));
            elsif Arg = "--index" and then I < Arg_Count then
               Index_Path := Ada.Strings.Unbounded.To_Unbounded_String (Ada.Command_Line.Argument (I + 1));
            end if;
         end;
      end loop;

      if Ada.Strings.Unbounded.Length (Input_Path) = 0 or else Ada.Strings.Unbounded.Length (Output_Path) = 0 then
         Put_Line ("Usage: stunir_code_slice --input <file.c> --output <slice.json> [--index <index.json>]");
         return;
      end if;

      Config.Input_File := Path_Strings.To_Bounded_String (Ada.Strings.Unbounded.To_String (Input_Path));
      Config.Output_Path := Path_Strings.To_Bounded_String (Ada.Strings.Unbounded.To_String (Output_Path));
      if Ada.Strings.Unbounded.Length (Index_Path) > 0 then
         Config.Index_Path := Path_Strings.To_Bounded_String (Ada.Strings.Unbounded.To_String (Index_Path));
         Config.Has_Index := True;
      end if;

      Slice_File (Config, Result);
      if Result.Status /= Success then
         Put_Line ("[ERROR] Code slice failed");
         return;
      end if;

      Write_Slice_JSON (Result.Slice, Ada.Strings.Unbounded.To_String (Output_Path), Output_OK);
      if not Output_OK then
         Put_Line ("[ERROR] Failed to write slice JSON");
      end if;
   end Run_Code_Slice;

end STUNIR_Code_Slice;

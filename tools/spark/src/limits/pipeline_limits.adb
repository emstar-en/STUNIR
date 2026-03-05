--  Pipeline Limits Configuration - Implementation
--  Provides configurable limits for STUNIR pipeline tools

with Ada.Text_IO; use Ada.Text_IO;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Strings.Fixed;

package body Pipeline_Limits is

   --  ========================================================================
   --  Helper: Parse integer from string safely
   --  ========================================================================

   function Safe_To_Integer (S : String) return Integer is
      Result : Integer;
      Trimmed : constant String := Ada.Strings.Fixed.Trim (S, Ada.Strings.Both);
   begin
      if Trimmed'Length = 0 then
         return -1;  --  Invalid
      end if;
      begin
         Result := Integer'Value (Trimmed);
         return Result;
      exception
         when others =>
            return -1;  --  Invalid
      end;
   end Safe_To_Integer;

   --  ========================================================================
   --  Default Limits
   --  ========================================================================

   function Get_Default_Limits return Limits_Record is
   begin
      return (Max_Parameters    => Default_Max_Parameters,
              Max_Statements    => Default_Max_Statements,
              Max_Functions     => Default_Max_Functions,
              Max_Type_Defs     => Default_Max_Type_Defs,
              Max_Case_Entries  => Default_Max_Case_Entries,
              Max_Catch_Blocks  => Default_Max_Catch_Blocks,
              Max_Imports       => Default_Max_Imports,
              Max_Exports       => Default_Max_Exports,
              Max_Type_Fields   => Default_Max_Type_Fields,
              Max_Constants     => Default_Max_Constants,
              Max_Dependencies  => Default_Max_Dependencies,
              Max_Steps         => Default_Max_Steps,
              Max_Block_Depth   => Default_Max_Block_Depth,
              Max_GPU_Binaries  => Default_Max_GPU_Binaries,
              Max_Microcode     => Default_Max_Microcode);
   end Get_Default_Limits;

   --  ========================================================================
   --  Current Limits Accessors
   --  ========================================================================

   function Get_Current_Limits return Limits_Record is
   begin
      return Current_Limits;
   end Get_Current_Limits;

   procedure Set_Current_Limits (Limits : Limits_Record) is
   begin
      Current_Limits := Limits;

      --  Emit warnings for limits exceeding thresholds
      if Exceeds_Warning ("Max_Parameters", Limits.Max_Parameters) then
         Put_Line (Standard_Error, "[WARN] Max_Parameters=" 
                   & Positive'Image (Limits.Max_Parameters) 
                   & " exceeds warning threshold of" 
                   & Positive'Image (Warn_Parameters));
      end if;

      if Exceeds_Warning ("Max_Statements", Limits.Max_Statements) then
         Put_Line (Standard_Error, "[WARN] Max_Statements=" 
                   & Positive'Image (Limits.Max_Statements) 
                   & " exceeds warning threshold of" 
                   & Positive'Image (Warn_Statements));
      end if;

      if Exceeds_Warning ("Max_Functions", Limits.Max_Functions) then
         Put_Line (Standard_Error, "[WARN] Max_Functions=" 
                   & Positive'Image (Limits.Max_Functions) 
                   & " exceeds warning threshold of" 
                   & Positive'Image (Warn_Functions));
      end if;
   end Set_Current_Limits;

   procedure Reset_Limits is
   begin
      Current_Limits := Get_Default_Limits;
   end Reset_Limits;

   --  ========================================================================
   --  Warning Helpers
   --  ========================================================================

   function Exceeds_Warning (Limit_Name : String; Value : Positive) return Boolean is
      Threshold : constant Natural := Get_Warning_Threshold (Limit_Name);
   begin
      --  Unknown limits don't trigger warnings
      if Threshold = 0 then
         return False;
      end if;
      return Value > Threshold;
   end Exceeds_Warning;

   function Get_Warning_Threshold (Limit_Name : String) return Natural is
      Name : constant String := Ada.Strings.Fixed.Trim (Limit_Name, Ada.Strings.Both);
   begin
      if Name = "Max_Parameters" then
         return Warn_Parameters;
      elsif Name = "Max_Statements" then
         return Warn_Statements;
      elsif Name = "Max_Functions" then
         return Warn_Functions;
      elsif Name = "Max_Type_Defs" then
         return Warn_Type_Defs;
      elsif Name = "Max_Case_Entries" then
         return Warn_Case_Entries;
      elsif Name = "Max_Catch_Blocks" then
         return Warn_Catch_Blocks;
      elsif Name = "Max_Imports" then
         return Warn_Imports;
      elsif Name = "Max_Exports" then
         return Warn_Exports;
      elsif Name = "Max_Type_Fields" then
         return Warn_Type_Fields;
      elsif Name = "Max_Constants" then
         return Warn_Constants;
      elsif Name = "Max_Dependencies" then
         return Warn_Dependencies;
      elsif Name = "Max_Steps" then
         return Warn_Steps;
      elsif Name = "Max_Block_Depth" then
         return Warn_Block_Depth;
      elsif Name = "Max_GPU_Binaries" then
         return Warn_GPU_Binaries;
      elsif Name = "Max_Microcode" then
         return Warn_Microcode;
      else
         return 0;  --  Unknown limit name
      end if;
   end Get_Warning_Threshold;

   --  ========================================================================
   --  JSON Generation
   --  ========================================================================

   function Generate_Limits_Json (Limits : Limits_Record) return String is
      Result : Unbounded_String := To_Unbounded_String ("");
   begin
      Append (Result, "{" & ASCII.LF);
      Append (Result, "  ""limits"": {" & ASCII.LF);
      Append (Result, "    ""Max_Parameters"": " 
              & Positive'Image (Limits.Max_Parameters) & "," & ASCII.LF);
      Append (Result, "    ""Max_Statements"": " 
              & Positive'Image (Limits.Max_Statements) & "," & ASCII.LF);
      Append (Result, "    ""Max_Functions"": " 
              & Positive'Image (Limits.Max_Functions) & "," & ASCII.LF);
      Append (Result, "    ""Max_Type_Defs"": " 
              & Positive'Image (Limits.Max_Type_Defs) & "," & ASCII.LF);
      Append (Result, "    ""Max_Case_Entries"": " 
              & Positive'Image (Limits.Max_Case_Entries) & "," & ASCII.LF);
      Append (Result, "    ""Max_Catch_Blocks"": " 
              & Positive'Image (Limits.Max_Catch_Blocks) & "," & ASCII.LF);
      Append (Result, "    ""Max_Imports"": " 
              & Positive'Image (Limits.Max_Imports) & "," & ASCII.LF);
      Append (Result, "    ""Max_Exports"": " 
              & Positive'Image (Limits.Max_Exports) & "," & ASCII.LF);
      Append (Result, "    ""Max_Type_Fields"": " 
              & Positive'Image (Limits.Max_Type_Fields) & "," & ASCII.LF);
      Append (Result, "    ""Max_Constants"": " 
              & Positive'Image (Limits.Max_Constants) & "," & ASCII.LF);
      Append (Result, "    ""Max_Dependencies"": " 
              & Positive'Image (Limits.Max_Dependencies) & "," & ASCII.LF);
      Append (Result, "    ""Max_Steps"": " 
              & Positive'Image (Limits.Max_Steps) & "," & ASCII.LF);
      Append (Result, "    ""Max_Block_Depth"": " 
              & Positive'Image (Limits.Max_Block_Depth) & "," & ASCII.LF);
      Append (Result, "    ""Max_GPU_Binaries"": " 
              & Positive'Image (Limits.Max_GPU_Binaries) & "," & ASCII.LF);
      Append (Result, "    ""Max_Microcode"": " 
              & Positive'Image (Limits.Max_Microcode) & ASCII.LF);
      Append (Result, "  }" & ASCII.LF);
      Append (Result, "}");
      return To_String (Result);
   end Generate_Limits_Json;

   --  ========================================================================
   --  JSON Parsing (Simple key-value extraction)
   --  ========================================================================

   function Parse_Limits_Json (Json_Str : String) return Limits_Record is
      Defaults : constant Limits_Record := Get_Default_Limits;
      Result   : Limits_Record := Defaults;

      --  Simple JSON value extraction
      function Extract_Int (Json : String; Key : String) return Integer is
         Search_Key : constant String := """" & Key & """:";
         Key_Start  : Integer;
         Value_Start : Integer;
         Value_End   : Integer;
      begin
         Key_Start := Ada.Strings.Fixed.Index (Json, Search_Key);
         if Key_Start = 0 then
            return -1;  --  Not found, use default
         end if;

         --  Find value after colon
         Value_Start := Key_Start + Search_Key'Length;
         --  Skip whitespace
         while Value_Start in Json'Range 
           and then Json (Value_Start) in ' ' | ASCII.HT | ASCII.LF | ASCII.CR loop
            Value_Start := Value_Start + 1;
         end loop;

         --  Find end of number
         Value_End := Value_Start;
         while Value_End in Json'Range 
           and then Json (Value_End) in '0' .. '9' | '-' loop
            Value_End := Value_End + 1;
         end loop;
         Value_End := Value_End - 1;

         if Value_Start > Value_End then
            return -1;  --  Invalid
         end if;

         return Safe_To_Integer (Json (Value_Start .. Value_End));
      end Extract_Int;

      Val : Integer;
   begin
      --  Parse each limit from JSON, use default if not found
      Val := Extract_Int (Json_Str, "Max_Parameters");
      if Val > 0 then
         Result.Max_Parameters := Val;
      end if;

      Val := Extract_Int (Json_Str, "Max_Statements");
      if Val > 0 then
         Result.Max_Statements := Val;
      end if;

      Val := Extract_Int (Json_Str, "Max_Functions");
      if Val > 0 then
         Result.Max_Functions := Val;
      end if;

      Val := Extract_Int (Json_Str, "Max_Type_Defs");
      if Val > 0 then
         Result.Max_Type_Defs := Val;
      end if;

      Val := Extract_Int (Json_Str, "Max_Case_Entries");
      if Val > 0 then
         Result.Max_Case_Entries := Val;
      end if;

      Val := Extract_Int (Json_Str, "Max_Catch_Blocks");
      if Val > 0 then
         Result.Max_Catch_Blocks := Val;
      end if;

      Val := Extract_Int (Json_Str, "Max_Imports");
      if Val > 0 then
         Result.Max_Imports := Val;
      end if;

      Val := Extract_Int (Json_Str, "Max_Exports");
      if Val > 0 then
         Result.Max_Exports := Val;
      end if;

      Val := Extract_Int (Json_Str, "Max_Type_Fields");
      if Val > 0 then
         Result.Max_Type_Fields := Val;
      end if;

      Val := Extract_Int (Json_Str, "Max_Constants");
      if Val > 0 then
         Result.Max_Constants := Val;
      end if;

      Val := Extract_Int (Json_Str, "Max_Dependencies");
      if Val > 0 then
         Result.Max_Dependencies := Val;
      end if;

      Val := Extract_Int (Json_Str, "Max_Steps");
      if Val > 0 then
         Result.Max_Steps := Val;
      end if;

      Val := Extract_Int (Json_Str, "Max_Block_Depth");
      if Val > 0 then
         Result.Max_Block_Depth := Val;
      end if;

      Val := Extract_Int (Json_Str, "Max_GPU_Binaries");
      if Val > 0 then
         Result.Max_GPU_Binaries := Val;
      end if;

      Val := Extract_Int (Json_Str, "Max_Microcode");
      if Val > 0 then
         Result.Max_Microcode := Val;
      end if;

      return Result;
   end Parse_Limits_Json;

   --  ========================================================================
   --  Receipt Loading
   --  ========================================================================

   function Load_Limits_From_Receipt (Receipt_Path : String) return Limits_Record is
      File    : File_Type;
      Content : Unbounded_String;
      Line    : String (1 .. 4096);
      Last    : Natural;
   begin
      --  Try to open receipt file
      begin
         Open (File, In_File, Receipt_Path);
      exception
         when others =>
            --  File not found, use defaults
            Put_Line (Standard_Error, "[INFO] Receipt not found: " 
                      & Receipt_Path & ", using default limits");
            return Get_Default_Limits;
      end;

      --  Read entire file
      while not End_Of_File (File) loop
         Get_Line (File, Line, Last);
         Append (Content, Line (1 .. Last));
         if not End_Of_File (File) then
            Append (Content, ASCII.LF);
         end if;
      end loop;
      Close (File);

      --  Parse limits from JSON
      declare
         Limits : constant Limits_Record := Parse_Limits_Json (To_String (Content));
      begin
         --  Emit warnings for high limits
         if Exceeds_Warning ("Max_Parameters", Limits.Max_Parameters) then
            Put_Line (Standard_Error, "[WARN] Loaded Max_Parameters=" 
                      & Positive'Image (Limits.Max_Parameters) 
                      & " exceeds threshold");
         end if;

         return Limits;
      end;

   exception
      when others =>
         Put_Line (Standard_Error, "[ERROR] Failed to parse receipt: " 
                   & Receipt_Path & ", using default limits");
         return Get_Default_Limits;
   end Load_Limits_From_Receipt;

end Pipeline_Limits;
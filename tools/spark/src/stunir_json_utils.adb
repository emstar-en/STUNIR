-------------------------------------------------------------------------------
--  STUNIR JSON Utilities - Ada SPARK Implementation Body
--  PRIMARY IMPLEMENTATION (Ada SPARK is the default language for STUNIR tools)
--
--  Lightweight JSON parsing and generation for DO-178C Level A compliance
--
--  Copyright (c) 2026 STUNIR Project
--  License: MIT
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with Ada.Text_IO;
with Ada.Exceptions;
with GNAT.SHA256;

package body STUNIR_JSON_Utils is

   use Ada.Text_IO;

   --  Helper: Find substring in JSON
   function Find_Field (JSON_Text : String; Field : String) return Natural is
      Search_Str : constant String := """" & Field & """:";
   begin
      for I in JSON_Text'First .. JSON_Text'Last - Search_Str'Length + 1 loop
         if JSON_Text (I .. I + Search_Str'Length - 1) = Search_Str then
            return I + Search_Str'Length;
         end if;
      end loop;
      return 0;
   end Find_Field;

   --  Helper: Extract string value after position
   function Get_String_After (JSON_Text : String; Pos : Positive) return String is
      Start_Pos : Natural := 0;
      End_Pos   : Natural := 0;
   begin
      --  Skip whitespace and find opening quote
      for I in Pos .. JSON_Text'Last loop
         if JSON_Text (I) = '"' then
            Start_Pos := I + 1;
            exit;
         end if;
      end loop;

      if Start_Pos = 0 then
         return "";
      end if;

      --  Find closing quote
      for I in Start_Pos .. JSON_Text'Last loop
         if JSON_Text (I) = '"' then
            End_Pos := I - 1;
            exit;
         end if;
      end loop;

      if End_Pos = 0 or End_Pos < Start_Pos then
         return "";
      end if;

      return JSON_Text (Start_Pos .. End_Pos);
   end Get_String_After;

   --  Extract string value from JSON field
   function Extract_String_Value
     (JSON_Text : String;
      Field_Name : String) return String
   is
      Pos : constant Natural := Find_Field (JSON_Text, Field_Name);
   begin
      if Pos = 0 then
         return "";
      end if;
      return Get_String_After (JSON_Text, Pos);
   end Extract_String_Value;

   --  Helper: Find array in JSON
   function Find_Array (JSON_Text : String; Field : String) return Natural is
      Search_Str : constant String := """" & Field & """:";
      Pos : Natural;
   begin
      Pos := Find_Field (JSON_Text, Field);
      if Pos = 0 then
         return 0;
      end if;
      
      -- Skip whitespace and find opening bracket
      for I in Pos .. JSON_Text'Last loop
         if JSON_Text (I) = '[' then
            return I;
         elsif JSON_Text (I) /= ' ' and JSON_Text (I) /= ASCII.HT and JSON_Text (I) /= ASCII.LF then
            return 0;  -- Not an array
         end if;
      end loop;
      return 0;
   end Find_Array;

   --  Helper: Extract object from array at position
   procedure Get_Next_Object
     (JSON_Text : String;
      Start_Pos : Natural;
      Obj_Start : out Natural;
      Obj_End   : out Natural)
   is
      Depth : Integer := 0;
      In_Obj : Boolean := False;
   begin
      Obj_Start := 0;
      Obj_End := 0;
      
      for I in Start_Pos .. JSON_Text'Last loop
         if JSON_Text (I) = '{' then
            if not In_Obj then
               Obj_Start := I;
               In_Obj := True;
            end if;
            Depth := Depth + 1;
         elsif JSON_Text (I) = '}' and Depth > 0 then
            Depth := Depth - 1;
            if Depth = 0 and In_Obj then
               Obj_End := I;
               return;
            end if;
         end if;
      end loop;
   end Get_Next_Object;

   --  Parse simple JSON spec into IR Module
   procedure Parse_Spec_JSON
     (JSON_Text : String;
      Module    : out IR_Module;
      Status    : out Parse_Status)
   is
      -- Extract fields from spec
      Module_Name_Str : constant String := Extract_String_Value (JSON_Text, "module");
      Desc_Str : constant String := Extract_String_Value (JSON_Text, "description");
      Funcs_Array_Pos : constant Natural := Find_Array (JSON_Text, "functions");
   begin
      Status := Error_Invalid_JSON;
      
      --  Initialize module with minimal defaults (avoid stack overflow)
      Module.IR_Version := "v1";
      Module.Module_Name := Name_Strings.Null_Bounded_String;
      Module.Docstring := Doc_Strings.Null_Bounded_String;
      Module.Type_Cnt := 0;
      Module.Func_Cnt := 0;

      --  Extract module name (try "module" first, then "name")
      if Module_Name_Str'Length = 0 then
         declare
            Alt_Name : constant String := Extract_String_Value (JSON_Text, "name");
         begin
            if Alt_Name'Length = 0 then
               Put_Line ("[ERROR] Missing 'module' or 'name' field in JSON spec");
               Status := Error_Missing_Field;
               return;
            end if;
            Module.Module_Name := Name_Strings.To_Bounded_String (Alt_Name);
         end;
      else
         if Module_Name_Str'Length > Max_Name_Length then
            Put_Line ("[ERROR] Module name too long");
            Status := Error_Too_Large;
            return;
         end if;
         Module.Module_Name := Name_Strings.To_Bounded_String (Module_Name_Str);
      end if;

      --  Extract docstring
      if Desc_Str'Length > 0 and Desc_Str'Length <= Max_Doc_Length then
         Module.Docstring := Doc_Strings.To_Bounded_String (Desc_Str);
      end if;

      --  Parse functions array
      if Funcs_Array_Pos > 0 then
         declare
            Func_Pos : Natural := Funcs_Array_Pos + 1;
            Obj_Start, Obj_End : Natural;
         begin
            --  Parse each function object in array
            while Module.Func_Cnt < Max_Functions loop
               Get_Next_Object (JSON_Text, Func_Pos, Obj_Start, Obj_End);
               exit when Obj_Start = 0 or Obj_End = 0;
               
               declare
                  Func_JSON : constant String := JSON_Text (Obj_Start .. Obj_End);
                  Func_Name : constant String := Extract_String_Value (Func_JSON, "name");
                  Func_Returns : constant String := Extract_String_Value (Func_JSON, "returns");
                  Params_Pos : constant Natural := Find_Array (Func_JSON, "params");
                  Body_Pos : constant Natural := Find_Array (Func_JSON, "body");
               begin
                  if Func_Name'Length > 0 then
                     Module.Func_Cnt := Module.Func_Cnt + 1;
                     Module.Functions (Module.Func_Cnt).Name := 
                       Name_Strings.To_Bounded_String (Func_Name);
                     
                     -- Set return type
                     if Func_Returns'Length > 0 then
                        Module.Functions (Module.Func_Cnt).Return_Type := 
                          Type_Strings.To_Bounded_String (Func_Returns);
                     else
                        Module.Functions (Module.Func_Cnt).Return_Type := 
                          Type_Strings.To_Bounded_String ("void");
                     end if;
                     
                     Module.Functions (Module.Func_Cnt).Arg_Cnt := 0;
                     Module.Functions (Module.Func_Cnt).Stmt_Cnt := 0;
                     Module.Functions (Module.Func_Cnt).Docstring := Doc_Strings.Null_Bounded_String;
                     
                     -- Parse params (simplified - just count for now)
                     if Params_Pos > 0 then
                        declare
                           Param_Pos : Natural := Params_Pos + 1;
                           Param_Start, Param_End : Natural;
                           Func_Idx : constant Positive := Module.Func_Cnt;
                        begin
                           while Module.Functions (Func_Idx).Arg_Cnt < Max_Args loop
                              Get_Next_Object (Func_JSON, Param_Pos, Param_Start, Param_End);
                              exit when Param_Start = 0 or Param_End = 0;
                              
                              declare
                                 Param_JSON : constant String := Func_JSON (Param_Start .. Param_End);
                                 Param_Name : constant String := Extract_String_Value (Param_JSON, "name");
                                 Param_Type : constant String := Extract_String_Value (Param_JSON, "type");
                              begin
                                 if Param_Name'Length > 0 then
                                    Module.Functions (Func_Idx).Arg_Cnt := Module.Functions (Func_Idx).Arg_Cnt + 1;
                                    Module.Functions (Func_Idx).Args (Module.Functions (Func_Idx).Arg_Cnt).Name :=
                                      Name_Strings.To_Bounded_String (Param_Name);
                                    if Param_Type'Length > 0 then
                                       Module.Functions (Func_Idx).Args (Module.Functions (Func_Idx).Arg_Cnt).Type_Ref :=
                                         Type_Strings.To_Bounded_String (Param_Type);
                                    else
                                       Module.Functions (Func_Idx).Args (Module.Functions (Func_Idx).Arg_Cnt).Type_Ref :=
                                         Type_Strings.To_Bounded_String ("i32");
                                    end if;
                                 end if;
                              end;
                              
                              Param_Pos := Param_End + 1;
                           end loop;
                        end;
                     end if;
                     
                     -- Parse body statements (simplified - just count for now)
                     if Body_Pos > 0 then
                        declare
                           Stmt_Pos : Natural := Body_Pos + 1;
                           Stmt_Start, Stmt_End : Natural;
                           Func_Idx : constant Positive := Module.Func_Cnt;
                        begin
                           while Module.Functions (Func_Idx).Stmt_Cnt < Max_Statements loop
                              Get_Next_Object (Func_JSON, Stmt_Pos, Stmt_Start, Stmt_End);
                              exit when Stmt_Start = 0 or Stmt_End = 0;
                              
                              Module.Functions (Func_Idx).Stmt_Cnt := Module.Functions (Func_Idx).Stmt_Cnt + 1;
                              Module.Functions (Func_Idx).Statements (Module.Functions (Func_Idx).Stmt_Cnt).Kind := Stmt_Nop;
                              Module.Functions (Func_Idx).Statements (Module.Functions (Func_Idx).Stmt_Cnt).Data :=
                                Code_Buffers.Null_Bounded_String;
                              
                              Stmt_Pos := Stmt_End + 1;
                           end loop;
                        end;
                     end if;
                  end if;
               end;
               
               Func_Pos := Obj_End + 1;
            end loop;
         end;
      end if;

      Put_Line ("[INFO] Parsed module: " & Name_Strings.To_String (Module.Module_Name) & 
                " with " & Natural'Image (Module.Func_Cnt) & " function(s)");
      Status := Success;

   exception
      when E : others =>
         Put_Line ("[ERROR] Exception during JSON parsing");
         Put_Line ("[ERROR] Exception: " & Ada.Exceptions.Exception_Information (E));
         Status := Error_Invalid_JSON;
   end Parse_Spec_JSON;

   --  Helper: Append string to bounded string buffer
   procedure Append_To_Buffer
     (Buffer : in out JSON_Buffer;
      Text   : String)
   is
      Current : constant String := JSON_Buffers.To_String (Buffer);
   begin
      if Current'Length + Text'Length <= Max_JSON_Size then
         Buffer := JSON_Buffers.To_Bounded_String (Current & Text);
      end if;
   end Append_To_Buffer;

   --  Serialize IR Module to JSON
   procedure IR_To_JSON
     (Module : IR_Module;
      Output : out JSON_Buffer;
      Status : out Parse_Status)
   is
   begin
      Output := JSON_Buffers.Null_Bounded_String;
      
      --  Start JSON object
      Append_To_Buffer (Output, "{""schema"":""stunir_ir_v1"",");
      Append_To_Buffer (Output, """ir_version"":""" & Module.IR_Version & """,");
      Append_To_Buffer (Output, """module_name"":""" & Name_Strings.To_String (Module.Module_Name) & """,");
      Append_To_Buffer (Output, """docstring"":""" & Doc_Strings.To_String (Module.Docstring) & """,");
      Append_To_Buffer (Output, """types"":[],");
      Append_To_Buffer (Output, """functions"":[");
      
      --  Serialize all functions
      for I in 1 .. Module.Func_Cnt loop
         if I > 1 then
            Append_To_Buffer (Output, ",");
         end if;
         
         Append_To_Buffer (Output, "{""name"":""" & Name_Strings.To_String (Module.Functions (I).Name) & """,");
         
         --  Serialize args
         Append_To_Buffer (Output, """args"":[");
         for J in 1 .. Module.Functions (I).Arg_Cnt loop
            if J > 1 then
               Append_To_Buffer (Output, ",");
            end if;
            Append_To_Buffer (Output, "{""name"":""" & Name_Strings.To_String (Module.Functions (I).Args (J).Name) & """,");
            Append_To_Buffer (Output, """type"":""" & Type_Strings.To_String (Module.Functions (I).Args (J).Type_Ref) & """}");
         end loop;
         Append_To_Buffer (Output, "],");
         
         Append_To_Buffer (Output, """return_type"":""" & Type_Strings.To_String (Module.Functions (I).Return_Type) & """,");
         
         --  Serialize steps (simplified - empty for now)
         Append_To_Buffer (Output, """steps"":[");
         for J in 1 .. Module.Functions (I).Stmt_Cnt loop
            if J > 1 then
               Append_To_Buffer (Output, ",");
            end if;
            -- Simplified step representation
            Append_To_Buffer (Output, "{""op"":""noop""}");
         end loop;
         Append_To_Buffer (Output, "]}");
      end loop;
      
      Append_To_Buffer (Output, "]}");
      
      Status := Success;

   exception
      when others =>
         Status := Error_Invalid_JSON;
         Output := JSON_Buffers.Null_Bounded_String;
   end IR_To_JSON;

   --  Compute SHA-256 hash of JSON (deterministic)
   function Compute_JSON_Hash (JSON_Text : String) return String is
      Context : GNAT.SHA256.Context := GNAT.SHA256.Initial_Context;
   begin
      GNAT.SHA256.Update (Context, JSON_Text);
      return GNAT.SHA256.Digest (Context);
   end Compute_JSON_Hash;

end STUNIR_JSON_Utils;

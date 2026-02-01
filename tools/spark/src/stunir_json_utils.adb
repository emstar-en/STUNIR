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

   --  Extract integer value from JSON field
   function Extract_Integer_Value
     (JSON_Text : String;
      Field_Name : String) return Natural
   is
      Pos : constant Natural := Find_Field (JSON_Text, Field_Name);
      Result : Natural := 0;
   begin
      if Pos = 0 then
         return 0;
      end if;
      
      --  Skip to the value (after ':' and whitespace)
      for I in Pos .. JSON_Text'Last loop
         if JSON_Text (I) in '0' .. '9' then
            --  Parse digits
            for J in I .. JSON_Text'Last loop
               if JSON_Text (J) in '0' .. '9' then
                  --  Convert digit and accumulate
                  Result := Result * 10 + (Character'Pos (JSON_Text (J)) - Character'Pos ('0'));
                  
                  --  Safety check to prevent overflow
                  if Result > 10_000 then
                     return 10_000;  --  Cap at reasonable limit
                  end if;
               else
                  --  End of number
                  return Result;
               end if;
            end loop;
            return Result;
         elsif JSON_Text (I) /= ' ' and JSON_Text (I) /= ':' and 
               JSON_Text (I) /= ASCII.HT and JSON_Text (I) /= ASCII.LF then
            --  Non-digit, non-whitespace found before number
            return 0;
         end if;
      end loop;
      
      return Result;
   end Extract_Integer_Value;

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
                     
                -- v0.8.2: Parse body statements with recursive multi-level nesting support
                if Body_Pos > 0 then
                      declare
                         Func_Idx : constant Positive := Module.Func_Cnt;
                         
                         -- v0.8.2: Recursive procedure to flatten nested statements
                         procedure Flatten_Block (Block_JSON : String; Array_Pos : Natural; Depth : Natural := 0) is
                            Stmt_Pos : Natural := Array_Pos + 1;
                            Stmt_Start, Stmt_End : Natural;
                         begin
                            -- Safety: Limit nesting depth to 5 levels
                            if Depth > 5 then
                               Put_Line ("[ERROR] Maximum nesting depth (5) exceeded");
                               return;
                            end if;
                            while Module.Functions (Func_Idx).Stmt_Cnt < Max_Statements loop
                               Get_Next_Object (Block_JSON, Stmt_Pos, Stmt_Start, Stmt_End);
                               exit when Stmt_Start = 0 or Stmt_End = 0;
                               declare
                                  Stmt_JSON : constant String := Block_JSON (Stmt_Start .. Stmt_End);
                                  Stmt_Type : constant String := Extract_String_Value (Stmt_JSON, "type");
                               begin
                                  -- Reserve slot for this statement
                                  Module.Functions (Func_Idx).Stmt_Cnt := Module.Functions (Func_Idx).Stmt_Cnt + 1;
                                  declare
                                     Current_Idx : constant Positive := Module.Functions (Func_Idx).Stmt_Cnt;
                                  begin
                                     -- Initialize statement with defaults
                                     Module.Functions (Func_Idx).Statements (Current_Idx).Kind := Stmt_Nop;
                                     Module.Functions (Func_Idx).Statements (Current_Idx).Data := Code_Buffers.Null_Bounded_String;
                                     Module.Functions (Func_Idx).Statements (Current_Idx).Target := Name_Strings.Null_Bounded_String;
                                     Module.Functions (Func_Idx).Statements (Current_Idx).Value := Code_Buffers.Null_Bounded_String;
                                     Module.Functions (Func_Idx).Statements (Current_Idx).Condition := Code_Buffers.Null_Bounded_String;
                                     Module.Functions (Func_Idx).Statements (Current_Idx).Init_Expr := Code_Buffers.Null_Bounded_String;
                                     Module.Functions (Func_Idx).Statements (Current_Idx).Incr_Expr := Code_Buffers.Null_Bounded_String;
                                     Module.Functions (Func_Idx).Statements (Current_Idx).Block_Start := 0;
                                     Module.Functions (Func_Idx).Statements (Current_Idx).Block_Count := 0;
                                     Module.Functions (Func_Idx).Statements (Current_Idx).Else_Start := 0;
                                     Module.Functions (Func_Idx).Statements (Current_Idx).Else_Count := 0;
                                     -- Parse based on statement type
                                     if Stmt_Type = "assign" or Stmt_Type = "var_decl" then
                                        Module.Functions (Func_Idx).Statements (Current_Idx).Kind := Stmt_Assign;
                                        declare
                                           Target_Str : constant String := Extract_String_Value (Stmt_JSON, "target");
                                           Var_Name : constant String := Extract_String_Value (Stmt_JSON, "var_name");
                                           Value_Str : constant String := Extract_String_Value (Stmt_JSON, "value");
                                           Init_Str : constant String := Extract_String_Value (Stmt_JSON, "init");
                                        begin
                                           if Target_Str'Length > 0 and Target_Str'Length <= Max_Name_Length then
                                              Module.Functions (Func_Idx).Statements (Current_Idx).Target :=
                                                Name_Strings.To_Bounded_String (Target_Str);
                                           elsif Var_Name'Length > 0 and Var_Name'Length <= Max_Name_Length then
                                              Module.Functions (Func_Idx).Statements (Current_Idx).Target :=
                                                Name_Strings.To_Bounded_String (Var_Name);
                                           end if;
                                           if Value_Str'Length > 0 and Value_Str'Length <= Max_Code_Length then
                                              Module.Functions (Func_Idx).Statements (Current_Idx).Value :=
                                                Code_Buffers.To_Bounded_String (Value_Str);
                                           elsif Init_Str'Length > 0 and Init_Str'Length <= Max_Code_Length then
                                              Module.Functions (Func_Idx).Statements (Current_Idx).Value :=
                                                Code_Buffers.To_Bounded_String (Init_Str);
                                           end if;
                                        end;
                                     elsif Stmt_Type = "return" then
                                        Module.Functions (Func_Idx).Statements (Current_Idx).Kind := Stmt_Return;
                                        declare
                                           Value_Str : constant String := Extract_String_Value (Stmt_JSON, "value");
                                        begin
                                           if Value_Str'Length > 0 and Value_Str'Length <= Max_Code_Length then
                                              Module.Functions (Func_Idx).Statements (Current_Idx).Value :=
                                                Code_Buffers.To_Bounded_String (Value_Str);
                                           end if;
                                        end;
                                     elsif Stmt_Type = "call" then
                                        Module.Functions (Func_Idx).Statements (Current_Idx).Kind := Stmt_Call;
                                        declare
                                           Func_Name : constant String := Extract_String_Value (Stmt_JSON, "func");
                                           Args_Str : constant String := Extract_String_Value (Stmt_JSON, "args");
                                           Assign_To : constant String := Extract_String_Value (Stmt_JSON, "assign_to");
                                        begin
                                           if Func_Name'Length > 0 then
                                              declare
                                                 Call_Expr : constant String := Func_Name & "(" & Args_Str & ")";
                                              begin
                                                 if Call_Expr'Length <= Max_Code_Length then
                                                    Module.Functions (Func_Idx).Statements (Current_Idx).Value :=
                                                      Code_Buffers.To_Bounded_String (Call_Expr);
                                                 end if;
                                              end;
                                           end if;
                                           if Assign_To'Length > 0 and Assign_To'Length <= Max_Name_Length then
                                              Module.Functions (Func_Idx).Statements (Current_Idx).Target :=
                                                Name_Strings.To_Bounded_String (Assign_To);
                                           end if;
                                        end;
                                     elsif Stmt_Type = "if" then
                                        -- v0.8.2: Recursive handling of nested if statements
                                        Module.Functions (Func_Idx).Statements (Current_Idx).Kind := Stmt_If;
                                        declare
                                           Cond_Str : constant String := Extract_String_Value (Stmt_JSON, "condition");
                                           -- Try "then_block" first (IR format), then "then" (spec format)
                                           Then_Array_Pos_1 : constant Natural := Find_Array (Stmt_JSON, "then_block");
                                           Then_Array_Pos_2 : constant Natural := Find_Array (Stmt_JSON, "then");
                                           Then_Array_Pos : constant Natural := (if Then_Array_Pos_1 > 0 then Then_Array_Pos_1 else Then_Array_Pos_2);
                                           -- Try "else_block" first (IR format), then "else" (spec format)
                                           Else_Array_Pos_1 : constant Natural := Find_Array (Stmt_JSON, "else_block");
                                           Else_Array_Pos_2 : constant Natural := Find_Array (Stmt_JSON, "else");
                                           Else_Array_Pos : constant Natural := (if Else_Array_Pos_1 > 0 then Else_Array_Pos_1 else Else_Array_Pos_2);
                                           Then_Start_Idx : Natural := 0;
                                           Then_Count_Val : Natural := 0;
                                           Else_Start_Idx : Natural := 0;
                                           Else_Count_Val : Natural := 0;
                                           Count_Before : Natural;
                                        begin
                                           -- Extract condition
                                           if Cond_Str'Length > 0 and Cond_Str'Length <= Max_Code_Length then
                                              Module.Functions (Func_Idx).Statements (Current_Idx).Condition :=
                                                Code_Buffers.To_Bounded_String (Cond_Str);
                                           end if;
                                           -- Recursively flatten then_block
                                           if Then_Array_Pos > 0 then
                                              Then_Start_Idx := Module.Functions (Func_Idx).Stmt_Cnt + 1;  -- 1-based
                                              Count_Before := Module.Functions (Func_Idx).Stmt_Cnt;
                                              Flatten_Block (Stmt_JSON, Then_Array_Pos, Depth + 1);  -- RECURSIVE CALL
                                              Then_Count_Val := Module.Functions (Func_Idx).Stmt_Cnt - Count_Before;
                                           end if;
                                           -- Recursively flatten else_block
                                           if Else_Array_Pos > 0 then
                                              Else_Start_Idx := Module.Functions (Func_Idx).Stmt_Cnt + 1;  -- 1-based
                                              Count_Before := Module.Functions (Func_Idx).Stmt_Cnt;
                                              Flatten_Block (Stmt_JSON, Else_Array_Pos, Depth + 1);  -- RECURSIVE CALL
                                              Else_Count_Val := Module.Functions (Func_Idx).Stmt_Cnt - Count_Before;
                                           end if;
                                           -- Fill in block indices
                                           Module.Functions (Func_Idx).Statements (Current_Idx).Block_Start := Then_Start_Idx;
                                           Module.Functions (Func_Idx).Statements (Current_Idx).Block_Count := Then_Count_Val;
                                           Module.Functions (Func_Idx).Statements (Current_Idx).Else_Start := Else_Start_Idx;
                                           Module.Functions (Func_Idx).Statements (Current_Idx).Else_Count := Else_Count_Val;
                                           Put_Line ("[INFO] Flattened if: then_block[" & Natural'Image(Then_Start_Idx) & ".." & 
                                                     Natural'Image(Then_Start_Idx + Then_Count_Val - 1) & "] else_block[" & 
                                                     Natural'Image(Else_Start_Idx) & ".." & Natural'Image(Else_Start_Idx + Else_Count_Val - 1) & "]");
                                        end;
                                     elsif Stmt_Type = "while" then
                                        -- v0.8.2: Recursive handling of nested while statements
                                        Module.Functions (Func_Idx).Statements (Current_Idx).Kind := Stmt_While;
                                        declare
                                           Cond_Str : constant String := Extract_String_Value (Stmt_JSON, "condition");
                                           Body_Array_Pos : constant Natural := Find_Array (Stmt_JSON, "body");
                                           Body_Start_Idx : Natural := 0;
                                           Body_Count_Val : Natural := 0;
                                           Count_Before : Natural;
                                        begin
                                           -- Extract condition
                                           if Cond_Str'Length > 0 and Cond_Str'Length <= Max_Code_Length then
                                              Module.Functions (Func_Idx).Statements (Current_Idx).Condition :=
                                                Code_Buffers.To_Bounded_String (Cond_Str);
                                           end if;
                                           -- Recursively flatten body
                                           if Body_Array_Pos > 0 then
                                              Body_Start_Idx := Module.Functions (Func_Idx).Stmt_Cnt + 1;
                                              Count_Before := Module.Functions (Func_Idx).Stmt_Cnt;
                                              Flatten_Block (Stmt_JSON, Body_Array_Pos, Depth + 1);  -- RECURSIVE CALL
                                              Body_Count_Val := Module.Functions (Func_Idx).Stmt_Cnt - Count_Before;
                                           end if;
                                           -- Fill in block indices
                                           Module.Functions (Func_Idx).Statements (Current_Idx).Block_Start := Body_Start_Idx;
                                           Module.Functions (Func_Idx).Statements (Current_Idx).Block_Count := Body_Count_Val;
                                           Put_Line ("[INFO] Flattened while: body[" & Natural'Image(Body_Start_Idx) & ".." & 
                                                     Natural'Image(Body_Start_Idx + Body_Count_Val - 1) & "]");
                                        end;
                                     elsif Stmt_Type = "for" then
                                        -- v0.8.2: Recursive handling of nested for statements
                                        Module.Functions (Func_Idx).Statements (Current_Idx).Kind := Stmt_For;
                                        declare
                                           Init_Str : constant String := Extract_String_Value (Stmt_JSON, "init");
                                           Cond_Str : constant String := Extract_String_Value (Stmt_JSON, "condition");
                                           Incr_Str : constant String := Extract_String_Value (Stmt_JSON, "increment");
                                           Body_Array_Pos : constant Natural := Find_Array (Stmt_JSON, "body");
                                           Body_Start_Idx : Natural := 0;
                                           Body_Count_Val : Natural := 0;
                                           Count_Before : Natural;
                                        begin
                                           -- Extract for loop components
                                           if Init_Str'Length > 0 and Init_Str'Length <= Max_Code_Length then
                                              Module.Functions (Func_Idx).Statements (Current_Idx).Init_Expr :=
                                                Code_Buffers.To_Bounded_String (Init_Str);
                                           end if;
                                           if Cond_Str'Length > 0 and Cond_Str'Length <= Max_Code_Length then
                                              Module.Functions (Func_Idx).Statements (Current_Idx).Condition :=
                                                Code_Buffers.To_Bounded_String (Cond_Str);
                                           end if;
                                           if Incr_Str'Length > 0 and Incr_Str'Length <= Max_Code_Length then
                                              Module.Functions (Func_Idx).Statements (Current_Idx).Incr_Expr :=
                                                Code_Buffers.To_Bounded_String (Incr_Str);
                                           end if;
                                           -- Recursively flatten body
                                           if Body_Array_Pos > 0 then
                                              Body_Start_Idx := Module.Functions (Func_Idx).Stmt_Cnt + 1;
                                              Count_Before := Module.Functions (Func_Idx).Stmt_Cnt;
                                              Flatten_Block (Stmt_JSON, Body_Array_Pos, Depth + 1);  -- RECURSIVE CALL
                                              Body_Count_Val := Module.Functions (Func_Idx).Stmt_Cnt - Count_Before;
                                           end if;
                                           -- Fill in block indices
                                           Module.Functions (Func_Idx).Statements (Current_Idx).Block_Start := Body_Start_Idx;
                                           Module.Functions (Func_Idx).Statements (Current_Idx).Block_Count := Body_Count_Val;
                                           Put_Line ("[INFO] Flattened for: body[" & Natural'Image(Body_Start_Idx) & ".." & 
                                                     Natural'Image(Body_Start_Idx + Body_Count_Val - 1) & "]");
                                        end;
                                     else
                                        -- Unknown statement type - keep as Stmt_Nop
                                        null;
                                     end if;
                                  end;
                               end;
                               Stmt_Pos := Stmt_End + 1;
                            end loop;
                         end Flatten_Block;
                         
                      begin
                         -- Call recursive flattening for function body
                         Flatten_Block (Func_JSON, Body_Pos);
                      end;
                end if;
                  end if;  -- Close the if from line 234 (if Func_Name'Length > 0)
               end;  -- Close declare from line 227
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
      
      --  Start JSON object (v0.8.1: Use stunir_flat_ir_v1 for flattened control flow)
      Append_To_Buffer (Output, "{""schema"":""stunir_flat_ir_v1"",");
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
         
          --  Serialize steps (v0.8.0: with proper control flow fields)
          Append_To_Buffer (Output, """steps"":[");
          for J in 1 .. Module.Functions (I).Stmt_Cnt loop
             if J > 1 then
                Append_To_Buffer (Output, ",");
             end if;
             
             declare
                Stmt : IR_Statement renames Module.Functions (I).Statements (J);
             begin
                Append_To_Buffer (Output, "{");
                
                -- Emit op field based on statement kind
                case Stmt.Kind is
                   when Stmt_Assign =>
                      Append_To_Buffer (Output, """op"":""assign""");
                      if Name_Strings.Length (Stmt.Target) > 0 then
                         Append_To_Buffer (Output, ",""target"":""" & Name_Strings.To_String (Stmt.Target) & """");
                      end if;
                      if Code_Buffers.Length (Stmt.Value) > 0 then
                         Append_To_Buffer (Output, ",""value"":""" & Code_Buffers.To_String (Stmt.Value) & """");
                      end if;
                   
                   when Stmt_Call =>
                      Append_To_Buffer (Output, """op"":""call""");
                      if Code_Buffers.Length (Stmt.Value) > 0 then
                         Append_To_Buffer (Output, ",""value"":""" & Code_Buffers.To_String (Stmt.Value) & """");
                      end if;
                      if Name_Strings.Length (Stmt.Target) > 0 then
                         Append_To_Buffer (Output, ",""target"":""" & Name_Strings.To_String (Stmt.Target) & """");
                      end if;
                   
                   when Stmt_Return =>
                      Append_To_Buffer (Output, """op"":""return""");
                      if Code_Buffers.Length (Stmt.Value) > 0 then
                         Append_To_Buffer (Output, ",""value"":""" & Code_Buffers.To_String (Stmt.Value) & """");
                      end if;
                   
                   when Stmt_If =>
                      Append_To_Buffer (Output, """op"":""if""");
                      if Code_Buffers.Length (Stmt.Condition) > 0 then
                         Append_To_Buffer (Output, ",""condition"":""" & Code_Buffers.To_String (Stmt.Condition) & """");
                      end if;
                      -- Emit block indices for flattened IR
                      if Stmt.Block_Start > 0 then
                         Append_To_Buffer (Output, ",""block_start"":" & Natural'Image (Stmt.Block_Start));
                         Append_To_Buffer (Output, ",""block_count"":" & Natural'Image (Stmt.Block_Count));
                      end if;
                      if Stmt.Else_Start > 0 then
                         Append_To_Buffer (Output, ",""else_start"":" & Natural'Image (Stmt.Else_Start));
                         Append_To_Buffer (Output, ",""else_count"":" & Natural'Image (Stmt.Else_Count));
                      end if;
                   
                   when Stmt_While =>
                      Append_To_Buffer (Output, """op"":""while""");
                      if Code_Buffers.Length (Stmt.Condition) > 0 then
                         Append_To_Buffer (Output, ",""condition"":""" & Code_Buffers.To_String (Stmt.Condition) & """");
                      end if;
                      if Stmt.Block_Start > 0 then
                         Append_To_Buffer (Output, ",""block_start"":" & Natural'Image (Stmt.Block_Start));
                         Append_To_Buffer (Output, ",""block_count"":" & Natural'Image (Stmt.Block_Count));
                      end if;
                   
                   when Stmt_For =>
                      Append_To_Buffer (Output, """op"":""for""");
                      if Code_Buffers.Length (Stmt.Init_Expr) > 0 then
                         Append_To_Buffer (Output, ",""init"":""" & Code_Buffers.To_String (Stmt.Init_Expr) & """");
                      end if;
                      if Code_Buffers.Length (Stmt.Condition) > 0 then
                         Append_To_Buffer (Output, ",""condition"":""" & Code_Buffers.To_String (Stmt.Condition) & """");
                      end if;
                      if Code_Buffers.Length (Stmt.Incr_Expr) > 0 then
                         Append_To_Buffer (Output, ",""increment"":""" & Code_Buffers.To_String (Stmt.Incr_Expr) & """");
                      end if;
                      if Stmt.Block_Start > 0 then
                         Append_To_Buffer (Output, ",""block_start"":" & Natural'Image (Stmt.Block_Start));
                         Append_To_Buffer (Output, ",""block_count"":" & Natural'Image (Stmt.Block_Count));
                      end if;
                   
                   when Stmt_Nop =>
                      Append_To_Buffer (Output, """op"":""noop""");
                end case;
                
                Append_To_Buffer (Output, "}");
             end;
          end loop;
          Append_To_Buffer (Output, "]");  -- Close steps array
         Append_To_Buffer (Output, "}");  -- Close function object
      end loop;
      
      Append_To_Buffer (Output, "]");  -- Close functions array
      Append_To_Buffer (Output, "}");  -- Close module object
      
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
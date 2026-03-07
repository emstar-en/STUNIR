-------------------------------------------------------------------------------
--  STUNIR Semantic IR Emitter Package Body
--  DO-178C Level A Compliant
--  SPARK 2014 Mode
--
--  Implementation of Semantic IR JSON emission.
--
--  Copyright (c) 2026 STUNIR Project
--  License: MIT
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with Ada.Text_IO;
use Ada.Text_IO;
with Semantic_IR.Nodes; use Semantic_IR.Nodes;
with Semantic_IR.Normalizer; use Semantic_IR.Normalizer;

package body Semantic_IR.Emitter is

   --  Escape a string for JSON output
   function Escape_JSON_String (S : String) return String is
      Result : String (1 .. S'Length * 2);  --  Worst case: all escapes
      Last   : Natural := 0;
   begin
      for I in S'Range loop
         case S (I) is
            when '"' =>
               Last := Last + 1;
               Result (Last) := '\';
               Last := Last + 1;
               Result (Last) := '"';
            when '\' =>
               Last := Last + 1;
               Result (Last) := '\';
               Last := Last + 1;
               Result (Last) := '\';
            when ASCII.LF =>
               Last := Last + 1;
               Result (Last) := '\';
               Last := Last + 1;
               Result (Last) := 'n';
            when ASCII.CR =>
               Last := Last + 1;
               Result (Last) := '\';
               Last := Last + 1;
               Result (Last) := 'r';
            when ASCII.HT =>
               Last := Last + 1;
               Result (Last) := '\';
               Last := Last + 1;
               Result (Last) := 't';
            when others =>
               Last := Last + 1;
               Result (Last) := S (I);
         end case;
      end loop;
      return Result (1 .. Last);
   end Escape_JSON_String;

   --  Convert a Semantic_Node_Kind to JSON string
   function Kind_To_String (Kind : Semantic_Node_Kind) return String is
   begin
      case Kind is
         when Kind_Module => return "module";
         when Kind_Function_Decl => return "function_decl";
         when Kind_Type_Decl => return "type_decl";
         when Kind_Const_Decl => return "const_decl";
         when Kind_Var_Decl => return "var_decl";
         when Kind_Block_Stmt => return "block_stmt";
         when Kind_Expr_Stmt => return "expr_stmt";
         when Kind_If_Stmt => return "if_stmt";
         when Kind_While_Stmt => return "while_stmt";
         when Kind_For_Stmt => return "for_stmt";
         when Kind_Return_Stmt => return "return_stmt";
         when Kind_Break_Stmt => return "break_stmt";
         when Kind_Continue_Stmt => return "continue_stmt";
         when Kind_Var_Decl_Stmt => return "var_decl_stmt";
         when Kind_Assign_Stmt => return "assign_stmt";
         when Kind_Integer_Literal => return "integer_literal";
         when Kind_Float_Literal => return "float_literal";
         when Kind_String_Literal => return "string_literal";
         when Kind_Bool_Literal => return "bool_literal";
         when Kind_Var_Ref => return "var_ref";
         when Kind_Binary_Expr => return "binary_expr";
         when Kind_Unary_Expr => return "unary_expr";
         when Kind_Function_Call => return "function_call";
         when Kind_Member_Expr => return "member_expr";
         when Kind_Array_Access => return "array_access";
         when Kind_Cast_Expr => return "cast_expr";
         when Kind_Ternary_Expr => return "ternary_expr";
         when Kind_Array_Init => return "array_init";
         when Kind_Struct_Init => return "struct_init";
      end case;
   end Kind_To_String;

   --  Convert a Type_Kind to JSON string
   function Type_Kind_To_String (Kind : Type_Kind) return String is
   begin
      case Kind is
         when TK_Primitive => return "primitive";
         when TK_Array => return "array";
         when TK_Pointer => return "pointer";
         when TK_Struct => return "struct";
         when TK_Function => return "function";
         when TK_Ref => return "ref";
      end case;
   end Type_Kind_To_String;

   --  Convert a Safety_Level to JSON string
   function Safety_Level_To_String (Level : Safety_Level) return String is
   begin
      case Level is
         when Level_None => return "none";
         when Level_DO178C_D => return "do178c_d";
         when Level_DO178C_C => return "do178c_c";
         when Level_DO178C_B => return "do178c_b";
         when Level_DO178C_A => return "do178c_a";
      end case;
   end Safety_Level_To_String;

   --  Convert a Target_Category to JSON string
   function Target_Category_To_String (Cat : Target_Category) return String is
   begin
      case Cat is
         when Target_Embedded => return "embedded";
         when Target_Realtime => return "realtime";
         when Target_Safety_Critical => return "safety_critical";
         when Target_Gpu => return "gpu";
         when Target_Wasm => return "wasm";
         when Target_Native => return "native";
      end case;
   end Target_Category_To_String;

   --  Emit a Semantic IR module to a JSON file
   procedure Emit_Module_To_File
     (Module      : in     Semantic_Module;
      Output_Path : in     Path_String;
      Config      : in     Emitter_Config;
      Result      :    out Emitter_Result)
   is
      JSON_Output : JSON_String;
      File        : File_Type;
      Temp_Status : Emitter_Status;
   begin
      --  Initialize result
      Result.Success := False;
      Result.Stats := (others => 0);
      Result.Message := Error_String_Strings.Null_Bounded_String;

      --  Validate module
      if not Is_Valid_Module (Module) then
         Result.Message := Error_String_Strings.To_Bounded_String ("Invalid module");
         return;
      end if;

      --  Check normal form if required
      if Config.Enforce_Normal and then not Is_In_Normal_Form (Module) then
         Result.Message := Error_String_Strings.To_Bounded_String ("Module not in normal form");
         return;
      end if;

      --  Emit to string
      Emit_Module_To_String (Module, JSON_Output, Config, Result);
      if not Result.Success then
         return;
      end if;

      --  Write to file
      begin
         Create (File, Out_File, Output_Path);
         Put_Line (File, JSON_String_Strings.To_String (JSON_Output));
         Close (File);
      exception
         when others =>
            Result.Message := Error_String_Strings.To_Bounded_String ("File I/O error");
            return;
      end;

      Result.Success := True;
   end Emit_Module_To_File;

   --  Emit a Semantic IR module to a JSON string
   procedure Emit_Module_To_String
     (Module      : in     Semantic_Module;
      JSON_Output :    out JSON_String;
      Config      : in     Emitter_Config;
      Result      :    out Emitter_Result)
   is
      --  Build JSON string incrementally
      use Ada.Strings.Unbounded;
      Builder : Unbounded_String;

      --  Indentation level
      Indent_Level : Natural := 0;

      procedure Indent is
      begin
         if Config.Pretty_Print then
            for I in 1 .. Indent_Level loop
               Append (Builder, "  ");
            end loop;
         end if;
      end Indent;

      procedure New_Line is
      begin
         if Config.Pretty_Print then
            Append (Builder, ASCII.LF);
         end if;
      end New_Line;

      procedure Emit_String (S : String) is
      begin
         Append (Builder, '"');
         Append (Builder, Escape_JSON_String (S));
         Append (Builder, '"');
      end Emit_String;

      procedure Emit_Key_Value (Key, Value : String; Is_Last : Boolean) is
      begin
         Indent;
         Emit_String (Key);
         Append (Builder, ": ");
         Emit_String (Value);
         if not Is_Last then
            Append (Builder, ",");
         end if;
         New_Line;
      end Emit_Key_Value;

   begin
      --  Initialize result
      Result.Success := False;
      Result.Stats := (others => 0);
      Result.Message := Error_String_Strings.Null_Bounded_String;

      --  Validate module
      if not Is_Valid_Module (Module) then
         Result.Message := Error_String_Strings.To_Bounded_String ("Invalid module");
         JSON_Output := JSON_String_Strings.Null_Bounded_String;
         return;
      end if;

      --  Check normal form if required
      if Config.Enforce_Normal and then not Is_In_Normal_Form (Module) then
         Result.Message := Error_String_Strings.To_Bounded_String ("Module not in normal form");
         JSON_Output := JSON_String_Strings.Null_Bounded_String;
         return;
      end if;

      --  Start JSON object
      Append (Builder, '{');
      New_Line;
      Indent_Level := Indent_Level + 1;

      --  Emit schema version
      Emit_Key_Value ("schema_version", "1.0.0", False);

      --  Emit IR version
      Emit_Key_Value ("ir_version", "1.0.0", False);

      --  Emit module name
      Emit_Key_Value ("module_name",
         Name_Strings.To_String (Module.Module_Name), False);

      --  Emit module hash if configured
      if Config.Include_Hashes then
         Indent;
         Append (Builder, '"');
         Append (Builder, "module_hash");
         Append (Builder, '"');
         Append (Builder, ": ");
         Emit_String (Hash_Strings.To_String (Module.Module_Hash));
         Append (Builder, ",");
         New_Line;
      end if;

      --  Emit imports (sorted in normal form)
      Indent;
      Append (Builder, '"');
      Append (Builder, "imports");
      Append (Builder, '"');
      Append (Builder, ": [");
      New_Line;
      Indent_Level := Indent_Level + 1;

      for I in 1 .. Module.Import_Count loop
         Indent;
         Append (Builder, '{');
         New_Line;
         Indent_Level := Indent_Level + 1;

         Emit_Key_Value ("module",
            Name_Strings.To_String (Module.Imports (I).Module_Name),
            Module.Imports (I).Symbol_Count = 0);

         if Module.Imports (I).Symbol_Count > 0 then
            Indent;
            Append (Builder, '"');
            Append (Builder, "symbols");
            Append (Builder, '"');
            Append (Builder, ": [");
            for J in 1 .. Module.Imports (I).Symbol_Count loop
               Emit_String (Name_Strings.To_String (Module.Imports (I).Symbols (J)));
               if J < Module.Imports (I).Symbol_Count then
                  Append (Builder, ", ");
               end if;
            end loop;
            Append (Builder, "]");
            Append (Builder, ",");
            New_Line;
         end if;

         Indent_Level := Indent_Level - 1;
         Indent;
         Append (Builder, '}');
         if I < Module.Import_Count then
            Append (Builder, ",");
         end if;
         New_Line;
      end loop;

      Indent_Level := Indent_Level - 1;
      Indent;
      Append (Builder, "],");
      New_Line;

      --  Emit exports (sorted in normal form)
      Indent;
      Append (Builder, '"');
      Append (Builder, "exports");
      Append (Builder, '"');
      Append (Builder, ": [");
      for I in 1 .. Module.Export_Count loop
         Emit_String (Name_Strings.To_String (Module.Exports (I)));
         if I < Module.Export_Count then
            Append (Builder, ", ");
         end if;
      end loop;
      Append (Builder, "],");
      New_Line;

      --  Emit metadata
      Indent;
      Append (Builder, '"');
      Append (Builder, "metadata");
      Append (Builder, '"');
      Append (Builder, ": {");
      New_Line;
      Indent_Level := Indent_Level + 1;

      Emit_Key_Value ("safety_level",
         Safety_Level_To_String (Module.Metadata.Module_Safety), False);

      Emit_Key_Value ("optimization_level",
         Natural'Image (Module.Metadata.Optimization_Level), False);

      --  Target categories
      Indent;
      Append (Builder, '"');
      Append (Builder, "target_categories");
      Append (Builder, '"');
      Append (Builder, ": [");
      for I in 1 .. Module.Metadata.Target_Count loop
         Emit_String (Target_Category_To_String (Module.Metadata.Target_Categories (I)));
         if I < Module.Metadata.Target_Count then
            Append (Builder, ", ");
         end if;
      end loop;
      Append (Builder, "]");
      New_Line;

      Indent_Level := Indent_Level - 1;
      Indent;
      Append (Builder, "},");
      New_Line;

      --  Emit declarations (sorted in normal form)
      Indent;
      Append (Builder, '"');
      Append (Builder, "declarations");
      Append (Builder, '"');
      Append (Builder, ": [");
      New_Line;
      Indent_Level := Indent_Level + 1;

      for I in 1 .. Module.Decl_Count loop
         Indent;
         Emit_String (Name_Strings.To_String (Module.Declarations (I)));
         if I < Module.Decl_Count then
            Append (Builder, ",");
         end if;
         New_Line;
      end loop;

      Indent_Level := Indent_Level - 1;
      Indent;
      Append (Builder, "]");
      New_Line;

      --  Close JSON object
      Indent_Level := Indent_Level - 1;
      Indent;
      Append (Builder, '}');
      New_Line;

      --  Set output
      JSON_Output := JSON_String_Strings.To_Bounded_String (To_String (Builder));
      Result.Success := True;
      Result.Stats.Nodes_Emitted := 1;  --  Module node
      Result.Stats.Declarations_Emitted := Module.Decl_Count;

   exception
      when others =>
         Result.Message := Error_String_Strings.To_Bounded_String ("Emission error");
         JSON_Output := JSON_String_Strings.Null_Bounded_String;
   end Emit_Module_To_String;

end Semantic_IR.Emitter;
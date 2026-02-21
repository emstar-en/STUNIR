-------------------------------------------------------------------------------
--  STUNIR JSON Utilities - Ada SPARK Implementation
--  PRIMARY IMPLEMENTATION (Ada SPARK is the default language for STUNIR tools)
--
--  Lightweight JSON parsing and generation for DO-178C Level A compliance
--  Focuses on the specific JSON structures needed for STUNIR semantic IR
--
--  Copyright (c) 2026 STUNIR Project
--  License: MIT
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with Ada.Strings.Bounded;
--  NOTE: STUNIR.Semantic_IR (old monolith) no longer exists.
--  The IR types used by this package are defined locally below.
--  The typed AST layer is in Semantic_IR.* (src/semantic_ir/).
--  This package uses its own lightweight flat IR for JSON parsing.

package STUNIR_JSON_Utils is

   --  =========================================================================
   --  Size constants
   --  =========================================================================
   Max_JSON_Size    : constant := 1_048_576;  --  1 MB max JSON
   Max_Name_Length  : constant := 128;
   Max_Type_Length  : constant := 128;
   Max_Doc_Length   : constant := 512;
   Max_Functions    : constant := 256;
   Max_Types        : constant := 128;
   Max_Args         : constant := 32;
   Max_Stmts        : constant := 128;
   Max_Type_Params  : constant := 16;
   Max_Type_Args    : constant := 16;

   --  =========================================================================
   --  Bounded string packages (flat IR layer — not the typed AST layer)
   --  =========================================================================
   package JSON_Buffers is new Ada.Strings.Bounded.Generic_Bounded_Length
     (Max => Max_JSON_Size);
   subtype JSON_Buffer is JSON_Buffers.Bounded_String;

   package Name_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length
     (Max => Max_Name_Length);
   subtype IR_Name is Name_Strings.Bounded_String;

   package Type_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length
     (Max => Max_Type_Length);
   subtype IR_Type_Name is Type_Strings.Bounded_String;

   package Doc_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length
     (Max => Max_Doc_Length);
   subtype IR_Doc is Doc_Strings.Bounded_String;

   --  =========================================================================
   --  Flat IR types (lightweight; used only within this JSON utility layer)
   --  For the full typed AST IR see src/semantic_ir/Semantic_IR.*.
   --  Field names match the body (stunir_json_utils.adb) exactly.
   --  =========================================================================

   --  Code buffer for statement expressions (larger than name strings)
   Max_Code_Length    : constant := 4_096;
   Max_Statements     : constant := Max_Stmts;   --  alias for body compatibility
   Max_Generic_Insts  : constant := 64;

   package Code_Buffers is new Ada.Strings.Bounded.Generic_Bounded_Length
     (Max => Max_Code_Length);
   subtype IR_Code is Code_Buffers.Bounded_String;

   --  Type parameter (declared first — used by IR_Stmt and IR_Function)
   type IR_Type_Param is record
      Name       : IR_Name;
      Constraint : IR_Type_Name;
   end record;

   type IR_Type_Param_Array is array (1 .. Max_Type_Params) of IR_Type_Param;

   --  Function argument (field name: Type_Ref, not Arg_Type)
   type IR_Arg is record
      Name     : IR_Name;
      Type_Ref : IR_Type_Name;
   end record;

   type IR_Arg_Array is array (1 .. Max_Args) of IR_Arg;

   --  Array of type name strings (used for generic call type arguments)
   type IR_Type_Name_Array is array (1 .. Max_Type_Args) of IR_Type_Name;

   --  Statement kind (all variants used in the body)
   type IR_Stmt_Kind is (Stmt_Return, Stmt_Assign, Stmt_Call, Stmt_If,
                         Stmt_While, Stmt_For, Stmt_Nop, Stmt_Other,
                         Stmt_Break, Stmt_Continue, Stmt_Switch,
                         Stmt_Generic_Call, Stmt_Type_Cast);

   --  Switch case entry
   Max_Cases : constant := 32;

   type IR_Case is record
      Case_Value  : IR_Code;
      Block_Start : Natural := 0;
      Block_Count : Natural := 0;
   end record;

   type IR_Case_Array is array (1 .. Max_Cases) of IR_Case;

   --  Statement record — fields match body usage exactly
   type IR_Stmt is record
      Kind         : IR_Stmt_Kind := Stmt_Nop;
      Target       : IR_Name;
      Data         : IR_Code;
      Value        : IR_Code;
      Condition    : IR_Code;
      Init_Expr    : IR_Code;
      Incr_Expr    : IR_Code;
      Cast_Type    : IR_Type_Name;
      Block_Start  : Natural := 0;
      Block_Count  : Natural := 0;
      Else_Start   : Natural := 0;
      Else_Count   : Natural := 0;
      Case_Cnt     : Natural range 0 .. Max_Cases     := 0;
      Cases        : IR_Case_Array;
      Type_Arg_Cnt : Natural range 0 .. Max_Type_Args := 0;
      Type_Args    : IR_Type_Name_Array;  --  array of type name strings (not IR_Type_Param records)
   end record;

   --  IR_Statement is an alias for IR_Stmt (body uses both names)
   subtype IR_Statement is IR_Stmt;

   type IR_Stmt_Array is array (1 .. Max_Stmts) of IR_Stmt;

   --  Generic instantiation record
   type IR_Generic_Inst is record
      Name         : IR_Name;
      Base_Type    : IR_Type_Name;
      Type_Arg_Cnt : Natural range 0 .. Max_Type_Args := 0;
      Type_Args    : IR_Type_Name_Array;  --  array of type name strings
   end record;

   type IR_Generic_Inst_Array is array (1 .. Max_Type_Args) of IR_Generic_Inst;

   --  Function record — fields match body usage exactly
   type IR_Function is record
      Name           : IR_Name;
      Return_Type    : IR_Type_Name;
      Docstring      : IR_Doc;
      Arg_Cnt        : Natural range 0 .. Max_Args        := 0;
      Stmt_Cnt       : Natural range 0 .. Max_Stmts       := 0;
      Type_Param_Cnt : Natural range 0 .. Max_Type_Params := 0;
      Args           : IR_Arg_Array;
      Statements     : IR_Stmt_Array;
      Type_Params    : IR_Type_Param_Array;
      Generic_Insts  : IR_Generic_Inst_Array;
   end record;

   type IR_Function_Array is array (1 .. Max_Functions) of IR_Function;

   --  Type definition record
   type IR_Type_Def is record
      Name      : IR_Name;
      Base_Type : IR_Type_Name;
      Docstring : IR_Doc;
      Param_Cnt : Natural range 0 .. Max_Type_Params := 0;
      Params    : IR_Type_Param_Array;
   end record;

   type IR_Type_Array is array (1 .. Max_Types) of IR_Type_Def;

   type IR_Generic_Inst_Module_Array is array (1 .. Max_Generic_Insts) of IR_Generic_Inst;

   --  Flat IR module (the primary output of Parse_Spec_JSON)
   type IR_Module is record
      IR_Version        : String (1 .. 2) := "v1";
      Module_Name       : IR_Name;
      Docstring         : IR_Doc;
      Func_Cnt          : Natural range 0 .. Max_Functions    := 0;
      Type_Cnt          : Natural range 0 .. Max_Types        := 0;
      Generic_Inst_Cnt  : Natural range 0 .. Max_Generic_Insts := 0;
      Functions         : IR_Function_Array;
      Types             : IR_Type_Array;
      Generic_Insts     : IR_Generic_Inst_Module_Array;
   end record;

   --  Module validity predicate
   function Is_Valid_Module (M : IR_Module) return Boolean is
     (Name_Strings.Length (M.Module_Name) > 0)
   with Ghost;

   --  =========================================================================
   --  Parse status
   --  =========================================================================
   type Parse_Status is (Success, Error_Invalid_JSON, Error_Missing_Field,
                         Error_Too_Large, Error_Invalid_Type);

   --  =========================================================================
   --  Public API
   --  =========================================================================

   --  Parse simple JSON spec into flat IR Module
   procedure Parse_Spec_JSON
     (JSON_Text : String;
      Module    : out IR_Module;
      Status    : out Parse_Status)
   with
     Pre  => JSON_Text'Length > 0 and JSON_Text'Length <= Max_JSON_Size,
     Post => (if Status = Success then Module.Func_Cnt > 0 or Module.Type_Cnt >= 0);

   --  Serialize flat IR Module to JSON
   procedure IR_To_JSON
     (Module : IR_Module;
      Output : out JSON_Buffer;
      Status : out Parse_Status)
   with
     Pre  => Is_Valid_Module (Module),
     Post => (if Status = Success then JSON_Buffers.Length (Output) > 0);

   --  Extract string value from JSON field (helper)
   function Extract_String_Value
     (JSON_Text  : String;
      Field_Name : String) return String
   with
     Pre  => JSON_Text'Length > 0 and Field_Name'Length > 0,
     Post => Extract_String_Value'Result'Length <= 1024;

   --  Extract integer value from JSON field (helper)
   --  Returns 0 if field not found or invalid
   function Extract_Integer_Value
     (JSON_Text  : String;
      Field_Name : String) return Natural
   with
     Pre => JSON_Text'Length > 0 and Field_Name'Length > 0;

   --  Find array in JSON and return position of opening bracket
   function Find_Array (JSON_Text : String; Field : String) return Natural
   with
     Pre => JSON_Text'Length > 0 and Field'Length > 0;

   --  Extract next object from array at position
   procedure Get_Next_Object
     (JSON_Text : String;
      Start_Pos : Natural;
      Obj_Start : out Natural;
      Obj_End   : out Natural)
   with
     Pre => JSON_Text'Length > 0 and Start_Pos <= JSON_Text'Last;

   --  Compute SHA-256 hash of JSON (deterministic)
   function Compute_JSON_Hash (JSON_Text : String) return String
   with
     Pre  => JSON_Text'Length > 0,
     Post => Compute_JSON_Hash'Result'Length = 64;

end STUNIR_JSON_Utils;

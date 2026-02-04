-- STUNIR Semantic IR Data Model (SPARK Implementation)
-- DO-178C Level A
-- Phase 3a: Core Category Emitters

with Ada.Strings.Bounded;

package STUNIR.Semantic_IR is
   pragma SPARK_Mode (On);

   -- Bounded strings for memory safety (reduced for v0.8.0 to prevent stack overflow)
   Max_Name_Length : constant := 128;
   Max_Type_Length : constant := 64;
   Max_Doc_Length  : constant := 512;
   Max_Code_Length : constant := 256;  -- Reduced from 4096 to 256 bytes

   package Name_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length (Max_Name_Length);
   package Type_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length (Max_Type_Length);
   package Doc_Strings  is new Ada.Strings.Bounded.Generic_Bounded_Length (Max_Doc_Length);
   package Code_Buffers is new Ada.Strings.Bounded.Generic_Bounded_Length (Max_Code_Length);

   subtype IR_Name_String is Name_Strings.Bounded_String;
   subtype IR_Type_String is Type_Strings.Bounded_String;
   subtype IR_Doc_String  is Doc_Strings.Bounded_String;
   subtype IR_Code_Buffer is Code_Buffers.Bounded_String;

   -- Primitive types
   type IR_Primitive_Type is
     (Type_String, Type_Int, Type_Float, Type_Bool, Type_Void,
      Type_I8, Type_I16, Type_I32, Type_I64,
      Type_U8, Type_U16, Type_U32, Type_U64,
      Type_F32, Type_F64);

   -- Field definition
   type IR_Field is record
      Name     : IR_Name_String;
      Type_Ref : IR_Type_String;
      Optional : Boolean := False;
   end record;

   -- Type definition
   Max_Fields : constant := 10;
   type Field_Array is array (Positive range <>) of IR_Field;

   type IR_Type_Def is record
      Name      : IR_Name_String;
      Docstring : IR_Doc_String;
      Fields    : Field_Array (1 .. Max_Fields);
      Field_Cnt : Natural range 0 .. Max_Fields := 0;
   end record
   with Dynamic_Predicate => Field_Cnt <= Max_Fields;

   -- Function argument
   type IR_Arg is record
      Name     : IR_Name_String;
      Type_Ref : IR_Type_String;
   end record;

   -- Statement (expanded for v0.9.0: added break, continue, switch/case)
   type IR_Statement_Kind is
     (Stmt_Assign, Stmt_Call, Stmt_Return, Stmt_If, Stmt_While, Stmt_For,
      Stmt_Break, Stmt_Continue, Stmt_Switch, Stmt_Nop,
      Stmt_Generic_Call, Stmt_Type_Cast);  -- v0.8.9: Generic calls and type casting

   -- Switch case entry (v0.9.0)
   type Switch_Case_Entry is record
      Case_Value  : IR_Code_Buffer;  -- Case value expression
      Block_Start : Natural := 0;    -- 1-based index of case block start
      Block_Count : Natural := 0;    -- Number of statements in case block
   end record;

   Max_Cases : constant := 10;  -- Maximum cases per switch (bounded for SPARK)
   type Case_Array is array (Positive range <>) of Switch_Case_Entry;

   -- Type argument for generic calls (v0.8.9)
   Max_Type_Args : constant := 4;
   type Type_Arg_Array is array (Positive range <>) of IR_Type_String;

   type IR_Statement is record
      Kind        : IR_Statement_Kind := Stmt_Nop;
      -- For all statements
      Data        : IR_Code_Buffer;  -- Legacy field, kept for compatibility
      -- For assign/call/return
      Target      : IR_Name_String;  -- Variable being assigned
      Value       : IR_Code_Buffer;  -- Expression value (also switch expr)
      -- For control flow (if/while/for)
      Condition   : IR_Code_Buffer;  -- Condition expression
      Init_Expr   : IR_Code_Buffer;  -- For loop initialization
      Incr_Expr   : IR_Code_Buffer;  -- For loop increment
      -- For flattened control flow blocks
      Block_Start : Natural := 0;    -- 1-based index of block start
      Block_Count : Natural := 0;    -- Number of statements in block
      Else_Start  : Natural := 0;    -- 1-based index of else/default block (0 if none)
      Else_Count  : Natural := 0;    -- Number of statements in else/default block
      -- For switch/case statements (v0.9.0)
      Case_Cnt    : Natural range 0 .. Max_Cases := 0;
      Cases       : Case_Array (1 .. Max_Cases);
      -- v0.8.9: Generic calls and type casting
      Type_Args   : Type_Arg_Array (1 .. Max_Type_Args);  -- Type arguments for generic calls
      Type_Arg_Cnt: Natural range 0 .. Max_Type_Args := 0;
      Cast_Type   : IR_Type_String;  -- Target type for type casting
   end record;

   -- Type parameter for generic functions (v0.8.9)
   type IR_Type_Param is record
      Name       : IR_Name_String;
      Constraint : IR_Type_String;  -- Type constraint (empty if none)
   end record;

   Max_Type_Params : constant := 4;
   type Type_Param_Array is array (Positive range <>) of IR_Type_Param;

   -- Generic instantiation (v0.8.9)
   type IR_Generic_Inst is record
      Name      : IR_Name_String;     -- Instantiated function name
      Base_Type : IR_Name_String;     -- Base generic function name
      Type_Args : Type_Arg_Array (1 .. Max_Type_Args);
      Type_Arg_Cnt : Natural range 0 .. Max_Type_Args := 0;
   end record;

   Max_Generic_Insts : constant := 10;
   type Generic_Inst_Array is array (Positive range <>) of IR_Generic_Inst;

   -- Function definition
   --  v0.8.6: Reduced limits to prevent stack overflow
   Max_Args       : constant := 8;
   Max_Statements : constant := 32;  -- Balanced for flattened control flow
   type Arg_Array is array (Positive range <>) of IR_Arg;
   type Statement_Array is array (Positive range <>) of IR_Statement;

   type IR_Function is record
      Name        : IR_Name_String;
      Docstring   : IR_Doc_String;
      Args        : Arg_Array (1 .. Max_Args);
      Return_Type : IR_Type_String;
      Statements  : Statement_Array (1 .. Max_Statements);
      Arg_Cnt     : Natural range 0 .. Max_Args := 0;
      Stmt_Cnt    : Natural range 0 .. Max_Statements := 0;
   end record
   with Dynamic_Predicate =>
     Arg_Cnt <= Max_Args and Stmt_Cnt <= Max_Statements;

   -- Module (top-level IR)
   Max_Types     : constant := 20;
   Max_Functions : constant := 20;
   type Type_Array is array (Positive range <>) of IR_Type_Def;
   type Function_Array is array (Positive range <>) of IR_Function;

   type IR_Module is record
      IR_Version            : String (1 .. 2) := "v1";
      Module_Name           : IR_Name_String;
      Docstring             : IR_Doc_String;
      Types                 : Type_Array (1 .. Max_Types);
      Functions             : Function_Array (1 .. Max_Functions);
      Type_Cnt              : Natural range 0 .. Max_Types := 0;
      Func_Cnt              : Natural range 0 .. Max_Functions := 0;
      -- v0.8.9: Generic instantiations
      Generic_Insts         : Generic_Inst_Array (1 .. Max_Generic_Insts);
      Generic_Inst_Cnt      : Natural range 0 .. Max_Generic_Insts := 0;
   end record
   with Dynamic_Predicate =>
     Type_Cnt <= Max_Types and Func_Cnt <= Max_Functions;

   -- Helper functions
   function Is_Valid_Module (Module : IR_Module) return Boolean is
     (Module.Func_Cnt > 0)
   with
     Global => null;

   function Get_Type_Name (T : IR_Type_Def) return String
   with
     Global => null,
     Post => Get_Type_Name'Result'Length <= Max_Name_Length;

   function Get_Function_Name (Func : IR_Function) return String
   with
     Global => null,
     Post => Get_Function_Name'Result'Length <= Max_Name_Length;

end STUNIR.Semantic_IR;

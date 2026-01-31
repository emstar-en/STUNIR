-- STUNIR Prolog Family Emitter (SPARK Specification)
-- DO-178C Level A
-- Phase 3b: Language Family Emitters

with STUNIR.Semantic_IR; use STUNIR.Semantic_IR;
with STUNIR.Emitters;    use STUNIR.Emitters;

package STUNIR.Emitters.Prolog is
   pragma SPARK_Mode (On);
   pragma Elaborate_Body;

   -- Prolog dialect enumeration
   type Prolog_Dialect is
     (SWI_Prolog,
      GNU_Prolog,
      SICStus,
      YAP,
      XSB,
      Ciao,
      BProlog,
      ECLiPSe);

   -- Prolog emitter configuration
   type Prolog_Config is record
      Dialect        : Prolog_Dialect := SWI_Prolog;
      Use_Tabling    : Boolean := False;
      Use_CLP        : Boolean := False;
      Use_Assertions : Boolean := False;
      Indent_Size    : Positive := 2;
      Max_Line_Width : Positive := 80;
   end record;

   -- Default configuration
   Default_Config : constant Prolog_Config :=
     (Dialect        => SWI_Prolog,
      Use_Tabling    => False,
      Use_CLP        => False,
      Use_Assertions => False,
      Indent_Size    => 2,
      Max_Line_Width => 80);

   -- Prolog emitter type
   type Prolog_Emitter is new Base_Emitter with record
      Config : Prolog_Config := Default_Config;
   end record;

   -- Override abstract methods from Base_Emitter
   overriding procedure Emit_Module
     (Self   : in out Prolog_Emitter;
      Module : in     IR_Module;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   with
     Pre'Class  => Is_Valid_Module (Module),
     Post'Class => (if Success then Code_Buffers.Length (Output) > 0);

   overriding procedure Emit_Type
     (Self   : in out Prolog_Emitter;
      T      : in     IR_Type_Def;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   with
     Pre'Class  => T.Field_Cnt > 0,
     Post'Class => (if Success then Code_Buffers.Length (Output) >= 0);

   overriding procedure Emit_Function
     (Self   : in out Prolog_Emitter;
      Func   : in     IR_Function;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   with
     Pre'Class  => Func.Arg_Cnt >= 0,
     Post'Class => (if Success then Code_Buffers.Length (Output) >= 0);

   -- Dialect-specific helpers
   function Get_Comment_Prefix (Dialect : Prolog_Dialect) return String
   with
     Global => null,
     Post => Get_Comment_Prefix'Result'Length in 1 .. 3;

   function Get_Module_Syntax (Dialect : Prolog_Dialect) return String
   with
     Global => null,
     Post => Get_Module_Syntax'Result'Length in 6 .. 10;

   function Map_Type_To_Prolog
     (Prim_Type : IR_Primitive_Type;
      Dialect   : Prolog_Dialect) return String
   with
     Global => null,
     Post => Map_Type_To_Prolog'Result'Length in 3 .. 15;

   function Supports_Tabling (Dialect : Prolog_Dialect) return Boolean
   with
     Global => null;

   function Supports_CLP (Dialect : Prolog_Dialect) return Boolean
   with
     Global => null;

   function Supports_Assertions (Dialect : Prolog_Dialect) return Boolean
   with
     Global => null;

   -- Prolog-specific utilities
   procedure Emit_Predicate_Head
     (Buffer         : in out IR_Code_Buffer;
      Predicate_Name : in     String;
      Arity          : in     Natural;
      Success        :    out Boolean)
   with
     Pre => Predicate_Name'Length > 0 and Predicate_Name'Length <= 128
            and Arity <= 20;

   procedure Emit_Clause_Separator
     (Buffer  : in out IR_Code_Buffer;
      Success :    out Boolean)
   with
     Post => (if Success then Code_Buffers.Length (Buffer) >= Code_Buffers.Length (Buffer'Old));

   procedure Emit_Body_Goal
     (Buffer  : in out IR_Code_Buffer;
      Goal    : in     String;
      Success :    out Boolean)
   with
     Pre => Goal'Length > 0 and Goal'Length <= 256;

   procedure Emit_Space
     (Buffer  : in out IR_Code_Buffer;
      Success :    out Boolean)
   with
     Post => (if Success then Code_Buffers.Length (Buffer) = Code_Buffers.Length (Buffer'Old) + 1);

   procedure Emit_Newline
     (Buffer  : in out IR_Code_Buffer;
      Success :    out Boolean)
   with
     Post => (if Success then Code_Buffers.Length (Buffer) = Code_Buffers.Length (Buffer'Old) + 1);

end STUNIR.Emitters.Prolog;

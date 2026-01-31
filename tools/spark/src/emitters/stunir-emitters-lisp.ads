-- STUNIR Lisp Family Emitter (SPARK Specification)
-- DO-178C Level A
-- Phase 3b: Language Family Emitters

with STUNIR.Semantic_IR; use STUNIR.Semantic_IR;
with STUNIR.Emitters;    use STUNIR.Emitters;

package STUNIR.Emitters.Lisp is
   pragma SPARK_Mode (On);
   pragma Elaborate_Body;

   -- Lisp dialect enumeration
   type Lisp_Dialect is
     (Common_Lisp,
      Scheme,
      Clojure,
      Racket,
      Emacs_Lisp,
      Guile,
      Hy,
      Janet);

   -- Scheme standard versions
   type Scheme_Standard is (R5RS, R6RS, R7RS);

   -- Lisp emitter configuration
   type Lisp_Config is record
      Dialect         : Lisp_Dialect := Common_Lisp;
      Scheme_Std      : Scheme_Standard := R7RS;
      Indent_Size     : Positive := 2;
      Max_Line_Width  : Positive := 80;
      Include_Types   : Boolean := True;
      Include_Docs    : Boolean := True;
   end record;

   -- Default configuration
   Default_Config : constant Lisp_Config :=
     (Dialect        => Common_Lisp,
      Scheme_Std     => R7RS,
      Indent_Size    => 2,
      Max_Line_Width => 80,
      Include_Types  => True,
      Include_Docs   => True);

   -- Lisp emitter type
   type Lisp_Emitter is new Base_Emitter with record
      Config : Lisp_Config := Default_Config;
   end record;

   -- Override abstract methods from Base_Emitter
   overriding procedure Emit_Module
     (Self   : in out Lisp_Emitter;
      Module : in     IR_Module;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   with
     Pre'Class  => Is_Valid_Module (Module),
     Post'Class => (if Success then Code_Buffers.Length (Output) > 0);

   overriding procedure Emit_Type
     (Self   : in out Lisp_Emitter;
      T      : in     IR_Type_Def;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   with
     Pre'Class  => T.Field_Cnt > 0,
     Post'Class => (if Success then Code_Buffers.Length (Output) >= 0);

   overriding procedure Emit_Function
     (Self   : in out Lisp_Emitter;
      Func   : in     IR_Function;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   with
     Pre'Class  => Func.Arg_Cnt >= 0,
     Post'Class => (if Success then Code_Buffers.Length (Output) >= 0);

   -- Dialect-specific helpers
   function Get_Comment_Prefix (Dialect : Lisp_Dialect) return String
   with
     Global => null,
     Post => Get_Comment_Prefix'Result'Length in 1 .. 4;

   function Get_Module_Syntax (Dialect : Lisp_Dialect) return String
   with
     Global => null,
     Post => Get_Module_Syntax'Result'Length in 6 .. 20;

   function Map_Type_To_Lisp
     (Prim_Type : IR_Primitive_Type;
      Dialect   : Lisp_Dialect) return String
   with
     Global => null,
     Post => Map_Type_To_Lisp'Result'Length in 3 .. 20;

   -- S-expression utilities
   procedure Emit_List_Start
     (Buffer  : in out IR_Code_Buffer;
      Success :    out Boolean)
   with
     Post => (if Success then Code_Buffers.Length (Buffer) > Code_Buffers.Length (Buffer'Old));

   procedure Emit_List_End
     (Buffer  : in out IR_Code_Buffer;
      Success :    out Boolean)
   with
     Post => (if Success then Code_Buffers.Length (Buffer) > Code_Buffers.Length (Buffer'Old));

   procedure Emit_Atom
     (Buffer  : in out IR_Code_Buffer;
      Atom    : in     String;
      Success :    out Boolean)
   with
     Pre => Atom'Length > 0 and Atom'Length <= 128,
     Post => (if Success then Code_Buffers.Length (Buffer) >= Code_Buffers.Length (Buffer'Old));

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

end STUNIR.Emitters.Lisp;

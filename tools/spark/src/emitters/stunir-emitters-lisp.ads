-- STUNIR Lisp Family Emitter (SPARK Specification)
-- DO-178C Level A
-- Phase 3b: Language Family Emitters

with STUNIR.Emitters;    use STUNIR.Emitters;
with STUNIR.Emitters.Node_Table;
with STUNIR.Emitters.CodeGen;    use STUNIR.Emitters.CodeGen;
with Semantic_IR.Modules;
with Semantic_IR.Declarations;
with Semantic_IR.Nodes;
with Semantic_IR.Types;

package STUNIR.Emitters.Lisp is
   pragma SPARK_Mode (On);
   pragma Elaborate_Body;

   -- Lisp dialect enumeration
   type Lisp_Dialect is
     (Common_Lisp,
      Scheme,
      Clojure,
      ClojureScript,
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

   --  Concrete emitter procedures (not overriding — base has no abstract procs)
   procedure Emit_Module
     (Self    : in out Lisp_Emitter;
      Module  : in     Semantic_IR.Modules.IR_Module;
      Nodes   : in     STUNIR.Emitters.Node_Table.Node_Table;
      Output  :    out STUNIR.Emitters.CodeGen.IR_Code_Buffer;
      Success :    out Boolean)
   with
     Pre  => Semantic_IR.Modules.Is_Valid_Module (Module),
     Post => (if Success then STUNIR.Emitters.CodeGen.Code_Buffers.Length (Output) > 0);

   procedure Emit_Type
     (Self    : in out Lisp_Emitter;
      T       : in     Semantic_IR.Declarations.Type_Declaration;
      Nodes   : in     STUNIR.Emitters.Node_Table.Node_Table;
      Output  :    out STUNIR.Emitters.CodeGen.IR_Code_Buffer;
      Success :    out Boolean)
   with
     Pre  => Semantic_IR.Nodes.Is_Valid_Node_ID (T.Base.Base.ID),
     Post => (if Success then STUNIR.Emitters.CodeGen.Code_Buffers.Length (Output) >= 0);

   procedure Emit_Function
     (Self    : in out Lisp_Emitter;
      Func    : in     Semantic_IR.Declarations.Function_Declaration;
      Nodes   : in     STUNIR.Emitters.Node_Table.Node_Table;
      Output  :    out STUNIR.Emitters.CodeGen.IR_Code_Buffer;
      Success :    out Boolean)
   with
     Pre  => Semantic_IR.Nodes.Is_Valid_Node_ID (Func.Base.Base.ID),
     Post => (if Success then STUNIR.Emitters.CodeGen.Code_Buffers.Length (Output) >= 0);

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
     (Prim_Type : Semantic_IR.Types.IR_Primitive_Type;
      Dialect   : Lisp_Dialect) return String
   with
     Global => null,
     Post => Map_Type_To_Lisp'Result'Length in 3 .. 20;

   -- S-expression utilities
   procedure Emit_List_Start
     (Buffer  : in out IR_Code_Buffer;
      Success :    out Boolean)
   with
     Post => (if Success then STUNIR.Emitters.CodeGen.Code_Buffers.Length (Buffer) > STUNIR.Emitters.CodeGen.Code_Buffers.Length (Buffer'Old));

   procedure Emit_List_End
     (Buffer  : in out IR_Code_Buffer;
      Success :    out Boolean)
   with
     Post => (if Success then STUNIR.Emitters.CodeGen.Code_Buffers.Length (Buffer) > STUNIR.Emitters.CodeGen.Code_Buffers.Length (Buffer'Old));

   procedure Emit_Atom
     (Buffer  : in out IR_Code_Buffer;
      Atom    : in     String;
      Success :    out Boolean)
   with
     Pre => Atom'Length > 0 and Atom'Length <= 128,
     Post => (if Success then STUNIR.Emitters.CodeGen.Code_Buffers.Length (Buffer) >= STUNIR.Emitters.CodeGen.Code_Buffers.Length (Buffer'Old));

   procedure Emit_Space
     (Buffer  : in out IR_Code_Buffer;
      Success :    out Boolean)
   with
     Post => (if Success then STUNIR.Emitters.CodeGen.Code_Buffers.Length (Buffer) = STUNIR.Emitters.CodeGen.Code_Buffers.Length (Buffer'Old) + 1);

   procedure Emit_Newline
     (Buffer  : in out IR_Code_Buffer;
      Success :    out Boolean)
   with
     Post => (if Success then STUNIR.Emitters.CodeGen.Code_Buffers.Length (Buffer) = STUNIR.Emitters.CodeGen.Code_Buffers.Length (Buffer'Old) + 1);

end STUNIR.Emitters.Lisp;

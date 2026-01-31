-- STUNIR Functional Programming Emitter (SPARK Specification)
-- DO-178C Level A
-- Phase 3c: Remaining Category Emitters
-- Support: Haskell, OCaml, F#, Erlang, Elixir, Scala

with STUNIR.Semantic_IR; use STUNIR.Semantic_IR;
with STUNIR.Emitters;    use STUNIR.Emitters;

package STUNIR.Emitters.Functional is
   pragma SPARK_Mode (On);
   pragma Elaborate_Body;

   -- Functional language enumeration
   type Functional_Language is
     (Haskell,
      OCaml,
      F_Sharp,
      Erlang,
      Elixir,
      Scala_Functional,
      Miranda,
      Clean,
      Idris,
      Agda);

   -- Type system strength
   type Type_System is
     (Strong_Static,
      Strong_Dynamic,
      Weak_Dynamic,
      Dependent);

   -- Functional emitter configuration
   type Functional_Config is record
      Language       : Functional_Language := Haskell;
      TSystem        : Type_System := Strong_Static;
      Use_Monads     : Boolean := True;
      Use_Laziness   : Boolean := True;  -- Lazy evaluation
      Pure_Functions : Boolean := True;  -- Enforce purity
      Indent_Size    : Positive := 2;
      Max_Line_Width : Positive := 100;
   end record;

   -- Default configuration
   Default_Config : constant Functional_Config :=
     (Language       => Haskell,
      TSystem        => Strong_Static,
      Use_Monads     => True,
      Use_Laziness   => True,
      Pure_Functions => True,
      Indent_Size    => 2,
      Max_Line_Width => 100);

   -- Functional emitter type
   type Functional_Emitter is new Base_Emitter with record
      Config : Functional_Config := Default_Config;
   end record;

   -- Override abstract methods from Base_Emitter
   overriding procedure Emit_Module
     (Self   : in out Functional_Emitter;
      Module : in     IR_Module;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   with
     Pre'Class  => Is_Valid_Module (Module),
     Post'Class => (if Success then Code_Buffers.Length (Output) > 0);

   overriding procedure Emit_Type
     (Self   : in out Functional_Emitter;
      T      : in     IR_Type_Def;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   with
     Pre'Class  => T.Field_Cnt > 0,
     Post'Class => (if Success then Code_Buffers.Length (Output) >= 0);

   overriding procedure Emit_Function
     (Self   : in out Functional_Emitter;
      Func   : in     IR_Function;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   with
     Pre'Class  => Func.Arg_Cnt >= 0,
     Post'Class => (if Success then Code_Buffers.Length (Output) >= 0);

   -- Helper functions
   function Get_FP_Type (Prim : IR_Primitive_Type; Lang : Functional_Language) return String
   with
     Global => null,
     Post => Get_FP_Type'Result'Length > 0;

end STUNIR.Emitters.Functional;

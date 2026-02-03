-- STUNIR Constraints Emitter (SPARK Specification)
-- DO-178C Level A
-- Phase 3c: Remaining Category Emitters
-- Support: Constraint programming, CSP solvers

with STUNIR.Semantic_IR; use STUNIR.Semantic_IR;
with STUNIR.Emitters;    use STUNIR.Emitters;

package STUNIR.Emitters.Constraints is
   pragma SPARK_Mode (On);
   pragma Elaborate_Body;

   -- Constraint solver enumeration
   type Constraint_Solver is
     (MiniZinc,
      Gecode,
      Choco,
      OR_Tools,
      Z3,
      CLP_FD,      -- Constraint Logic Programming over Finite Domains
      ECLiPSe_CLP,
      JaCoP);

   -- Constraint type
   type Constraint_Type is
     (Finite_Domain,
      Integer,
      Boolean,
      Set,
      Real);

   -- Constraints emitter configuration
   type Constraints_Config is record
      Solver         : Constraint_Solver := MiniZinc;
      CType          : Constraint_Type := Finite_Domain;
      Generate_Search: Boolean := True;
      Optimize       : Boolean := False;
      Indent_Size    : Positive := 2;
      Max_Line_Width : Positive := 100;
   end record;

   -- Default configuration
   Default_Config : constant Constraints_Config :=
     (Solver         => MiniZinc,
      CType          => Finite_Domain,
      Generate_Search=> True,
      Optimize       => False,
      Indent_Size    => 2,
      Max_Line_Width => 100);

   -- Constraints emitter type
   type Constraints_Emitter is new Base_Emitter with record
      Config : Constraints_Config := Default_Config;
   end record;

   -- Override abstract methods from Base_Emitter
   overriding procedure Emit_Module
     (Self   : in out Constraints_Emitter;
      Module : in     IR_Module;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   with
     Pre'Class  => Is_Valid_Module (Module),
     Post'Class => (if Success then Code_Buffers.Length (Output) > 0);

   overriding procedure Emit_Type
     (Self   : in out Constraints_Emitter;
      T      : in     IR_Type_Def;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   with
     Pre'Class  => T.Field_Cnt > 0,
     Post'Class => (if Success then Code_Buffers.Length (Output) >= 0);

   overriding procedure Emit_Function
     (Self   : in out Constraints_Emitter;
      Func   : in     IR_Function;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   with
     Pre'Class  => Func.Arg_Cnt >= 0,
     Post'Class => (if Success then Code_Buffers.Length (Output) >= 0);

end STUNIR.Emitters.Constraints;

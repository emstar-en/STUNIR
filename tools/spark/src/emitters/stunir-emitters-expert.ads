-- STUNIR Expert Systems Emitter (SPARK Specification)
-- DO-178C Level A
-- Phase 3c: Remaining Category Emitters
-- Support: Rule-based systems, Forward/Backward chaining

with STUNIR.Semantic_IR; use STUNIR.Semantic_IR;
with STUNIR.Emitters;    use STUNIR.Emitters;

package STUNIR.Emitters.Expert is
   pragma SPARK_Mode (On);
   pragma Elaborate_Body;

   -- Expert system type enumeration
   type Expert_System_Type is
     (CLIPS,          -- C Language Integrated Production System
      Jess,           -- Java Expert System Shell
      Drools,         -- Business Rules Management System
      RETE,           -- RETE algorithm
      OPS5,           -- Official Production System
      SOAR,           -- State Operator And Result
      Prolog_Based,   -- Prolog-based expert system
      Custom_Rules);

   -- Inference method
   type Inference_Method is
     (Forward_Chaining,
      Backward_Chaining,
      Mixed_Chaining);

   -- Expert system emitter configuration
   type Expert_Config is record
      System_Type    : Expert_System_Type := CLIPS;
      Inference      : Inference_Method := Forward_Chaining;
      Use_Certainty  : Boolean := False;  -- Certainty factors
      Use_Fuzzy      : Boolean := False;  -- Fuzzy logic
      Indent_Size    : Positive := 2;
      Max_Line_Width : Positive := 100;
   end record;

   -- Default configuration
   Default_Config : constant Expert_Config :=
     (System_Type    => CLIPS,
      Inference      => Forward_Chaining,
      Use_Certainty  => False,
      Use_Fuzzy      => False,
      Indent_Size    => 2,
      Max_Line_Width => 100);

   -- Expert emitter type
   type Expert_Emitter is new Base_Emitter with record
      Config : Expert_Config := Default_Config;
   end record;

   -- Override abstract methods from Base_Emitter
   overriding procedure Emit_Module
     (Self   : in out Expert_Emitter;
      Module : in     IR_Module;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   with
     Pre'Class  => Is_Valid_Module (Module),
     Post'Class => (if Success then Code_Buffers.Length (Output) > 0);

   overriding procedure Emit_Type
     (Self   : in out Expert_Emitter;
      T      : in     IR_Type_Def;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   with
     Pre'Class  => T.Field_Cnt > 0,
     Post'Class => (if Success then Code_Buffers.Length (Output) >= 0);

   overriding procedure Emit_Function
     (Self   : in out Expert_Emitter;
      Func   : in     IR_Function;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   with
     Pre'Class  => Func.Arg_Cnt >= 0,
     Post'Class => (if Success then Code_Buffers.Length (Output) >= 0);

end STUNIR.Emitters.Expert;

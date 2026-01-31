-- STUNIR Planning Language Emitter (SPARK Specification)
-- DO-178C Level A
-- Phase 3c: Remaining Category Emitters
-- Support: PDDL, STRIPS, SHOP2

with STUNIR.Semantic_IR; use STUNIR.Semantic_IR;
with STUNIR.Emitters;    use STUNIR.Emitters;

package STUNIR.Emitters.Planning is
   pragma SPARK_Mode (On);
   pragma Elaborate_Body;

   type Planning_Language is (PDDL, PDDL_2_1, PDDL_3_0, STRIPS, ADL, SHOP2, HTN);

   type Planning_Config is record
      Language       : Planning_Language := PDDL;
      Use_Temporal   : Boolean := False;
      Use_Numeric    : Boolean := False;
      Indent_Size    : Positive := 2;
      Max_Line_Width : Positive := 100;
   end record;

   Default_Config : constant Planning_Config :=
     (Language => PDDL, Use_Temporal => False, Use_Numeric => False, Indent_Size => 2, Max_Line_Width => 100);

   type Planning_Emitter is new Base_Emitter with record
      Config : Planning_Config := Default_Config;
   end record;

   overriding procedure Emit_Module (Self : in out Planning_Emitter; Module : in IR_Module; Output : out IR_Code_Buffer; Success: out Boolean)
   with Pre'Class => Is_Valid_Module (Module), Post'Class => (if Success then Code_Buffers.Length (Output) > 0);

   overriding procedure Emit_Type (Self : in out Planning_Emitter; T : in IR_Type_Def; Output : out IR_Code_Buffer; Success: out Boolean)
   with Pre'Class => T.Field_Cnt > 0, Post'Class => (if Success then Code_Buffers.Length (Output) >= 0);

   overriding procedure Emit_Function (Self : in out Planning_Emitter; Func : in IR_Function; Output : out IR_Code_Buffer; Success: out Boolean)
   with Pre'Class => Func.Arg_Cnt >= 0, Post'Class => (if Success then Code_Buffers.Length (Output) >= 0);

end STUNIR.Emitters.Planning;

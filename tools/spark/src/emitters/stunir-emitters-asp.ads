-- STUNIR Answer Set Programming Emitter (SPARK Specification)
-- DO-178C Level A
-- Phase 3c: Remaining Category Emitters
-- Support: Clingo, DLV, Potassco

with STUNIR.Semantic_IR; use STUNIR.Semantic_IR;
with STUNIR.Emitters;    use STUNIR.Emitters;

package STUNIR.Emitters.ASP is
   pragma SPARK_Mode (On);
   pragma Elaborate_Body;

   type ASP_Solver is (Clingo, DLV, Potassco, Smodels, Clasp);

   type ASP_Config is record
      Solver         : ASP_Solver := Clingo;
      Use_Optimization : Boolean := False;
      Use_Aggregates : Boolean := True;
      Indent_Size    : Positive := 2;
      Max_Line_Width : Positive := 100;
   end record;

   Default_Config : constant ASP_Config :=
     (Solver => Clingo, Use_Optimization => False, Use_Aggregates => True, Indent_Size => 2, Max_Line_Width => 100);

   type ASP_Emitter is new Base_Emitter with record
      Config : ASP_Config := Default_Config;
   end record;

   overriding procedure Emit_Module (Self : in out ASP_Emitter; Module : in IR_Module; Output : out IR_Code_Buffer; Success: out Boolean)
   with Pre'Class => Is_Valid_Module (Module), Post'Class => (if Success then Code_Buffers.Length (Output) > 0);

   overriding procedure Emit_Type (Self : in out ASP_Emitter; T : in IR_Type_Def; Output : out IR_Code_Buffer; Success: out Boolean)
   with Pre'Class => T.Field_Cnt > 0, Post'Class => (if Success then Code_Buffers.Length (Output) >= 0);

   overriding procedure Emit_Function (Self : in out ASP_Emitter; Func : in IR_Function; Output : out IR_Code_Buffer; Success: out Boolean)
   with Pre'Class => Func.Arg_Cnt >= 0, Post'Class => (if Success then Code_Buffers.Length (Output) >= 0);

end STUNIR.Emitters.ASP;

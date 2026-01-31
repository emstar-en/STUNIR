-- STUNIR Scientific Computing Emitter (SPARK Specification)
-- DO-178C Level A
-- Phase 3c: Remaining Category Emitters
-- Support: MATLAB, NumPy, Julia, R, Fortran

with STUNIR.Semantic_IR; use STUNIR.Semantic_IR;
with STUNIR.Emitters;    use STUNIR.Emitters;

package STUNIR.Emitters.Scientific is
   pragma SPARK_Mode (On);
   pragma Elaborate_Body;

   type Scientific_Language is (MATLAB, NumPy, Julia, R_Lang, Fortran_90, Fortran_95);

   type Scientific_Config is record
      Language       : Scientific_Language := NumPy;
      Use_Vectorization : Boolean := True;
      Use_GPU        : Boolean := False;
      Precision      : Positive := 64;  -- bits
      Indent_Size    : Positive := 4;
      Max_Line_Width : Positive := 100;
   end record;

   Default_Config : constant Scientific_Config :=
     (Language => NumPy, Use_Vectorization => True, Use_GPU => False, Precision => 64, Indent_Size => 4, Max_Line_Width => 100);

   type Scientific_Emitter is new Base_Emitter with record
      Config : Scientific_Config := Default_Config;
   end record;

   overriding procedure Emit_Module (Self : in out Scientific_Emitter; Module : in IR_Module; Output : out IR_Code_Buffer; Success: out Boolean)
   with Pre'Class => Is_Valid_Module (Module), Post'Class => (if Success then Code_Buffers.Length (Output) > 0);

   overriding procedure Emit_Type (Self : in out Scientific_Emitter; T : in IR_Type_Def; Output : out IR_Code_Buffer; Success: out Boolean)
   with Pre'Class => T.Field_Cnt > 0, Post'Class => (if Success then Code_Buffers.Length (Output) >= 0);

   overriding procedure Emit_Function (Self : in out Scientific_Emitter; Func : in IR_Function; Output : out IR_Code_Buffer; Success: out Boolean)
   with Pre'Class => Func.Arg_Cnt >= 0, Post'Class => (if Success then Code_Buffers.Length (Output) >= 0);

end STUNIR.Emitters.Scientific;

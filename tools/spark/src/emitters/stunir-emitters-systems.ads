-- STUNIR Systems Programming Emitter (SPARK Specification)
-- DO-178C Level A
-- Phase 3c: Remaining Category Emitters
-- Support: Ada, D, Nim, Zig

with STUNIR.Semantic_IR; use STUNIR.Semantic_IR;
with STUNIR.Emitters;    use STUNIR.Emitters;

package STUNIR.Emitters.Systems is
   pragma SPARK_Mode (On);
   pragma Elaborate_Body;

   type Systems_Language is (Ada_2012, Ada_2022, D_Lang, Nim, Zig, Carbon);

   type Systems_Config is record
      Language       : Systems_Language := Ada_2012;
      Use_SPARK      : Boolean := True;  -- For Ada
      Runtime_Checks : Boolean := True;
      Memory_Safety  : Boolean := True;
      Indent_Size    : Positive := 3;
      Max_Line_Width : Positive := 80;
   end record;

   Default_Config : constant Systems_Config :=
     (Language => Ada_2012, Use_SPARK => True, Runtime_Checks => True, Memory_Safety => True, Indent_Size => 3, Max_Line_Width => 80);

   type Systems_Emitter is new Base_Emitter with record
      Config : Systems_Config := Default_Config;
   end record;

   overriding procedure Emit_Module (Self : in out Systems_Emitter; Module : in IR_Module; Output : out IR_Code_Buffer; Success: out Boolean)
   with Pre'Class => Is_Valid_Module (Module), Post'Class => (if Success then Code_Buffers.Length (Output) > 0);

   overriding procedure Emit_Type (Self : in out Systems_Emitter; T : in IR_Type_Def; Output : out IR_Code_Buffer; Success: out Boolean)
   with Pre'Class => T.Field_Cnt > 0, Post'Class => (if Success then Code_Buffers.Length (Output) >= 0);

   overriding procedure Emit_Function (Self : in out Systems_Emitter; Func : in IR_Function; Output : out IR_Code_Buffer; Success: out Boolean)
   with Pre'Class => Func.Arg_Cnt >= 0, Post'Class => (if Success then Code_Buffers.Length (Output) >= 0);

end STUNIR.Emitters.Systems;

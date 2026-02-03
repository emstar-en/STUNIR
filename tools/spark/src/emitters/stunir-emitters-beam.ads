-- STUNIR BEAM VM Emitter (SPARK Specification)
-- DO-178C Level A
-- Phase 3c: Remaining Category Emitters
-- Support: Erlang BEAM, Elixir BEAM bytecode

with STUNIR.Semantic_IR; use STUNIR.Semantic_IR;
with STUNIR.Emitters;    use STUNIR.Emitters;

package STUNIR.Emitters.BEAM is
   pragma SPARK_Mode (On);
   pragma Elaborate_Body;

   type BEAM_Language is (Erlang, Elixir, LFE, Gleam);

   type BEAM_Config is record
      Language       : BEAM_Language := Erlang;
      Emit_Bytecode  : Boolean := False;  -- Emit .beam bytecode
      Use_OTP        : Boolean := True;   -- OTP behaviors
      Indent_Size    : Positive := 2;
      Max_Line_Width : Positive := 100;
   end record;

   Default_Config : constant BEAM_Config :=
     (Language => Erlang, Emit_Bytecode => False, Use_OTP => True, Indent_Size => 2, Max_Line_Width => 100);

   type BEAM_Emitter is new Base_Emitter with record
      Config : BEAM_Config := Default_Config;
   end record;

   overriding procedure Emit_Module (Self : in out BEAM_Emitter; Module : in IR_Module; Output : out IR_Code_Buffer; Success: out Boolean)
   with Pre'Class => Is_Valid_Module (Module), Post'Class => (if Success then Code_Buffers.Length (Output) > 0);

   overriding procedure Emit_Type (Self : in out BEAM_Emitter; T : in IR_Type_Def; Output : out IR_Code_Buffer; Success: out Boolean)
   with Pre'Class => T.Field_Cnt > 0, Post'Class => (if Success then Code_Buffers.Length (Output) >= 0);

   overriding procedure Emit_Function (Self : in out BEAM_Emitter; Func : in IR_Function; Output : out IR_Code_Buffer; Success: out Boolean)
   with Pre'Class => Func.Arg_Cnt >= 0, Post'Class => (if Success then Code_Buffers.Length (Output) >= 0);

end STUNIR.Emitters.BEAM;

-- STUNIR Parser Emitter (SPARK Specification)
-- DO-178C Level A
-- Phase 3c: Remaining Category Emitters
-- Support: Various parser generators

with STUNIR.Semantic_IR; use STUNIR.Semantic_IR;
with STUNIR.Emitters;    use STUNIR.Emitters;

package STUNIR.Emitters.Parser is
   pragma SPARK_Mode (On);
   pragma Elaborate_Body;

   -- Parser generator enumeration
   type Parser_Generator is
     (Yacc,
      Bison,
      ANTLR_Parser,
      JavaCC,
      CUP,
      Happy,
      Menhir,
      Hand_Written);

   -- Parser type
   type Parser_Type is
     (Top_Down,
      Bottom_Up,
      Recursive_Descent,
      Predictive,
      Shift_Reduce);

   -- Parser emitter configuration
   type Parser_Config is record
      Generator      : Parser_Generator := ANTLR_Parser;
      PType          : Parser_Type := Recursive_Descent;
      Generate_AST   : Boolean := True;
      Error_Recovery : Boolean := True;
      Indent_Size    : Positive := 2;
      Max_Line_Width : Positive := 100;
   end record;

   -- Default configuration
   Default_Config : constant Parser_Config :=
     (Generator      => ANTLR_Parser,
      PType          => Recursive_Descent,
      Generate_AST   => True,
      Error_Recovery => True,
      Indent_Size    => 2,
      Max_Line_Width => 100);

   -- Parser emitter type
   type Parser_Emitter is new Base_Emitter with record
      Config : Parser_Config := Default_Config;
   end record;

   -- Override abstract methods from Base_Emitter
   overriding procedure Emit_Module
     (Self   : in out Parser_Emitter;
      Module : in     IR_Module;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   with
     Pre'Class  => Is_Valid_Module (Module),
     Post'Class => (if Success then Code_Buffers.Length (Output) > 0);

   overriding procedure Emit_Type
     (Self   : in out Parser_Emitter;
      T      : in     IR_Type_Def;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   with
     Pre'Class  => T.Field_Cnt > 0,
     Post'Class => (if Success then Code_Buffers.Length (Output) >= 0);

   overriding procedure Emit_Function
     (Self   : in out Parser_Emitter;
      Func   : in     IR_Function;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   with
     Pre'Class  => Func.Arg_Cnt >= 0,
     Post'Class => (if Success then Code_Buffers.Length (Output) >= 0);

end STUNIR.Emitters.Parser;

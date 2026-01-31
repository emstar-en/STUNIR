-- STUNIR Grammar Emitter (SPARK Specification)
-- DO-178C Level A
-- Phase 3c: Remaining Category Emitters
-- Support: ANTLR, PEG, BNF, EBNF, Yacc, Bison

with STUNIR.Semantic_IR; use STUNIR.Semantic_IR;
with STUNIR.Emitters;    use STUNIR.Emitters;

package STUNIR.Emitters.Grammar is
   pragma SPARK_Mode (On);
   pragma Elaborate_Body;

   -- Grammar specification language enumeration
   type Grammar_Language is
     (ANTLR_v4,
      ANTLR_v3,
      PEG,
      BNF,
      EBNF,
      Yacc,
      Bison,
      LALR,
      LL_Star);

   -- Grammar type
   type Grammar_Type is
     (LL,       -- Left-to-right, Leftmost derivation
      LR,       -- Left-to-right, Rightmost derivation
      LALR_1,   -- Look-Ahead LR
      SLR,      -- Simple LR
      Recursive_Descent);

   -- Grammar emitter configuration
   type Grammar_Config is record
      Language       : Grammar_Language := ANTLR_v4;
      GType          : Grammar_Type := LL;
      Generate_AST   : Boolean := True;
      Generate_Visitor : Boolean := True;
      Package_Name   : IR_Name_String;
      Indent_Size    : Positive := 2;
      Max_Line_Width : Positive := 100;
   end record;

   -- Default configuration
   Default_Config : constant Grammar_Config :=
     (Language       => ANTLR_v4,
      GType          => LL,
      Generate_AST   => True,
      Generate_Visitor => True,
      Package_Name   => Name_Strings.To_Bounded_String ("MyGrammar"),
      Indent_Size    => 2,
      Max_Line_Width => 100);

   -- Grammar emitter type
   type Grammar_Emitter is new Base_Emitter with record
      Config : Grammar_Config := Default_Config;
   end record;

   -- Override abstract methods from Base_Emitter
   overriding procedure Emit_Module
     (Self   : in out Grammar_Emitter;
      Module : in     IR_Module;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   with
     Pre'Class  => Is_Valid_Module (Module),
     Post'Class => (if Success then Code_Buffers.Length (Output) > 0);

   overriding procedure Emit_Type
     (Self   : in out Grammar_Emitter;
      T      : in     IR_Type_Def;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   with
     Pre'Class  => T.Field_Cnt > 0,
     Post'Class => (if Success then Code_Buffers.Length (Output) >= 0);

   overriding procedure Emit_Function
     (Self   : in out Grammar_Emitter;
      Func   : in     IR_Function;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   with
     Pre'Class  => Func.Arg_Cnt >= 0,
     Post'Class => (if Success then Code_Buffers.Length (Output) >= 0);

   -- Helper functions
   function Get_Grammar_Extension (Lang : Grammar_Language) return String
   with
     Global => null,
     Post => Get_Grammar_Extension'Result'Length > 0;

end STUNIR.Emitters.Grammar;

-- STUNIR Business Languages Emitter (SPARK Specification)
-- DO-178C Level A
-- Phase 3c: Remaining Category Emitters
-- Support: COBOL, BASIC

with STUNIR.Semantic_IR; use STUNIR.Semantic_IR;
with STUNIR.Emitters;    use STUNIR.Emitters;

package STUNIR.Emitters.Business is
   pragma SPARK_Mode (On);
   pragma Elaborate_Body;

   -- Business language enumeration
   type Business_Language is
     (COBOL_85,
      COBOL_2002,
      COBOL_2014,
      BASIC,
      Visual_Basic,
      VBScript,
      PowerBASIC,
      FreeBASIC);

   -- COBOL dialect
   type COBOL_Dialect is
     (IBM_COBOL,
      GnuCOBOL,
      Micro_Focus_COBOL,
      ANSI_COBOL);

   -- Business emitter configuration
   type Business_Config is record
      Language       : Business_Language := COBOL_85;
      Dialect        : COBOL_Dialect := GnuCOBOL;
      Use_Divisions  : Boolean := True;  -- COBOL divisions
      Fixed_Format   : Boolean := True;  -- COBOL fixed format
      Line_Numbers   : Boolean := False; -- BASIC line numbers
      Indent_Size    : Positive := 4;
      Max_Line_Width : Positive := 72;   -- COBOL standard
   end record;

   -- Default configuration
   Default_Config : constant Business_Config :=
     (Language       => COBOL_85,
      Dialect        => GnuCOBOL,
      Use_Divisions  => True,
      Fixed_Format   => True,
      Line_Numbers   => False,
      Indent_Size    => 4,
      Max_Line_Width => 72);

   -- Business emitter type
   type Business_Emitter is new Base_Emitter with record
      Config : Business_Config := Default_Config;
   end record;

   -- Override abstract methods from Base_Emitter
   overriding procedure Emit_Module
     (Self   : in out Business_Emitter;
      Module : in     IR_Module;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   with
     Pre'Class  => Is_Valid_Module (Module),
     Post'Class => (if Success then Code_Buffers.Length (Output) > 0);

   overriding procedure Emit_Type
     (Self   : in out Business_Emitter;
      T      : in     IR_Type_Def;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   with
     Pre'Class  => T.Field_Cnt > 0,
     Post'Class => (if Success then Code_Buffers.Length (Output) >= 0);

   overriding procedure Emit_Function
     (Self   : in out Business_Emitter;
      Func   : in     IR_Function;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   with
     Pre'Class  => Func.Arg_Cnt >= 0,
     Post'Class => (if Success then Code_Buffers.Length (Output) >= 0);

   -- Helper functions for COBOL generation
   function Get_COBOL_Type (Prim : IR_Primitive_Type) return String
   with
     Global => null,
     Post => Get_COBOL_Type'Result'Length > 0;

   function Get_BASIC_Type (Prim : IR_Primitive_Type) return String
   with
     Global => null,
     Post => Get_BASIC_Type'Result'Length > 0;

end STUNIR.Emitters.Business;

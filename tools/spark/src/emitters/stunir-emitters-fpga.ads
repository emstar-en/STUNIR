-- STUNIR FPGA Hardware Description Emitter (SPARK Specification)
-- DO-178C Level A
-- Phase 3c: Remaining Category Emitters
-- Support: VHDL, Verilog, SystemVerilog

with STUNIR.Semantic_IR; use STUNIR.Semantic_IR;
with STUNIR.Emitters;    use STUNIR.Emitters;

package STUNIR.Emitters.FPGA is
   pragma SPARK_Mode (On);
   pragma Elaborate_Body;

   -- HDL language enumeration
   type HDL_Language is
     (VHDL_87,
      VHDL_93,
      VHDL_2008,
      Verilog_1995,
      Verilog_2001,
      Verilog_2005,
      SystemVerilog_2012);

   -- VHDL style
   type VHDL_Style is
     (Structural,
      Behavioral,
      Dataflow,
      RTL);

   -- FPGA emitter configuration
   type FPGA_Config is record
      Language       : HDL_Language := VHDL_2008;
      Style          : VHDL_Style := RTL;
      Use_Generics   : Boolean := True;  -- VHDL generics
      Use_Components : Boolean := True;  -- Component declarations
      Clock_Name     : IR_Name_String;   -- Clock signal name
      Reset_Name     : IR_Name_String;   -- Reset signal name
      Indent_Size    : Positive := 2;
      Max_Line_Width : Positive := 100;
   end record;

   -- Default configuration
   Default_Config : constant FPGA_Config :=
     (Language       => VHDL_2008,
      Style          => RTL,
      Use_Generics   => True,
      Use_Components => True,
      Clock_Name     => Name_Strings.To_Bounded_String ("clk"),
      Reset_Name     => Name_Strings.To_Bounded_String ("rst"),
      Indent_Size    => 2,
      Max_Line_Width => 100);

   -- FPGA emitter type
   type FPGA_Emitter is new Base_Emitter with record
      Config : FPGA_Config := Default_Config;
   end record;

   -- Override abstract methods from Base_Emitter
   overriding procedure Emit_Module
     (Self   : in out FPGA_Emitter;
      Module : in     IR_Module;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   with
     Pre'Class  => Is_Valid_Module (Module),
     Post'Class => (if Success then Code_Buffers.Length (Output) > 0);

   overriding procedure Emit_Type
     (Self   : in out FPGA_Emitter;
      T      : in     IR_Type_Def;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   with
     Pre'Class  => T.Field_Cnt > 0,
     Post'Class => (if Success then Code_Buffers.Length (Output) >= 0);

   overriding procedure Emit_Function
     (Self   : in out FPGA_Emitter;
      Func   : in     IR_Function;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   with
     Pre'Class  => Func.Arg_Cnt >= 0,
     Post'Class => (if Success then Code_Buffers.Length (Output) >= 0);

   -- Helper functions
   function Get_VHDL_Type (Prim : IR_Primitive_Type) return String
   with
     Global => null,
     Post => Get_VHDL_Type'Result'Length > 0;

   function Get_Verilog_Type (Prim : IR_Primitive_Type) return String
   with
     Global => null,
     Post => Get_Verilog_Type'Result'Length > 0;

end STUNIR.Emitters.FPGA;

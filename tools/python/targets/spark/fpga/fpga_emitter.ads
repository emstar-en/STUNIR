--  STUNIR FPGA Emitter - Ada SPARK Specification
--  VHDL/Verilog HDL emitter
--  DO-178C Level A compliant

pragma SPARK_Mode (On);

with Emitter_Types; use Emitter_Types;

package FPGA_Emitter is

   type HDL_Language is (VHDL, Verilog, SystemVerilog);

   type FPGA_Config is record
      Language : HDL_Language;
   end record;

   Default_Config : constant FPGA_Config := (Language => VHDL);

   procedure Emit_Entity (
      Entity_Name : in Identifier_String;
      Ports       : in String;
      Content     : out Content_String;
      Config      : in FPGA_Config;
      Status      : out Emitter_Status);

   procedure Emit_Process (
      Process_Name : in Identifier_String;
      Sensitivity  : in String;
      Body_Code    : in String;
      Content      : out Content_String;
      Config       : in FPGA_Config;
      Status       : out Emitter_Status);

end FPGA_Emitter;

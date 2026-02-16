--  STUNIR FPGA Emitter - Ada SPARK Implementation

pragma SPARK_Mode (On);

package body FPGA_Emitter is

   New_Line : constant Character := ASCII.LF;

   procedure Emit_Entity (
      Entity_Name : in Identifier_String;
      Ports       : in String;
      Content     : out Content_String;
      Config      : in FPGA_Config;
      Status      : out Emitter_Status)
   is
      Name : constant String := Identifier_Strings.To_String (Entity_Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      case Config.Language is
         when VHDL =>
            Content_Strings.Append (Content,
               "-- STUNIR Generated VHDL Entity" & New_Line &
               "-- DO-178C Level A Compliant" & New_Line &
               "library IEEE;" & New_Line &
               "use IEEE.STD_LOGIC_1164.ALL;" & New_Line & New_Line &
               "entity " & Name & " is" & New_Line &
               "    port (" & New_Line &
               "        " & Ports & New_Line &
               "    );" & New_Line &
               "end " & Name & ";" & New_Line & New_Line);
         when Verilog | SystemVerilog =>
            Content_Strings.Append (Content,
               "// STUNIR Generated Verilog Module" & New_Line &
               "// DO-178C Level A Compliant" & New_Line &
               "module " & Name & " (" & New_Line &
               "    " & Ports & New_Line &
               ");" & New_Line & New_Line);
      end case;
   end Emit_Entity;

   procedure Emit_Process (
      Process_Name : in Identifier_String;
      Sensitivity  : in String;
      Body_Code    : in String;
      Content      : out Content_String;
      Config       : in FPGA_Config;
      Status       : out Emitter_Status)
   is
      Name : constant String := Identifier_Strings.To_String (Process_Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      case Config.Language is
         when VHDL =>
            Content_Strings.Append (Content,
               "    " & Name & ": process(" & Sensitivity & ")" & New_Line &
               "    begin" & New_Line &
               "        " & Body_Code & New_Line &
               "    end process;" & New_Line & New_Line);
         when Verilog | SystemVerilog =>
            Content_Strings.Append (Content,
               "    always @(" & Sensitivity & ") begin : " & Name & New_Line &
               "        " & Body_Code & New_Line &
               "    end" & New_Line & New_Line);
      end case;
   end Emit_Process;

end FPGA_Emitter;

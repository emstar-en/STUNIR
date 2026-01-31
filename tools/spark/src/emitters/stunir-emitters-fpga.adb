-- STUNIR FPGA Hardware Description Emitter (SPARK Body)
-- DO-178C Level A
-- Phase 3c: Remaining Category Emitters

with Ada.Strings; use Ada.Strings;

package body STUNIR.Emitters.FPGA is
   pragma SPARK_Mode (On);

   -- Map IR primitive types to VHDL types
   function Get_VHDL_Type (Prim : IR_Primitive_Type) return String is
   begin
      case Prim is
         when Type_String => return "string";
         when Type_Int | Type_I32 => return "signed(31 downto 0)";
         when Type_I8  => return "signed(7 downto 0)";
         when Type_I16 => return "signed(15 downto 0)";
         when Type_I64 => return "signed(63 downto 0)";
         when Type_U8  => return "unsigned(7 downto 0)";
         when Type_U16 => return "unsigned(15 downto 0)";
         when Type_U32 => return "unsigned(31 downto 0)";
         when Type_U64 => return "unsigned(63 downto 0)";
         when Type_Float | Type_F32 => return "real";
         when Type_F64 => return "real";
         when Type_Bool => return "std_logic";
         when Type_Void => return "";
      end case;
   end Get_VHDL_Type;

   -- Map IR primitive types to Verilog types
   function Get_Verilog_Type (Prim : IR_Primitive_Type) return String is
   begin
      case Prim is
         when Type_String => return "string";
         when Type_Int | Type_I32 => return "reg signed [31:0]";
         when Type_I8  => return "reg signed [7:0]";
         when Type_I16 => return "reg signed [15:0]";
         when Type_I64 => return "reg signed [63:0]";
         when Type_U8  => return "reg [7:0]";
         when Type_U16 => return "reg [15:0]";
         when Type_U32 => return "reg [31:0]";
         when Type_U64 => return "reg [63:0]";
         when Type_Float | Type_F32 => return "real";
         when Type_F64 => return "real";
         when Type_Bool => return "reg";
         when Type_Void => return "";
      end case;
   end Get_Verilog_Type;

   -- Emit complete module
   overriding procedure Emit_Module
     (Self   : in out FPGA_Emitter;
      Module : in     IR_Module;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   is
   begin
      Code_Buffers.Set_Bounded_String
        (Target => Output,
         Source => "",
         Drop   => Ada.Strings.Right);

      if Self.Config.Language in VHDL_87 .. VHDL_2008 then
         -- Emit VHDL entity
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "-- STUNIR Generated VHDL" & ASCII.LF);
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "-- DO-178C Level A" & ASCII.LF & ASCII.LF);
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "library IEEE;" & ASCII.LF);
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "use IEEE.STD_LOGIC_1164.ALL;" & ASCII.LF);
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "use IEEE.NUMERIC_STD.ALL;" & ASCII.LF & ASCII.LF);
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "entity " & Name_Strings.To_String (Module.Module_Name) & " is" & ASCII.LF);
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "  port (" & ASCII.LF);
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "    " & Name_Strings.To_String (Self.Config.Clock_Name) & " : in std_logic;" & ASCII.LF);
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "    " & Name_Strings.To_String (Self.Config.Reset_Name) & " : in std_logic" & ASCII.LF);
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "  );" & ASCII.LF);
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "end " & Name_Strings.To_String (Module.Module_Name) & ";" & ASCII.LF & ASCII.LF);
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "architecture " & VHDL_Style'Image (Self.Config.Style) & " of "
                        & Name_Strings.To_String (Module.Module_Name) & " is" & ASCII.LF);
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "begin" & ASCII.LF);
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "  -- STUNIR generated logic" & ASCII.LF);
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "end " & VHDL_Style'Image (Self.Config.Style) & ";" & ASCII.LF);
      else
         -- Emit Verilog module
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "// STUNIR Generated Verilog" & ASCII.LF);
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "// DO-178C Level A" & ASCII.LF & ASCII.LF);
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "module " & Name_Strings.To_String (Module.Module_Name) & " (" & ASCII.LF);
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "  input " & Name_Strings.To_String (Self.Config.Clock_Name) & "," & ASCII.LF);
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "  input " & Name_Strings.To_String (Self.Config.Reset_Name) & ASCII.LF);
         Code_Buffers.Append
           (Source   => Output,
            New_Item => ");" & ASCII.LF & ASCII.LF);
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "  // STUNIR generated logic" & ASCII.LF & ASCII.LF);
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "endmodule" & ASCII.LF);
      end if;

      Success := True;
   end Emit_Module;

   -- Emit type definition
   overriding procedure Emit_Type
     (Self   : in out FPGA_Emitter;
      T      : in     IR_Type_Def;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   is
   begin
      Code_Buffers.Set_Bounded_String
        (Target => Output,
         Source => "",
         Drop   => Ada.Strings.Right);

      if Self.Config.Language in VHDL_87 .. VHDL_2008 then
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "  type " & Name_Strings.To_String (T.Name) & " is record" & ASCII.LF);
         for I in 1 .. T.Field_Cnt loop
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "    " & Name_Strings.To_String (T.Fields (I).Name) & " : std_logic;" & ASCII.LF);
         end loop;
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "  end record;" & ASCII.LF);
      else
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "  typedef struct {" & ASCII.LF);
         for I in 1 .. T.Field_Cnt loop
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "    logic " & Name_Strings.To_String (T.Fields (I).Name) & ";" & ASCII.LF);
         end loop;
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "  } " & Name_Strings.To_String (T.Name) & ";" & ASCII.LF);
      end if;

      Success := True;
   end Emit_Type;

   -- Emit function definition (process in VHDL, always block in Verilog)
   overriding procedure Emit_Function
     (Self   : in out FPGA_Emitter;
      Func   : in     IR_Function;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   is
   begin
      Code_Buffers.Set_Bounded_String
        (Target => Output,
         Source => "",
         Drop   => Ada.Strings.Right);

      if Self.Config.Language in VHDL_87 .. VHDL_2008 then
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "  " & Name_Strings.To_String (Func.Name) & "_proc: process("
                        & Name_Strings.To_String (Self.Config.Clock_Name) & ", "
                        & Name_Strings.To_String (Self.Config.Reset_Name) & ")" & ASCII.LF);
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "  begin" & ASCII.LF);
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "    -- STUNIR generated process" & ASCII.LF);
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "  end process;" & ASCII.LF);
      else
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "  always @(posedge " & Name_Strings.To_String (Self.Config.Clock_Name)
                        & " or posedge " & Name_Strings.To_String (Self.Config.Reset_Name) & ") begin" & ASCII.LF);
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "    // STUNIR generated always block" & ASCII.LF);
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "  end" & ASCII.LF);
      end if;

      Success := True;
   end Emit_Function;

end STUNIR.Emitters.FPGA;

//! FPGA emitters
//!
//! Supports: Verilog, VHDL, SystemVerilog

use crate::types::*;

/// FPGA HDL language
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HDLLanguage {
    Verilog,
    VHDL,
    SystemVerilog,
}

impl std::fmt::Display for HDLLanguage {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            HDLLanguage::Verilog => write!(f, "Verilog"),
            HDLLanguage::VHDL => write!(f, "VHDL"),
            HDLLanguage::SystemVerilog => write!(f, "SystemVerilog"),
        }
    }
}

/// Emit FPGA HDL code
pub fn emit(language: HDLLanguage, module_name: &str) -> EmitterResult<String> {
    match language {
        HDLLanguage::Verilog => emit_verilog(module_name),
        HDLLanguage::VHDL => emit_vhdl(module_name),
        HDLLanguage::SystemVerilog => emit_systemverilog(module_name),
    }
}

fn emit_verilog(module_name: &str) -> EmitterResult<String> {
    let mut code = String::new();
    
    code.push_str("// STUNIR Generated Verilog\n");
    code.push_str(&format!("// Module: {}\n", module_name));
    code.push_str("// Generator: Rust Pipeline\n\n");
    
    code.push_str(&format!("module {} (\n", module_name));
    code.push_str("    input wire clk,\n");
    code.push_str("    input wire rst,\n");
    code.push_str("    input wire [31:0] data_in,\n");
    code.push_str("    output reg [31:0] data_out\n");
    code.push_str(");\n\n");
    
    code.push_str("    always @(posedge clk or posedge rst) begin\n");
    code.push_str("        if (rst)\n");
    code.push_str("            data_out <= 32'h0;\n");
    code.push_str("        else\n");
    code.push_str("            data_out <= data_in;\n");
    code.push_str("    end\n\n");
    
    code.push_str("endmodule\n");
    
    Ok(code)
}

fn emit_vhdl(module_name: &str) -> EmitterResult<String> {
    let mut code = String::new();
    
    code.push_str("-- STUNIR Generated VHDL\n");
    code.push_str(&format!("-- Module: {}\n", module_name));
    code.push_str("-- Generator: Rust Pipeline\n\n");
    
    code.push_str("library IEEE;\n");
    code.push_str("use IEEE.STD_LOGIC_1164.ALL;\n");
    code.push_str("use IEEE.NUMERIC_STD.ALL;\n\n");
    
    code.push_str(&format!("entity {} is\n", module_name));
    code.push_str("    Port (\n");
    code.push_str("        clk : in STD_LOGIC;\n");
    code.push_str("        rst : in STD_LOGIC;\n");
    code.push_str("        data_in : in STD_LOGIC_VECTOR(31 downto 0);\n");
    code.push_str("        data_out : out STD_LOGIC_VECTOR(31 downto 0)\n");
    code.push_str("    );\n");
    code.push_str(&format!("end {};\n\n", module_name));
    
    code.push_str(&format!("architecture Behavioral of {} is\n", module_name));
    code.push_str("begin\n");
    code.push_str("    process(clk, rst)\n");
    code.push_str("    begin\n");
    code.push_str("        if rst = '1' then\n");
    code.push_str("            data_out <= (others => '0');\n");
    code.push_str("        elsif rising_edge(clk) then\n");
    code.push_str("            data_out <= data_in;\n");
    code.push_str("        end if;\n");
    code.push_str("    end process;\n");
    code.push_str("end Behavioral;\n");
    
    Ok(code)
}

fn emit_systemverilog(module_name: &str) -> EmitterResult<String> {
    let mut code = String::new();
    
    code.push_str("// STUNIR Generated SystemVerilog\n");
    code.push_str(&format!("// Module: {}\n", module_name));
    code.push_str("// Generator: Rust Pipeline\n\n");
    
    code.push_str(&format!("module {} (\n", module_name));
    code.push_str("    input logic clk,\n");
    code.push_str("    input logic rst,\n");
    code.push_str("    input logic [31:0] data_in,\n");
    code.push_str("    output logic [31:0] data_out\n");
    code.push_str(");\n\n");
    
    code.push_str("    always_ff @(posedge clk or posedge rst) begin\n");
    code.push_str("        if (rst)\n");
    code.push_str("            data_out <= 32'h0;\n");
    code.push_str("        else\n");
    code.push_str("            data_out <= data_in;\n");
    code.push_str("    end\n\n");
    
    code.push_str("endmodule\n");
    
    Ok(code)
}

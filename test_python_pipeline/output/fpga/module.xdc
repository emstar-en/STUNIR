# STUNIR FPGA Constraints: module
# Epoch: 1769938964

# Clock constraint (100 MHz)
create_clock -period 10.000 -name clk [get_ports clk]

# Input delay constraints
set_input_delay -clock clk -max 2.0 [get_ports start]
set_input_delay -clock clk -min 0.5 [get_ports start]

# Output delay constraints
set_output_delay -clock clk -max 2.0 [get_ports done]
set_output_delay -clock clk -max 2.0 [get_ports result*]

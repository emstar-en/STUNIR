// STUNIR FPGA Testbench: module
`timescale 1ns/1ps

module module_tb;
    reg clk = 0;
    reg rst_n = 0;
    reg start = 0;
    wire done;
    wire [31:0] result;
    
    module_top uut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .done(done),
        .result(result)
    );
    
    always #5 clk = ~clk;  // 100 MHz
    
    initial begin
        $dumpfile("module.vcd");
        $dumpvars(0, module_tb);
        
        #20 rst_n = 1;
        #10 start = 1;
        #10 start = 0;
        
        wait(done);
        $display("Result: %d", result);
        #20 $finish;
    end
endmodule

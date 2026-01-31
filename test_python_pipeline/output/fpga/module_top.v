// STUNIR FPGA Top Module: module
// Epoch: 1769856341

module module_top (
    input wire clk,
    input wire rst_n,
    input wire start,
    output wire done,
    output wire [31:0] result
);

    // Instantiate first function module
    parse_heartbeat u_parse_heartbeat (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .done(done),
        .result(result)
    );

endmodule

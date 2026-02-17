// STUNIR Generated Verilog Module: parse_heartbeat
// Epoch: 1769938964

module parse_heartbeat (
    input wire clk,
    input wire rst_n,
    input wire start,
    output reg done,
    output reg [31:0] result
);

    // State machine
    localparam IDLE = 2'b00;
    localparam EXEC = 2'b01;
    localparam DONE_STATE = 2'b10;
    reg [1:0] state;


    // Sequential logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            done <= 1'b0;
            result <= 32'd0;
        end else begin
            case (state)
                IDLE: begin
                    if (start) begin
                        state <= EXEC;
                        done <= 1'b0;
                    end
                end
                EXEC: begin
                    // Computation
                    state <= DONE_STATE;
                end
                DONE_STATE: begin
                    done <= 1'b1;
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule
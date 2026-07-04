`default_nettype none
`timescale 1ns/1ns

// COMPUTE CORE — stage 6: 驻留 WARPS_PER_CORE 个 warp（switch-on-stall 交错调度）。
// 共享：1 fetcher（被调度的前端）。每 warp：1 decoder + warp_instruction 锁存 +
// T×{registers,alu,lsu,pc}，各由自身 warp_state[w] 驱动。所有 per-(warp,thread) 内部信号
// 拍平成 1D [W*T]（sv2v 友好），索引 p = w*THREADS_PER_BLOCK + i。
module core #(
    parameter DATA_MEM_ADDR_BITS = 8,
    parameter DATA_MEM_DATA_BITS = 8,
    parameter PROGRAM_MEM_ADDR_BITS = 8,
    parameter PROGRAM_MEM_DATA_BITS = 16,
    parameter THREADS_PER_BLOCK = 4,
    parameter WARPS_PER_CORE = 2
) (
    input wire clk,
    input wire reset,
    input wire start,
    output wire done,

    // 每 warp block 元数据
    input wire [7:0] warp_block_id [WARPS_PER_CORE-1:0],
    input wire [$clog2(THREADS_PER_BLOCK):0] warp_thread_count [WARPS_PER_CORE-1:0],

    // 程序内存（共享 fetcher）
    output reg program_mem_read_valid,
    output reg [PROGRAM_MEM_ADDR_BITS-1:0] program_mem_read_address,
    input reg program_mem_read_ready,
    input reg [PROGRAM_MEM_DATA_BITS-1:0] program_mem_read_data,

    // 数据内存：每 (warp,thread) 一个端口，拍平 W*T
    output reg [WARPS_PER_CORE*THREADS_PER_BLOCK-1:0] data_mem_read_valid,
    output reg [DATA_MEM_ADDR_BITS-1:0] data_mem_read_address [WARPS_PER_CORE*THREADS_PER_BLOCK-1:0],
    input reg [WARPS_PER_CORE*THREADS_PER_BLOCK-1:0] data_mem_read_ready,
    input reg [DATA_MEM_DATA_BITS-1:0] data_mem_read_data [WARPS_PER_CORE*THREADS_PER_BLOCK-1:0],
    output reg [WARPS_PER_CORE*THREADS_PER_BLOCK-1:0] data_mem_write_valid,
    output reg [DATA_MEM_ADDR_BITS-1:0] data_mem_write_address [WARPS_PER_CORE*THREADS_PER_BLOCK-1:0],
    output reg [DATA_MEM_DATA_BITS-1:0] data_mem_write_data [WARPS_PER_CORE*THREADS_PER_BLOCK-1:0],
    input reg [WARPS_PER_CORE*THREADS_PER_BLOCK-1:0] data_mem_write_ready
);
    // 共享前端
    reg [2:0] fetcher_state;
    reg [15:0] instruction;
    reg [15:0] warp_instruction [WARPS_PER_CORE-1:0];

    // scheduler <-> core
    wire [7:0] current_pc;
    wire [2:0] core_state;
    wire [$clog2(WARPS_PER_CORE)-1:0] current_warp;
    wire [7:0] warp_pc [WARPS_PER_CORE-1:0];
    wire [2:0] warp_state [WARPS_PER_CORE-1:0];
    wire [7:0] warp_next_pc [WARPS_PER_CORE-1:0];

    // per-warp decoded 信号
    wire [3:0] dec_rd [WARPS_PER_CORE-1:0];
    wire [3:0] dec_rs [WARPS_PER_CORE-1:0];
    wire [3:0] dec_rt [WARPS_PER_CORE-1:0];
    wire [2:0] dec_nzp [WARPS_PER_CORE-1:0];
    wire [7:0] dec_imm [WARPS_PER_CORE-1:0];
    wire dec_reg_we [WARPS_PER_CORE-1:0];
    wire dec_mem_re [WARPS_PER_CORE-1:0];
    wire dec_mem_we [WARPS_PER_CORE-1:0];
    wire dec_mem_pred [WARPS_PER_CORE-1:0];
    wire dec_nzp_we [WARPS_PER_CORE-1:0];
    wire [1:0] dec_reg_mux [WARPS_PER_CORE-1:0];
    wire [1:0] dec_alu_arith [WARPS_PER_CORE-1:0];
    wire dec_alu_out_mux [WARPS_PER_CORE-1:0];
    wire dec_pc_mux [WARPS_PER_CORE-1:0];
    wire dec_ret [WARPS_PER_CORE-1:0];

    // per-(warp,thread) 数据通路信号（拍平 W*T）
    wire [7:0] rs [WARPS_PER_CORE*THREADS_PER_BLOCK-1:0];
    wire [7:0] rt [WARPS_PER_CORE*THREADS_PER_BLOCK-1:0];
    wire [7:0] alu_out [WARPS_PER_CORE*THREADS_PER_BLOCK-1:0];
    wire [7:0] lsu_out [WARPS_PER_CORE*THREADS_PER_BLOCK-1:0];
    wire [1:0] lsu_state [WARPS_PER_CORE*THREADS_PER_BLOCK-1:0];
    wire [7:0] next_pc [WARPS_PER_CORE*THREADS_PER_BLOCK-1:0];
    wire [2:0] thread_nzp [WARPS_PER_CORE*THREADS_PER_BLOCK-1:0];

    // current warp 的 decoded 选择（喂给 scheduler）
    wire cur_mem_re = dec_mem_re[current_warp];
    wire cur_mem_we = dec_mem_we[current_warp];
    wire cur_ret    = dec_ret[current_warp];

    // 共享 fetcher（取 current warp）
    fetcher #(
        .PROGRAM_MEM_ADDR_BITS(PROGRAM_MEM_ADDR_BITS),
        .PROGRAM_MEM_DATA_BITS(PROGRAM_MEM_DATA_BITS)
    ) fetcher_instance (
        .clk(clk), .reset(reset),
        .core_state(core_state),
        .current_pc(current_pc),
        .mem_read_valid(program_mem_read_valid),
        .mem_read_address(program_mem_read_address),
        .mem_read_ready(program_mem_read_ready),
        .mem_read_data(program_mem_read_data),
        .fetcher_state(fetcher_state),
        .instruction(instruction)
    );

    // 取回时把指令锁存进 current warp 的槽
    always @(posedge clk) begin
        if (!reset && fetcher_state == 3'b010)
            warp_instruction[current_warp] <= instruction;
    end

    // warp 调度器
    scheduler #(
        .THREADS_PER_BLOCK(THREADS_PER_BLOCK),
        .WARPS_PER_CORE(WARPS_PER_CORE)
    ) scheduler_instance (
        .clk(clk), .reset(reset), .start(start),
        .warp_thread_count(warp_thread_count),
        .decoded_mem_read_enable(cur_mem_re),
        .decoded_mem_write_enable(cur_mem_we),
        .decoded_ret(cur_ret),
        .fetcher_state(fetcher_state),
        .lsu_state(lsu_state),
        .warp_next_pc(warp_next_pc),
        .current_pc(current_pc),
        .warp_pc(warp_pc),
        .warp_state(warp_state),
        .current_warp(current_warp),
        .core_state(core_state),
        .done(done)
    );

    genvar w, i;
    generate
        for (w = 0; w < WARPS_PER_CORE; w = w + 1) begin : warps
            // 每 warp 一份 decoder，吃自身锁存指令，由自身 warp_state 驱动
            decoder decoder_instance (
                .clk(clk), .reset(reset),
                .core_state(warp_state[w]),
                .instruction(warp_instruction[w]),
                .decoded_rd_address(dec_rd[w]),
                .decoded_rs_address(dec_rs[w]),
                .decoded_rt_address(dec_rt[w]),
                .decoded_nzp(dec_nzp[w]),
                .decoded_immediate(dec_imm[w]),
                .decoded_reg_write_enable(dec_reg_we[w]),
                .decoded_mem_read_enable(dec_mem_re[w]),
                .decoded_mem_write_enable(dec_mem_we[w]),
                .decoded_mem_pred_enable(dec_mem_pred[w]),
                .decoded_nzp_write_enable(dec_nzp_we[w]),
                .decoded_reg_input_mux(dec_reg_mux[w]),
                .decoded_alu_arithmetic_mux(dec_alu_arith[w]),
                .decoded_alu_output_mux(dec_alu_out_mux[w]),
                .decoded_pc_mux(dec_pc_mux[w]),
                .decoded_ret(dec_ret[w])
            );

            // 代表线程（最低索引）的 next_pc 作为该 warp 的 warp_next_pc
            assign warp_next_pc[w] = next_pc[w*THREADS_PER_BLOCK + 0];

            for (i = 0; i < THREADS_PER_BLOCK; i = i + 1) begin : threads
                localparam integer p = w*THREADS_PER_BLOCK + i;

                alu alu_instance (
                    .clk(clk), .reset(reset),
                    .enable(i < warp_thread_count[w]),
                    .core_state(warp_state[w]),
                    .decoded_alu_arithmetic_mux(dec_alu_arith[w]),
                    .decoded_alu_output_mux(dec_alu_out_mux[w]),
                    .rs(rs[p]), .rt(rt[p]), .alu_out(alu_out[p])
                );

                lsu lsu_instance (
                    .clk(clk), .reset(reset),
                    .enable(i < warp_thread_count[w]),
                    .core_state(warp_state[w]),
                    .decoded_mem_read_enable(dec_mem_re[w]),
                    .decoded_mem_write_enable(dec_mem_we[w]),
                    .mem_write_predicate_ok((!dec_mem_pred[w]) || ((thread_nzp[p] & dec_nzp[w]) != 3'b0)),
                    .mem_read_valid(data_mem_read_valid[p]),
                    .mem_read_address(data_mem_read_address[p]),
                    .mem_read_ready(data_mem_read_ready[p]),
                    .mem_read_data(data_mem_read_data[p]),
                    .mem_write_valid(data_mem_write_valid[p]),
                    .mem_write_address(data_mem_write_address[p]),
                    .mem_write_data(data_mem_write_data[p]),
                    .mem_write_ready(data_mem_write_ready[p]),
                    .rs(rs[p]), .rt(rt[p]),
                    .lsu_state(lsu_state[p]), .lsu_out(lsu_out[p])
                );

                registers #(
                    .THREADS_PER_BLOCK(THREADS_PER_BLOCK),
                    .THREAD_ID(i),
                    .DATA_BITS(DATA_MEM_DATA_BITS)
                ) register_instance (
                    .clk(clk), .reset(reset),
                    .enable(i < warp_thread_count[w]),
                    .block_id(warp_block_id[w]),
                    .core_state(warp_state[w]),
                    .decoded_reg_write_enable(dec_reg_we[w]),
                    .decoded_reg_input_mux(dec_reg_mux[w]),
                    .decoded_rd_address(dec_rd[w]),
                    .decoded_rs_address(dec_rs[w]),
                    .decoded_rt_address(dec_rt[w]),
                    .decoded_immediate(dec_imm[w]),
                    .alu_out(alu_out[p]), .lsu_out(lsu_out[p]),
                    .rs(rs[p]), .rt(rt[p])
                );

                pc #(
                    .DATA_MEM_DATA_BITS(DATA_MEM_DATA_BITS),
                    .PROGRAM_MEM_ADDR_BITS(PROGRAM_MEM_ADDR_BITS)
                ) pc_instance (
                    .clk(clk), .reset(reset),
                    .enable(i < warp_thread_count[w]),
                    .core_state(warp_state[w]),
                    .decoded_nzp(dec_nzp[w]),
                    .decoded_immediate(dec_imm[w]),
                    .decoded_nzp_write_enable(dec_nzp_we[w]),
                    .decoded_pc_mux(dec_pc_mux[w]),
                    .alu_out(alu_out[p]),
                    .current_pc(warp_pc[w]),
                    .next_pc(next_pc[p]),
                    .nzp(thread_nzp[p])
                );
            end
        end
    endgenerate
endmodule

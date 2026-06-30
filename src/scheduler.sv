`default_nettype none
`timescale 1ns/1ns

// SCHEDULER (branch divergence via min-PC active mask)
// > 管理单个 core 处理 1 个 block 的控制流。
// > 线程可分叉：每线程持有自己的 PC (thread_pc[i])。每拍取仍在运行线程的最小 PC，
//   只有处于该 PC 的线程(active_mask)执行；其余线程冻结。PC 重合时自动重收敛。
//   执行 RET 的线程置 done_mask[i] 退休；全部退休则该 block 完成。
module scheduler #(
    parameter THREADS_PER_BLOCK = 4
) (
    input wire clk,
    input wire reset,
    input wire start,

    // Block metadata
    input wire [$clog2(THREADS_PER_BLOCK):0] thread_count,

    // Control Signals
    input reg decoded_mem_read_enable,
    input reg decoded_mem_write_enable,
    input reg decoded_ret,

    // Memory Access State
    input reg [2:0] fetcher_state,
    input reg [1:0] lsu_state [THREADS_PER_BLOCK-1:0],

    // PC / divergence interface
    output reg [7:0] current_pc,                          // fetch PC = 运行线程的最小 PC
    output reg [7:0] thread_pc [THREADS_PER_BLOCK-1:0],   // per-thread PC
    output reg [THREADS_PER_BLOCK-1:0] active_mask,       // 本指令执行的线程
    input reg [7:0] next_pc [THREADS_PER_BLOCK-1:0],      // per-thread next PC (来自 pc.sv)

    // Execution State
    output reg [2:0] core_state,
    output reg done
);
    localparam IDLE = 3'b000,
        FETCH = 3'b001,
        DECODE = 3'b010,
        REQUEST = 3'b011,
        WAIT = 3'b100,
        EXECUTE = 3'b101,
        UPDATE = 3'b110,
        DONE = 3'b111;

    // 已退休（执行过 RET）的线程
    reg [THREADS_PER_BLOCK-1:0] done_mask;

    always @(posedge clk) begin
        if (reset) begin
            current_pc <= 0;
            core_state <= IDLE;
            done <= 0;
            done_mask <= 0;
            active_mask <= 0;
            for (int i = 0; i < THREADS_PER_BLOCK; i = i + 1) begin
                thread_pc[i] <= 0;
            end
        end else begin
            case (core_state)
                IDLE: begin
                    if (start) begin
                        // 所有线程从 PC 0 开始，i<thread_count 的线程为活跃
                        current_pc <= 0;
                        for (int i = 0; i < THREADS_PER_BLOCK; i = i + 1) begin
                            active_mask[i] <= (i < thread_count) ? 1'b1 : 1'b0;
                        end
                        core_state <= FETCH;
                    end
                end
                FETCH: begin
                    if (fetcher_state == 3'b010) begin
                        core_state <= DECODE;
                    end
                end
                DECODE: begin
                    core_state <= REQUEST;
                end
                REQUEST: begin
                    core_state <= WAIT;
                end
                WAIT: begin
                    reg any_lsu_waiting = 1'b0;
                    for (int i = 0; i < THREADS_PER_BLOCK; i++) begin
                        if (lsu_state[i] == 2'b01 || lsu_state[i] == 2'b10) begin
                            any_lsu_waiting = 1'b1;
                            break;
                        end
                    end
                    if (!any_lsu_waiting) begin
                        core_state <= EXECUTE;
                    end
                end
                EXECUTE: begin
                    core_state <= UPDATE;
                end
                UPDATE: begin
                    // 临时变量（声明在块顶，与 WAIT 风格一致）
                    reg [7:0] min_pc;
                    reg found;
                    reg [7:0] eff;
                    reg eligible;

                    // ---- 提交本指令：active 线程退休或推进自己的 PC ----
                    for (int i = 0; i < THREADS_PER_BLOCK; i = i + 1) begin
                        if (active_mask[i]) begin
                            if (decoded_ret) begin
                                done_mask[i] <= 1'b1;        // 该线程退休
                            end else begin
                                thread_pc[i] <= next_pc[i];  // 推进到自己的 next PC
                            end
                        end
                    end

                    // ---- 计算下一个 fetch PC = 仍在运行线程的最小 PC ----
                    // 仍在运行 = 本 block 内 && 未退休 && 不在本指令退休
                    min_pc = 8'hFF;
                    found = 1'b0;
                    for (int i = 0; i < THREADS_PER_BLOCK; i = i + 1) begin
                        eligible = (i < thread_count) && !done_mask[i]
                                   && !(decoded_ret && active_mask[i]);
                        if (eligible) begin
                            // 线程 i 的有效 next PC
                            eff = (active_mask[i] && !decoded_ret) ? next_pc[i] : thread_pc[i];
                            if (!found || (eff < min_pc)) begin
                                min_pc = eff;
                                found = 1'b1;
                            end
                        end
                    end

                    if (!found) begin
                        // 本 block 所有线程已退休 -> 完成
                        done <= 1;
                        core_state <= DONE;
                    end else begin
                        current_pc <= min_pc;
                        // 有效 PC 等于 min_pc 的线程下一拍执行
                        for (int i = 0; i < THREADS_PER_BLOCK; i = i + 1) begin
                            eligible = (i < thread_count) && !done_mask[i]
                                       && !(decoded_ret && active_mask[i]);
                            eff = (active_mask[i] && !decoded_ret) ? next_pc[i] : thread_pc[i];
                            active_mask[i] <= (eligible && (eff == min_pc)) ? 1'b1 : 1'b0;
                        end
                        core_state <= FETCH;
                    end
                end
                DONE: begin
                    // no-op
                end
            endcase
        end
    end
endmodule

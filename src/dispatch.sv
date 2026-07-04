`default_nettype none
`timescale 1ns/1ns

// BLOCK DISPATCH — stage 6: 一次给一个 core 分配 ≤WARPS_PER_CORE 个连续 block（每 warp 一个）。
// core_done[c] 表示该 core 本批 W 个 warp 全部 DONE。未用 warp 槽 thread_count=0。
module dispatch #(
    parameter NUM_CORES = 2,
    parameter THREADS_PER_BLOCK = 4,
    parameter WARPS_PER_CORE = 2
) (
    input wire clk,
    input wire reset,
    input wire start,

    input wire [7:0] thread_count,

    input reg [NUM_CORES-1:0] core_done,
    output reg [NUM_CORES-1:0] core_start,
    output reg [NUM_CORES-1:0] core_reset,
    output reg [7:0] core_warp_block_id [NUM_CORES*WARPS_PER_CORE-1:0],
    output reg [$clog2(THREADS_PER_BLOCK):0] core_warp_thread_count [NUM_CORES*WARPS_PER_CORE-1:0],

    output reg done
);
    wire [7:0] total_blocks;
    assign total_blocks = (thread_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    reg [7:0] blocks_dispatched;
    reg [7:0] blocks_done;
    reg [7:0] core_batch_count [NUM_CORES-1:0]; // 本核本批分到几个 warp
    reg start_execution;

    integer c, w, cnt;
    reg [7:0] blk;

    always @(posedge clk) begin
        if (reset) begin
            done <= 0;
            blocks_dispatched = 0;
            blocks_done = 0;
            start_execution <= 0;
            for (c = 0; c < NUM_CORES; c = c + 1) begin
                core_start[c] <= 0;
                core_reset[c] <= 1;
                core_batch_count[c] <= 0;
                for (w = 0; w < WARPS_PER_CORE; w = w + 1) begin
                    core_warp_block_id[c*WARPS_PER_CORE + w] <= 0;
                    core_warp_thread_count[c*WARPS_PER_CORE + w] <= 0;
                end
            end
        end else if (start) begin
            if (!start_execution) begin
                start_execution <= 1;
                for (c = 0; c < NUM_CORES; c = c + 1) core_reset[c] <= 1;
            end

            if (blocks_done == total_blocks) done <= 1;

            // 刚复位的 core：若还有 block，成批（≤W）分配
            for (c = 0; c < NUM_CORES; c = c + 1) begin
                if (core_reset[c]) begin
                    core_reset[c] <= 0;
                    if (blocks_dispatched < total_blocks) begin
                        cnt = 0;
                        for (w = 0; w < WARPS_PER_CORE; w = w + 1) begin
                            blk = blocks_dispatched + w[7:0];
                            if (blk < total_blocks) begin
                                core_warp_block_id[c*WARPS_PER_CORE + w] <= blk;
                                core_warp_thread_count[c*WARPS_PER_CORE + w] <=
                                    (blk == total_blocks - 1)
                                        ? (thread_count - blk*THREADS_PER_BLOCK)
                                        : THREADS_PER_BLOCK[$clog2(THREADS_PER_BLOCK):0];
                                cnt = cnt + 1;
                            end else begin
                                core_warp_thread_count[c*WARPS_PER_CORE + w] <= 0;
                            end
                        end
                        core_batch_count[c] <= cnt;
                        blocks_dispatched = blocks_dispatched + cnt; // = min(W, total_blocks - blocks_dispatched)
                        core_start[c] <= 1;
                    end
                end
            end

            // core 完成本批：回收，blocks_done 加上本批 warp 数
            for (c = 0; c < NUM_CORES; c = c + 1) begin
                if (core_start[c] && core_done[c]) begin
                    core_reset[c] <= 1;
                    core_start[c] <= 0;
                    blocks_done = blocks_done + core_batch_count[c];
                end
            end
        end
    end
endmodule

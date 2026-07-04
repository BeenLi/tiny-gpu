`default_nettype none
`timescale 1ns/1ns

// WARP SCHEDULER — stage 7: multi-warp residency (switch-on-stall) + intra-warp
// branch divergence. 每 warp 一套 per-thread 分叉状态：thread_pc[p]/active_mask[p]/
// done_mask[p]（拍平 p=w*T+i）。UPDATE 用 min-PC 重收敛：取仍在运行线程的最小 PC，
// 只有处于该 PC 的线程下拍执行，PC 重合即自动重收敛；执行 RET 的线程置 done_mask 退休，
// 该 warp 全部退休才 -> DONE。warp 选择（ready/live 轮转、park-on-stall）与分叉正交。
module scheduler #(
    parameter THREADS_PER_BLOCK = 4,
    parameter WARPS_PER_CORE = 2
) (
    input wire clk,
    input wire reset,
    input wire start,

    // 每 warp block 元数据；slot 有效 <=> warp_thread_count[w] != 0
    input reg [$clog2(THREADS_PER_BLOCK):0] warp_thread_count [WARPS_PER_CORE-1:0],

    // 当前 warp 指令的 decoded 控制（core 用 decoded_*[current_warp] 选出）
    input reg decoded_mem_read_enable,
    input reg decoded_mem_write_enable,
    input reg decoded_ret,

    // 共享 fetcher 状态 + 拍平的 per-(warp,thread) LSU 状态
    input reg [2:0] fetcher_state,
    input reg [1:0] lsu_state [WARPS_PER_CORE*THREADS_PER_BLOCK-1:0],

    // 每 (warp,thread) 的 next PC（来自 pc.sv），min-PC 重收敛用
    input reg [7:0] next_pc [WARPS_PER_CORE*THREADS_PER_BLOCK-1:0],

    output wire [7:0] current_pc,
    output reg  [7:0] warp_pc [WARPS_PER_CORE-1:0],                     // 每 warp fetch PC = 运行线程最小 PC
    output reg  [7:0] thread_pc [WARPS_PER_CORE*THREADS_PER_BLOCK-1:0], // per-thread PC
    output reg  [WARPS_PER_CORE*THREADS_PER_BLOCK-1:0] active_mask,     // 本指令执行的线程（拍平）
    output reg  [2:0] warp_state [WARPS_PER_CORE-1:0],
    output reg  [(WARPS_PER_CORE > 1 ? $clog2(WARPS_PER_CORE) : 1)-1:0] current_warp,
    output wire [2:0] core_state,
    output reg  done
);
    localparam IDLE=3'b000, FETCH=3'b001, DECODE=3'b010, REQUEST=3'b011,
               WAIT=3'b100, EXECUTE=3'b101, UPDATE=3'b110, DONE=3'b111;
    localparam LSU_REQUESTING=2'b01, LSU_WAITING=2'b10;

    assign core_state = warp_state[current_warp];
    assign current_pc = warp_pc[current_warp];

    integer w, i, k, p;
    reg [WARPS_PER_CORE-1:0] lsu_busy;   // per warp: 任一活跃 LSU 仍 REQUESTING/WAITING
    reg [WARPS_PER_CORE*THREADS_PER_BLOCK-1:0] done_mask;  // 已执行 RET 退休的线程（拍平）
    reg ready_found; reg [(WARPS_PER_CORE > 1 ? $clog2(WARPS_PER_CORE) : 1)-1:0] ready_warp; // ready = FETCH | (WAIT & !busy)
    reg live_found;  reg [(WARPS_PER_CORE > 1 ? $clog2(WARPS_PER_CORE) : 1)-1:0] live_warp;  // live  = state != DONE
    reg [(WARPS_PER_CORE > 1 ? $clog2(WARPS_PER_CORE) : 1)-1:0] cand;
    // min-PC 重收敛扫描临时量
    reg [7:0] min_pc, eff;
    reg found, eligible;

    always @(posedge clk) begin
        if (reset) begin
            done <= 0;
            current_warp <= 0;
            for (w=0; w<WARPS_PER_CORE; w=w+1) begin
                warp_pc[w] <= 0;
                warp_state[w] <= IDLE;
            end
            for (p=0; p<WARPS_PER_CORE*THREADS_PER_BLOCK; p=p+1) begin
                thread_pc[p] <= 0;
                active_mask[p] <= 1'b0;
                done_mask[p] <= 1'b0;
            end
        end else begin
            // 每拍算：per-warp LSU-busy + 轮转就绪/存活选择
            for (w=0; w<WARPS_PER_CORE; w=w+1) begin
                lsu_busy[w] = 1'b0;
                for (i=0; i<THREADS_PER_BLOCK; i=i+1)
                    if ((i < warp_thread_count[w]) &&
                        (lsu_state[w*THREADS_PER_BLOCK+i]==LSU_REQUESTING ||
                         lsu_state[w*THREADS_PER_BLOCK+i]==LSU_WAITING))
                        lsu_busy[w] = 1'b1;
            end
            ready_found = 1'b0; ready_warp = current_warp;
            live_found  = 1'b0; live_warp  = current_warp;
            // 从 current_warp+1 起轮转扫描；k=W 时回到 current 自身（用于 WAIT 自我 resume）
            for (k=1; k<=WARPS_PER_CORE; k=k+1) begin
                cand = ((current_warp + k) >= WARPS_PER_CORE)
                        ? (current_warp + k - WARPS_PER_CORE) : (current_warp + k);
                // 这拍就能干活的 warp
                if (!ready_found &&
                    ((warp_state[cand]==FETCH) || (warp_state[cand]==WAIT && !lsu_busy[cand]))) begin
                    ready_found = 1'b1; ready_warp = cand;
                end
                // 还没退休的 warp (自己也算)
                if (!live_found && (warp_state[cand]!=DONE)) begin
                    live_found = 1'b1; live_warp = cand;
                end
            end

            case (warp_state[current_warp])
                IDLE: begin
                    if (start) begin
                        for (w=0; w<WARPS_PER_CORE; w=w+1) begin
                            warp_pc[w] <= 0;
                            warp_state[w] <= (warp_thread_count[w] != 0) ? FETCH : DONE;
                            for (i=0; i<THREADS_PER_BLOCK; i=i+1) begin
                                thread_pc[w*THREADS_PER_BLOCK+i] <= 0;
                                done_mask[w*THREADS_PER_BLOCK+i] <= 1'b0;
                                active_mask[w*THREADS_PER_BLOCK+i] <= (i < warp_thread_count[w]) ? 1'b1 : 1'b0;
                            end
                        end
                        current_warp <= 0; // dispatch 从 slot 0 起填，warp 0 必有效
                    end
                end
                FETCH: begin
                    if (fetcher_state == 3'b010) warp_state[current_warp] <= DECODE;
                end
                DECODE: begin
                    warp_state[current_warp] <= REQUEST;
                end
                REQUEST: begin
                    if (decoded_mem_read_enable || decoded_mem_write_enable) begin
                        warp_state[current_warp] <= WAIT;               // park
                        if (ready_found) begin
                            current_warp <= ready_warp;
                            if (warp_state[ready_warp]==WAIT) warp_state[ready_warp] <= EXECUTE;
                        end else if (live_found) begin
                            current_warp <= live_warp;                  // 其余都忙，落到存活 warp 轮询
                        end
                    end else begin
                        warp_state[current_warp] <= EXECUTE;            // 非访存指令直通
                    end
                end
                EXECUTE: begin
                    warp_state[current_warp] <= UPDATE;
                end
                UPDATE: begin
                    // ---- 1) 提交 active 线程：RET 退休，否则推进自己的 PC ----
                    for (i=0; i<THREADS_PER_BLOCK; i=i+1) begin
                        p = current_warp*THREADS_PER_BLOCK + i;
                        if (active_mask[p]) begin
                            if (decoded_ret) done_mask[p] <= 1'b1;
                            else thread_pc[p] <= next_pc[p];
                        end
                    end
                    // ---- 2) min-PC over 仍在运行线程（排除本拍退休的） ----
                    min_pc = 8'hFF; found = 1'b0;
                    for (i=0; i<THREADS_PER_BLOCK; i=i+1) begin
                        p = current_warp*THREADS_PER_BLOCK + i;
                        eligible = (i < warp_thread_count[current_warp]) && !done_mask[p]
                                   && !(decoded_ret && active_mask[p]);
                        if (eligible) begin
                            eff = (active_mask[p] && !decoded_ret) ? next_pc[p] : thread_pc[p];
                            if (!found || (eff < min_pc)) begin min_pc = eff; found = 1'b1; end
                        end
                    end
                    // ---- 3) 决策 ----
                    if (!found) begin
                        warp_state[current_warp] <= DONE;              // 本 warp 全部退休
                        if (ready_found) begin
                            current_warp <= ready_warp;
                            if (warp_state[ready_warp]==WAIT) warp_state[ready_warp] <= EXECUTE;
                        end else if (live_found) begin
                            current_warp <= live_warp;
                        end
                    end else begin
                        warp_pc[current_warp] <= min_pc;
                        for (i=0; i<THREADS_PER_BLOCK; i=i+1) begin
                            p = current_warp*THREADS_PER_BLOCK + i;
                            eligible = (i < warp_thread_count[current_warp]) && !done_mask[p]
                                       && !(decoded_ret && active_mask[p]);
                            eff = (active_mask[p] && !decoded_ret) ? next_pc[p] : thread_pc[p];
                            active_mask[p] <= (eligible && (eff == min_pc)) ? 1'b1 : 1'b0;
                        end
                        warp_state[current_warp] <= FETCH;             // greedy 续发
                    end
                end
                WAIT: begin
                    if (ready_found) begin                              // 挂起轮询：内存回来则 resume
                        current_warp <= ready_warp;
                        if (warp_state[ready_warp]==WAIT) warp_state[ready_warp] <= EXECUTE;
                    end
                end
                DONE: begin
                    if (ready_found) begin
                        current_warp <= ready_warp;
                        if (warp_state[ready_warp]==WAIT) warp_state[ready_warp] <= EXECUTE;
                    end else if (live_found) begin
                        current_warp <= live_warp;
                    end else begin
                        done <= 1;
                    end
                end
            endcase
        end
    end
endmodule

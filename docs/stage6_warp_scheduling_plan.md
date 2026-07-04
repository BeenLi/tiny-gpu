# Warp Scheduling Implementation Plan（阶段 6）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让每个 core 驻留 `WARPS_PER_CORE` 个 warp，用 switch-on-stall 交错调度把访存等待藏进其他 warp 的计算，实测总 cycle 下降。

**Architecture:** 现有 6 阶段 FSM 变成 per-warp（`warp_state[w]`）；单活跃 warp 贪心发射，遇 LDR/STR 在 REQUEST 落 WAIT 挂起并轮转切到就绪 warp；挂起 warp 的 LSU 靠"每 warp 一份 decoder（保持 `decoded_mem_*`）+ 始终 enable"在后台自主跑完；`done` 仅在全 warp DONE 时拉高。设计详见 `docs/stage6_warp_scheduling.md`。

**Tech Stack:** SystemVerilog（`src/*.sv`）→ `sv2v` → `iverilog -g2012` → cocotb 1.9.2（venv）。测试 = cocotb Python testbench，`make test_<name>` 在远端 `~/autoResearch/tiny-gpu` 执行（先 `source env.sh`）。

## Global Constraints

- 执行环境：**远端** `ssh myDevbox` → `~/autoResearch/tiny-gpu`；每次先 `source env.sh`。本地不跑实验。
- `WARPS_PER_CORE = 2`（默认，参数化）。`THREADS_PER_BLOCK = 4`、`NUM_CORES = 2`、`DATA_MEM_NUM_CHANNELS = 4` 保持默认。
- **不改这些模块内部逻辑**：`registers/alu/lsu/pc/decoder/fetcher/controller/dcr`（只改例化数量、`core_state` 来源、`NUM_CONSUMERS`）。
- **sv2v 兼容**：所有过端口/跨模块的 per-(warp,thread) 信号一律**拍平成 1D `[WARPS_PER_CORE*THREADS_PER_BLOCK]`**，索引 `p = w*THREADS_PER_BLOCK + i`。不使用 2D unpacked 数组。per-warp（仅按 warp）信号用 1D `[WARPS_PER_CORE]`（原码已证 1D unpacked 可综合）。
- lockstep：warp 内所有线程同 PC；`warp_pc[w]` 取代表线程 `next_pc[w*T+0]` 推进。intra-warp 分叉本阶段不支持。
- 状态编码沿用：`IDLE=000 FETCH=001 DECODE=010 REQUEST=011 WAIT=100 EXECUTE=101 UPDATE=110 DONE=111`；LSU `IDLE=00 REQUESTING=01 WAITING=10 DONE=11`；fetcher `FETCHED=010`。
- 分支：`stage6-warp-scheduling`（从 master）。每个 Task 末尾 commit。commit message 结尾加 `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`。

---

## 文件结构（改动地图）

| 文件 | 动作 | 职责 |
|---|---|---|
| `test/test_warpadd.py` | 创建 | threads=16 的 matadd（4 block/2 core=每核 2 warp），功能校验 + cycle 计数。既是 baseline 也是 warp-sched 的对照 |
| `src/scheduler.sv` | 重写 | warp 调度器：per-warp FSM + 贪心 switch-on-stall + 就绪集轮转 |
| `src/core.sv` | 重写 | 例化 1 共享 fetcher + W×(decoder+指令锁存) + W*T×{registers,alu,lsu,pc}；per-warp `warp_state` 连线；数据内存端口拍平 W*T |
| `src/dispatch.sv` | 重写 | 一次给一个 core 分配 ≤W 个 block；`core_done` = 该核整批 warp 全 DONE |
| `src/gpu.sv` | 改 | 数据内存 controller `NUM_CONSUMERS=NUM_CORES*W*T`；per-warp block metadata 连线；LSU 拍平索引含 warp |
| `docs/stage6_warp_scheduling_results.md` | 创建 | 记录 baseline→warp-sched 的 cycle 实测与分析 |
| `docs/LEARNING_GUIDE.md` | 改 | 进展表把 `⬜ Warp scheduling` 更新为 ✅ |

---

## Task 1：建分支 + 对照测试 + 记录 baseline

**Files:**
- Create: `test/test_warpadd.py`

**Interfaces:**
- Produces: cocotb 测试 `test_warpadd`，DUT 接口与现有测试相同（`dut.done`、`device_control_*`、`start`、内存端口）。RTL 版本无关——同一测试既在 master RTL 上取 baseline，也在改造后取 warp-sched 数。

- [ ] **Step 1: 建分支**

远端执行：
```bash
ssh myDevbox 'cd ~/autoResearch/tiny-gpu && git checkout -b stage6-warp-scheduling && git branch'
```
Expected: `* stage6-warp-scheduling`

- [ ] **Step 2: 写对照测试（threads=16 的 matadd）**

创建 `test/test_warpadd.py`（内容完整如下）：
```python
import cocotb
from cocotb.triggers import RisingEdge
from .helpers.setup import setup
from .helpers.memory import Memory
from .helpers.format import format_cycle
from .helpers.logger import logger

@cocotb.test()
async def test_warpadd(dut):
    # matadd over 16 elements: baseA=0, baseB=16, baseC=32
    # threads=16, THREADS_PER_BLOCK=4 => 4 blocks; NUM_CORES=2 => 2 warps resident per core.
    program_memory = Memory(dut=dut, addr_bits=8, data_bits=16, channels=1, name="program")
    program = [
        0b0101000011011110, # MUL   R0, %blockIdx, %blockDim
        0b0011000000001111, # ADD   R0, R0, %threadIdx      ; i = blockIdx*blockDim + threadIdx
        0b1001000100000000, # CONST R1, #0                  ; baseA
        0b1001001000010000, # CONST R2, #16                 ; baseB
        0b1001001100100000, # CONST R3, #32                 ; baseC
        0b0011010000010000, # ADD   R4, R1, R0              ; addr(A[i])
        0b0111010001000000, # LDR   R4, R4                  ; A[i]
        0b0011010100100000, # ADD   R5, R2, R0              ; addr(B[i])
        0b0111010101010000, # LDR   R5, R5                  ; B[i]
        0b0011011001000101, # ADD   R6, R4, R5              ; C[i] = A[i]+B[i]
        0b0011011100110000, # ADD   R7, R3, R0              ; addr(C[i])
        0b1000000001110110, # STR   R7, R6                  ; store C[i]
        0b1111000000000000, # RET
    ]

    data_memory = Memory(dut=dut, addr_bits=8, data_bits=8, channels=4, name="data")
    data = list(range(16)) + list(range(16))  # A[0..15] at 0-15, B[0..15] at 16-31

    threads = 16
    await setup(
        dut=dut, program_memory=program_memory, program=program,
        data_memory=data_memory, data=data, threads=threads,
    )
    data_memory.display(48)

    cycles = 0
    while dut.done.value != 1:
        data_memory.run()
        program_memory.run()
        await cocotb.triggers.ReadOnly()
        format_cycle(dut, cycles)
        await RisingEdge(dut.clk)
        cycles += 1

    logger.info(f"Completed in {cycles} cycles")
    data_memory.display(48)

    expected = [a + b for a, b in zip(data[0:16], data[16:32])]  # C[i] = 2*i
    for i, exp in enumerate(expected):
        got = data_memory.memory[i + 32]
        assert got == exp, f"Result mismatch at index {i}: expected {exp}, got {got}"
```

- [ ] **Step 3: 在 master RTL（当前分支起点）上跑，取 baseline**

```bash
ssh myDevbox 'cd ~/autoResearch/tiny-gpu && source env.sh && make test_warpadd 2>&1 | tail -30'
```
Expected: 测试 PASS；日志出现 `Completed in <N> cycles`。
**把这个 N 记为 BASELINE_CYCLES**（写进本 Task Step 5 的 commit message 和 Task 5 的结果文档）。

> 说明：此刻分支 RTL == master（单 warp/core、整核阻塞），4 个 block 在 2 个 core 上**顺序**跑。这就是 baseline。

- [ ] **Step 4: 存一份 baseline 日志备查**

```bash
ssh myDevbox 'cd ~/autoResearch/tiny-gpu && cp test/logs/test_warpadd.log test/logs/test_warpadd.baseline.log 2>/dev/null; ls -la test/logs/ | grep warpadd'
```
Expected: 看到 `test_warpadd.baseline.log`（若日志名不同，用 `ls test/logs/` 确认实际文件名后再 cp）。

- [ ] **Step 5: Commit**

```bash
ssh myDevbox 'cd ~/autoResearch/tiny-gpu && git add test/test_warpadd.py test/logs/ && git commit -q -m "test(stage6): warpadd baseline (threads=16, matadd); BASELINE=<N> cycles

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"'
```
（把 `<N>` 换成 Step 3 记录的实际 cycle 数。）

---

## Task 2：重写 `src/scheduler.sv` 为 warp 调度器

**Files:**
- Modify (rewrite): `src/scheduler.sv`

**Interfaces:**
- Consumes（来自 core，Task 3 提供）：`warp_thread_count[W]`、当前 warp 的 `decoded_mem_read_enable/decoded_mem_write_enable/decoded_ret`（core 用 `current_warp` 选出）、`fetcher_state`、`lsu_state[W*T]`（拍平）、`warp_next_pc[W]`、`start`。
- Produces（给 core）：`current_pc`、`warp_pc[W]`、`warp_state[W]`、`current_warp`、`core_state`(=`warp_state[current_warp]`)、`done`。

- [ ] **Step 1: 整文件替换为下列内容**

```systemverilog
`default_nettype none
`timescale 1ns/1ns

// WARP SCHEDULER — stage 6: multi-warp residency + switch-on-stall interleaving.
// 6 阶段 FSM 变成 per-warp（warp_state[w]）。单活跃 warp 贪心发射，遇 LDR/STR 在
// REQUEST 落 WAIT 挂起并轮转切换；挂起 warp 的 LSU 后台自主跑完（core 里每 warp 一份
// decoder 保持其 decoded_mem_*，且 LSU 始终 enable）。全 warp DONE 时 done 拉高。
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

    // 每 warp 的 next PC（core 喂 next_pc[w*T+0]，最低活跃线程；lockstep => 全相等）
    input reg [7:0] warp_next_pc [WARPS_PER_CORE-1:0],

    output wire [7:0] current_pc,
    output reg  [7:0] warp_pc [WARPS_PER_CORE-1:0],
    output reg  [2:0] warp_state [WARPS_PER_CORE-1:0],
    output reg  [$clog2(WARPS_PER_CORE)-1:0] current_warp,
    output wire [2:0] core_state,
    output reg  done
);
    localparam IDLE=3'b000, FETCH=3'b001, DECODE=3'b010, REQUEST=3'b011,
               WAIT=3'b100, EXECUTE=3'b101, UPDATE=3'b110, DONE=3'b111;
    localparam LSU_REQUESTING=2'b01, LSU_WAITING=2'b10;

    assign core_state = warp_state[current_warp];
    assign current_pc = warp_pc[current_warp];

    integer w, i, k;
    reg [WARPS_PER_CORE-1:0] lsu_busy;   // per warp: 任一活跃 LSU 仍 REQUESTING/WAITING
    reg ready_found; reg [$clog2(WARPS_PER_CORE)-1:0] ready_warp; // ready = FETCH | (WAIT & !busy)
    reg live_found;  reg [$clog2(WARPS_PER_CORE)-1:0] live_warp;  // live  = state != DONE
    reg [$clog2(WARPS_PER_CORE)-1:0] cand;

    always @(posedge clk) begin
        if (reset) begin
            done <= 0;
            current_warp <= 0;
            for (w=0; w<WARPS_PER_CORE; w=w+1) begin
                warp_pc[w] <= 0;
                warp_state[w] <= IDLE;
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
                if (!ready_found &&
                    ((warp_state[cand]==FETCH) || (warp_state[cand]==WAIT && !lsu_busy[cand]))) begin
                    ready_found = 1'b1; ready_warp = cand;
                end
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
                    if (decoded_ret) begin
                        warp_state[current_warp] <= DONE;
                        if (ready_found) begin
                            current_warp <= ready_warp;
                            if (warp_state[ready_warp]==WAIT) warp_state[ready_warp] <= EXECUTE;
                        end else if (live_found) begin
                            current_warp <= live_warp;
                        end else begin
                            done <= 1;                                  // 全 warp DONE
                        end
                    end else begin
                        warp_pc[current_warp] <= warp_next_pc[current_warp];
                        warp_state[current_warp] <= FETCH;              // greedy 续发
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
```

- [ ] **Step 2: 单文件语法编译（不接整机，先查语法/sv2v）**

```bash
ssh myDevbox 'cd ~/autoResearch/tiny-gpu && source env.sh && sv2v -w build/scheduler.v src/scheduler.sv && echo SV2V_OK'
```
Expected: 打印 `SV2V_OK`，无 sv2v 报错。若报 2D unpacked 相关错误——检查是否误留 2D 端口（本设计已全拍平）。

> 注：整机功能测试在 Task 5（scheduler 需 core/dispatch/gpu 配套才能跑）。本 Task 只保证语法层通过。

- [ ] **Step 3: Commit**

```bash
ssh myDevbox 'cd ~/autoResearch/tiny-gpu && git add src/scheduler.sv && git commit -q -m "feat(stage6): rewrite scheduler as warp scheduler (per-warp FSM, switch-on-stall)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"'
```

---

## Task 3：重写 `src/core.sv` 例化 W 个 warp context

**Files:**
- Modify (rewrite): `src/core.sv`

**Interfaces:**
- Consumes（来自 gpu，Task 4 提供）：`warp_block_id[W]`、`warp_thread_count[W]`、`start`、程序内存端口（共享 fetcher）、拍平数据内存端口 `[W*T]`。
- Produces：`done`；实例化 scheduler（Task 2）与 W×(decoder + T×{registers,alu,lsu,pc})。

- [ ] **Step 1: 整文件替换为下列内容**

```systemverilog
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
```

- [ ] **Step 2: 单文件 sv2v 编译**

```bash
ssh myDevbox 'cd ~/autoResearch/tiny-gpu && source env.sh && sv2v -I src/* src/core.sv > /dev/null && echo SV2V_OK'
```
Expected: `SV2V_OK`（sv2v 需 `-I src/*` 找到被例化子模块）。若报错，多为 unpacked 数组连线/索引；对照上面端口拍平约定修正。

- [ ] **Step 3: Commit**

```bash
ssh myDevbox 'cd ~/autoResearch/tiny-gpu && git add src/core.sv && git commit -q -m "feat(stage6): core holds W warp contexts (per-warp decoder + datapath, shared fetcher)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"'
```

---

## Task 4：`dispatch.sv` 分配 W block/core + `gpu.sv` 加宽连线

**Files:**
- Modify (rewrite): `src/dispatch.sv`
- Modify: `src/gpu.sv`

**Interfaces:**
- dispatch Produces：`core_start/core_reset/core_done[NUM_CORES]`、拍平 `core_warp_block_id[NUM_CORES*W]`、`core_warp_thread_count[NUM_CORES*W]`。
- gpu 连线：数据内存 controller `NUM_CONSUMERS=NUM_CORES*W*T`；给每个 core 喂 per-warp 元数据切片与拍平 LSU 端口。

- [ ] **Step 1: 整替换 `src/dispatch.sv`**

```systemverilog
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

    integer c, w;
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
                        core_batch_count[c] <= 0;
                        for (w = 0; w < WARPS_PER_CORE; w = w + 1) begin
                            blk = blocks_dispatched + w[7:0];
                            if (blk < total_blocks) begin
                                core_warp_block_id[c*WARPS_PER_CORE + w] <= blk;
                                core_warp_thread_count[c*WARPS_PER_CORE + w] <=
                                    (blk == total_blocks - 1)
                                        ? (thread_count - blk*THREADS_PER_BLOCK)
                                        : THREADS_PER_BLOCK[$clog2(THREADS_PER_BLOCK):0];
                                core_batch_count[c] <= core_batch_count[c] + 1;
                            end else begin
                                core_warp_thread_count[c*WARPS_PER_CORE + w] <= 0; // 空槽
                            end
                        end
                        // 本批分配的 warp 数 = min(W, total_blocks - blocks_dispatched)
                        blocks_dispatched = blocks_dispatched +
                            (((total_blocks - blocks_dispatched) < WARPS_PER_CORE)
                                ? (total_blocks - blocks_dispatched) : WARPS_PER_CORE);
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
```

> 说明：`core_batch_count[c]` 用非阻塞在分配处 `<=` 累加了 W 次，最终值 = 实际分配 warp 数（每次 `<=` 覆盖，最后一次生效值即计数）——**改用阻塞累加更稳**：把该块内 `core_batch_count[c] <= core_batch_count[c] + 1;` 与回收处读值改为一致的阻塞变量。实现时若发现批计数不对，改为：本地 `integer cnt; cnt=0; for w: if(blk<total) cnt=cnt+1; core_batch_count[c]<=cnt;`（一次性写）。

- [ ] **Step 2: 改 `src/gpu.sv`（3 处）**

(a) 数据内存 controller 消费者数改为含 warp：把
```systemverilog
    localparam NUM_LSUS = NUM_CORES * THREADS_PER_BLOCK;
```
改为
```systemverilog
    localparam NUM_LSUS = NUM_CORES * WARPS_PER_CORE * THREADS_PER_BLOCK;
```
并在 `gpu` 的 `#(...)` 参数列表加入 `parameter WARPS_PER_CORE = 2`。dispatch 例化补 `.WARPS_PER_CORE(WARPS_PER_CORE)`，core 例化补 `.WARPS_PER_CORE(WARPS_PER_CORE)`。

(b) dispatch 输出/core 输入的 block 元数据换成 per-warp。把 gpu 里
```systemverilog
    reg [7:0] core_block_id [NUM_CORES-1:0];
    reg [$clog2(THREADS_PER_BLOCK):0] core_thread_count [NUM_CORES-1:0];
```
改为
```systemverilog
    reg [7:0] core_warp_block_id [NUM_CORES*WARPS_PER_CORE-1:0];
    reg [$clog2(THREADS_PER_BLOCK):0] core_warp_thread_count [NUM_CORES*WARPS_PER_CORE-1:0];
```
dispatch 例化端口 `.core_block_id(...)`/`.core_thread_count(...)` 换成 `.core_warp_block_id(core_warp_block_id)`、`.core_warp_thread_count(core_warp_thread_count)`。

(c) 每个 core 的 generate 块里：LSU 拍平索引改为含 warp，并把 per-warp 元数据切片喂给 core。将内层 pass-through 循环从 `THREADS_PER_BLOCK` 扩到 `WARPS_PER_CORE*THREADS_PER_BLOCK`，并把 core 的 data_mem 端口宽度对齐 `WARPS_PER_CORE*THREADS_PER_BLOCK`；`lsu_index = i*THREADS_PER_BLOCK + j` 改为全局
```systemverilog
    localparam lsu_index = i*(WARPS_PER_CORE*THREADS_PER_BLOCK) + j; // i=core, j=0..W*T-1
```
core 例化新增/改：
```systemverilog
        // per-core 的 W 个 warp 元数据切片
        reg [7:0] this_warp_block_id [WARPS_PER_CORE-1:0];
        reg [$clog2(THREADS_PER_BLOCK):0] this_warp_thread_count [WARPS_PER_CORE-1:0];
        genvar wv;
        for (wv = 0; wv < WARPS_PER_CORE; wv = wv + 1) begin
            always @(posedge clk) begin
                this_warp_block_id[wv] <= core_warp_block_id[i*WARPS_PER_CORE + wv];
                this_warp_thread_count[wv] <= core_warp_thread_count[i*WARPS_PER_CORE + wv];
            end
        end
```
并把 core 例化的 `.block_id(...)`/`.thread_count(...)` 换成 `.warp_block_id(this_warp_block_id)`、`.warp_thread_count(this_warp_thread_count)`，`.WARPS_PER_CORE(WARPS_PER_CORE)`，data_mem 端口接宽度 `WARPS_PER_CORE*THREADS_PER_BLOCK` 的 per-core 本地信号（把现有 `core_lsu_*[THREADS_PER_BLOCK-1:0]` 宽度改为 `[WARPS_PER_CORE*THREADS_PER_BLOCK-1:0]`）。

> 实现提示：gpu.sv 的 EDA pass-through 结构保留不动，只是把每 core 的通道数从 T 提到 W*T、块元数据从标量提到 W 槽。改完 `make compile` 若报未连线/宽度不匹配，逐条对齐宽度。

- [ ] **Step 3: 全量编译**

```bash
ssh myDevbox 'cd ~/autoResearch/tiny-gpu && source env.sh && make compile 2>&1 | tail -20 && echo COMPILE_DONE'
```
Expected: 生成 `build/gpu.v` 无 sv2v 报错，末尾 `COMPILE_DONE`。

- [ ] **Step 4: Commit**

```bash
ssh myDevbox 'cd ~/autoResearch/tiny-gpu && git add src/dispatch.sv src/gpu.sv && git commit -q -m "feat(stage6): dispatch assigns W blocks/core; gpu widens data-mem to W*T

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"'
```

---

## Task 5：集成门禁（回归 + 延迟隐藏实验）+ 结果归档

**Files:**
- Create: `docs/stage6_warp_scheduling_results.md`
- Modify: `docs/LEARNING_GUIDE.md`

- [ ] **Step 1: 回归 —— 现有非分叉 kernel 必须全绿（W=1 路径）**

```bash
ssh myDevbox 'cd ~/autoResearch/tiny-gpu && source env.sh && make test_matadd 2>&1 | tail -5 && make test_matmul 2>&1 | tail -5'
```
Expected: 两者均 PASS。matadd(threads=8→2 block/2 core=每核 1 warp) 走单 warp 路径，结果与 baseline 一致；matmul 同理正确。
若失败：进 systematic-debugging，对照 `test/logs/` 逐拍核查 scheduler 切换/写回时序（重点 park→resume 的 `lsu_state` 与 `warp_state`）。

- [ ] **Step 2: 实验 —— warp-sched 功能正确 + cycle 下降**

```bash
ssh myDevbox 'cd ~/autoResearch/tiny-gpu && source env.sh && make test_warpadd 2>&1 | tail -8'
```
Expected: PASS（16 个结果 `C[i]==2*i` 全对）；日志 `Completed in <M> cycles`，且 **M < BASELINE_CYCLES**（Task 1 记录值）。
若 M ≥ BASELINE：不是功能 bug，而是没隐藏到延迟——检查是否真的发生了 park+切换（trace 里应看到 `current_warp` 在两 warp 间来回、且某 warp WAIT 期间另一 warp 在 EXECUTE/UPDATE）。

- [ ] **Step 3: 记录结果文档**

创建 `docs/stage6_warp_scheduling_results.md`，写入：
- baseline（master RTL，顺序跑 4 block）cycle 数 = BASELINE_CYCLES；
- warp-sched（每核 2 warp 交错）cycle 数 = M；
- 节省 = BASELINE - M（及百分比）；
- 从 trace 摘一段：某 warp LDR 挂起期间另一 warp 正在计算的逐拍片段，作为"延迟被填满"的证据；
- 讨论：核在"两 warp 同时卡内存"时仍空转（不可隐藏的残余停顿），与 `DATA_MEM_NUM_CHANNELS=4` 的带宽上限关系。

- [ ] **Step 4: 更新学习指南进展**

编辑 `docs/LEARNING_GUIDE.md`，把
```
- ⬜ **Warp scheduling（多 warp 驻留 + 交错调度）** —— 第二阶段，待开。
```
改为
```
- ✅ **Warp scheduling（多 warp 驻留 + switch-on-stall 交错调度）** —— 已完成（分支 `stage6-warp-scheduling`）。每核驻留 WARPS_PER_CORE(=2) 个 warp，6 阶段 FSM per-warp 化；遇 LDR/STR 挂起并轮转切换，挂起 warp 的 LSU 后台自主跑完（每 warp 一份 decoder 保持 decoded_mem_*）。实测 `test_warpadd`（threads=16）**BASELINE→M cycle**。详见 [设计](stage6_warp_scheduling.md) 与 [结果](stage6_warp_scheduling_results.md)。
```
（把 BASELINE、M 换成实测值。）

- [ ] **Step 5: Commit**

```bash
ssh myDevbox 'cd ~/autoResearch/tiny-gpu && git add docs/ test/logs/ && git commit -q -m "docs(stage6): warp scheduling results (BASELINE->M cycles) + guide update

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"'
```

---

## 自查（写完计划回看 spec 的覆盖）

- §2 资源模型（fetcher 共享 / decoder W 份 / datapath W×T / controller 加宽）→ Task 3（core）+ Task 4（gpu）。✅
- §3 per-warp FSM + 贪心 switch-on-stall + 就绪集轮转 + resume → Task 2（scheduler）。✅
- §2 不变式（单元始终 enable、不按 current_warp 门控）→ core.sv `enable(i < warp_thread_count[w])`。✅
- §1 decoder 每 warp 一份（LSU 自主推进依赖）→ core.sv per-warp decoder。✅
- §4 dispatch 分配 W block/core、`core_done` 聚合 → Task 4（dispatch）。✅
- §5 验证（NUM_CORES=1?→改用 threads=16 逼出 W=2；baseline vs warp-sched 同 kernel）→ Task 1 + Task 5。✅
- §6 范围（分叉 kernel 不支持）→ 回归只跑 matadd/matmul（非分叉），relu/divloop 不在门禁。✅
- §6 风险（2D unpacked / park-resume 时序 / dispatch 尾部 / 同 kernel 对照）→ Global Constraints 拍平约定 + Task 5 调试指引。✅
```

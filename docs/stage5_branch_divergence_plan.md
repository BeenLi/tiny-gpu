# Stage 5 (一) Branch Divergence 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让一个 core 内同一 warp 的线程能走不同分支路径并在 PC 重合时自动重收敛（min-PC active mask），同时保持 matadd/matmul 逐拍行为不变。

**Architecture:** scheduler 新增 per-thread PC 数组 `thread_pc[i]` 与 per-thread 退休位 `done_mask[i]`。每拍 fetch 最小 PC（min over 仍在运行的线程），只有 `thread_pc[i]==fetch_pc` 的线程进入 `active_mask` 真正执行；其余线程通过 `enable=0` 全状态冻结。线程执行 `RET` 即置 `done_mask` 退出；全部退休则 core done。

**Tech Stack:** SystemVerilog (sv2v v0.0.13 → iverilog 10.2) + cocotb 1.9.2（Python 3.11 venv）。

## Global Constraints

- 所有编译/运行在**远端**执行：`ssh myDevbox` → `cd ~/autoResearch/tiny-gpu && source env.sh`（必须先 source env.sh，用 venv 的 cocotb 1.9.2，不要用 ~/.local 的 cocotb 2.0）。
- 运行单个测试：`make test_<name>`（如 `make test_relu`）。`make` 会先 `make compile`（sv2v 编译 `src/*.sv` → `build/gpu.v`）再跑 iverilog + cocotb。新增测试文件 `test/test_<name>.py` 无需改 Makefile。
- 回归基线：**`make test_matadd` 与 `make test_matmul` 必须始终 PASS**。任何改动以不破坏它们为前提。
- 硬件参数固定：`NUM_CORES=2`，`THREADS_PER_BLOCK=4`，故 `%blockDim`(R14)=4、`%blockIdx`=R13、`%threadIdx`=R15。
- ISA 编码：`[15:12]=opcode,[11:8]=rd,[7:4]=rs,[3:0]=rt`；CONST/imm 用 `[7:0]`；BRnzp 用 `[11:9]=nzp,[7:0]=目标pc`。opcode：BRnzp=0001, CMP=0010, ADD=0011, SUB=0100, MUL=0101, DIV=0110, LDR=0111, STR=1000, CONST=1001, RET=1111。CMP 设 `alu_out[2:0]={N=(rs<rt),Z=(rs==rt),P=(rs>rt)}`。无条件跳转 = `BRnzp` 且 nzp=111（CMP 后必有一位置位，故恒匹配）。
- 设计依据：`docs/stage5_branch_divergence.md`。

---

### Task 1: Branch divergence RTL（由 ReLU 分叉测试驱动）

实现 min-PC active mask。先写一个会真正分叉的 ReLU kernel 测试（当前 lockstep 下会算错 → red），再改 `scheduler.sv` + `core.sv` 使其通过（green），并确认 matadd/matmul 回归通过。

**Files:**
- Create: `test/test_relu.py`
- Modify: `src/scheduler.sv`（整文件替换）
- Modify: `src/core.sv`（连线 + enable gating）

**Interfaces:**
- Produces（scheduler 新端口，core 据此连线）：
  - `input wire [$clog2(THREADS_PER_BLOCK):0] thread_count`
  - `output reg [7:0] thread_pc [THREADS_PER_BLOCK-1:0]`
  - `output reg [THREADS_PER_BLOCK-1:0] active_mask`
  - `output reg [7:0] current_pc`（语义改为 fetch PC = min）
- Consumes：`next_pc[i]`（来自每线程 `pc.sv`，签名不变）。

- [ ] **Step 1: 写 ReLU 分叉测试（会失败）**

创建 `test/test_relu.py`：

```python
import cocotb
from cocotb.triggers import RisingEdge
from .helpers.setup import setup
from .helpers.memory import Memory
from .helpers.format import format_cycle
from .helpers.logger import logger

@cocotb.test()
async def test_relu(dut):
    # 阈值 ReLU:  Y[i] = (X[i] < T) ? 0 : X[i]
    # 同一 block 内线程依据 X[i] 是否 < T 走不同分支 -> 分叉，末尾 STORE/RET 处重收敛
    program_memory = Memory(dut=dut, addr_bits=8, data_bits=16, channels=1, name="program")
    program = [
        0b0101000011011110, # 0  MUL  R0, %blockIdx, %blockDim
        0b0011000000001111, # 1  ADD  R0, R0, %threadIdx      ; i = blockIdx*blockDim + threadIdx
        0b1001000100000000, # 2  CONST R1, #0                 ; baseX = 0
        0b1001001000001000, # 3  CONST R2, #8                 ; baseY = 8
        0b1001001100000100, # 4  CONST R3, #4                 ; T = 4
        0b0011010000010000, # 5  ADD  R4, R1, R0              ; addr X[i] = baseX + i
        0b0111010001000000, # 6  LDR  R4, R4                  ; R4 = X[i]
        0b0011010100100000, # 7  ADD  R5, R2, R0              ; addr Y[i] = baseY + i
        0b0010000001000011, # 8  CMP  R4, R3                  ; X[i] vs T  (N = X<T)
        0b0001100000001100, # 9  BRn  #12                     ; if X<T goto ZERO(12)
        0b1000000001010100, # 10 STR  R5, R4                  ; (else) Y[i] = X[i]
        0b0001111000001110, # 11 BRnzp #14                    ; goto END(14) 无条件
        0b1001011000000000, # 12 CONST R6, #0                 ; (ZERO) R6 = 0
        0b1000000001010110, # 13 STR  R5, R6                  ; Y[i] = 0
        0b1111000000000000, # 14 RET
    ]

    data_memory = Memory(dut=dut, addr_bits=8, data_bits=8, channels=4, name="data")
    X = [2, 5, 3, 7, 1, 6, 0, 4]   # 跨越阈值 T=4，使每个 block 内线程分叉
    data = X + [0] * 8             # addr 0-7 = X, addr 8-15 = Y(结果区)

    threads = 8
    await setup(dut=dut, program_memory=program_memory, program=program,
                data_memory=data_memory, data=data, threads=threads)

    data_memory.display(16)

    cycles = 0
    while dut.done.value != 1:
        data_memory.run()
        program_memory.run()
        await cocotb.triggers.ReadOnly()
        format_cycle(dut, cycles)
        await RisingEdge(dut.clk)
        cycles += 1

    logger.info(f"Completed in {cycles} cycles")
    data_memory.display(16)

    T = 4
    expected = [0 if x < T else x for x in X]   # [0,5,0,7,0,6,0,4]
    for i, exp in enumerate(expected):
        result = data_memory.memory[i + 8]
        assert result == exp, f"Y[{i}] mismatch: expected {exp}, got {result}"
```

- [ ] **Step 2: 跑测试确认它失败（red）**

Run: `ssh myDevbox 'cd ~/autoResearch/tiny-gpu && source env.sh && make test_relu'`
Expected: FAIL。当前 lockstep 用 `next_pc[THREADS_PER_BLOCK-1]` 把最后一个线程的分支强加给全体，所有线程走 else 路径 → `Y[0] mismatch: expected 0, got 2`（AssertionError）。测试能跑完（到达 RET），不会挂死。

- [ ] **Step 3: 整文件替换 `src/scheduler.sv`**

用以下完整内容覆盖 `src/scheduler.sv`：

```systemverilog
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
```

- [ ] **Step 4: 改 `src/core.sv`（连线 + enable gating）**

改动 4 处：

(a) 在 `wire [7:0] next_pc[THREADS_PER_BLOCK-1:0];`（约 line 49）之后新增两行声明：

```systemverilog
    wire [7:0] next_pc[THREADS_PER_BLOCK-1:0];
    wire [7:0] thread_pc[THREADS_PER_BLOCK-1:0];
    wire [THREADS_PER_BLOCK-1:0] active_mask;
```

(b) scheduler 例化（约 line 114-129）增加 `thread_count` / `thread_pc` / `active_mask` 端口。把整段替换为：

```systemverilog
    // Scheduler
    scheduler #(
        .THREADS_PER_BLOCK(THREADS_PER_BLOCK),
    ) scheduler_instance (
        .clk(clk),
        .reset(reset),
        .start(start),
        .thread_count(thread_count),
        .fetcher_state(fetcher_state),
        .core_state(core_state),
        .decoded_mem_read_enable(decoded_mem_read_enable),
        .decoded_mem_write_enable(decoded_mem_write_enable),
        .decoded_ret(decoded_ret),
        .lsu_state(lsu_state),
        .current_pc(current_pc),
        .thread_pc(thread_pc),
        .active_mask(active_mask),
        .next_pc(next_pc),
        .done(done)
    );
```

(c) 在 generate 的 for-loop 里，把 alu / lsu / registers / pc **四个**实例的
`.enable(i < thread_count),`
统一改为
`.enable((i < thread_count) && active_mask[i]),`
（共 4 处，约 line 139 / 152 / 178 / 200）。

(d) pc 实例（约 line 207）把
`.current_pc(current_pc),`
改为
`.current_pc(thread_pc[i]),`
（注意：fetcher 实例的 `.current_pc(current_pc)` 保持不变——fetch 仍按 min PC 取指。）

- [ ] **Step 5: 跑 ReLU 测试确认通过（green）**

Run: `ssh myDevbox 'cd ~/autoResearch/tiny-gpu && source env.sh && make test_relu'`
Expected: PASS。日志可见 `Completed in N cycles`，结果区 Y = `[0,5,0,7,0,6,0,4]`，cocotb 汇总打印 `test.test_relu.test_relu ... PASS`，make 退出码 0。
（若编译报 sv2v/iverilog 错误，先核对 scheduler 端口与 core 连线一致；若结果错，查 `enable` 是否四处都加了 `&& active_mask[i]`、pc 实例是否接了 `thread_pc[i]`。）

- [ ] **Step 6: 回归——matadd / matmul 仍通过**

Run: `ssh myDevbox 'cd ~/autoResearch/tiny-gpu && source env.sh && make test_matadd && make test_matmul'`
Expected: 两者均 PASS。非分叉 kernel 下所有活跃线程 `thread_pc` 恒相等 → min=该 PC、mask=全体活跃线程 → 与原 lockstep 逐拍一致。

- [ ] **Step 7: 提交**

```bash
ssh myDevbox 'cd ~/autoResearch/tiny-gpu && git add src/scheduler.sv src/core.sv test/test_relu.py && git -c user.name="wanli" -c user.email="wanli990802@gmail.com" commit -m "feat: branch divergence via min-PC active mask

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"'
```

---

### Task 2: 变长循环测试（验证 partial done_mask / 不同周期退出）

ReLU 的所有线程在 RET 前已重收敛。本 task 用一个变长循环 kernel：每线程循环 `%threadIdx` 次后**各自**存结果并 `RET`——线程在**不同周期**退休（`done_mask` 逐步置位），并通过 min-PC 在循环中重收敛。验证 Task 1 RTL 的 per-thread RET 路径。

**Files:**
- Create: `test/test_divloop.py`

**Interfaces:**
- Consumes：Task 1 实现的 scheduler/core（`done_mask`、min-PC 逻辑）。

- [ ] **Step 1: 写变长循环测试**

创建 `test/test_divloop.py`：

```python
import cocotb
from cocotb.triggers import RisingEdge
from .helpers.setup import setup
from .helpers.memory import Memory
from .helpers.format import format_cycle
from .helpers.logger import logger

@cocotb.test()
async def test_divloop(dut):
    # 每线程循环 %threadIdx 次 (acc+=1)，然后在各自的退出路径存 acc 并 RET。
    # 退出路径(8-10)的 PC 低于循环体(11-13)：先完成的线程在低 PC 处 RET 退休，
    # 其余线程继续在高 PC 循环 -> 不同周期退休(partial done_mask) + min-PC 重收敛。
    # 结果 Y[i] = threadIdx。
    program_memory = Memory(dut=dut, addr_bits=8, data_bits=16, channels=1, name="program")
    program = [
        0b0101000011011110, # 0  MUL  R0, %blockIdx, %blockDim
        0b0011000000001111, # 1  ADD  R0, R0, %threadIdx     ; i
        0b1001000100000001, # 2  CONST R1, #1                ; increment
        0b1001001000001000, # 3  CONST R2, #8                ; baseY = 8
        0b1001001100000000, # 4  CONST R3, #0                ; acc = 0
        0b1001010000000000, # 5  CONST R4, #0                ; k = 0
        0b0010000001001111, # 6  CMP  R4, %threadIdx         ; (LOOP_CHECK) k vs tid
        0b0001100000001011, # 7  BRn  #11                    ; if k<tid goto BODY(11)
        0b0011010100100000, # 8  ADD  R5, R2, R0             ; (EXIT) addr Y[i] = baseY + i
        0b1000000001010011, # 9  STR  R5, R3                 ; Y[i] = acc
        0b1111000000000000, # 10 RET
        0b0011001100110001, # 11 ADD  R3, R3, R1             ; (BODY) acc += 1
        0b0011010001000001, # 12 ADD  R4, R4, R1             ; k += 1
        0b0001111000000110, # 13 BRnzp #6                    ; goto LOOP_CHECK(6) 无条件
    ]

    data_memory = Memory(dut=dut, addr_bits=8, data_bits=8, channels=4, name="data")
    data = [0] * 16   # addr 8-11 接收 Y

    threads = 4       # 单 block，便于在 trace 中观察逐线程退出
    await setup(dut=dut, program_memory=program_memory, program=program,
                data_memory=data_memory, data=data, threads=threads)

    data_memory.display(16)

    cycles = 0
    while dut.done.value != 1:
        data_memory.run()
        program_memory.run()
        await cocotb.triggers.ReadOnly()
        format_cycle(dut, cycles)
        await RisingEdge(dut.clk)
        cycles += 1

    logger.info(f"Completed in {cycles} cycles")
    data_memory.display(16)

    expected = [0, 1, 2, 3]   # Y[i] = threadIdx
    for i, exp in enumerate(expected):
        result = data_memory.memory[i + 8]
        assert result == exp, f"Y[{i}] mismatch: expected {exp}, got {result}"
```

- [ ] **Step 2: 跑测试（预期通过）**

Run: `ssh myDevbox 'cd ~/autoResearch/tiny-gpu && source env.sh && make test_divloop'`
Expected: PASS，Y(addr 8-11) = `[0,1,2,3]`。
（若 FAIL：说明 Task 1 的 partial done_mask 逻辑有误——重点查 scheduler UPDATE 里 `eligible` 的 `!(decoded_ret && active_mask[i])` 项，以及 `done` 仅在 `!found`（无可运行线程）时才置位。修 Task 1 文件后重跑。）

- [ ] **Step 3: 提交**

```bash
ssh myDevbox 'cd ~/autoResearch/tiny-gpu && git add test/test_divloop.py && git -c user.name="wanli" -c user.email="wanli990802@gmail.com" commit -m "test: variable-length-loop kernel exercises per-thread RET (partial done_mask)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"'
```

---

### Task 3: 更新学习文档

把第一阶段成果归档：在设计文档补充实测周期数，在 LEARNING_GUIDE 链接并标注进展。

**Files:**
- Modify: `docs/stage5_branch_divergence.md`（§8 补实测）
- Modify: `docs/LEARNING_GUIDE.md`（归档链接 + 阶段5进展）

- [ ] **Step 1: 记录实测周期数**

从 Task 1/2 的运行日志取 `Completed in N cycles`，在 `docs/stage5_branch_divergence.md` 第 8 节末尾追加一小段：列出 test_relu / test_divloop / test_matadd / test_matmul 各自周期数与 PASS 状态，并贴一段 test_relu 的 trace 片段（显示 active_mask 从全体→子集→重收敛）。

- [ ] **Step 2: 更新 LEARNING_GUIDE.md**

在「精读笔记归档」列表加一行链接到 `stage5_branch_divergence.md`；在「阶段 5」小节标注 branch divergence（min-PC active mask）已完成、warp scheduling 待第二阶段。

- [ ] **Step 3: 提交**

```bash
ssh myDevbox 'cd ~/autoResearch/tiny-gpu && git add docs/stage5_branch_divergence.md docs/LEARNING_GUIDE.md && git -c user.name="wanli" -c user.email="wanli990802@gmail.com" commit -m "docs: record stage5 branch divergence results

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"'
```

---

## Self-Review

**1. Spec coverage（对照 `docs/stage5_branch_divergence.md`）**
- §3 min-PC active mask 机制 → Task 1 Step 3（scheduler）。
- §4 逐文件改动（scheduler/core/pc/lsu-registers-alu/fetcher）→ Task 1 Step 3-4；pc/lsu/registers/alu 无内部改动，仅 core 改 enable，符合设计。
- §6 向后兼容 → Task 1 Step 6 回归。
- §7 边界（per-thread RET、WAIT、thread_count、初始化）→ scheduler 实现覆盖；per-thread RET 由 Task 2 验证；thread_count/初始化由 IDLE + eligible 项覆盖。
- §8 验证（relu + divloop + 回归）→ Task 1/2/Step6。
- §9 接口预留 → 留待第二阶段，无需本计划任务。

**2. Placeholder scan**：无 TBD/TODO；每个改动步骤含完整代码与确切命令。

**3. Type consistency**：scheduler 新端口 `thread_count`/`thread_pc`/`active_mask`/`current_pc` 在 core 连线（Step 4）与 scheduler 定义（Step 3）一致；`thread_pc` 两处均为 `[7:0] ... [THREADS_PER_BLOCK-1:0]`；`active_mask` 两处均为 `[THREADS_PER_BLOCK-1:0]`。测试中内存读取统一 `data_memory.memory[i+8]`。

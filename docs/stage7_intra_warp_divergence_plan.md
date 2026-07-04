# Stage 7 — Intra-Warp Branch Divergence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore per-thread branch divergence inside each warp of the stage-6 switch-on-stall warp scheduler, so `test_relu` and `test_divloop` pass and a new multi-warp-per-core divergence test passes, with all non-divergent regressions bit-identical.

**Architecture:** Port the pre-warp `bd18e12` per-thread mechanism (thread_pc / active_mask / done_mask + min-PC reconvergence at UPDATE) into each warp of the scheduler, flattened to 1-D `[W*T]` (index `p = w*THREADS_PER_BLOCK + i`). The switch-on-stall warp selection (ready/live round-robin, park-on-mem) is orthogonal and stays untouched; only scheduler UPDATE and a few core.sv wires change.

**Tech Stack:** SystemVerilog → `sv2v` → `iverilog -g2012` → cocotb 1.9.2 (venv). Simulation on remote only.

## Global Constraints

- ALL compile/run/experiments on remote: `ssh myDevbox` → `cd ~/autoResearch/tiny-gpu && source env.sh` (cocotb 1.9.2 venv). NEVER run locally.
- Parameters (unchanged): `THREADS_PER_BLOCK=4`, `WARPS_PER_CORE=2`, `NUM_CORES=2`.
- sv2v-safe: no 2-D unpacked arrays at module ports. Flatten per-(warp,thread) signals to 1-D `[WARPS_PER_CORE*THREADS_PER_BLOCK]`, index `p = w*THREADS_PER_BLOCK + i`.
- Regressions `test_warpadd`, `test_matadd`, `test_matadd_tail`, `test_matadd_multibatch` are non-divergent (uniform active_mask) and MUST stay green with identical results.
- cycles = SIM_TIME / 25000 (25us clock period).
- Run a single test: `make test_<name>`. Compile only: `make compile`.
- Commit messages end with a trailing `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` line.

## File Structure

- `src/scheduler.sv` — MODIFY (full rewrite). Per-warp FSM + NEW per-thread divergence state (thread_pc/active_mask/done_mask) + min-PC reconvergence in UPDATE. Interface: drop `warp_next_pc`, add `next_pc [W*T]` input, add `thread_pc [W*T]` + `active_mask [W*T]` outputs.
- `src/core.sv` — MODIFY (5 wiring edits). Drop warp_next_pc; feed per-thread PC into `pc`; gate every unit `enable` by active_mask; wire new scheduler ports.
- `test/helpers/format.py` — MODIFY. Print per-thread `active_mask` + `thread_pc` in the cycle trace (debug aid).
- `test/test_relu_warpsched.py` — CREATE. relu variant, 16 threads ⇒ 2 diverging warps per core.
- `docs/stage7_intra_warp_divergence_results.md` — CREATE. Cycle table + reconvergence trace + notes.
- `docs/LEARNING_GUIDE.md` — MODIFY. Add stage-7 ✅ bullet after line 105.

No changes to `gpu.sv`, `dispatch.sv`, `pc.sv`, `lsu.sv`, `decoder.sv`, `alu.sv`, `registers.sv`, `fetcher.sv`.

---

### Task 1: Per-warp divergence RTL (scheduler + core + trace)

The stage-6 lockstep scheduler makes each warp follow thread 0's PC, so `test_relu` currently fails (`Y[1]` expected 5, got 0). This task adds per-thread divergence so it passes. scheduler.sv and core.sv change together (the port change breaks compilation until both are done), so they share one test cycle.

**Files:**
- Modify: `src/scheduler.sv` (full rewrite)
- Modify: `src/core.sv` (5 edits)
- Modify: `test/helpers/format.py` (trace)
- Test (pre-existing, no edit): `test/test_relu.py`, `test/test_divloop.py`

**Interfaces:**
- Consumes: `pc.sv` output `next_pc[p]` (per-thread next PC, EXECUTE-updated), `lsu.sv` `lsu_state[p]`, decoder `dec_*[current_warp]`.
- Produces (scheduler ports later consumed by core.sv):
  - input `reg [7:0] next_pc [WARPS_PER_CORE*THREADS_PER_BLOCK-1:0]`
  - output `reg [7:0] thread_pc [WARPS_PER_CORE*THREADS_PER_BLOCK-1:0]`
  - output `reg [WARPS_PER_CORE*THREADS_PER_BLOCK-1:0] active_mask` (packed)
  - output `reg [7:0] warp_pc [WARPS_PER_CORE-1:0]` (now = min-PC of running threads, drives current_pc)
  - (removed: `warp_next_pc`)

- [ ] **Step 1: Enhance the cycle trace to show divergence (format.py)**

In `test/helpers/format.py`, inside `format_cycle`, capture the scheduler handle and print per-thread `active_mask`/`thread_pc`. Replace the thread loop block:

```python
        warps = _safe(lambda: list(ci.warps), [])
        for w, warp in enumerate(warps):
            wstate = _safe(lambda: format_core_state(str(warp.threads[0].alu_instance.core_state.value)))
            marker = " <== current" if current_warp == w else ""
            logger.debug(f"  warp {w}: state={wstate}{marker}")

            threads = _safe(lambda: list(warp.threads), [])
            for t, thread in enumerate(threads):
                lsu = _safe(lambda: format_lsu_state(str(thread.lsu_instance.lsu_state.value)))
                rs = _safe(lambda: int(str(thread.register_instance.rs.value), 2))
                rt = _safe(lambda: int(str(thread.register_instance.rt.value), 2))
                regs = _safe(lambda: format_registers([str(item.value) for item in thread.register_instance.registers]))
                logger.debug(f"    thread {w}.{t}: lsu={lsu}  RS={rs} RT={rt}  [{regs}]")
```

with (adds `sched = ...` and `act`/`tpc` per thread):

```python
        sched = _safe(lambda: ci.scheduler_instance, None)
        warps = _safe(lambda: list(ci.warps), [])
        for w, warp in enumerate(warps):
            wstate = _safe(lambda: format_core_state(str(warp.threads[0].alu_instance.core_state.value)))
            marker = " <== current" if current_warp == w else ""
            logger.debug(f"  warp {w}: state={wstate}{marker}")

            threads = _safe(lambda: list(warp.threads), [])
            for t, thread in enumerate(threads):
                p = w * len(threads) + t
                lsu = _safe(lambda: format_lsu_state(str(thread.lsu_instance.lsu_state.value)))
                act = _safe(lambda: int(sched.active_mask[p].value))
                tpc = _safe(lambda: int(str(sched.thread_pc[p].value), 2))
                rs = _safe(lambda: int(str(thread.register_instance.rs.value), 2))
                rt = _safe(lambda: int(str(thread.register_instance.rt.value), 2))
                regs = _safe(lambda: format_registers([str(item.value) for item in thread.register_instance.registers]))
                logger.debug(f"    thread {w}.{t}: act={act} tpc={tpc} lsu={lsu}  RS={rs} RT={rt}  [{regs}]")
```

- [ ] **Step 2: Rewrite `src/scheduler.sv`**

Replace the ENTIRE file with:

```systemverilog
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
```

- [ ] **Step 3: Edit `src/core.sv` — remove warp_next_pc wire, add thread_pc/active_mask wires**

Find:

```systemverilog
    wire [7:0] warp_next_pc [WARPS_PER_CORE-1:0];
```

Replace with:

```systemverilog
    wire [7:0] thread_pc [WARPS_PER_CORE*THREADS_PER_BLOCK-1:0];
    wire [WARPS_PER_CORE*THREADS_PER_BLOCK-1:0] active_mask;
```

(The `wire [7:0] next_pc [WARPS_PER_CORE*THREADS_PER_BLOCK-1:0];` declaration already exists a few lines below — keep it.)

- [ ] **Step 4: Edit `src/core.sv` — scheduler port map**

Find:

```systemverilog
        .lsu_state(lsu_state),
        .warp_next_pc(warp_next_pc),
        .current_pc(current_pc),
        .warp_pc(warp_pc),
        .warp_state(warp_state),
```

Replace with:

```systemverilog
        .lsu_state(lsu_state),
        .next_pc(next_pc),
        .current_pc(current_pc),
        .warp_pc(warp_pc),
        .thread_pc(thread_pc),
        .active_mask(active_mask),
        .warp_state(warp_state),
```

- [ ] **Step 5: Edit `src/core.sv` — remove the thread-0 warp_next_pc shortcut**

Find and DELETE this line (and its comment line just above it):

```systemverilog
            // 代表线程（最低索引）的 next_pc 作为该 warp 的 warp_next_pc
            assign warp_next_pc[w] = next_pc[w*THREADS_PER_BLOCK + 0];
```

- [ ] **Step 6: Edit `src/core.sv` — gate all four unit enables by active_mask**

There are four `.enable(i < warp_thread_count[w]),` lines (in `alu`, `lsu`, `registers`, `pc`). Change EACH of the four to:

```systemverilog
                    .enable((i < warp_thread_count[w]) && active_mask[p]),
```

(`p` is the `localparam integer p = w*THREADS_PER_BLOCK + i;` already declared at the top of the `threads` generate block.)

- [ ] **Step 7: Edit `src/core.sv` — feed per-thread PC into the pc unit**

In the `pc pc_instance (...)` instantiation, find:

```systemverilog
                    .alu_out(alu_out[p]),
                    .current_pc(warp_pc[w]),
                    .next_pc(next_pc[p]),
```

Replace with:

```systemverilog
                    .alu_out(alu_out[p]),
                    .current_pc(thread_pc[p]),
                    .next_pc(next_pc[p]),
```

- [ ] **Step 8: Compile**

Run: `cd ~/autoResearch/tiny-gpu && source env.sh && make compile`
Expected: `sv2v` + `iverilog` succeed, no errors, `build/gpu.v` regenerated.

- [ ] **Step 9: Run test_relu (the target)**

Run: `make test_relu 2>&1 | grep -iE "cycles|PASS|FAIL|mismatch"`
Expected: `PASS=1 FAIL=0`; Y = `[0,5,0,7,0,6,0,4]` (relu of X=`[2,5,3,7,1,6,0,4]`, T=4). No `mismatch`.

- [ ] **Step 10: Run test_divloop (staggered per-thread retire)**

Run: `make test_divloop 2>&1 | grep -iE "cycles|PASS|FAIL|mismatch"`
Expected: `PASS=1 FAIL=0`; Y = `[0,1,2,3]`. This exercises partial `done_mask` (threads RET at different cycles) + min-PC reconvergence.

- [ ] **Step 11: Run all four regressions**

Run: `for t in warpadd matadd matadd_tail matadd_multibatch; do echo "== $t =="; make test_$t 2>&1 | grep -iE "cycles|PASS|FAIL|mismatch"; done`
Expected: every one `PASS=1 FAIL=0`, no `mismatch`. Record each cycle count (from the `Completed in N cycles` log line) for the results doc.

- [ ] **Step 12: Commit**

```bash
cd ~/autoResearch/tiny-gpu && git add src/scheduler.sv src/core.sv test/helpers/format.py
git commit -F - <<"MSG"
feat(stage7): per-warp intra-warp divergence (min-PC reconvergence)

Restore per-thread thread_pc/active_mask/done_mask + min-PC reconvergence
inside each warp of the switch-on-stall scheduler. UPDATE trigger moves from
"any RET" to "no runners left"; enables gated by active_mask; parked warp's
active_mask frozen -> latency hiding preserved. relu + divloop pass; regressions
unchanged. format.py trace now prints per-thread act/tpc.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
MSG
```

---

### Task 2: Combined multi-warp-per-core divergence test

Proves divergence and switch-on-stall interleaving compose: 16 threads ⇒ 4 blocks ⇒ dispatch gives each of the 2 cores 2 warps, and each 4-thread warp diverges on the ReLU threshold.

**Files:**
- Create: `test/test_relu_warpsched.py`

**Interfaces:**
- Consumes: same DUT + helpers as `test/test_relu.py` (`setup`, `Memory`, `format_cycle`).

- [ ] **Step 1: Create `test/test_relu_warpsched.py`**

```python
import cocotb
from cocotb.triggers import RisingEdge
from .helpers.setup import setup
from .helpers.memory import Memory
from .helpers.format import format_cycle
from .helpers.logger import logger

@cocotb.test()
async def test_relu_warpsched(dut):
    # 阈值 ReLU（baseY=16 变体）：16 线程 -> 4 block -> dispatch 每 core 2 个 warp。
    # 每个 4-线程 warp 内部按 X<T 分叉 -> 同时压 intra-warp divergence + switch-on-stall。
    program_memory = Memory(dut=dut, addr_bits=8, data_bits=16, channels=1, name="program")
    program = [
        0b0101000011011110, # 0  MUL  R0, %blockIdx, %blockDim
        0b0011000000001111, # 1  ADD  R0, R0, %threadIdx      ; i = blockIdx*blockDim + threadIdx
        0b1001000100000000, # 2  CONST R1, #0                 ; baseX = 0
        0b1001001000010000, # 3  CONST R2, #16                ; baseY = 16
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
    # 每 4 线程一组（= 一个 warp），组内都横跨阈值 T=4 以强制 warp 内分叉
    X = [2, 5, 3, 7,  1, 6, 0, 4,  3, 8, 2, 5,  6, 1, 7, 0]
    data = X + [0] * 16            # addr 0-15 = X, addr 16-31 = Y(结果区)

    threads = 16
    await setup(dut=dut, program_memory=program_memory, program=program,
                data_memory=data_memory, data=data, threads=threads)

    data_memory.display(32)

    cycles = 0
    while dut.done.value != 1:
        data_memory.run()
        program_memory.run()
        await cocotb.triggers.ReadOnly()
        format_cycle(dut, cycles)
        await RisingEdge(dut.clk)
        cycles += 1

    logger.info(f"Completed in {cycles} cycles")
    data_memory.display(32)

    T = 4
    expected = [0 if x < T else x for x in X]   # [0,5,0,7,0,6,0,4,0,8,0,5,6,0,7,0]
    for i, exp in enumerate(expected):
        result = data_memory.memory[i + 16]
        assert result == exp, f"Y[{i}] mismatch: expected {exp}, got {result}"
```

- [ ] **Step 2: Run it**

Run: `cd ~/autoResearch/tiny-gpu && source env.sh && make test_relu_warpsched 2>&1 | grep -iE "cycles|PASS|FAIL|mismatch"`
Expected: `PASS=1 FAIL=0`; Y (addr 16-31) = `[0,5,0,7,0,6,0,4,0,8,0,5,6,0,7,0]`. Record the cycle count.

- [ ] **Step 3: Commit**

```bash
cd ~/autoResearch/tiny-gpu && git add test/test_relu_warpsched.py
git commit -F - <<"MSG"
test(stage7): multi-warp-per-core divergence (relu x16, 2 warps/core)

16 threads -> 4 blocks -> 2 diverging warps per core; each 4-thread warp
straddles T=4. Proves intra-warp divergence and switch-on-stall interleaving
compose. Y[i] == relu(X[i]).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
MSG
```

---

### Task 3: Documentation

**Files:**
- Create: `docs/stage7_intra_warp_divergence_results.md`
- Modify: `docs/LEARNING_GUIDE.md` (add one ✅ bullet after line 105)

**Interfaces:** none (docs only).

- [ ] **Step 1: Create `docs/stage7_intra_warp_divergence_results.md`**

Use this structure. Fill the CYCLE CELLS marked `<n>` with the numbers recorded in Task 1 Step 11, Task 1 Step 9/10, and Task 2 Step 2. The trace snippet: paste ~6-10 lines from a `test_relu` debug log (enable via the logger) showing one warp where `active_mask` splits after the `BRn` at PC 9 (some threads go to PC 10, others to PC 12) and re-converges to all-active at the `RET` (PC 14).

```markdown
# Stage 7 — Intra-Warp Divergence: Results

> Recombining branch divergence with stage-6 switch-on-stall warp scheduling.
> Design: [stage7_intra_warp_divergence_design.md](stage7_intra_warp_divergence_design.md).

## What changed

Each warp now carries per-thread `thread_pc` / `active_mask` / `done_mask`
(flattened `[W*T]`). Scheduler UPDATE does min-PC reconvergence over the current
warp's running threads; the warp retires only when no runners remain. `core.sv`
feeds each thread its own PC and gates every unit's `enable` by `active_mask`.
Warp selection (ready/live round-robin, park-on-stall) is unchanged.

## Correctness

| Test | Threads | Warps/core | Property exercised | Result | Cycles |
|------|---------|-----------|--------------------|--------|--------|
| test_relu            | 8  | 1 | if/else divergence + reconverge at STR/RET | PASS | <n> |
| test_divloop         | 4  | 1 | staggered per-thread RET (partial done_mask) | PASS | <n> |
| test_relu_warpsched  | 16 | 2 | divergence x switch-on-stall together | PASS | <n> |

## Regressions (non-divergent, must be identical)

| Test | Cycles | Result |
|------|--------|--------|
| test_warpadd            | <n> | PASS |
| test_matadd             | <n> | PASS |
| test_matadd_tail        | <n> | PASS |
| test_matadd_multibatch  | <n> | PASS |

## Reconvergence trace (test_relu, one warp)

```
<paste 6-10 debug lines: active_mask splits after BRn@9 (some -> 10, some -> 12),
 re-converges to all-active at RET@14; note tpc per thread>
```

## Why it composes with latency hiding

A parked warp's `active_mask` only changes in UPDATE, which a parked warp never
enters, so its active threads' LSUs keep draining while another warp runs — the
stage-6 latency-hiding property is untouched by divergence.
```

- [ ] **Step 2: Append a stage-7 bullet to `docs/LEARNING_GUIDE.md`**

After the existing warp-scheduling bullet (line 105, the one starting `- ✅ **Warp scheduling`), insert a new line:

```markdown
- ✅ **Intra-warp divergence 复合 warp scheduling（stage 7）** —— 已完成。把 per-thread `thread_pc/active_mask/done_mask` + min-PC 重收敛重新折进每个 warp：UPDATE 从「任一 RET 即完」改为「无运行线程才完」，`enable` 受 `active_mask` 门控，挂起 warp 的 `active_mask` 冻结 -> 延迟隐藏不受影响。验证：`test_relu`（if/else）+ `test_divloop`（变长循环 partial done_mask）+ 新增 `test_relu_warpsched`（16 线程 -> 每 core 2 个分叉 warp）；四个非分叉回归零偏移。详见 [设计](stage7_intra_warp_divergence_design.md)、[实现计划](stage7_intra_warp_divergence_plan.md)、[结果](stage7_intra_warp_divergence_results.md)。
```

- [ ] **Step 3: Commit**

```bash
cd ~/autoResearch/tiny-gpu && git add docs/stage7_intra_warp_divergence_results.md docs/LEARNING_GUIDE.md
git commit -F - <<"MSG"
docs(stage7): intra-warp divergence results + guide update

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
MSG
```

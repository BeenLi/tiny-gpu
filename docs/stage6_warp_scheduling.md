# 阶段 6：Warp Scheduling —— 多 warp 驻留 + switch-on-stall 交错调度（设计）

> 目标：把每个 core 从「单 warp 驻留、整核阻塞」改造成「多 warp 驻留、遇内存等待就切换」，
> 用**延迟隐藏**把访存等待的空转拍填满，实测总 cycle 下降。
>
> 本文是设计文档（spec）。实现计划见后续 `stage6_warp_scheduling_plan.md`（writing-plans 产出）。

---

## 0. 背景与动机

当前 `scheduler.sv` 的 6 阶段状态机 `FETCH→DECODE→REQUEST→WAIT→EXECUTE→UPDATE`
**整核共享一份**：只要本 block 内任一线程的 LSU 还在忙（`lsu_state ∈ {REQUESTING, WAITING}`），
整个 core 就卡在 `WAIT`，直到内存返回——**零延迟隐藏**。

```
baseline(单warp):  A:LDR REQ|WAIT WAIT WAIT WAIT|EXE UPD   A:ADD ...
                             └──── 整核干等 N 拍 ────┘
```

真实 GPU 用 **warp scheduling** 解决：一个 SM 驻留多个 warp，某 warp 卡在内存等待时，
调度器切到另一个就绪 warp 发射，用别的 warp 的计算填满这段等待。本阶段在 tiny-gpu 上
实现其最小忠实形态。

### 阶段决策（brainstorming 结论）

| 维度 | 选择 |
|---|---|
| 深度 | **全量实现 + cocotb 仿真验证**（对标 branch divergence / predication 阶段）|
| 基线 | **简化 lockstep warp**：warp 内锁步、单 warp PC，本阶段不做 intra-warp 分叉 |
| 架构 | **A. switch-on-stall 交错调度**（单活跃 warp + `current_warp` 指针，遇内存挂起切换）|
| 策略 | **单策略**：greedy switch-on-stall + 就绪集轮转；`WARPS_PER_CORE` 参数化（默认 2）|

被否决/延后：全流水 barrel（路线 B，过度设计）；LRR vs GTO 双策略对比（W≥4 才有意义，延后）；
intra-warp 分叉与 warp 调度的重组（延后）。

---

## 1. 关键时序发现（设计的物理基础）

读 `src/lsu.sv` 得到两条决定骨架的事实：

1. **LSU 一旦在 `REQUEST` 被踢起就自主推进**。
   `IDLE →(core_state==REQUEST)→ REQUESTING → WAITING →(mem_read_ready)→ DONE`，
   中间各拍**不依赖** core_state 停在 WAIT——只靠 `mem_read_ready` 往前走。
   ⇒ 一个被"挂起"的 warp，它的 LSU 会在后台把内存请求自己跑完。**这就是延迟隐藏成立的物理基础。**

2. **LSU 的整个 case 包在 `if (decoded_mem_read_enable)`（写路径同理）里**。
   它推进的每一拍都要求"自己那条访存指令的 decoded 信号仍然拉高"。
   ⇒ 若 decoder 全核共享，`current_warp` 一切换，`decoded_mem_read_enable` 就变成别的 warp 的指令，
   **挂起 warp 的 LSU 会僵死在 WAITING**。
   ⇒ **推论：decoder 必须每 warp 一份**，各自锁存自己的 in-flight 指令。
   这正对应真实 GPU 里每 warp 的 instruction buffer。

---

## 2. 资源模型：什么复制、什么共享

每 core 驻留 `WARPS_PER_CORE`（参数，默认 2，下记 W）个 warp；**warp = block**，
warp 内 T=`THREADS_PER_BLOCK` 个线程锁步执行（本阶段无 intra-warp 分叉）。

| 部件 | 现在 | 改造后 | 理由 |
|---|---|---|---|
| fetcher | 1/core | **1/core（共享，被调度的稀缺资源）** | 单活跃 warp 取指，warp scheduler 决定谁 fetch |
| decoder | 1/core | **W/core（每 warp 一份 + `warp_instruction[w]` 锁存）** | 见 §1-2：挂起 warp 的 LSU 靠自己的 decoded 信号自主跑完 |
| registers / alu / lsu / pc | T/core | **W×T/core** | 每 warp 独立上下文 + 独立在途内存 |
| warp PC / 状态 | 1 组 | **每 warp**：`warp_pc[w]` `wstate[w]` | 独立控制流 |
| data memory controller | `NUM_CORES*T` 消费者 | **`NUM_CORES*W*T`** | 复用现有 channel 仲裁作为跨 warp 的内存带宽竞争 |
| program memory controller | `NUM_CORES` 消费者 | **不变**（fetcher 仍 1/core） | — |

> **关键不变式**：每 warp 的 registers/alu/lsu/pc **始终 enable**：
> `enable = (i < warp_thread_count[w])`，**不**按 `current_warp` 门控。
> 否则挂起 warp 的 LSU 会因 enable=0 而冻结，无法自主推进到 DONE，延迟隐藏失效。

---

## 3. per-warp 六阶段 FSM + warp scheduler

### 3.1 核心洞察

现有 6 阶段 `FETCH→DECODE→REQUEST→WAIT→EXECUTE→UPDATE` **原封不动地变成 per-warp**
（`wstate[w]`，沿用相同 3-bit 编码 `FETCH=001 … UPDATE=110`），
所以 `registers/alu/lsu/pc/fetcher` 模块**内部逻辑零改动**——它们只是把 `.core_state`
接到各自 warp 的 `wstate[w]`。真正新增的只有"**谁来推进**"这层调度。

### 3.2 单活跃 warp + greedy + 就绪集轮转

- **单活跃 warp**：任一时刻只有 `current_warp` 在走 FETCH…UPDATE；
  其余 warp 要么 **WAIT（挂起，LSU 后台自主跑）**、要么 **DONE**、要么 **READY（fresh，待取指）**。
- **greedy**：`current_warp` 背靠背连续发射，直到
  ① 遇访存指令（LDR/STR）→ REQUEST 后落到 WAIT **挂起并让出**；或
  ② 执行 RET → DONE。
- **让出时的选择（就绪集轮转）**：从 `current_warp+1` 轮转扫描，选第一个"可推进"的 warp：
  - `READY`（fresh，要 fetch 下一条），或
  - `WAIT` 且其 LSU 已全部完成（内存回来了，可 resume）。
  - 跳过 LSU 仍忙的挂起 warp。
  - 全 `DONE` → core 完成；有 warp 但都还卡在内存 → 核空转（**不可隐藏的残余停顿**，量化带宽瓶颈）。
- **resume**：被选中的挂起 warp 从 `WAIT → EXECUTE → UPDATE`。
  此时它的 LSU 已 DONE、`lsu_out` 就绪；写回目标寄存器用**它自己 decoder 的** `decoded_rd`——
  因 decoder 每 warp 一份且锁存了该 warp 的指令，天然正确（无需额外保存 writeback 记录）。
  写回并 `warp_pc <= next_pc` 后，greedy 继续把它当 current 往下发。

### 3.3 判断"挂起 warp 内存已回"

沿用现有 WAIT 的判据，但下沉到 per-warp：warp w 的所有活跃线程
（`i < warp_thread_count[w]`）的 `lsu_state[w][i] ∈ {IDLE, DONE}` 即为"就绪可 resume"；
任一线程 `∈ {REQUESTING, WAITING}` 即"仍忙"。

### 3.4 延迟隐藏时序（W=2，A/B）

```
baseline(单warp):  A:LDR REQ|WAIT WAIT WAIT WAIT|EXE UPD   A:ADD ...        (WAIT 空转 N 拍)
warp-sched(W=2):   A:LDR REQ|·······park·······|resume EXE UPD  A:ADD ...
                            B:FETCH DEC REQ EXE UPD  B:...                  (B 填满 A 的等待)
```

A 发 LDR 到 REQUEST 后挂起、切到 B；A 的 LSU 在后台跑那 N 拍内存延迟，同时 B 在 fetch/compute；
A 内存回来后择机 resume 写回。原来整核干等的 N 拍被 B 的计算填满 → 总 cycle 下降。
核只有在**所有驻留 warp 都卡在内存**时才真的空转。

---

## 4. 逐模块改动

### `scheduler.sv` → warp scheduler（改动最大，基本重写）
- 新增 per-warp 数组：`wstate[w]`、`warp_pc[w]`、`warp_done[w]`、
  `warp_block_id[w]`、`warp_thread_count[w]`；标量 `current_warp`、`rr_ptr`（轮转指针）。
- 输入 `lsu_state` 由 `[T]` 扩成 `[W][T]`；用于 §3.3 的 resume 判据。
- 逻辑：步进 `current_warp.wstate`；REQUEST 后按 `decoded_mem_*[current_warp]` 决定 park 还是直通 EXECUTE；
  park/RET 时跑就绪集轮转选新 current；resume 挂起 warp 到 EXECUTE。
- 输出 per-warp `wstate[w]`、`warp_pc[w]`、`current_warp`（fetcher 取指选择）、`done`（全 warp DONE）。
- **`warp_pc[w]` 推进**：lockstep 下同一 warp 的 T 个 `next_pc[w][i]` 相等，取代表活跃线程（最低索引活跃线程）的 `next_pc` 作为新 `warp_pc[w]`；inter-thread 分叉本阶段不支持（见 §6）。

### `core.sv` → 例化 W 个 warp context（连线为主）
- `generate` 外层 `for w in 0..W-1`：每 warp 1 个 decoder + `warp_instruction[w]` 锁存 +
  内层 `for i in 0..T-1` 的 registers/alu/lsu/pc。
- 每 warp 单元 `.core_state(wstate[w])`；decoder 吃 `warp_instruction[w]`；
  `enable = (i < warp_thread_count[w])`（**不**含 current_warp，见 §2 不变式）。
- 共享 1 个 fetcher：`.current_pc(warp_pc[current_warp])`、`.core_state(wstate[current_warp])`；
  FETCHED 时 `warp_instruction[current_warp] <= instruction`。
- data memory 端口由 `[T]` 扩成 `[W][T]`（拍平后接 controller）。

### `dispatch.sv` → 一次给一个 core 分配 W 个 block
- 每 core 每批分配 ≤W 个连续 block（尾部不足按实际数）；
  输出 `core_block_id[core][w]`、`core_thread_count[core][w]`、`core_num_warps[core]`。
- `core_done[core]` = 该 core 本批 W 个 warp 全 DONE；`blocks_done += 本批 warp 数`。
  保留原有 block 计数骨架，只是按 W 跨步。

### `gpu.sv` → 顶层加宽连线
- data memory controller `NUM_CONSUMERS`: `NUM_CORES*T` → `NUM_CORES*W*T`；
  per-warp block metadata 连线；program memory 不变。

### 零改动模块
`registers / alu / lsu / pc / decoder / fetcher / controller` **内部逻辑不动**——
只是例化数量、`core_state` 来源、`NUM_CONSUMERS` 变了。这是本设计"低风险"的关键。

---

## 5. 验证方案（cocotb）

**基线对照**：master（单 warp/core、整核阻塞）为 baseline；新分支为 warp-sched。
同一 kernel、同一数据，对比总 cycle，隔离"调度"这一个变量。

**如何逼出"每 core ≥2 warp"**：`total_blocks = ceil(thread_count / T)`。
设 `NUM_CORES=1, T=4, thread_count=8` → 2 个 block 全落到 1 个 core → W=2 driven；
或 `thread_count=16, W=4`。

**验证 kernel**：
1. `test_matadd`（LDR,LDR,ADD,STR，访存重）——功能回归 + cycle 对照，**预期明显省 cycle**。
2. `test_matmul`（多次 LDR 循环）——延迟隐藏收益更大。
3. （可选）LDR 密集合成 kernel，把收益最大化便于讲解。

**通过标准**：
- 功能：所有 kernel 输出与 baseline 逐元素一致（调度不能改变正确性）。
- 性能：warp-sched 总 cycle < baseline；文档记录 `baseline → warp-sched` 的 cycle 数
  （沿用 predication 阶段"198→176"式实测记录）。
- 回归：现有非分叉 kernel 全绿。

**TDD 节奏**：先写能观测 cycle 计数的 test（红）→ 逐步实现 scheduler/core 至功能正确（绿）→ 再看 cycle 收益。

---

## 6. 范围与风险

**范围内**：lockstep warp、W 参数化（默认 2）、单策略（greedy switch-on-stall + 就绪集轮转）、cycle 对照实验。

**明确不做（YAGNI / 未来工作）**：
- intra-warp 分支分叉（min-PC）与 warp scheduling 的**重组**——本分支从 master 出发但把 scheduler 换成
  lockstep-warp 版；分叉 kernel（relu/divloop）本分支不支持，留作后续。
- LRR vs GTO 双策略对比（W≥4 才有意义）——留作后续实验。
- 全流水 barrel（路线 B）。

**主要风险**：
1. **拍平多维端口**（`[W][T]` 数据内存、`lsu_state[W][T]`）在 `sv2v → iverilog` 下的可综合性——先做最小连线冒烟测试。
2. **park/resume 边界时序**：resume 进 EXECUTE 时 LSU 必须已 DONE 且 `lsu_out` 稳定；
   靠 per-warp `wstate` + 始终 enable 保证，但需在 trace 里逐拍核对。
3. **dispatch W-block 分配**的尾部（thread_count 非 T 整数倍）与 `core_done` 聚合正确性。
4. baseline 对照必须跑在**同一 kernel/数据**下，隔离调度这一个变量。

---

## 7. 术语对照（GPU 概念 ↔ tiny-gpu 实现）

| 真实 GPU | 本设计 |
|---|---|
| SM（流多处理器）| core |
| warp（32 线程锁步）| block = W 个之一，T 线程锁步 |
| warp scheduler | 重写后的 `scheduler.sv` |
| 常驻 warp 上下文 / 寄存器分区 | W×T registers 实例 + `warp_pc[w]`/`wstate[w]` |
| instruction buffer（每 warp）| 每 warp 的 decoder + `warp_instruction[w]` |
| 访存延迟隐藏 | switch-on-stall：LSU 后台自主跑，切到就绪 warp |
| GTO/LRR 调度策略 | 本阶段：greedy + 就绪集轮转（单策略）|

# 阶段 5（一）：Branch Divergence —— min-PC active mask 改造设计

> 目标：让同一个 warp（= 一个 core 内 `THREADS_PER_BLOCK` 个线程）里的线程能够**走不同的分支路径**，并在 PC 重新相等时**自动重收敛**；同时保持非分叉 kernel（matadd/matmul）逐拍行为不变。
>
> 本文是 stage 5 第一阶段的设计 spec。第二阶段（warp scheduling，多 warp 驻留 + 交错调度）单独成文，本阶段只为其预留接口、不实现。

---

## 1. 背景：当前 lockstep 的根

当前架构（见 [stage2](stage2_architecture.md) / [stage3](stage3_execution.md)）：

```
core = 1 block = 1 "warp"，纯 lockstep
  ├─ 1× fetcher      ← 每拍只取 1 条指令（按单一 current_pc）
  ├─ 1× scheduler    ← 单一 core_state、单一 current_pc
  └─ THREADS_PER_BLOCK 条 lane（各自 alu/lsu/registers/pc）
```

不支持 divergence 的根在 `src/scheduler.sv` 的 UPDATE 阶段：

```systemverilog
// TODO: Branch divergence. For now assume all next_pc converge
current_pc <= next_pc[THREADS_PER_BLOCK-1];
```

它把**最后一个线程**算出的 `next_pc` 强加给所有线程。每个线程其实**已经有独立的 `pc.sv` 在算自己的 `next_pc[i]`**（含各自的 NZP 寄存器），只是这一步把除最后一个之外的结果全丢弃了。这是本次改造最有利的现成基础。

现有 kernel 都不会真正分叉：matadd 无分支；matmul 的循环所有线程次数相同（同一矩阵维度），始终收敛。因此**验证必须新写会分叉的 kernel**。

## 2. 设计目标 / 非目标

**目标**
- 同一 warp 内线程可处于不同 PC，分别推进各自路径。
- PC 重新相等时自动重收敛（无需编译器/ISA 提供重收敛点）。
- 支持 per-thread 提前 `RET`（线程在不同时刻结束）。
- 零 ISA / 汇编器改动。改动面集中在 `scheduler.sv` 与 `core.sv` 的连线。
- 非分叉 kernel 逐拍行为与现状完全一致（matadd/matmul 回归 PASS）。

**非目标（YAGNI，留给后续阶段）**
- 多 warp 驻留与 warp 间调度（stage 5 第二阶段）。
- SIMT 重收敛栈 / IPDOM（本设计用 min-PC，不需要）。
- 对非结构化（不可归约）CFG 的最优重收敛——tiny-gpu 的 kernel 都是结构化 if/循环，min-PC 足够。

## 3. 机制：min-PC active mask

唯一新增的硬件状态：scheduler 里的 **per-thread PC 数组** `thread_pc[i]` 与 **per-thread done 位** `done_mask[i]`。

每拍（组合逻辑）：

```
fetch_pc    = min{ thread_pc[i] | i < thread_count 且 !done_mask[i] }
active_mask = { i | thread_pc[i] == fetch_pc 且 !done_mask[i] 且 i < thread_count }
```

- `fetch_pc` 喂给 fetcher（取指地址），同时作为本拍"正在执行的 PC"。
- `active_mask` 内的 lane 才真正提交：写寄存器 / 写 NZP / 发内存请求 / 推进自己的 PC。
- `active_mask` 外的 lane 本拍**全状态冻结**（寄存器、NZP、PC 都不变）。
- **重收敛自动发生**：当多个线程的 `thread_pc` 再次相等，它们自然回到同一个 mask。
- **前进性保证**：每拍总在推进当前"最小 PC"的那批线程，不会卡死。

> 为什么 min（最小 PC）能保证重收敛？对结构化控制流（if/else、while），分叉点之后两条路径终将汇合到同一条后续指令（post-dominator），其 PC 大于两条路径内部的 PC。持续优先推进较小 PC 的一侧，会让落后的一侧逐步追上，最终所有线程在汇合点 PC 相等 → 同一 mask → 重收敛。

## 4. 逐文件改动

| 文件 | 改动 |
|---|---|
| **scheduler.sv** | 新增 `reg [7:0] thread_pc[THREADS_PER_BLOCK-1:0]`、`reg [THREADS_PER_BLOCK-1:0] done_mask`。新增组合逻辑算 `fetch_pc=min` 与 `active_mask`。端口新增输出 `thread_pc[]`、`active_mask`；`current_pc` 改为输出 `fetch_pc`。UPDATE 阶段：`decoded_ret` 时把 `active_mask` 内线程置 `done_mask`，否则 `active_mask` 内线程 `thread_pc[i] <= next_pc[i]`；当 `i<thread_count` 的线程全部 done → `done <= 1, core_state <= DONE`。**删除** `current_pc <= next_pc[THREADS_PER_BLOCK-1];` 这行 hack。 |
| **core.sv** | 每个 `pc_instance[i].current_pc` 从共享 `current_pc` 改接 `thread_pc[i]`；fetcher 仍接 `current_pc(=fetch_pc)`。每条 lane 的 `enable` 由 `(i < thread_count)` 改为 `(i < thread_count) && active_mask[i]`，统一 gating alu/lsu/registers/pc。新增 `wire [7:0] thread_pc[...]`、`wire [...] active_mask` 连线。 |
| **pc.sv** | 几乎不动：本就用输入 `current_pc` 算 `next_pc`、持有自己的 NZP。现在它收到的 `current_pc` 变成"本线程的 PC"即可，语义自然正确。 |
| **lsu.sv / registers.sv / alu.sv** | 内部不改：写操作本就被 `enable` 守卫。mask 通过 `enable` 自动生效——mask 外 lane 不发内存请求、不写寄存器、不污染 NZP，状态被冻结。 |
| **fetcher.sv** | 不改：照常按 `current_pc(=fetch_pc)` 取指。 |

> 关键不变量：mask 外 lane 的所有写路径都经 `enable` 守卫，置 `enable=0` 即可"冻结全状态"。这是本设计能把改动面压到极小的原因。

## 5. 每拍数据流（if/else kernel 示例）

4 线程，`t0/t1` 走 THEN（如 `x>=0`），`t2/t3` 走 ELSE（`x<0`）。程序布局：

```
...5:CMP  6:BRnzp→ELSE(9)  7:THEN C=A+B  8:BR→STORE(10)  9:ELSE C=A-B  10:STORE  11:RET
分叉后 thread_pc = [7,7,9,9]
拍① fetch=7   mask={0,1}       跑 THEN        → thread_pc=[8,8,9,9]
拍② fetch=8   mask={0,1}       无条件跳        → thread_pc=[10,10,9,9]
拍③ fetch=9   mask={2,3}       跑 ELSE        → thread_pc=[10,10,10,10]
拍④ fetch=10  mask={0,1,2,3} ★重收敛 STORE 全体 → [11,11,11,11]
拍⑤ fetch=11  mask=all         RET → 全 done → core done
```

trace 中可直接观察到 mask：`{0,1,2,3} → {0,1} → {2,3} → 收回 {0,1,2,3}`。无条件跳转用 `BRnzp` 且 `decoded_nzp=3'b111`（匹配 CMP 后任意 NZP 位）实现。

## 6. 向后兼容

非分叉 kernel 下，所有活跃线程的 `thread_pc` 永远相等 → `fetch_pc` = 该 PC、`active_mask` = 全体活跃线程 → 行为与现状 lockstep **逐拍完全一致**。因此：

- `make test_matadd` 必须仍 PASS。
- `make test_matmul` 必须仍 PASS。

这是改造的回归基线，先于新功能验证。

## 7. 边界情况

- **per-thread 提前 RET**：先到 `RET` 的线程（在 `active_mask` 内执行到 RET）置 `done_mask`，退出 active 集；其余线程继续；`i<thread_count` 全部 done 才 core done。变长循环 kernel 专门验证此路径。
- **WAIT 阶段**：mask 外 lane 的 `lsu_state` 停在 IDLE，不会拖住 WAIT；只等 mask 内确实发了请求的 LSU。
- **`thread_count` 不满 block**：`i >= thread_count` 的 lane 从一开始就不进 active 集，与 mask 逻辑叠加无冲突。
- **初始化**：reset/start 时 `thread_pc[*]=0`、`done_mask=0`；首拍 `fetch_pc=0`、`active_mask` = 全体活跃线程。
- **全部 done 的判定时机**：在 UPDATE 更新 `done_mask`/`thread_pc` 后判断；若 `i<thread_count` 全 done → 进 DONE，否则据更新后的 `thread_pc` 重算 `fetch_pc`/`active_mask` 供下一次 FETCH。

## 8. 验证方案

1. **`test_relu.py`（if/else 主验收）**：构造让部分线程 `x<0` 的输入向量；kernel 计算 `max(0,x)`（`x<0` 与 `x>=0` 走不同路径，末尾 STORE 处重收敛）。断言输出 = 逐元素 `max(0,x)`；读 trace 确认 mask 缩放与重收敛。
2. **`test_divloop.py`（变长循环压测）**：每线程循环 `threadIdx` 次做累加，结果应为 `0,1,2,...`（或等差和）。验证 per-thread 提前退出 + 循环内重收敛。
3. **回归**：`test_matadd` / `test_matmul` 仍 PASS（第 6 节基线）。

验证均在远端执行：`cd ~/autoResearch/tiny-gpu && source env.sh && make test_*`。

### 8.1 实测结果（2026-06-30，分支 `stage5-branch-divergence`）

| 测试 | 周期数 | 状态 | 说明 |
|---|---|---|---|
| `test_relu`（if/else 分叉） | 198 | PASS | Y = `[0,5,0,7,0,6,0,4]` |
| `test_divloop`（变长循环） | 418 | PASS | Y = `[0,1,2,3]`，partial done_mask |
| `test_matadd`（回归） | 178 | PASS | 与改造前日志一致（178） |
| `test_matmul`（回归） | 491 | PASS | 与改造前日志一致（491） |

**回归零偏移**：matadd/matmul 周期数与改造前完全相同，验证了非分叉 kernel 逐拍行为不变（§6）。

**分叉证据（来自 `test_relu` trace）**：同一次运行中 fetcher 同时执行了
- else 路径 `STR R5, R4`（Y=X[i]，出现 144 次）
- then 路径 `CONST R6, #0` + `STR R5, R6`（Y=0，分别出现 88 / 144 次）

而 bug 版 lockstep 只会执行 else 路径（then 路径指令一次都不出现）。trace 片段（Core 0）显示 `%threadIdx=0` 的线程（`R4=X[0]=2 < T=4`）停在 **PC 12（then/ZERO 路径）**，与停在 else 路径的线程分道——min-PC 先跑完低 PC 的 else 段（pc10-11），再跑 then 段（pc12-13），最终在 `STR`/`RET` 处重收敛。

**代价**：分叉把两条路径串行化，relu 比"假装收敛"的 lockstep 多花约 28 周期（170 → 198），符合 SIMT 分叉的本质开销。

## 9. 已知局限与对下一阶段的接口预留

- min-PC 重收敛对**不可归约**控制流非最优，但 tiny-gpu kernel 均为结构化，不受影响。
- 为 **stage 5 第二阶段 warp scheduling** 预留：`active_mask` + per-warp 状态的思路可直接推广为"每 warp 一套 `thread_pc[]/done_mask/active_mask` + core 级 warp 选择器"。本阶段把 mask 逻辑收敛在 scheduler 内，正是为下一阶段按 warp 复制这套状态做准备。

---

### 改动文件清单（实现阶段据此展开）

- `src/scheduler.sv`（主改）
- `src/core.sv`（连线 + enable gating）
- `test/test_relu.py`（新增）
- `test/test_divloop.py`（新增）
- 可能需要的小汇编 kernel 内联在测试里（参照 `test/test_matadd.py` 的 `program=[...]` 写法）

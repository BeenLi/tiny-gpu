# Stage 6 结果：Warp Scheduling — 延迟隐藏实测（348→274 cycle，约 21%）

> 结论：每核驻留 2 个 warp（switch-on-stall），在 `test_warpadd`（threads=16）上使 baseline 的 348 cycle 降至 274 cycle，**净省约 74 拍（~21%）**；5 个测试（含尾部部分-warp 和多-batch 分派）全部通过。

---

## 1. Cycle 对照表

| 测试 | 调度策略 | cycle 数 | 备注 |
|---|---|---|---|
| test_warpadd（threads=16）| baseline（单 warp，整核阻塞）| **348** | master 分支原始实现 |
| test_warpadd（threads=16）| warp-sched（每核驻留 2 warp，switch-on-stall）| **~274** | 省 74 cyc，**~21%** |
| test_matadd（threads=8）| warp-sched | ~267 | 回归验证 |
| test_matmul（threads=8）| warp-sched | ~457 | 回归验证 |
| test_matadd_tail（threads=6）| warp-sched | ~267 | 尾部部分-warp（tc=2）覆盖 |
| test_matadd_multibatch（threads=20）| warp-sched | ~427 | 多-batch（5 blocks > 4 slots）覆盖 |

---

## 2. 延迟隐藏 Trace 证据

以下原始 trace 摘自 `test_warpadd` 仿真日志（Core 0，Cycle 70–80），完整体现了 warp 调度"用计算填满等待"的核心机制。

### 原始 trace（Core 0，Cycle 70–80）

```
================================== Cycle 70 ==================================

+--------------------- Core 0 ---------------------+
current_warp=1  core_state=FETCH  fetcher_state=IDLE  PC=0  instr=LDR R4, R4  done=0
  warp 0: state=WAIT          ← warp 0 发出 LDR 请求，进入 WAIT；调度器切换到 warp 1
    thread 0.0: lsu=REQUESTING  RS=0 RT=0  [R0=0, R2=16, R3=32, R4=0, ...]
    thread 0.1: lsu=REQUESTING  RS=1 RT=1  [R0=1, R2=16, R3=32, R4=1, ...]
    thread 0.2: lsu=REQUESTING  RS=2 RT=2  [R0=2, R2=16, R3=32, R4=2, ...]
    thread 0.3: lsu=REQUESTING  RS=3 RT=3  [R0=3, R2=16, R3=32, R4=3, ...]
  warp 1: state=FETCH  <== current  ← warp 1 接管流水线，开始 FETCH
    thread 1.0: lsu=IDLE  [R0=0, %blockIdx=1, ...]
    ...

================================== Cycle 71–76 ==================================
（共 6 拍；每拍 warp 0 均 state=WAIT / lsu=WAITING；warp 1 推进 FETCH→...）
current_warp=1  core_state=FETCH  fetcher_state=FETCHING  ...
  warp 0: state=WAIT
    thread 0.0: lsu=WAITING  ...   ← 内存请求在途，warp 0 挂起等待
    thread 0.1: lsu=WAITING  ...
    ...
  warp 1: state=FETCH  <== current  ← warp 1 持续 FETCH，占用流水线

================================== Cycle 77 ==================================

current_warp=1  core_state=FETCH  fetcher_state=FETCHED  PC=0  instr=MUL R0, %blockIdx, %blockDim  done=0
  warp 0: state=WAIT          ← 内存响应已到，LSU 转为 DONE；但调度器还给 warp 1 继续
    thread 0.0: lsu=DONE  RS=0 RT=0  [R4=0, ...]   ← 内存数据就绪
    thread 0.1: lsu=DONE  RS=1 RT=1  [R4=1, ...]
    thread 0.2: lsu=DONE  RS=2 RT=2  [R4=2, ...]
    thread 0.3: lsu=DONE  RS=3 RT=3  [R4=3, ...]
  warp 1: state=FETCH  <== current  ← warp 1 完成取指（FETCHED）

================================== Cycle 78 ==================================

current_warp=1  core_state=DECODE  fetcher_state=FETCHED  PC=0  instr=MUL R0, %blockIdx, %blockDim
  warp 0: state=WAIT   lsu=DONE   ← 仍挂起（等下一轮调度）
  warp 1: state=DECODE  <== current  ← warp 1 进入译码阶段

================================== Cycle 79 ==================================

current_warp=1  core_state=REQUEST  ...
  warp 0: state=WAIT   lsu=DONE
  warp 1: state=REQUEST  <== current  ← warp 1 进入请求阶段

================================== Cycle 80 ==================================

current_warp=1  core_state=EXECUTE  ...
  warp 0: state=WAIT   lsu=DONE   ← warp 0 内存等待期内，warp 1 已走完 FETCH→DECODE→REQUEST
  warp 1: state=EXECUTE  <== current  ← warp 1 执行 MUL（第一条真正的计算指令）
    thread 1.0: lsu=IDLE  RS=1 RT=4  [%blockIdx=1, %blockDim=4, ...]
    thread 1.1: lsu=IDLE  RS=1 RT=4  ...
    thread 1.2: lsu=IDLE  RS=1 RT=4  ...
    thread 1.3: lsu=IDLE  RS=1 RT=4  ...
```

### 逐拍解读

| Cycle | warp 0（挂起方）| warp 1（推进方）| 说明 |
|---|---|---|---|
| 70 | state=WAIT, lsu=REQUESTING | state=FETCH（current_warp 切换到 1）| **switch-on-stall 触发**：LDR 发送请求，立即让出调度权 |
| 71–76 | state=WAIT, lsu=WAITING | state=FETCH（持续）| warp 1 在内存等待期内推进 FETCH；warp 0 的 LSU 后台异步等待 |
| 77 | state=WAIT, lsu=**DONE** | state=FETCH（FETCHED）| 内存响应到达，warp 0 数据就绪；但 warp 1 仍持有调度权 |
| 78 | state=WAIT, lsu=DONE | state=**DECODE** | warp 1 译码，warp 0 继续等待下一次轮转 |
| 79 | state=WAIT, lsu=DONE | state=**REQUEST** | warp 1 继续推进 |
| 80 | state=WAIT, lsu=DONE | state=**EXECUTE** | warp 1 执行 MUL；这整段 ~10 拍的内存等待被 warp 1 的指令填满，**延迟隐藏成功** |

**关键点**：baseline 中 warp 0 的这段 WAIT 会让整个 core 空转；warp scheduling 后，这些周期被 warp 1 的 FETCH/DECODE/EXECUTE 充分利用。

---

## 3. 残余停顿讨论

延迟隐藏并非完全，仍有残余停顿（348→274 而非→0）。原因：

- **所有驻留 warp 同时卡在内存时**，core 才真正空转。本测试只驻留 2 个 warp；若两个 warp 的 LDR 请求在时间上高度重叠，则内存等待仍留有空洞。
- **内存带宽瓶颈**：`DATA_MEM_NUM_CHANNELS=4`，4 个通道并发处理最多 4 路请求。warp 越多越能填满等待，但最终受限于内存带宽——即使增加 WARPS_PER_CORE，当带宽饱和后 cycle 节省趋于平坦。
- **取指带宽**：fetcher 仍为每 core 1 个，各 warp 串行取指，轮转本身有开销。

---

## 4. 根因修复简述

实现过程中暴露并修复了两个根因 bug：

### Bug 1：format.py 层级适配
- **现象**：多 warp 后 trace 打印崩溃（KeyError / index 越界）。
- **根因**：原始 `format.py` 的 `log_cycle()` 假设 `lsu_state` 为 `[T]`，改为 `[W][T]` 后层级不匹配。
- **修复**：在 `format.py` 中用 `isinstance` 判断 `lsu_state` 是否为二维嵌套列表，正确索引到 `lsu_state[w][t]`。

### Bug 2：gpu.sv warp 元数据组合传递
- **现象**：所有 warp 的 `thread_count` 都被误判为 0，warp scheduler 把所有 warp 都视为 DONE，导致 GPU 立即退出。
- **根因**：`gpu.sv` 中 warp 元数据（`warp_thread_count`、`warp_block_id`）从 dispatch 信号走了一拍寄存器（`<=`）再送给 scheduler；scheduler 在 dispatch 的同一拍执行 start 判定时读到的是上一拍的旧值（全 0）。
- **修复**：将 warp 元数据的传递改为组合逻辑（`=` 直通），消除一拍延迟，让 scheduler 在 start 拍直接读到有效值。

---

## 5. 覆盖测试说明

| 测试 | 覆盖点 | 结果 |
|---|---|---|
| test_warpadd（threads=16）| 基础对照，4 core × 2 warp | PASS |
| test_matadd（threads=8）| 回归，含 LDR/STR | PASS |
| test_matmul（threads=8）| 回归，乘加密集型 | PASS |
| test_matadd_tail（threads=6）| 尾部部分-warp（warp 1 只有 tc=2 线程）| PASS |
| test_matadd_multibatch（threads=20）| 多-batch（5 blocks > 4 slots，dispatch 轮转分配）| PASS |

---

## 6. 参考文档

- [设计文档](stage6_warp_scheduling.md)
- [实现计划](stage6_warp_scheduling_plan.md)

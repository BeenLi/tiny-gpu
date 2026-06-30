# 阶段 5（一）理论专题：SIMT Branch Divergence 与 Reconvergence

> 配套 [stage5_branch_divergence.md](stage5_branch_divergence.md)（本仓库的 min-PC 实现）。本文把"分叉/重收敛"放回完整的 GPU 体系结构理论里：为什么会分叉、硬件怎么收敛、代价从哪来、真实 GPU 怎么做、我们的实现处在理论谱系的哪一格。

---

## 1. SIMT 执行模型：lockstep 是省晶体管的代价

GPU 的核心权衡是**用并行度换取单线程的简单**。一个 warp（NVIDIA=32 线程，AMD wavefront=64 或 RDNA 的 wave32）里的所有线程**共享一套取指/译码/调度逻辑**，只在执行级有各自的 ALU/寄存器/数据通路。这就是 **SIMT（Single Instruction, Multiple Thread）**：

- 一份 PC、一份指令、一个 scheduler → N 条数据通路（lane）。
- 取指/译码的硬件成本被 N 条 lane 摊薄——这是 GPU 能堆出上万条 lane 的根本原因。
- 代价：一个 warp 每拍只能执行**一条**指令。只要所有线程走同一条指令流（同一 PC），这套模型就完美。

tiny-gpu 改造前正是最纯粹的 SIMT lockstep：`core` 只有一个 `current_pc`、一个 fetcher，所有线程被强制同 PC（`scheduler.sv` 旧代码 `current_pc <= next_pc[THREADS_PER_BLOCK-1]`——直接拿最后一个线程的分支结果强加给全体）。

## 2. Branch Divergence：warp 内的控制流分叉

问题出在**数据相关的控制流**：

```c
if (x[i] < T) y[i] = 0;     // 一部分线程走这条
else          y[i] = x[i];  // 另一部分线程走那条
```

同一 warp 里，`x[i]` 因线程而异 → `CMP` 之后每个线程的 NZP 不同 → `BRnzp` 让它们算出**不同的 next PC**。此刻"一份 PC 跑全体"的假设破裂：线程在控制流上**分叉（diverge）**。

关键认知：**divergence 是 warp _内_ 的现象**（intra-warp）。它和 warp _间_ 的调度（warp scheduling，第二阶段）是正交的两个轴：

| | 轴 | 单位 | 解决的问题 |
|---|---|---|---|
| Branch divergence | warp **内** | 线程 | 同一 warp 线程走不同路径 |
| Warp scheduling | warp **间** | warp | 多 warp 交错跑、隐藏延迟 |

SIMT 硬件**无法**真正"同时"执行两条不同的指令。所以分叉的本质处理只有一个办法：**把不同路径串行化**——先执行一条路径（屏蔽走另一条路径的线程），再执行另一条。屏蔽用的就是 **active mask**（活跃掩码）：一个 per-lane 的位向量，指明"本指令哪些线程真正提交结果"。被屏蔽的线程消耗周期但不写任何状态（寄存器/内存/PC 全冻结）。

## 3. Reconvergence：在哪里、为什么要收回来

如果只分叉不收敛，利用率会随每个分支指数级崩塌（嵌套两层 if 就可能掉到 1/4）。所以分叉后必须尽早让线程**重收敛（reconverge）**回到同一 PC、恢复满 mask。

理论上的最佳收敛点是分支的 **immediate post-dominator（IPDOM，直接后必经点）**：从分支出发、两条路径**必然都会经过**的第一个点。对 `if/else`，IPDOM 就是 `if` 之后的汇合语句；对 `while`，是循环出口。在 IPDOM 处恢复满 mask，能保证"只要还能并行就并行"。

> **后必经点（post-dominator）**：若从节点 A 出发到达函数出口的**所有**路径都经过 B，则 B 后必经 A。最近的那个 B 就是 IPDOM。它是控制流图（CFG）上的纯静态属性，通常由**编译器**计算。

## 4. 三类硬件机制（理论谱系）

### 4.1 SIMT Reconvergence Stack（IPDOM 栈）— 真实 GPU 的经典做法（NVIDIA Volta 之前）

每个 warp 维护一个**重收敛栈**，每个栈项 = `(重收敛PC=IPDOM, active_mask, 该路径的下一PC)`。遇到分叉分支时：

1. 压入一个"重收敛项"（PC=IPDOM，mask=分叉前的全集）。
2. 把两条路径各压一项（各自的 mask 与起始 PC）。
3. 弹栈、执行栈顶路径直到其 PC 到达 IPDOM；再弹下一条路径；最后弹重收敛项 → 恢复满 mask。

优点：在 IPDOM 精确收敛，对任意结构都正确。代价：需要**编译器提供 IPDOM 信息**（ISA 里带重收敛标记，如 NVIDIA 的 `SSY`/`.S` 标志位）、需要每 warp 一个栈、控制复杂。

### 4.2 min-PC / PC-based Reconvergence（栈-free 启发式）— **本仓库实现的就是这个**

不维护栈，也不需要编译器给 IPDOM。规则极简：

```
每拍 fetch_pc = min{ 仍在运行线程的 PC }
active_mask  = { PC == fetch_pc 的线程 }
```

**为什么对结构化代码必然收敛？** 两条直觉：

- **前进性**：每拍总在推进"当前最小 PC"的那批线程 → 永不卡死。
- **自动汇合**：对**可归约（reducible）CFG**——也就是正常的 `if/else`、`while`、`for`——重收敛点的 PC 总是大于分叉两条路径内部的 PC（前向分支跳过一段、循环体在 check 与 exit 之间）。持续优先推进较小 PC 的一侧，会让落后的一侧逐步追上，最终所有线程在汇合点 PC 相等 → 同一 mask → 收敛。

代价/局限：

- 需要一个对 PC 的 **min 归约**（小比较器树）和 per-thread PC 寄存器。
- 对**不可归约（irreducible）/ 非结构化** CFG（如 goto 进入循环中部）不保证最优收敛——但高级语言编译出的代码几乎都是可归约的，tiny-gpu 的 kernel 也都是。
- 它本质是 IPDOM 栈的一个"够用版"近似：对结构化代码两者收敛点一致，对病态 CFG 才有差异。

### 4.3 Independent Thread Scheduling（ITS）— NVIDIA Volta(2017) 之后

Volta 给**每个线程独立的 PC 和调用栈**，并引入硬件"convergence optimizer"在运行时寻找收敛机会。它打破了 Volta 之前"warp 内隐式同步"的假设——因此现代 CUDA 必须显式用 `__syncwarp()` 而不能再依赖 warp-synchronous 编程。ITS 的好处是能交错执行分叉的两条路径（例如让一条路径在等内存时跑另一条），代价是每线程状态更多、收敛不再保证立刻发生。

> 谱系小结：**栈（精确，需编译器）→ min-PC（启发式，自给自足）→ per-thread PC（最灵活，最贵）**。我们选 min-PC，正是因为它在"最小硬件改动"和"真实可用的重收敛"之间最契合 tiny-gpu。

## 5. 性能代价：SIMT 利用率与串行化

衡量分叉损失的标准指标是 **SIMT/warp execution efficiency**（也叫 lane utilization）：

```
效率 = (各拍活跃 lane 数之和) / (总拍数 × warp 宽度)
```

- 无分叉：每拍满 mask → 100%。
- 2 路均分的 `if/else`：分叉区每拍只有半数 lane 活跃 → 该区 50%。
- 最坏情况：32 线程各走唯一路径 → 分叉区跌到 1/32 ≈ 3%。
- 嵌套分叉会**相乘**衰减（两层各半 → 25%）。

代价的物理本质是**路径串行化**：硬件没法同时跑两条指令，只能一条接一条。本仓库 `test_relu` 实测：

- bug 版 lockstep（只跑 else 路径，结果错）：170 cycles。
- 正确分叉版（else 段 + then 段串行 + 重收敛）：198 cycles，**多出的 ~28 cycle 就是 then 路径被串行执行的开销**。

这把抽象的"divergence 有代价"变成了你能数出来的周期数——这正是写 GPU kernel 时"尽量让一个 warp 内线程走同一分支"的硬件根源。

## 6. 编译器侧的对策

硬件机制之外，编译器也在抑制分叉：

- **Predication（谓词化）**：对很短的分支体，编译器不生成跳转，而是让所有 lane 都执行两边的指令，再用谓词位决定是否写回。没有 PC 分叉 → 没有 mask 串行化的控制开销，但两边都算（浪费算力）。短分支用谓词、长分支用真分叉，是经典权衡。tiny-gpu 没有谓词化，全靠真分叉。**详见专题 [Predication（谓词化）](stage5_predication.md)。**
- **结构化 + IPDOM 标注**：编译器保证 CFG 可归约，并为栈式硬件标注重收敛点。
- **分支最小化 / 数据布局**：让同一 warp 的线程尽量走同一路径（例如按条件排序数据），把 divergence 消灭在编译/数据准备阶段。

## 7. 进阶研究方向（超出 tiny-gpu，建立视野）

分叉收敛是 GPU 体系结构的长青研究点，代表性思路：

- **Dynamic Warp Formation**（Fung et al., 2007）：把不同 warp 里走同一 PC 的线程**动态重组**成新满 warp，提高利用率。
- **Thread Block Compaction**（Fung & Aamodt, 2011）：在 block 粒度对分叉后的线程做压缩重组。
- **Thread Frontiers**（Diamos et al., 2011）：用更细的收敛点排序，改善非结构化控制流。
- **Dual-Path / Multi-Path Execution**：让栈上多条路径**交错**发射，而非严格串行，借此在一条路径停顿时跑另一条。
- 这些都在回答同一个问题：**分叉之后，如何更快地把利用率拉回来。**

## 8. 映射回 tiny-gpu：我们做了什么、简化了什么

| 理论要素 | tiny-gpu 第一阶段实现 |
|---|---|
| active mask | `scheduler.active_mask`，经 `core` 的 `enable=(i<thread_count)&&active_mask[i]` 屏蔽 lane |
| per-thread PC | `scheduler.thread_pc[i]`（复用每线程已有的 `pc.sv`/`next_pc[i]`）|
| 重收敛机制 | min-PC（§4.2），无栈、无 IPDOM、无 ISA 改动 |
| per-thread 退出 | `done_mask`，线程执行 RET 即退出 active 集；全退出才 core done |
| 收敛点 | 隐式——由 min-PC 在 PC 重合处自动达成（无需标注）|
| 简化/未做 | 谓词化、IPDOM 栈、ITS、动态 warp 重组、不可归约 CFG 的保证 |

一句话定位：**我们实现了 SIMT 分叉/重收敛的最小正确内核（min-PC + active mask + per-thread done），它对结构化 kernel 等价于真实 GPU 的 IPDOM 栈，但硬件代价小一个量级。**

## 9. 通向第二阶段：divergence 与 warp scheduling 如何叠加

第二阶段要让一个 core **驻留多个 warp** 并在它们之间交错调度（隐藏内存延迟）。两者的叠加关系很清晰：

- divergence 的状态（`thread_pc[]` / `active_mask` / `done_mask`）是**每个 warp 一套**。
- warp scheduling 在这些"带 mask 的 warp"**之间**做选择（round-robin / GTO 等策略），让一个 warp 在 WAIT 时另一个 warp 顶上。
- 也就是说：第一阶段把"一个 warp 的分叉状态"收敛进了 scheduler——第二阶段只需把这套状态**按 warp 复制**，再加一个 warp 选择器。这正是我们第一阶段设计时刻意预留的接口。

---

### 一页速记

- SIMT = 一份取指喂多条 lane；省晶体管，代价是 warp 每拍一条指令。
- divergence = warp 内线程因数据走不同 PC；硬件只能**串行化 + active mask**。
- reconvergence = 尽快收回满 mask；理论最优点是 **IPDOM**。
- 三档机制：**IPDOM 栈**（真·精确，需编译器）／ **min-PC**（启发式，我们用的）／ **per-thread PC/ITS**（Volta+，最灵活最贵）。
- 代价 = SIMT 利用率下降 + 路径串行（relu 实测 170→198 cycle）。
- divergence 是 warp 内、scheduling 是 warp 间——正交，可叠加（per-warp mask）。

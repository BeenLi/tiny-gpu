# tiny-gpu 学习指南

一份由浅入深、可动手验证的学习路线。整个仓库约 1800 行代码、15 个 Verilog 文件，文档极其完整，非常适合系统学习 GPU 的硬件原理。

## 学习全景：先建立心智模型

这个 GPU 的层次结构是：**GPU → 多个 Core → 每个 Core 内每个 Thread 有独立的 ALU/LSU/PC/寄存器**。

所有学习都应围绕这个层次展开。推荐的总策略是：**先跑通仿真看现象，再自顶向下读代码，最后自底向上抠细节。**

## 精读笔记归档

逐阶段精读时积累的详细笔记：

- [阶段 1：ISA 精读](stage1_isa.md) —— 指令格式、字段复用、控制信号、CMP+BRnzp 与 NZP 独热掩码
- [阶段 2：自顶向下读架构](stage2_architecture.md) —— gpu/dcr/dispatch/controller/core，valid/ready 握手，EDA 痕迹与累加器时序代价
- [阶段 3：核内状态机](stage3_execution.md) —— scheduler/fetcher/registers/alu/lsu/pc，含 **ALU NZP bug 的发现与修复**
- [阶段 4：仿真测试框架](stage4_simulation.md) —— Makefile/setup/memory/logger，cocotb 仿真闭环
- [阶段 5（一）：Branch Divergence 设计](stage5_branch_divergence.md) / [实现计划](stage5_branch_divergence_plan.md) / [理论专题](stage5_branch_divergence_theory.md) / [Predication 专题](stage5_predication.md) —— min-PC active mask，per-thread PC + done_mask，自动重收敛；理论篇涵盖 SIMT/IPDOM 栈/min-PC/ITS 谱系与利用率代价；谓词篇讲 if-conversion 与 active mask 的同构

---

## 阶段 0：先把它跑起来（半天）

环境已配好（见 [SETUP_NOTES.md](../SETUP_NOTES.md)），先获得"它能算出正确结果"的直观感受：

```sh
cd ~/autoResearch/tiny-gpu
source env.sh
make test_matadd
make test_matmul
```

然后**精读一份执行 trace 日志**（在 [test/logs/](../test/logs/)）。这是整个仓库最有价值的学习材料——它逐周期打印了每个 core、每个 thread 的 PC、寄存器、状态机状态。对照日志看"一条指令是怎么一拍一拍走完的"，比读代码更直观。

---

## 阶段 1：读懂 README + ISA（1 天，纯文档）

完整读一遍 [README.md](../README.md)，重点理解三个抽象：

1. **执行模型**：6 个阶段 `FETCH → DECODE → REQUEST → WAIT → EXECUTE → UPDATE`
2. **ISA**：11 条指令（看 `docs/images/isa.png`），特别是 `%blockIdx/%blockDim/%threadIdx` 三个只读寄存器如何实现 SIMD
3. **两个 kernel**：把 `matadd.asm` 和 `matmul.asm` 在纸上手动"执行"一遍，算出地址和结果

**检验点**：你能不能解释 `i = blockIdx * blockDim + threadIdx` 为什么能让 8 个线程各处理矩阵的一个元素？

---

## 阶段 2：自顶向下读代码（2-3 天）

按"从大到小、从外到内"的顺序读，每个文件都对照 README 对应章节：

| 顺序 | 文件 | 行数 | 看什么 |
|---|---|---|---|
| 1 | [src/gpu.sv](../src/gpu.sv) | 217 | **顶层模块**。看它如何例化 dcr、dispatch、core，以及 memory controller 的连线。这是全局接线图 |
| 2 | [src/dcr.sv](../src/dcr.sv) | 28 | 最简单的模块，存 `thread_count`。先读它建立信心 |
| 3 | [src/dispatch.sv](../src/dispatch.sv) | 91 | 如何把 threads 切成 block 分发给各 core，如何判断 kernel 结束 |
| 4 | [src/controller.sv](../src/controller.sv) | 133 | 内存控制器：多核请求如何按带宽节流、仲裁 |
| 5 | [src/core.sv](../src/core.sv) | 212 | **核心**。看它如何例化 scheduler + 每个 thread 的 fetcher/decoder/alu/lsu/pc/registers |

---

## 阶段 3：钻进 Core 内部的状态机（2-3 天）

这是整个项目的精髓，也是最难的部分。按这个顺序：

| 顺序 | 文件 | 看什么 |
|---|---|---|
| 1 | [src/scheduler.sv](../src/scheduler.sv) | **最关键**。6 阶段状态机就在这里。重点理解它如何处理 LSU 的异步等待（`WAIT` 状态），以及所有线程"同步推进"的假设 |
| 2 | [src/fetcher.sv](../src/fetcher.sv) | 异步从 program memory 取指的小状态机 |
| 3 | [src/decoder.sv](../src/decoder.sv) | 16 位指令如何译码成一堆控制信号（连接 ISA 和硬件的桥梁） |
| 4 | [src/pc.sv](../src/pc.sv) | PC 默认 +1，以及 `CMP`/`BRnzp` 如何用 NZP 寄存器实现分支 |
| 5 | [src/registers.sv](../src/registers.sv) | 寄存器堆，注意那 3 个只读 SIMD 寄存器的特殊处理 |
| 6 | [src/alu.sv](../src/alu.sv) | 最简单，ADD/SUB/MUL/DIV/CMP |
| 7 | [src/lsu.sv](../src/lsu.sv) | LDR/STR，**异步内存访问**的状态机——理解了它就理解了 scheduler 为什么要 WAIT |

**学习方法**：选 trace 日志里的一条 `LDR` 指令，跟着它走完 scheduler → lsu → controller → memory → 回来 的完整路径。这一条线打通，整个 GPU 就懂了。

---

## 阶段 4：读懂仿真测试框架（1 天）

理解"汇编 kernel 是怎么变成硬件输入，结果又怎么读出来"：

- [Makefile](../Makefile) — 编译流程：`sv2v`（SystemVerilog→Verilog）→ `iverilog` → `cocotb`
- [test/test_matadd.py](../test/test_matadd.py) — 如何加载程序/数据内存、设置 thread_count、拉高 start、读结果
- [test/helpers/](../test/helpers/) — `memory.py`（模拟外部内存+带宽）、`format.py`（trace 打印）、`setup.py`

---

## 阶段 5：动手改造（巩固，开放式）

读懂之后，最好的巩固是动手。README 末尾的 "Next Steps" 是现成的练习清单，从易到难：

1. **写一个新 kernel**（比如向量点积、ReLU），跑通仿真 —— 最容易，巩固 ISA
2. **给 ISA 加一条新指令**（比如 `MOD` 或 `AND`）—— 需要改 decoder + alu，理解数据通路
3. **加一个指令 cache** —— README 列的第一个 TODO，中等难度
4. 更难的：branch divergence、memory coalescing、pipelining

### 进展

- ✅ **Branch divergence（min-PC active mask）** —— 已完成（分支 `stage5-branch-divergence`）。每线程持有 `thread_pc[i]`，每拍 fetch 最小 PC、只执行 `active_mask` 内线程，PC 重合自动重收敛；支持 per-thread RET。验证 kernel：`test_relu`（if/else）+ `test_divloop`（变长循环，partial done_mask）；matadd/matmul 回归零偏移。详见 [设计文档](stage5_branch_divergence.md) 与 [实现计划](stage5_branch_divergence_plan.md)。
- ⬜ **Warp scheduling（多 warp 驻留 + 交错调度）** —— 第二阶段，待开。

---

## 几条建议

- **trace 日志是你最好的老师**，遇到任何看不懂的代码，去日志里找对应周期对照
- 读 `.sv` 时重点关注 `state` 信号和 `always @(posedge clk)` 块——这是状态机的骨架
- 阶段 2-3 是重点，**别在顶层连线上花太多时间**，核心理解在 scheduler + lsu 的异步协作
- 卡住时，画时序图或状态转移图，硬件逻辑画出来就清晰了

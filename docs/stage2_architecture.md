# 阶段 2：自顶向下读架构笔记

> 自顶向下：先看整机接线图 [gpu.sv](../src/gpu.sv)，再逐个部件下钻。本笔记覆盖 gpu / dcr / dispatch，以及途中的 EDA 痕迹与时序讨论。

## 一、顶层 gpu.sv

### 参数：整机规模

[gpu.sv:10-18](../src/gpu.sv#L10-L18)：

| 参数 | 默认 | 含义 |
|---|---|---|
| `NUM_CORES` | 2 | 计算核数 |
| `THREADS_PER_BLOCK` | 4 | 每核每 block 线程数 → 决定 ALU/LSU/寄存器份数 |
| `DATA_MEM_NUM_CHANNELS` | 4 | 数据内存并发通道（带宽） |
| `PROGRAM_MEM_NUM_CHANNELS` | 1 | 程序内存并发通道 |

派生量 [gpu.sv:58](../src/gpu.sv#L58)：`NUM_LSUS = NUM_CORES * THREADS_PER_BLOCK = 8`。
**8 个 LSU 抢 4 条数据通道** → 这就是内存控制器要做仲裁限流的根源。

### 五大部件（README 架构图的代码版）

| 部件 | 位置 | 作用 |
|---|---|---|
| dcr | [76-83](../src/gpu.sv#L76-L83) | 存 `thread_count` |
| data controller | [86-112](../src/gpu.sv#L86-L112) | 8 LSU ↔ 4 通道，可读写 |
| program controller | [115-134](../src/gpu.sv#L115-L134) | 2 fetcher ↔ 1 通道，只读（`WRITE_ENABLE(0)`） |
| dispatch | [137-151](../src/gpu.sv#L137-L151) | 派 block、汇总 done |
| core × N | [155-216](../src/gpu.sv#L155-L216) | generate 循环例化 |

**同一个 `controller` 模块复用两次**（数据内存可读写4通道 / 程序内存只读1通道），区别只在参数。

### 贯穿全机的 valid/ready 握手

所有内部内存访问统一用这套异步握手：
```
consumer 侧            controller 侧
  read_valid  ──"我要读"──→
  read_address ──地址──→
            ←──"好了"── read_ready
            ←──数据──── read_data
```
这就是 LSU `REQUESTING→WAITING→DONE` 状态机的物理含义。controller / lsu / fetcher 全是它的变体。

### generate 块要点

- 一维展平寻址 [gpu.sv:171](../src/gpu.sv#L171)：`lsu_index = i*THREADS_PER_BLOCK + j`，core0 占 LSU 0-3，core1 占 4-7（与 trace 线程编号一致）。
- per-core 中转信号 [gpu.sv:159-184](../src/gpu.sv#L159-L184)：core 内用 `[THREADS_PER_BLOCK]` 局部数组，顶层用 `[NUM_LSUS]` 全局数组，中间逐位搬运。**这是 EDA 妥协**（OpenLane 用 Verilog 2005，不许切片顶层信号），学习时可忽略，当成 LSU 直连 controller。

## 二、dcr.sv（最简单）

全文一个 8 位寄存器：`device_control_write_enable` 高时存入 `device_control_data`，常驱动到 `thread_count`（[dcr.sv:19-27](../src/dcr.sv#L19-L27)）。这是 setup 启动前写 `threads=8` 的落点。（第16行 `device_conrol_register` 是拼写错误，不影响功能。）

## 三、dispatch.sv（block 调度器，≈ NVIDIA GigaThread）

### 算 block 数

[dispatch.sv:30-31](../src/dispatch.sv#L30-L31)：`total_blocks = (thread_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK`（向上取整除法）。matadd：`(8+3)/4=2`。

### 两个计数器

- `blocks_dispatched`：已派出
- `blocks_done`：已完成
- 结束条件 [dispatch.sv:61-63](../src/dispatch.sv#L61-L63)：`blocks_done == total_blocks` → `done`

### 每核生命周期：reset → start → done → reset

- **派发** [65-80](../src/dispatch.sv#L65-L80)：core 刚 reset 且有未派 block → 设 `core_start/core_block_id/core_thread_count`，`blocks_dispatched++`
- **回收** [82-89](../src/dispatch.sv#L82-L89)：core 在跑且报 done → reset 它，`blocks_done++`

**core 是可复用资源**，谁先空出谁接下一个 block —— 这是 "block 数可远多于物理 core 数" 的根基。

### 最后一个 block 的零头

[73-75](../src/dispatch.sv#L73-L75)：若线程不能整除，最后一块用 `thread_count - dispatched*THREADS_PER_BLOCK` 算实际线程数；core 内用它屏蔽多余线程（对应 [pc.sv](../src/pc.sv) 的 `enable`）。

## 四、两个 EDA / 时序知识点（重要）

### ① `start_execution` 一次性触发器 —— 真·EDA 妥协

想表达"`start` 第一次拉高时复位所有 core 踢动流水线"。自然写法 `@(posedge start)` 会把 `start` 当时钟 → 多时钟域/多驱动冲突，且 `start` 非合法时钟。

解法（[dispatch.sv:53-58](../src/dispatch.sv#L53-L58)）：全部塞进唯一 `@(posedge clk)`，用 `start_execution` 标志自己做边沿检测，等价于一个 **one-shot**。纯为迁就 Verilog 2005 流片工具，**可忽略**。

### ② 计数器用阻塞赋值 `=` —— 不是 EDA，是 Verilog 语义正确用法

`blocks_dispatched`/`blocks_done` 用阻塞 `=`，其余用非阻塞 `<=`。原因：**一个时钟沿内可能多个 core 同时改同一计数器**，for 循环顺序遍历时后一次必须看到前一次的更新值。

```
阻塞 =（正确）：i=0 派 block0 后计数立刻=1；i=1 看到 1 → 派 block1 ✓
非阻塞 <=（错误）：i=0、i=1 都读到旧值 0 → 都派 block0，计数只 +1 ✗
```

阻塞让 for 循环像软件顺序累加，处理"一拍内可变数量的并行事件改共享累加器"。**这是值得记住的硬件设计知识点**，与 EDA 工具无关。

### ③ 该写法的硬件代价（时序）

可综合，但综合成一条**串行进位组合链**：for 循环被展开成 N 个串联条件加法器，`blocks_dispatched` 只有一个寄存器（沿末更新），中间累加全是组合中间线网。

- 关键路径延迟 ≈ `N × (比较+加法+选择)`，**O(N) 线性增长** → 限制 f_max
- tiny-gpu N=2，可忽略；扩到几十上百 core 就成瓶颈
- 真实硬件改法：**popcount 加法树**（把"本拍要派的 core"做成 mask，树形数 1，O(log N)）、流水线化、或每核独立计数+末端归约
- 思想：用面积/复杂度换时钟频率，把 O(N) 串行进位链换成 O(log N) 并行归约树

> 启示：行为级 Verilog（for + 阻塞累加）写起来像顺序软件，综合出来却是一条物理延迟链。README "Next Steps" 的 *"Optimize control flow ... to improve cycle time"* 即指此类。

## 五、controller.sv（内存控制器：多消费者→少通道的限流漏斗）

最体现"内存带宽约束"的模块，是 trace 里访存排队的根源。

### 两侧接口

```
消费者侧 (NUM_CONSUMERS)          内存侧 (NUM_CHANNELS)
  8 个 LSU      ──────►  controller  ──────►  4 条数据通道
  或 2 个 fetcher                              或 1 条程序通道
```

漏斗窄口 = 限流。同时最多 `NUM_CHANNELS` 个访存在飞，其余消费者等待。**带宽 = 通道数**。

### 核心数据结构（[controller.sv:44-47](../src/controller.sv#L44-L47)）

- `controller_state[NUM_CHANNELS]`：每条通道独立的状态机
- `current_consumer[NUM_CHANNELS]`：每条通道正在服务谁
- `channel_serving_consumer[NUM_CONSUMERS]`：认领板，防止多通道抢同一请求

每条通道 = 一个独立"搬运工"，一次只搬一个消费者请求，全程跟到底。外层 `for(i in NUM_CHANNELS)` 让所有通道并发。

### 每条通道的 4 阶段状态机（与 [format.py:78-86](../test/helpers/format.py#L78-L86) 对应）

以读为例：
```
IDLE ──找到请求──► READ_WAITING ──内存返回──► READ_RELAYING ──消费者确认──► IDLE
```

- **IDLE** [70-96](../src/controller.sv#L70-L96)：从 j=0 扫描，找第一个"有请求且未被认领"的消费者 → 认领它、向内存发起读、`break`
- **READ_WAITING** [97-105](../src/controller.sv#L97-L105)：等 `mem_read_ready` → 把数据转交消费者、拉高 `consumer_read_ready`
- **READ_RELAYING** [115-121](../src/controller.sv#L115-L121)：等消费者撤掉 `valid`（确认收到）→ 释放认领、回 IDLE

写请求(WRITE_WAITING/RELAYING)对称，方向相反、无返回数据。

### 限流如何发生（trace 排队的根源）

数据 controller 只有 4 通道，8 个 LSU 同发请求时前 4 个被接走，**后 4 个的 valid 挂着没人理** → 卡在各自 LSU 的 WAITING，直到某通道走完一轮空出来。

### 两个细节

1. **固定优先级**：IDLE 扫描从 j=0 起 + `break`，低编号消费者优先（非公平轮询，简化）。
2. **`channel_serving_consumer` 用阻塞 `=`**（[controller.sv:74,84,117,124](../src/controller.sv#L74)）：一拍内多通道并发认领，通道 i 的认领须立刻对 i+1 可见，否则抢同一请求。同 dispatch 计数器的道理。

## 六、core.sv（装配图：1 套控制平面 + N 套数据平面 = SIMD）

**全是结构化例化，没有一个 `always` 块**。读它的关键是看清资源共享 vs 每线程一份。

### 资源分两类

| | 数量 | 模块 | 角色 |
|---|---|---|---|
| 共享（每核 1 份） | 1 | fetcher、decoder、scheduler | 控制平面：取指/译码/调度 |
| 每线程一份 | 4 | ALU、LSU、registers、PC | 数据平面：各算各的数据 |

这就是 SIMD 的硬件定义：一条指令译码一次（共享 decoder），控制信号 `decoded_*` 同时广播给 4 个线程的所有单元；4 个线程各有独立的 `rs[i]/rt[i]/寄存器/alu_out[i]`（[core.sv:50-54](../src/core.sv#L50-L54) 全是数组）。差异仅来自各自寄存器数据，尤其只读的 `%threadIdx`（按 `THREAD_ID(i)` 注入，[core.sv:173](../src/core.sv#L173)）。

> A100 类比：1 个 scheduler = sub-partition 的单 warp scheduler；4 个 per-thread ALU = SIMD lanes。

### enable 屏蔽多余线程

每个 per-thread 单元接 `enable(i < thread_count)`（[core.sv:139,152,178,200](../src/core.sv#L139)）。这是 dispatch 传来的 `thread_count` 的落点：不满的 block 里多余线程的 ALU/LSU/PC/寄存器全部 `enable=0`，不工作。闭环了"分发器算零头 → core 屏蔽线程"。

### 共享 PC 与线程收敛假设

- `current_pc` 单份共享（[core.sv:48](../src/core.sv#L48)）—— 整 block 共用一个当前 PC
- `next_pc[i]` 每线程一份（[core.sv:49](../src/core.sv#L49)）—— 各线程 PC 单元各算各的

硬件**有**每线程 next_pc 计算能力，但**只有一个 current_pc**，所以假定所有线程收敛到同一 PC → 这就是"无 branch divergence"的结构体现。支持 divergence 需让 current_pc 也每线程一份 + 掩码调度。

### 大脑是 scheduler

`core_state`（6 阶段状态机）声明在 core，但由 scheduler 驱动（[core.sv:113-129](../src/core.sv#L113-L129)）。它据 `fetcher_state/lsu_state/decoded_ret` 决定核走到哪个阶段、何时推进 PC、何时报 done。所有子单元都看 `core_state` 行事 → 阶段 3 的第一个钻研对象。

## 阶段 2 进度

- [x] gpu.sv（顶层接线）
- [x] dcr.sv
- [x] dispatch.sv
- [x] controller.sv（内存控制器：8 LSU 抢 4 通道的仲裁限流）
- [x] core.sv（核内例化 scheduler + 每线程 ALU/LSU/PC/寄存器）

**阶段 2 通关。** 下一步进入阶段 3，从 [scheduler.sv](../src/scheduler.sv) 钻 6 阶段状态机。

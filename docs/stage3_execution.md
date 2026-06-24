# 阶段 3：钻进 Core 内部的状态机

> 核内是整个项目的精髓。本笔记覆盖 scheduler / fetcher / registers / alu，其中 **ALU 的 NZP bug 发现与修复**是阶段 3 最有价值的内容。

## 一、scheduler.sv（核的大脑：8 状态机）

[scheduler.sv](../src/scheduler.sv) 驱动 trace 里的 `Core State`，管一个 block 从启动到结束。

8 状态：`IDLE → FETCH → DECODE → REQUEST → WAIT → EXECUTE → UPDATE →(回 FETCH 或 DONE)`。6 个执行阶段处理**一条指令**，走完回 FETCH 取下一条。

**两类状态**：
- 同步（1 拍过）：DECODE、REQUEST、EXECUTE、UPDATE
- 阻塞（等异步）：**FETCH**（等 `fetcher_state==FETCHED`）、**WAIT**（等所有 LSU 完成）

**WAIT 是核心机制**（[scheduler.sv:77-92](../src/scheduler.sv#L77-L92)）：只要任一线程 LSU 还在 REQUESTING/WAITING，整核卡住 → SIMD 锁步代价。非访存指令 1 拍通过；LDR/STR 才真正等待。controller 的 4 通道限流，最终反映为 scheduler 在 WAIT 多停留。

**UPDATE 的收敛假设**（[scheduler.sv:104](../src/scheduler.sv#L104)）：`current_pc <= next_pc[THREADS_PER_BLOCK-1]` —— 只取最后一个线程的 next_pc，假设所有线程收敛。这是"无 branch divergence"的字面证据。遇 `RET` 则 `done<=1` → 回 dispatch 回收。

**178 拍 vs 理论 78 拍**的差 = FETCH/WAIT 的访存等待，正是 cache/pipelining 要优化的靶子。

## 二、fetcher.sv（取指：3 状态机）

[fetcher.sv](../src/fetcher.sv)：`IDLE → FETCHING → FETCHED`，跟着 `core_state` 跳舞。
- core 进 FETCH → fetcher 向程序内存发起读
- 程序内存返回 → 存 `instruction`，进 FETCHED
- core 看到 FETCHED 才进 DECODE；fetcher 看到 core 进 DECODE 才复位

FETCHED 是"等 scheduler 确认"的握手停顿（同 controller 的 RELAYING 思想）。用全机统一的 valid/ready 协议，与 LSU 取数完全对称。FETCH 多拍等待 → 指令 cache 的优化动机。

## 三、registers.sv（寄存器堆：SIMD 真正发生处）

[registers.sv](../src/registers.sv)：每线程 16 寄存器 = 13 自由(R0-R12) + 3 只读。

**SIMD 命脉**：`%threadIdx` 来自参数 `THREAD_ID(i)`（[core.sv:173](../src/core.sv#L173) generate 下标），每份寄存器堆的 `%threadIdx` 被硬编成不同值 → 同程序处理不同数据。`%blockDim`=THREADS_PER_BLOCK（参数），`%blockIdx`=block_id（运行期）。

两个 core_state 门控动作：
- REQUEST 拍：读 `registers[rs/rt]` → `rs/rt`
- UPDATE 拍：按 `decoded_reg_input_mux`（ALU/LSU/立即数三选一）写回 `registers[rd]`

写保护 `decoded_rd_address < 13` 挡住对只读寄存器的写。`[Bad Solution]` 注释：每拍重写 %blockIdx 是已知浪费。

## 四、alu.sv（算术 + 比较）—— 含一个真实 bug 的发现与修复

[alu.sv](../src/alu.sv) 只在 EXECUTE 拍工作，两模式由 `decoded_alu_output_mux` 选：
- 算术（mux=0）：ADD/SUB/MUL/DIV，由 `decoded_alu_arithmetic_mux` 四选一
- 比较（mux=1）：CMP，输出 NZP 三位

### 🐛 发现的 bug（原始代码）

```verilog
alu_out_reg <= {5'b0, (rs - rt > 0), (rs - rt == 0), (rs - rt < 0)};
```

这里有**双重错位**：

1. **无符号减法**：`rs`/`rt` 是 `reg [7:0]` 无符号。`rs - rt` 回绕，`(rs - rt < 0)` **恒为假** → N 位是死位；`(rs - rt > 0)` 在 `rs<rt` 时也为真（回绕成大正数）→ 实际变成 `(rs != rt)`。
2. **位序反了**：ALU 位序 `{>0, ==0, <0}` 与指令 nzp 字段 `{N,Z,P}` 相反（[format.py:19-21](../test/helpers/format.py#L19-L21) 确认 bit11=N、bit9=P；pc.sv 按下标对齐 AND）。

实测（iverilog）：`rs=1,rt=2`（rs<rt）原始输出 `P=1,N=0` ❌（本该 N=1）。

### 为什么 matmul 仍然 PASS（bug 被掩盖）

- `BRn LOOP` 编码 `decoded_nzp = instruction[11:9] = 100`，测 bit[2]
- 原始 ALU 的 bit[2] = `(rs-rt>0)` = `(rs≠rt)`（无符号下）
- 所以 `BRn` 实际执行"**k ≠ N 就循环**"，而非"k < N"
- matmul 的 `k` 从 0 单调 +1，必然恰好等于 N 退出 → "≠ 退出"≡"< 退出"，结果正确
- 实测 `k=3,N=2` 仍会跳转，暴露真相：它测的是 ≠ 不是 <

三重错位（无符号 bug + 位序反 + 递增计数器）恰好抵消，结果正确。任何需要**真正有符号"小于"**的 kernel 都会暴露此 bug。

### ✅ 修复

```verilog
// Bit order matches instruction nzp field {N,Z,P}: bit2=N, bit1=Z, bit0=P.
// rs/rt unsigned → (rs-rt<0) never true; compare rs and rt directly.
alu_out_reg <= {5'b0, (rs < rt), (rs == rt), (rs > rt)};
```

- bit2=N=`(rs<rt)` 对齐指令 bit2=N ✓
- bit1=Z=`(rs==rt)` ✓
- bit0=P=`(rs>rt)` 对齐指令 bit0=P ✓

修复后实测：`rs=1,rt=2`→N=1；`rs=2,rt=2`→Z=1；`rs=5,rt=2`→P=1，语义全对。`BRn` 现在真正测"小于"。两个 kernel（matadd/matmul）回归测试均 PASS。

### 三个教训

1. **Verilog 有符号/无符号陷阱**：`reg [7:0]` 默认无符号，`(a-b)<0` 静默失效；需要 `$signed()` 或直接比较。
2. **测试通过 ≠ 没 bug**：matmul PASS 只覆盖了它用到的路径，N 位 bug 全程没触发。
3. **实证优先**：靠 iverilog 微测试 + 真实 trace 才看清"bug 存在但被掩盖"的全貌，而非停在推理。

## 五、lsu.sv（异步访存状态机，阶段 3 压轴）

[lsu.sv](../src/lsu.sv)：每线程私有，正是 scheduler `WAIT` 死等的对象。4 状态：`IDLE → REQUESTING → WAITING → DONE`，读(LDR)/写(STR)两段对称 case。

### 读路径逐拍（与 core_state 咬合）

| LSU 状态 | 触发 | 动作 |
|---|---|---|
| IDLE | `core_state==REQUEST` | 进 REQUESTING |
| REQUESTING | 下一拍 | 发请求 `mem_read_valid<=1`, `mem_read_address<=rs` → WAITING |
| WAITING | 等 `mem_read_ready` | 收数据 `lsu_out<=mem_read_data` → DONE |
| DONE | `core_state==UPDATE` | 复位回 IDLE |

`rs`=地址、`rt`=数据（写时），落实阶段 1 的 `LDR rd,rs`=`rd=mem[rs]` 和 `STR rs,rt`=`mem[rs]=rt`。

### 与 scheduler WAIT 咬合

scheduler 在 WAIT 轮询 `lsu_state`，只要任一 LSU 在 REQUESTING/WAITING 就卡住。LSU 卡多久 scheduler 卡多久。非访存指令 LSU 恒 IDLE，WAIT 1 拍过。

### req/ack 握手（反直觉点）

等读数据时等的是 `mem_read_ready` 而非 valid——因为这不是 AXI 数据流握手，而是请求/响应：`valid`=发起方"我有请求"，`ready`=响应方"处理完了"。读写共用此语义，详见 [stage2 文档握手小节](stage2_architecture.md)。

### 完整访存流水线（端到端闭环 LDR）

```
① REQUEST 拍   registers 读 registers[rs] → rs（地址）
② REQUEST→WAIT  LSU IDLE→REQUESTING→WAITING, 发 mem_read_valid
③ WAIT 中       controller: 8 LSU 抢 4 通道, 接走请求→访问外部内存
④ WAIT 中       外部内存返回 → controller 回 ready+data → LSU
⑤ WAIT 末       LSU WAITING→DONE, lsu_out <= mem_read_data
⑥ scheduler     所有 LSU DONE → 离开 WAIT 进 EXECUTE
⑦ UPDATE 拍     registers: reg_input_mux=MEMORY, registers[rd] <= lsu_out
```

**因果链闭环**：controller 的 4 通道限流 → 第③④步多耗拍 → LSU 在 WAITING 多停 → scheduler 在 WAIT 多停 → kernel 总拍数变多（matadd 178 拍）。这就是"访存延迟是 GPU 核心瓶颈"在代码里的完整体现。

小瑕疵（不影响功能）：[lsu.sv:16](../src/lsu.sv#L16) 拼写 `Sgiansl`；[lsu.sv:84](../src/lsu.sv#L84) 写路径注释复制粘贴成 `Only read when...`。

## 六、pc.sv（程序计数器：两阶段时序 + NZP 跨指令存活）

[pc.sv](../src/pc.sv) 同一 always 块里两个 `if`，被不同 core_state 门控（互斥）：

| 阶段 | 动作 | 代码 |
|---|---|---|
| EXECUTE(101) | **算** next_pc（用当前 nzp 判分支） | [pc.sv:42-55](../src/pc.sv#L42-L55) |
| UPDATE(110) | **存** nzp（把 ALU 比较结果锁进寄存器） | [pc.sv:57-65](../src/pc.sv#L57-L65) |

### 为何分两阶段（数据依赖逼出来的）

- **存 nzp 必须在 UPDATE**：nzp 来自 `alu_out`，而 ALU 只在 EXECUTE 才算出 → 只能在之后的 UPDATE 存。
- **算 next_pc 必须在 EXECUTE**：scheduler 在 UPDATE 消费 next_pc（[scheduler.sv:104](../src/scheduler.sv#L104)）→ 必须在之前的 EXECUTE 备好。

生产-消费时序把两件事钉在相邻的 EXECUTE/UPDATE 两拍。

### 关键洞察：nzp 跨指令存活

算 next_pc 时读的 nzp，是**更早某条指令**存的，绝不是本条。因为本条的 nzp 要到 UPDATE 才存，而读它的 next_pc 计算在更早的 EXECUTE 已跑完。所以 CMP 和 BRnzp 必须是**两条独立指令、CMP 在前**：

```
指令 N   = CMP R9, R2
  EXECUTE: ALU 算 alu_out=比较结果; pc 算 next_pc=PC+1
  UPDATE : pc 存 nzp <= alu_out  ──┐ 新结果落进寄存器
指令 N+1 = BRn LOOP                 │ nzp 跨指令存活
  EXECUTE: pc 读 nzp 算 next_pc ────┘ 用的就是上面这个值
           (nzp & decoded_nzp)!=0 ? 跳LOOP : PC+1
  UPDATE : scheduler current_pc <= next_pc
```

nzp 被 CMP 的 UPDATE 写、被后续 BRnzp 的 EXECUTE 读，中间隔几条指令都行（只要没新 CMP 覆盖）。

### next_pc 三出路

```
pc_mux==0           → next_pc = current_pc + 1
pc_mux==1 & 命中     → next_pc = decoded_immediate（跳转）
pc_mux==1 & 未命中   → next_pc = current_pc + 1
```
命中判断 `(nzp & decoded_nzp)!=0` 即阶段 1 的独热掩码集合判断。

### 无 divergence 的 pc 侧证据

每线程都有自己的 PC 单元算 `next_pc[i]`，但 scheduler 只取最后一个喂单一 current_pc——**算 4 份只用 1 份**。ALU 侧产生分歧、pc 侧丢弃分歧。

## 阶段 3 进度（通关）

- [x] scheduler.sv（6 阶段状态机、WAIT 异步、收敛假设）
- [x] fetcher.sv（取指 3 状态机）
- [x] decoder.sv（阶段 1 已精读）
- [x] registers.sv（寄存器堆 + SIMD 注入）
- [x] alu.sv（算术 + 比较；**发现并修复 NZP bug**）
- [x] lsu.sv（异步访存状态机，端到端流水线闭环）
- [x] pc.sv（两阶段时序 + NZP 跨指令存活）

**15 个源文件全部读完。** 整机数据通路、控制流、访存流水线、SIMD 机制全部打通。

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

## 阶段 3 进度

- [x] scheduler.sv（6 阶段状态机、WAIT 异步、收敛假设）
- [x] fetcher.sv（取指 3 状态机）
- [x] decoder.sv（阶段 1 已精读）
- [x] registers.sv（寄存器堆 + SIMD 注入）
- [x] alu.sv（算术 + 比较；**发现并修复 NZP bug**）
- [ ] pc.sv（next_pc 计算 + NZP 存储；阶段 1 已部分覆盖）
- [ ] lsu.sv（异步访存状态机，阶段 3 压轴，闭环 scheduler 的 WAIT）

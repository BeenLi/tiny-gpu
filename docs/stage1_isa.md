# 阶段 1：ISA 精读笔记

> ISA 是"汇编 kernel"与"硬件控制信号"之间的桥梁。精读它最好的方式是对照译码器源码 [src/decoder.sv](../src/decoder.sv) 看每一位如何被硬件解读。

## 一、16 位指令格式

每条指令固定 16 位（字段切分见 [decoder.sv:66-70](../src/decoder.sv#L66-L70)）：

```
 位:  15 14 13 12 | 11 10 9 8 | 7 6 5 4 | 3 2 1 0
      └─ opcode ─┘ └── rd ──┘ └─ rs ─┘ └─ rt ─┘
                   └ nzp ┘[11:9]
                   └──── immediate ────┘[7:0]
```

| 字段 | 位 | 用途 |
|---|---|---|
| `opcode` | [15:12] | 操作码 |
| `rd` | [11:8] | 目标寄存器 |
| `rs` | [7:4] | 源寄存器 1 |
| `rt` | [3:0] | 源寄存器 2 |
| `immediate` | [7:0] | 8 位立即数（CONST/BRnzp 复用 rs+rt 这 8 位） |
| `nzp` | [11:9] | 分支条件（BRnzp 复用 rd 高 3 位） |

**字段复用**：同样的位在不同指令里含义不同。硬件无条件把所有切法都译出（[decoder.sv:66-70](../src/decoder.sv#L66-L70)），再由 opcode 的 case 决定用哪些。

## 二、寄存器编码（4 位 → 16 个）

来自 [format.py:4-12](../test/helpers/format.py#L4-L12)：

| 编号 | 名字 | 读写 |
|---|---|---|
| 0-12 | `R0`–`R12` | 可读写（13 个自由寄存器） |
| 13 | `%blockIdx` | 只读 |
| 14 | `%blockDim` | 只读 |
| 15 | `%threadIdx` | 只读 |

最后 3 个只读寄存器是 SIMD 命脉——硬件在每个线程自动填入不同 `%threadIdx`，让同一份程序处理不同数据。

## 三、亲手译码示例

`MUL R0, %blockIdx, %blockDim` = `0b0101000011011110`：
```
0101 | 0000 | 1101 | 1110
MUL    R0   13=%blkIdx 14=%blkDim
```

`CONST R2, #8` = `0b1001001000001000`：
```
1001 | 0010 | 0000 1000
CONST  R2    imm=8
```

`STR R7, R6` = `0b1000000001110110`：
```
1000 | 0000 | 0111 | 0110
STR   (空)   7=R7   6=R6      → mem[R7] = R6（地址在前，数据在后）
```

## 四、11 条指令 → 控制信号

核心开关：
- `reg_write_enable`：是否写回 rd
- `reg_input_mux`：写回值来源 —— `00`=ALU、`01`=内存(LSU)、`10`=立即数
- `alu_arithmetic_mux`：`00`=加 `01`=减 `10`=乘 `11`=除
- `alu_output_mux`：`0`=算术结果、`1`=比较结果(给 NZP)
- `mem_read_enable` / `mem_write_enable`：访存
- `nzp_write_enable`：写 NZP
- `pc_mux`：`0`=PC+1、`1`=跳转
- `ret`：线程结束

对照表（[decoder.sv:84-130](../src/decoder.sv#L84-L130)）：

| 指令 | 干什么 | 关键信号 |
|---|---|---|
| `NOP` | 空操作 | 无 |
| `ADD/SUB/MUL/DIV` | rd = rs ⊕ rt | `reg_write`, `input_mux=00`, `arith_mux=00/01/10/11` |
| `LDR rd, rs` | rd = mem[rs] | `reg_write`, `input_mux=01`, `mem_read` |
| `STR rs, rt` | mem[rs] = rt | `mem_write` |
| `CONST rd, #imm` | rd = imm | `reg_write`, `input_mux=10` |
| `CMP rs, rt` | 比较存 NZP | `alu_output_mux=1`, `nzp_write` |
| `BRnzp #imm` | 条件跳转 | `pc_mux=1` |
| `RET` | 线程结束 | `ret` |

## 五、唯一的控制流机制：CMP + BRnzp

tiny-gpu 没有专门比较器，循环/分支全靠这对组合：

1. **`CMP rs, rt`**：ALU 算 `rs - rt`，但 `alu_output_mux=1` 让它输出**符号**（3 位 `N/Z/P`：负/零/正），写进 PC 单元的 NZP 寄存器。
2. **`BRnzp #imm`**：指令 [11:9] 带 nzp 掩码，若与上次 CMP 的 NZP 匹配，`pc_mux=1` 跳到 imm，否则 PC+1。

### NZP 匹配为何用 `(nzp & decoded_nzp) != 0` 而非 `==`

见 [pc.sv:44](../src/pc.sv#L44)。关键：**NZP 是独热编码**。

- `nzp`（CMP 存的实际结果）：永远只有 1 位为 1 —— `100`=负、`010`=零、`001`=正
- `decoded_nzp`（分支的条件掩码）：可多位为 1 —— `BRzp`=`011`（>=）、`BRnp`=`101`（!=）、`BRnzp`=`111`（无条件）

分支语义是"实际结果是否**属于**允许集合"，即集合成员判断 = 按位与非零。用 `==` 会让多条件分支（>=、!= 等）失效：
```
实际=相等 010, 掩码 BRzp=011
按位与：010 & 011 = 010 ≠ 0 → 跳转 ✓
相等：  010 == 011 ? 否       → 不跳 ✗（错）
```

matmul 中 `CMP R9, R2` + `BRn LOOP`（掩码 `100`）：k<N 时 nzp=100 → `100&100≠0` → 跳回 LOOP；k==N 时 nzp=010 → `010&100=0` → 退出。

### 重要简化：单 PC + 线程收敛

tiny-gpu 假设一个 block 内所有线程每条指令后收敛到同一 PC（trace 里每个 Core 只有一个 `current_pc`），因此**无法处理真正的 branch divergence**。matmul 能跑只因其分支恰好所有线程同进同出。

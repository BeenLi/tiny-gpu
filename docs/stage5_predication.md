# 阶段 5 理论专题：Predication（谓词化）

> 配套 [stage5_branch_divergence_theory.md](stage5_branch_divergence_theory.md)。分叉/重收敛是"遇到分支后硬件怎么办"；predication 是"干脆不生成分支"——它和 active mask 是同一枚硬币的两面，是理解 GPU 控制流的另一半。

---

## 1. 一句话定义

**Predication（谓词化）：把控制依赖变成数据依赖。** 不再用分支跳过某条指令，而是**总是执行**这条指令，但给它挂一个**谓词（predicate，1 位布尔守卫）**——谓词为真才提交结果（写回寄存器/内存），为假则**作废（nullify）**，等价于一条 no-op。

形式上（PTX 风格，`@p` 是守卫谓词）：

```
@p  add r1, r2, r3   // 谓词 p 为真才执行写回；否则这条指令对该 lane 是 no-op
```

把 `if/else` 改写成"无分支的直线代码 + 谓词"的编译变换，叫 **if-conversion（分支转换）**。

## 2. 核心机制

三个要件：

1. **谓词寄存器**：每个线程一组 1 位寄存器。比较指令（PTX 的 `setp`）写谓词；tiny-gpu 的 `CMP` 写 NZP 已是其雏形。
2. **守卫字段**：指令里带一个谓词号（及取反位 `@!p`），指明"由哪个谓词守卫"。
3. **Nullification（作废）**：被守卫关闭的指令**照样流过流水线**（占一个发射槽），但**抑制一切副作用**——不写寄存器、不访存、不抛异常。这与 tiny-gpu 里 `enable=0` 让一条 lane"全状态冻结"是**完全相同**的硬件动作。

## 3. SIMT 里：predication 与 active mask 是同一枚硬币

在 SIMD/SIMT 数据通路上，让结果正确的唯一手段就是**按 lane 屏蔽写回**。predication 和 divergence 的 active mask 做的是同一件事，区别只在**谁来决定 mask、指令流里有没有分支**：

| | 谁算 mask | 指令流 | 两条路径 |
|---|---|---|---|
| **真分支 + 重收敛** | 硬件（按 PC 是否相等，运行时）| 含分支，一次只取一条路径的指令 | 串行，靠重收敛收回满 mask |
| **predication / if-conversion** | 编译器（按谓词，编译时）| 无分支，两路径直线内联成一条流 | 全取、全发射，按 lane 谓词决定写回 |

所以一个干净的认知：**SIMT 的 active mask 就是硬件自动施加的谓词化**。predication 把"该不该写回"的决定，从运行时的控制流，搬到了编译时的数据（谓词位）。

真实硬件里两者还会**叠加**：一条 lane 最终是否写回 = `active_mask[lane] AND guard_predicate[lane]`。

## 4. 同一个 if/else：真分支版 vs 谓词化版

源代码：
```c
if (a < b) c = a + b;
else       c = a - b;
```

**真分支版（tiny-gpu 现在的做法，CMP + BRnzp + 重收敛）：**
```
        CMP   a, b            ; 设 NZP
        BRn   ELSE            ; a<b 跳走
        ADD   c, a, b         ; then
        BRnzp END             ; 跳过 else
ELSE:   SUB   c, a, b         ; else
END:    ...                   ; 重收敛点
```
→ 一份 PC、有跳转；分叉时 then/else 段被**串行**执行，靠 min-PC 在 END 处重收敛。

**谓词化版（if-conversion，无分支直线流）：**
```
        setp.lt  p, a, b      ; p = (a < b)
@p      add      c, a, b      ; p 为真的 lane: c=a+b
@!p     sub      c, a, b      ; p 为假的 lane: c=a-b
```
→ 没有任何跳转；`add` 和 `sub` 都被发射；每条 lane 按自己的 `p` 只提交一条。**没有 PC 分叉，自然不需要重收敛栈/min-PC。**

## 5. 代价权衡：什么时候用谓词、什么时候用真分支

谓词化把"控制开销"换成了"全员都算两边"：

- **谓词化的代价**：所有 lane **都执行两条路径的指令**，谓词为假的 lane 白算。固定 `2N` 条指令（N=每边指令数），与是否发散无关。
- **真分支的代价**：
  - warp **一致**（所有 lane 走同一边）→ 只执行 `N`，另一边整段跳过——**真分支大赢**。
  - warp **发散** → 两边串行，约 `2N` warp-指令 + 跳转/重收敛开销——此时谓词化反而更省（省掉了控制机器）。

结论（也是编译器 `ptxas`/`nvcc` 的启发式）：

> **分支体短 / 大概率发散 → 谓词化；分支体长 / 大概率一致 → 真分支。** 编译器按分支体的指令数设一个阈值，短的自动 if-convert，长的留作真分支。

这解释了 GPU 编程里一条经验：小的 `if` 不必怕——编译器谓词化掉了，几乎零控制开销；要警惕的是 warp 内**长且发散**的分支。

## 6. 真实 ISA 里的谓词化

谓词化最早在 **CPU** 上为对抗分支预测失败而生，GPU 把它发挥到极致：

- **Itanium (IA-64)**：**全谓词**架构，几乎每条指令都带 64 个谓词寄存器之一作守卫——谓词化的教科书。
- **ARM**：经典 A32 模式**每条指令条件执行**——最高 4 位是条件域（`EQ/NE/GT/LT/…`），故 `ADDEQ`=相等才加、`MOVNE`=不等才传；条件查 CPSR 标志（由 `CMP` 或带 `S` 后缀的算术置位）。教科书例子 GCD（整段 if/else 零跳转）：

  ```arm
  ; gcd(a,b): a in r0, b in r1
  gcd:    CMP   r0, r1        ; 比较 → 置标志
          SUBGT r0, r0, r1    ; if a>b : a -= b   （无 S，不改标志）
          SUBLT r1, r1, r0    ; if a<b : b -= a   （仍看同一次 CMP 的标志）
          BNE   gcd           ; a!=b 继续
  ```
  关键细节：`SUBGT/SUBLT` **不带 `S`** 故不覆盖标志，才能"一次比较驱动多条互斥条件指令"。AArch64 取消了通用条件执行，收敛为条件选择 `CSEL`/`CSINC`/`CCMP`；Thumb-2 用 `IT`(If-Then) 块守卫随后最多 4 条指令。A32 是"每条标量指令都带谓词"的最彻底形态——和 GPU 把谓词放进每条 lane 是同一回事。
- **x86**：有限谓词化 `CMOV`（条件传送）。
- **NVIDIA PTX/SASS**：几乎每条指令都可带守卫谓词 `@p`/`@!p`；`setp` 写谓词；`ptxas` 对短分支自动 if-conversion。
- **AMD GCN/RDNA**：把 mask 显式化为 **`EXEC` 寄存器**（每 wavefront 一个、按 lane 的活跃掩码）。发散控制流由编译器**软件管理**：用向量比较把条件写进 `VCC`/SGPR，再 AND 进 `EXEC`，分支前后保存/恢复 `EXEC`。AMD 这套把"predication"和"divergence mask"**统一成对同一个 EXEC 寄存器的操作**——是理解"两者本是一回事"的最佳实例。

## 7. 映射回 tiny-gpu：现状与一个自然的扩展练习

**现状：tiny-gpu 没有谓词化。** 它的 ISA 没有 `@p` 守卫字段，每条指令的写回只受 divergence 的 `active_mask`（即 `enable`）门控。所以 `if/else` 全靠真分支 + min-PC 重收敛（我们第一阶段实现的）。

**如果要加谓词化（可选的 ISA 扩展练习）：**

1. 给每线程加 1 个谓词位寄存器（甚至直接复用现有 NZP——`CMP` 已经在写它）。
2. 在指令编码里挪出 1~2 位作守卫字段（谓词号 + 取反位）。
3. 把每条 lane 的写回条件改成 `active_mask[i] && guard[i]`（`guard` 由谓词位经 decoder 给出）。
4. 写一个"谓词化 ReLU" kernel：用 `CMP` 设位，再用守卫的 `STR`/`ADD` 直线写回，**不带任何 BRnzp**。

这样短 if/else 就能走"无分支直线流"，省掉 min-PC 的路径串行——正好和第一阶段的真分支版形成对照实验（同一 kernel，数 cycle、比 SIMT 利用率）。

## 8. 与 branch divergence 的关系（一表收束）

| 维度 | Branch divergence（真分支） | Predication（谓词化）|
|---|---|---|
| 指令流 | 含跳转，按 PC 取一条路径 | 无跳转，两路径直线内联 |
| mask 来源 | 硬件运行时（PC 匹配）| 编译器编译时（谓词）|
| 不取的那条路径 | 一致时可整段跳过 | 永远也要发射（白算）|
| 需要重收敛机器 | 需要（栈 / min-PC）| 不需要 |
| 适合 | 长体、一致分支 | 短体、易发散分支 |
| 本质 | 都靠**按 lane 屏蔽写回**得到 SIMD 上的正确结果 | 同左 |

---

### 一页速记

- predication = 不跳转，指令照发，**按谓词决定是否写回**；把控制依赖变数据依赖。
- if-conversion = 编译器把 `if/else` 改写成"两路径直线 + 守卫谓词"。
- SIMT 里 **active mask 就是硬件自动谓词化**；最终写回 = `active_mask AND guard`。
- 权衡：**短/易发散用谓词，长/一致用真分支**；编译器按分支体长度自动选。
- 实例：Itanium 全谓词、ARM 条件执行、x86 CMOV、PTX `@p`、AMD `EXEC` 掩码。
- tiny-gpu 暂无谓词化（全靠真分支 + min-PC）；加一个守卫字段即可做对照实验。

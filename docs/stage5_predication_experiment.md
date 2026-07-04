# 阶段 5(一·补):谓词化 ReLU 对照实验 —— PSTR 实测

> 把 [Predication 专题](stage5_predication.md) 的理论落回硬件:给 tiny-gpu 加一条**显式谓词指令 `PSTR`**,把阈值 ReLU 写成**无分支**版本,和 [Branch Divergence](stage5_branch_divergence.md) 的真分支版逐拍对比。
>
> 分支 `stage5-predication`,提交 `4b99055`。日期 2026-07-04。

---

## 1. 目标与假设

分支版 `test_relu` 用 `CMP + BRn/BRnzp` 让线程分叉,靠 min-PC 把两条路径**串行化**,实测 **198 cycles**。

本实验加一条**谓词化存储 `PSTR`**,让所有 lane **直线**跑完、不跳转,两条条件存储都执行、但每条只在匹配 NZP 的 lane 真正写内存。假设:**消除分支 + 消除分叉串行的控制开销 → 更少周期**;但也要用实测暴露谓词化的反向权衡(收敛块反而更慢)。

## 2. 设计:`PSTR`(predicated store)

### 2.1 指令(零破坏,复用空闲编码 + 现成判据)

新增操作码 `PSTR = 4'b1010`(原空闲)。**复用 STR 未用的 `rd` 字段 `[11:9]` 放谓词条件**,判据与 BRnzp 完全同构:

```
[15:12]=1010(PSTR)  [11:9]=nzp_cond  [8]=0  [7:4]=rs(addr)  [3:0]=rt(data)
语义:  if ((本线程 nzp & nzp_cond) != 0)  mem[R[rs]] <= R[rt]
```

`nzp` 是 CMP 设的每线程 N/Z/P 寄存器(N=rs<rt, Z==, P=>)。BRnzp 用 `(nzp & cond)!=0` 决定「跳不跳」,PSTR 用同一判据决定「写不写内存」—— 把控制相关变成一次条件提交,**无 PC 分叉**。

### 2.2 无分支 ReLU kernel(13 条,两条 PSTR)

```
 8  CMP  R4, R3            ; N=(X<T), Z=(X==T), P=(X>T)
 9  CONST R6, #0
10  PSTR.!n R5, R4   cond=011(Z|P) ; X>=T 的 lane: Y[i]=X
11  PSTR.n  R5, R6   cond=100(N)   ; X<T  的 lane: Y[i]=0
12  RET
```

对比分支版(15 条):第 9 条 `BRn #12`、第 11 条 `BRnzp #14` 两条跳转被消掉,ReLU 从 15 → 13 条,且全程 lockstep(无分叉)。

### 2.3 RTL 改动(4 文件,纯新增)

| 文件 | 改动 |
|---|---|
| `decoder.sv` | 译码 `PSTR`;新增 `decoded_mem_pred_enable`(条件已由现成的 `decoded_nzp<=instr[11:9]` 提供) |
| `pc.sv` | 把每线程内部 `reg [2:0] nzp` 暴露成 `output`(引出谓词源) |
| `lsu.sv` | 写路径加守卫:`if (decoded_mem_write_enable && mem_write_predicate_ok)`;守卫为 0 时 LSU 停在 IDLE —— 不发请求、不占 WAIT、不写内存 |
| `core.sv` | 连线:`pc.nzp -> thread_nzp[i]`;每个 LSU 的 `mem_write_predicate_ok` 接 `(!decoded_mem_pred_enable) \|\| ((thread_nzp[i] & decoded_nzp)!=0)` |

**关键不变量:只守卫「写内存」这一步,绝不动 lane 的 `enable`。** 被谓词 squash 的 lane 仍照常推进 PC、和其他 lane 保持 lockstep(否则 PC 会错位、min-PC 会误判)。plain STR 走 `pred_enable=0 → 守卫恒 1`,旧路径逐拍不变。

## 3. 实测结果

| 测试 | 周期 | 状态 | 说明 |
|---|---|---|---|
| `test_relu`(真分支) | 198 | PASS | min-PC 串行两条路径 |
| **`test_relu_pred`(谓词化)** | **176** | **PASS** | 无分支直线,两条 PSTR 条件提交 |
| `test_matadd`(回归) | 178 | PASS | 与改造前一致 |
| `test_matmul`(回归) | 491 | PASS | 与改造前一致 |
| `test_divloop`(回归) | 418 | PASS | 与改造前一致 |

**谓词化 ReLU 省 22 拍(198 → 176,~11%)。回归零偏移** —— 现有 4 个测试周期数与改造前逐一相同,证明 PSTR 是纯新增、不碰旧数据通路。

## 4. 分析:22 拍从哪来,又为什么不砍半

两个版本**都**要付两次内存往返:本数据集 X=[2,5,3,7,1,6,0,4]、T=4 让**每个 block 都真分叉**(既有 X>=T 的 lane 要存 X、又有 X<T 的 lane 要存 0)。所以:

- 分支版:分叉后 min-PC 先跑 else 段(存 X)、再跑 then 段(存 0),两次内存往返 + **两条跳转指令 + 分叉/重收敛的多趟 mask 推进**。
- 谓词版:两条 PSTR 在同一趟 lockstep 里各发一次内存往返,**没有跳转、没有分叉多趟**。

→ **省下的 22 拍 = 被消除的分支指令槽 + 分叉串行的控制开销;两次内存往返的代价没省**(两版都要付)。这解释了为什么是「省一截」而非「砍半」。

### 关键洞察:这份省是「捡来的」,换个数据就翻盘

- 本例每个 block 都分叉 → 两条存储**反正都要跑**,谓词化只是顺手抹掉了控制开销 → **纯赚**。
- 若某 block **收敛**(全 X>=T 或全 X<T):真分支只跑一条路径、**整段跳过另一条**,只付**一次**内存往返;谓词版却被迫两条 PSTR 都跑 → **两次**往返 → **谓词化反而更慢**。

这正是理论篇讲的短分支/谓词 vs 长分支/收敛的权衡,现在能在自己的硬件上用周期数验证:**谓词化把「控制流代价」换成「无条件多算」;当那份多算本来就要付(分叉块)时它赢,当本可跳过(收敛块)时它输。** GPU 编译器的 if-conversion 启发式,判的就是这笔账。

## 5. 映射回真实 GPU

这条 PSTR 补上了 tiny-gpu 缺的**第二层谓词**(见 [Predication 专题](stage5_predication.md) §"真实 GPU 的两层谓词"):

| 层 | 真实 GPU | tiny-gpu |
|---|---|---|
| active mask 底座 | AMD `EXEC` / NVIDIA active mask | `active_mask`(阶段一 min-PC) |
| **显式 per-instruction 谓词** | NVIDIA PTX `@p`/SASS 谓词域、Intel flag | **`PSTR` 的 `[11:9]` 条件字段(本实验)** |

PSTR ≈ NVIDIA 的 `@p st`(带谓词的存储)。我们只谓词化了 store;真实 GPU 的谓词覆盖几乎所有指令(`@p add` 等),由编译器 if-conversion 自动生成。tiny-gpu 无编译器,故手写汇编直接填 `[11:9]` 条件位。

## 6. 局限与后续

- **仅 store 谓词化**:未做 predicated 寄存器写(`@p CONST/ADD`)。原因是 16 位 ISA 里 CONST/ADD 已占满,加 3 位条件域需重新编码;STR 的 `rd` 字段恰好空闲,故先做 PSTR(YAGNI)。
- **谓词源固定为 NZP**:没有独立谓词寄存器,复用 CMP 的 N/Z/P。够用且省硬件。
- **没有 if-conversion 自动化**:条件位靠手写。真实价值在编译器把短分支自动谓词化。
- 可选延伸:①加一组**收敛**输入数据,实测「谓词化反而更慢」的反向权衡(第 4 节的预测);②加 predicated 寄存器写做更通用的 `@p` 演示。

---

### 一页速记

- `PSTR`(opcode 1010):谓词化存储,复用 BRnzp 的 `(nzp & cond)!=0`,`[11:9]` 放条件。
- 无分支 ReLU:两条 PSTR 直线跑,每条条件提交 → 无 PC 分叉、无 min-PC 串行。
- 实测 176 vs 198,**省 22 拍**;回归零偏移。
- 省的是控制开销;两次内存往返两版都付 → 分叉块纯赚、收敛块会亏。
- PSTR = tiny-gpu 的第二层谓词,对应 NVIDIA `@p st`。

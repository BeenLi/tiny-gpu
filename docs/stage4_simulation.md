# 阶段 4：仿真测试框架

> 回答三个问题：汇编 kernel 怎么变成硬件输入、结果怎么读出来、"外部内存"是谁在模拟。理解这些就看懂了整个 cocotb 仿真闭环。

## 一、Makefile：三步编译流水线

[Makefile](../Makefile) 把 `make test_matadd` 拆成编译 + 运行：

```
SystemVerilog ──sv2v──► Verilog ──iverilog──► sim.vvp ──cocotb/vvp──► 运行+trace
   src/*.sv            build/gpu.v          可执行仿真      Python 驱动
```

1. `compile`：`sv2v` 把 SystemVerilog 转成 iverilog 能吃的 Verilog 2005
2. `iverilog -s gpu -g2012`：以 gpu 为顶层编译成 `build/sim.vvp`
3. `MODULE=test.test_matadd vvp ...`：cocotb 加载 Python 测试模块驱动仿真

### alu 为什么单独编译

[Makefile:11-14](../Makefile#L11-L14) 先 `compile_alu` 单独转 alu，再追加到 gpu.v 末尾。原因：`sv2v -I src/*` 经 shell 展开成 `sv2v -I src/alu.sv <其余文件>`（alu.sv **字母序第一**），`-I`（include 路径参数）把 alu.sv 吃掉，使它不进 gpu.v。所以才单独编译再拼回。是 glob + `-I` 的副作用，非刻意设计。

> 改 alu.sv 能生效正是走 `compile_alu` 这条单独路径。

## 二、setup.py：硬件"开机流程"

[setup.py](../test/helpers/setup.py) = README 启动 kernel 的 4 步：

| 步骤 | 代码 | 对应硬件 |
|---|---|---|
| 起时钟 | [setup.py:16-17](../test/helpers/setup.py#L16-L17) | 25us 周期时钟 |
| 复位 | [setup.py:20-22](../test/helpers/setup.py#L20-L22) | reset 拉高一拍再放低 |
| 装程序+数据内存 | [setup.py:25-28](../test/helpers/setup.py#L25-L28) | 写进 Memory 模型数组 |
| 设 thread_count | [setup.py:31-34](../test/helpers/setup.py#L31-L34) | 写 DCR |
| 启动 | [setup.py:37](../test/helpers/setup.py#L37) | `start=1` |

接上阶段 2 的 dcr.sv（thread_count 落点）和 dispatch.sv（start 触发）。

## 三、memory.py：模拟"外部异步内存"（最关键）

**DUT 里没有真正的内存，外部内存是 Python 模拟的。** [Memory](../test/helpers/memory.py) 类：

- `self.memory = [0] * 2**addr_bits`（[memory.py:9](../test/helpers/memory.py#L9)）：真正存数据的数组（256 行）
- `run()`（[memory.py:24-69](../test/helpers/memory.py#L24-L69)）：**每个时钟周期被测试循环调用一次**，扮演 req/ack 握手的响应侧

```python
for i in range(self.channels):
    if mem_read_valid[i] == 1:                # DUT 发起读请求
        mem_read_data[i] = self.memory[addr]  # 取数
        mem_read_ready[i] = 1                  # 拉 ready：处理完了
```

两个要点：

1. **内存"快但窄"**：`run()` 同一拍就对每个 valid 拉 ready（~1 拍延迟）。慢**不在内存延迟**，在**通道数（带宽）**——数据内存 `channels=4`、程序内存 `channels=1`，与 controller 通道数严格对应。
2. **限流不在这里，在 controller**：memory.py 只暴露 channels 个通道逐个响应；把 8 个 LSU 节流到 4 通道的是 controller 仲裁。memory.py 提供"窄带宽外部内存"，controller 负责"在窄口排队"，两者合起来产生访存等待。

> `int(str(...value)[i:i+bits], 2)` 位切片（[memory.py:30-33](../test/helpers/memory.py#L30-L33)）是在解包"多通道打包成一根总线"的信号，对应 gpu.sv 顶层铺平 per-channel 数组。

## 四、driver loop + logger：每拍快照

[test_matadd.py:50-58](../test/test_matadd.py#L50-L58)：

```python
while dut.done.value != 1:
    data_memory.run()           # Python 内存模型响应这一拍请求
    program_memory.run()
    await ReadOnly()
    format_cycle(dut, cycles)   # dump DUT 这一拍所有状态进日志
    await RisingEdge(dut.clk)    # 推进时钟
```

[logger.py](../test/helpers/logger.py) 以 debug 级把每拍 `format_cycle` 输出全写进 `test/logs/log_<时间戳>.txt` → 日志上万行（178 拍 × 所有线程）。最后 `data_memory.display()` 打印最终内存 + 断言验证。

## 五、整个仿真闭环

```
       ┌─────────────── Python (cocotb 测试) ───────────────┐
       │  setup: 装内存/设线程/start                          │
       │  每拍: memory.run() 模拟外部内存  ←──req/ack──┐       │
       │        format_cycle() dump 状态 → logger     │       │
       └──────────────────────┬───────────────────────┼──────┘
                              驱动 clk/信号            valid/ready
                               ▼                       │
       ┌─────────────── Verilog DUT (gpu) ─────────────┘
       │  dispatch → core → scheduler → lsu/fetcher → controller
       └────────────────────────────────────────────────────┘
```

Python 扮演"外部世界"（内存 + 控制 + 观测），Verilog 是被测 GPU，两者每拍通过 valid/ready 交互——这就是 cocotb 仿真的本质。

## 阶段 4 小结

- ✅ Makefile：sv2v → iverilog → cocotb 三步；alu 因 glob+`-I` 副作用单独编译
- ✅ setup.py：时钟/复位/装内存/设线程/start = README 启动 4 步
- ✅ **memory.py：外部内存是 Python 模拟的**，快(1拍)但窄(通道数=带宽)，限流在 controller
- ✅ driver loop 每拍 `memory.run()` + `format_cycle()`，logger 写出万行 trace
- ✅ 整个仿真 = Python(外部世界) ↔ Verilog(GPU) 每拍 valid/ready 交互

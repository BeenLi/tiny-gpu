import cocotb
from cocotb.triggers import RisingEdge
from .helpers.setup import setup
from .helpers.memory import Memory
from .helpers.format import format_cycle
from .helpers.logger import logger

@cocotb.test()
async def test_relu_pred(dut):
    # 谓词化(无分支)阈值 ReLU:  Y[i] = (X[i] < T) ? 0 : X[i]
    # 不用 BRnzp 跳转，而用 PSTR (predicated store) 让所有线程直线执行两条条件存储，
    # 每条只在匹配 NZP 的 lane 真正写内存 —— 无 PC 分叉、无 min-PC 串行。
    #   PSTR 格式: [15:12]=1010  [11:9]=nzp_cond  [8]=0  [7:4]=rs(addr)  [3:0]=rt(data)
    #   语义: if ((thread_nzp & nzp_cond) != 0) mem[R[rs]] <= R[rt]
    program_memory = Memory(dut=dut, addr_bits=8, data_bits=16, channels=1, name="program")
    program = [
        0b0101000011011110, # 0  MUL  R0, %blockIdx, %blockDim
        0b0011000000001111, # 1  ADD  R0, R0, %threadIdx      ; i = blockIdx*blockDim + threadIdx
        0b1001000100000000, # 2  CONST R1, #0                 ; baseX = 0
        0b1001001000001000, # 3  CONST R2, #8                 ; baseY = 8
        0b1001001100000100, # 4  CONST R3, #4                 ; T = 4
        0b0011010000010000, # 5  ADD  R4, R1, R0              ; addr X[i] = baseX + i
        0b0111010001000000, # 6  LDR  R4, R4                  ; R4 = X[i]
        0b0011010100100000, # 7  ADD  R5, R2, R0              ; addr Y[i] = baseY + i
        0b0010000001000011, # 8  CMP  R4, R3                  ; N=(X<T), Z=(X==T), P=(X>T)
        0b1001011000000000, # 9  CONST R6, #0                 ; R6 = 0
        0b1010011001010100, # 10 PSTR.!n R5, R4  cond=011(Z|P); if X>=T: Y[i]=X
        0b1010100001010110, # 11 PSTR.n  R5, R6  cond=100(N)  ; if X<T : Y[i]=0
        0b1111000000000000, # 12 RET
    ]

    data_memory = Memory(dut=dut, addr_bits=8, data_bits=8, channels=4, name="data")
    X = [2, 5, 3, 7, 1, 6, 0, 4]   # 跨越阈值 T=4，使每个 block 内线程条件不同
    data = X + [0] * 8             # addr 0-7 = X, addr 8-15 = Y(结果区)

    threads = 8
    await setup(dut=dut, program_memory=program_memory, program=program,
                data_memory=data_memory, data=data, threads=threads)

    data_memory.display(16)

    cycles = 0
    while dut.done.value != 1:
        data_memory.run()
        program_memory.run()
        await cocotb.triggers.ReadOnly()
        format_cycle(dut, cycles)
        await RisingEdge(dut.clk)
        cycles += 1

    logger.info(f"[PRED] Completed in {cycles} cycles (branch-based test_relu baseline = 198)")
    data_memory.display(16)

    T = 4
    expected = [0 if x < T else x for x in X]   # [0,5,0,7,0,6,0,4]
    for i, exp in enumerate(expected):
        result = data_memory.memory[i + 8]
        assert result == exp, f"Y[{i}] mismatch: expected {exp}, got {result}"

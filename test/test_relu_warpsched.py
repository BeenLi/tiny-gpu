import cocotb
from cocotb.triggers import RisingEdge
from .helpers.setup import setup
from .helpers.memory import Memory
from .helpers.format import format_cycle
from .helpers.logger import logger

@cocotb.test()
async def test_relu_warpsched(dut):
    # 阈值 ReLU（baseY=16 变体）：16 线程 -> 4 block -> dispatch 每 core 2 个 warp。
    # 每个 4-线程 warp 内部按 X<T 分叉 -> 同时压 intra-warp divergence + switch-on-stall。
    program_memory = Memory(dut=dut, addr_bits=8, data_bits=16, channels=1, name="program")
    program = [
        0b0101000011011110, # 0  MUL  R0, %blockIdx, %blockDim
        0b0011000000001111, # 1  ADD  R0, R0, %threadIdx      ; i = blockIdx*blockDim + threadIdx
        0b1001000100000000, # 2  CONST R1, #0                 ; baseX = 0
        0b1001001000010000, # 3  CONST R2, #16                ; baseY = 16
        0b1001001100000100, # 4  CONST R3, #4                 ; T = 4
        0b0011010000010000, # 5  ADD  R4, R1, R0              ; addr X[i] = baseX + i
        0b0111010001000000, # 6  LDR  R4, R4                  ; R4 = X[i]
        0b0011010100100000, # 7  ADD  R5, R2, R0              ; addr Y[i] = baseY + i
        0b0010000001000011, # 8  CMP  R4, R3                  ; X[i] vs T  (N = X<T)
        0b0001100000001100, # 9  BRn  #12                     ; if X<T goto ZERO(12)
        0b1000000001010100, # 10 STR  R5, R4                  ; (else) Y[i] = X[i]
        0b0001111000001110, # 11 BRnzp #14                    ; goto END(14) 无条件
        0b1001011000000000, # 12 CONST R6, #0                 ; (ZERO) R6 = 0
        0b1000000001010110, # 13 STR  R5, R6                  ; Y[i] = 0
        0b1111000000000000, # 14 RET
    ]

    data_memory = Memory(dut=dut, addr_bits=8, data_bits=8, channels=4, name="data")
    # 每 4 线程一组（= 一个 warp），组内都横跨阈值 T=4 以强制 warp 内分叉
    X = [2, 5, 3, 7,  1, 6, 0, 4,  3, 8, 2, 5,  6, 1, 7, 0]
    data = X + [0] * 16            # addr 0-15 = X, addr 16-31 = Y(结果区)

    threads = 16
    await setup(dut=dut, program_memory=program_memory, program=program,
                data_memory=data_memory, data=data, threads=threads)

    data_memory.display(32)

    cycles = 0
    while dut.done.value != 1:
        data_memory.run()
        program_memory.run()
        await cocotb.triggers.ReadOnly()
        format_cycle(dut, cycles)
        await RisingEdge(dut.clk)
        cycles += 1

    logger.info(f"Completed in {cycles} cycles")
    data_memory.display(32)

    T = 4
    expected = [0 if x < T else x for x in X]   # [0,5,0,7,0,6,0,4,0,8,0,5,6,0,7,0]
    for i, exp in enumerate(expected):
        result = data_memory.memory[i + 16]
        assert result == exp, f"Y[{i}] mismatch: expected {exp}, got {result}"

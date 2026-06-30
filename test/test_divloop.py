import cocotb
from cocotb.triggers import RisingEdge
from .helpers.setup import setup
from .helpers.memory import Memory
from .helpers.format import format_cycle
from .helpers.logger import logger

@cocotb.test()
async def test_divloop(dut):
    # 每线程循环 %threadIdx 次 (acc+=1)，然后在各自的退出路径存 acc 并 RET。
    # 退出路径(8-10)的 PC 低于循环体(11-13)：先完成的线程在低 PC 处 RET 退休，
    # 其余线程继续在高 PC 循环 -> 不同周期退休(partial done_mask) + min-PC 重收敛。
    # 结果 Y[i] = threadIdx。
    program_memory = Memory(dut=dut, addr_bits=8, data_bits=16, channels=1, name="program")
    program = [
        0b0101000011011110, # 0  MUL  R0, %blockIdx, %blockDim
        0b0011000000001111, # 1  ADD  R0, R0, %threadIdx     ; i
        0b1001000100000001, # 2  CONST R1, #1                ; increment
        0b1001001000001000, # 3  CONST R2, #8                ; baseY = 8
        0b1001001100000000, # 4  CONST R3, #0                ; acc = 0
        0b1001010000000000, # 5  CONST R4, #0                ; k = 0
        0b0010000001001111, # 6  CMP  R4, %threadIdx         ; (LOOP_CHECK) k vs tid
        0b0001100000001011, # 7  BRn  #11                    ; if k<tid goto BODY(11)
        0b0011010100100000, # 8  ADD  R5, R2, R0             ; (EXIT) addr Y[i] = baseY + i
        0b1000000001010011, # 9  STR  R5, R3                 ; Y[i] = acc
        0b1111000000000000, # 10 RET
        0b0011001100110001, # 11 ADD  R3, R3, R1             ; (BODY) acc += 1
        0b0011010001000001, # 12 ADD  R4, R4, R1             ; k += 1
        0b0001111000000110, # 13 BRnzp #6                    ; goto LOOP_CHECK(6) 无条件
    ]

    data_memory = Memory(dut=dut, addr_bits=8, data_bits=8, channels=4, name="data")
    data = [0] * 16   # addr 8-11 接收 Y

    threads = 4       # 单 block，便于在 trace 中观察逐线程退出
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

    logger.info(f"Completed in {cycles} cycles")
    data_memory.display(16)

    expected = [0, 1, 2, 3]   # Y[i] = threadIdx
    for i, exp in enumerate(expected):
        result = data_memory.memory[i + 8]
        assert result == exp, f"Y[{i}] mismatch: expected {exp}, got {result}"

import cocotb
from cocotb.triggers import RisingEdge
from .helpers.setup import setup
from .helpers.memory import Memory
from .helpers.format import format_cycle
from .helpers.logger import logger

@cocotb.test()
async def test_warpadd(dut):
    # matadd over 16 elements: baseA=0, baseB=16, baseC=32
    # threads=16, THREADS_PER_BLOCK=4 => 4 blocks; NUM_CORES=2 => 2 warps resident per core.
    program_memory = Memory(dut=dut, addr_bits=8, data_bits=16, channels=1, name="program")
    program = [
        0b0101000011011110, # MUL   R0, %blockIdx, %blockDim
        0b0011000000001111, # ADD   R0, R0, %threadIdx      ; i = blockIdx*blockDim + threadIdx
        0b1001000100000000, # CONST R1, #0                  ; baseA
        0b1001001000010000, # CONST R2, #16                 ; baseB
        0b1001001100100000, # CONST R3, #32                 ; baseC
        0b0011010000010000, # ADD   R4, R1, R0              ; addr(A[i])
        0b0111010001000000, # LDR   R4, R4                  ; A[i]
        0b0011010100100000, # ADD   R5, R2, R0              ; addr(B[i])
        0b0111010101010000, # LDR   R5, R5                  ; B[i]
        0b0011011001000101, # ADD   R6, R4, R5              ; C[i] = A[i]+B[i]
        0b0011011100110000, # ADD   R7, R3, R0              ; addr(C[i])
        0b1000000001110110, # STR   R7, R6                  ; store C[i]
        0b1111000000000000, # RET
    ]

    data_memory = Memory(dut=dut, addr_bits=8, data_bits=8, channels=4, name="data")
    data = list(range(16)) + list(range(16))  # A[0..15] at 0-15, B[0..15] at 16-31

    threads = 16
    await setup(
        dut=dut, program_memory=program_memory, program=program,
        data_memory=data_memory, data=data, threads=threads,
    )
    data_memory.display(48)

    cycles = 0
    while dut.done.value != 1:
        data_memory.run()
        program_memory.run()
        await cocotb.triggers.ReadOnly()
        format_cycle(dut, cycles)
        await RisingEdge(dut.clk)
        cycles += 1

    logger.info(f"Completed in {cycles} cycles")
    data_memory.display(48)

    expected = [a + b for a, b in zip(data[0:16], data[16:32])]  # C[i] = 2*i
    for i, exp in enumerate(expected):
        got = data_memory.memory[i + 32]
        assert got == exp, f"Result mismatch at index {i}: expected {exp}, got {got}"

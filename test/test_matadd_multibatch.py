import cocotb
from cocotb.triggers import RisingEdge
from .helpers.setup import setup
from .helpers.memory import Memory
from .helpers.format import format_cycle
from .helpers.logger import logger

@cocotb.test()
async def test_matadd_multibatch(dut):
    # matadd over 20 elements: baseA=0, baseB=20, baseC=40
    # threads=20, THREADS_PER_BLOCK=4 => total_blocks=ceil(20/4)=5
    # NUM_CORES=2, WARPS_PER_CORE=2 => first batch fills 4 blocks (cores 0,1 each get 2 warps)
    # 5th block triggers second dispatch batch => core_reset + re-dispatch path
    program_memory = Memory(dut=dut, addr_bits=8, data_bits=16, channels=1, name="program")
    program = [
        0b0101000011011110, # MUL   R0, %blockIdx, %blockDim
        0b0011000000001111, # ADD   R0, R0, %threadIdx      ; i = blockIdx*blockDim + threadIdx
        0b1001000100000000, # CONST R1, #0                  ; baseA
        0b1001001000010100, # CONST R2, #20                 ; baseB
        0b1001001100101000, # CONST R3, #40                 ; baseC
        0b0011010000010000, # ADD   R4, R1, R0              ; addr(A[i])
        0b0111010001000000, # LDR   R4, R4                  ; A[i]
        0b0011010100100000, # ADD   R5, R2, R0              ; addr(B[i])
        0b0111010101010000, # LDR   R5, R5                  ; B[i]
        0b0011011001000101, # ADD   R6, R4, R5              ; C[i] = A[i]+B[i]
        0b0011011100110000, # ADD   R7, R3, R0              ; addr(C[i])
        0b1000000001110110, # STR   R7, R6                  ; store C[i]
        0b1111000000000000, # RET
    ]

    # A[0..19] at addr 0-19, B[0..19] at addr 20-39, C at addr 40-59
    data_memory = Memory(dut=dut, addr_bits=8, data_bits=8, channels=4, name="data")
    data = list(range(20)) + list(range(20))

    threads = 20

    await setup(
        dut=dut,
        program_memory=program_memory,
        program=program,
        data_memory=data_memory,
        data=data,
        threads=threads
    )

    data_memory.display(60)

    cycles = 0
    while dut.done.value != 1:
        data_memory.run()
        program_memory.run()

        await cocotb.triggers.ReadOnly()
        format_cycle(dut, cycles)

        await RisingEdge(dut.clk)
        cycles += 1

    logger.info(f"Completed in {cycles} cycles")
    data_memory.display(60)

    for i in range(20):
        assert data_memory.memory[i + 40] == 2 * i, \
            f"Result mismatch at index {i}: expected {2*i}, got {data_memory.memory[i+40]}"

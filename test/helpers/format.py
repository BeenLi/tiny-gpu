from typing import List, Optional
from .logger import logger

def format_register(register: int) -> str:
    if register < 13:
        return f"R{register}"
    if register == 13:
        return f"%blockIdx"
    if register == 14:
        return f"%blockDim"
    if register == 15:
        return f"%threadIdx"

def format_instruction(instruction: str) -> str:
    opcode = instruction[0:4]
    rd = format_register(int(instruction[4:8], 2))
    rs = format_register(int(instruction[8:12], 2))
    rt = format_register(int(instruction[12:16], 2))
    n = "N" if instruction[4] == "1" else ""
    z = "Z" if instruction[5] == "1" else ""
    p = "P" if instruction[6] == "1" else ""
    imm = f"#{int(instruction[8:16], 2)}"

    if opcode == "0000":
        return "NOP"
    elif opcode == "0001":
        return f"BRnzp {n}{z}{p}, {imm}"
    elif opcode == "0010":
        return f"CMP {rs}, {rt}"
    elif opcode == "0011":
        return f"ADD {rd}, {rs}, {rt}"
    elif opcode == "0100":
        return f"SUB {rd}, {rs}, {rt}"
    elif opcode == "0101":
        return f"MUL {rd}, {rs}, {rt}"
    elif opcode == "0110":
        return f"DIV {rd}, {rs}, {rt}"
    elif opcode == "0111":
        return f"LDR {rd}, {rs}"
    elif opcode == "1000":
        return f"STR {rs}, {rt}"
    elif opcode == "1001":
        return f"CONST {rd}, {imm}"
    elif opcode == "1111":
        return "RET"
    return "UNKNOWN"

def format_core_state(core_state: str) -> str:
    core_state_map = {
        "000": "IDLE",
        "001": "FETCH",
        "010": "DECODE",
        "011": "REQUEST",
        "100": "WAIT",
        "101": "EXECUTE",
        "110": "UPDATE",
        "111": "DONE"
    }
    return core_state_map[core_state]

def format_fetcher_state(fetcher_state: str) -> str:
    fetcher_state_map = {
        "000": "IDLE",
        "001": "FETCHING",
        "010": "FETCHED"
    }
    return fetcher_state_map[fetcher_state]

def format_lsu_state(lsu_state: str) -> str:
    lsu_state_map = {
        "00": "IDLE",
        "01": "REQUESTING",
        "10": "WAITING",
        "11": "DONE"
    }
    return lsu_state_map[lsu_state]

def format_memory_controller_state(controller_state: str) -> str:
    controller_state_map = {
        "000": "IDLE",
        "010": "READ_WAITING",
        "011": "WRITE_WAITING",
        "100": "READ_RELAYING",
        "101": "WRITE_RELAYING"
    }
    return controller_state_map[controller_state]

def format_registers(registers: List[str]) -> str:
    formatted_registers = []
    for i, reg_value in enumerate(registers):
        decimal_value = int(reg_value, 2)  # Convert binary string to decimal
        reg_idx = 15 - i # Register data is provided in reverse order
        formatted_registers.append(f"{format_register(reg_idx)} = {decimal_value}")
    formatted_registers.reverse()
    return ", ".join(formatted_registers)

def _safe(fn, default="?"):
    """Evaluate fn() and swallow any error (uninitialized signal, missing handle, x/z)."""
    try:
        return fn()
    except Exception:
        return default

# Stage 6 warp-scheduling hierarchy:
#   dut.cores[c].core_instance
#       .current_warp / .core_state / .fetcher_state / .current_pc / .done / .instruction  (core-level scalars)
#       .warps[w].decoder_instance                                                          (per-warp decoder)
#       .warps[w].threads[t].alu_instance.core_state  == warp_state[w] (FSM state of warp w)
#       .warps[w].threads[t].lsu_instance.lsu_state
#       .warps[w].threads[t].register_instance.rs / .rt / .registers
def format_cycle(dut, cycle_id: int, thread_id: Optional[int] = None):
    logger.debug(f"\n================================== Cycle {cycle_id} ==================================")

    cores = _safe(lambda: list(dut.cores), [])
    for core_idx, core in enumerate(cores):
        ci = _safe(lambda: core.core_instance, None)
        if ci is None:
            continue

        current_warp = _safe(lambda: int(str(ci.current_warp.value), 2))
        core_state = _safe(lambda: format_core_state(str(ci.core_state.value)))
        fetcher_state = _safe(lambda: format_fetcher_state(str(ci.fetcher_state.value)))
        current_pc = _safe(lambda: int(str(ci.current_pc.value), 2))
        done = _safe(lambda: str(ci.done.value))
        instr = _safe(lambda: format_instruction(str(ci.instruction.value)))

        logger.debug(
            f"\n+--------------------- Core {core_idx} ---------------------+"
        )
        logger.debug(
            f"current_warp={current_warp}  core_state={core_state}  "
            f"fetcher_state={fetcher_state}  PC={current_pc}  instr={instr}  done={done}"
        )

        sched = _safe(lambda: ci.scheduler_instance, None)
        warps = _safe(lambda: list(ci.warps), [])
        for w, warp in enumerate(warps):
            wstate = _safe(lambda: format_core_state(str(warp.threads[0].alu_instance.core_state.value)))
            marker = " <== current" if current_warp == w else ""
            logger.debug(f"  warp {w}: state={wstate}{marker}")

            threads = _safe(lambda: list(warp.threads), [])
            for t, thread in enumerate(threads):
                p = w * len(threads) + t
                lsu = _safe(lambda: format_lsu_state(str(thread.lsu_instance.lsu_state.value)))
                act = _safe(lambda: int(sched.active_mask[p].value))
                tpc = _safe(lambda: int(str(sched.thread_pc[p].value), 2))
                rs = _safe(lambda: int(str(thread.register_instance.rs.value), 2))
                rt = _safe(lambda: int(str(thread.register_instance.rt.value), 2))
                regs = _safe(lambda: format_registers([str(item.value) for item in thread.register_instance.registers]))
                logger.debug(f"    thread {w}.{t}: act={act} tpc={tpc} lsu={lsu}  RS={rs} RT={rt}  [{regs}]")

# Stage 7 — Intra-Warp Divergence: Results

> Recombining branch divergence with stage-6 switch-on-stall warp scheduling.
> Design: [stage7_intra_warp_divergence_design.md](stage7_intra_warp_divergence_design.md).

## What changed

Each warp now carries per-thread `thread_pc` / `active_mask` / `done_mask`
(flattened `[W*T]`). Scheduler UPDATE does min-PC reconvergence over the current
warp's running threads; the warp retires only when no runners remain. `core.sv`
feeds each thread its own PC and gates every unit's `enable` by `active_mask`.
Warp selection (ready/live round-robin, park-on-stall) is unchanged.

## Correctness

| Test | Threads | Warps/core | Property exercised | Result | Cycles |
|------|---------|-----------|--------------------|--------|--------|
| test_relu            | 8  | 1 | if/else divergence + reconverge at STR/RET | PASS | 307 |
| test_divloop         | 4  | 1 | staggered per-thread RET (partial done_mask) | PASS | 389 |
| test_relu_warpsched  | 16 | 2 | divergence x switch-on-stall together | PASS | 306 |

## Regressions (non-divergent, must be identical)

| Test | Cycles | Result |
|------|--------|--------|
| test_warpadd            | 274 | PASS |
| test_matadd             | 267 | PASS |
| test_matadd_tail        | 267 | PASS |
| test_matadd_multibatch  | 427 | PASS |

## Reconvergence trace (test_relu, warp 0 of Core 0)

The ReLU kernel branches at PC 9 (`BRn #12`): threads whose X[i]<T=4 go to
the ZERO path (PC 12-13), the rest take the else path (PC 10-11).  The four
threads in block 0 carry X = [2, 5, 3, 7] with T=4, so threads 0 and 2
diverge to ZERO while threads 1 and 3 take the else path.  After both arms
complete, all four reconverge at the `RET` (PC 14).

The trace below is taken verbatim from the cocotb debug log
(`test/logs/log_20260704234733.txt`).  Per-thread PC (`tpc`) is omitted
(Icarus VPI limitation on unpacked arrays); divergence is visible from the
core-level `PC=` field together with the per-thread `act=` bits.

```
================================== Cycle 169 ==================================
+--------------------- Core 0 ---------------------+
current_warp=0  core_state=UPDATE  fetcher_state=IDLE  PC=9  instr=BRnzp N, #12  done=0
  warp 0: state=UPDATE <== current
    thread 0.0: act=1 lsu=IDLE  RS=0 RT=0  [R0=0, R1=0, R2=8, R3=4, R4=2, R5=8, %blockIdx=0, %threadIdx=0]
    thread 0.1: act=1 lsu=IDLE  RS=1 RT=0  [R0=1, R1=0, R2=8, R3=4, R4=5, R5=9, %blockIdx=0, %threadIdx=1]
    thread 0.2: act=1 lsu=IDLE  RS=2 RT=0  [R0=2, R1=0, R2=8, R3=4, R4=3, R5=10, %blockIdx=0, %threadIdx=2]
    thread 0.3: act=1 lsu=IDLE  RS=3 RT=0  [R0=3, R1=0, R2=8, R3=4, R4=7, R5=11, %blockIdx=0, %threadIdx=3]
================================== Cycle 170 ==================================   ← DIVERGED
+--------------------- Core 0 ---------------------+
current_warp=0  core_state=FETCH  fetcher_state=IDLE  PC=10  instr=BRnzp N, #12  done=0
  warp 0: state=FETCH <== current
    thread 0.0: act=0 lsu=IDLE  RS=0 RT=0  [R0=0, R1=0, R2=8, R3=4, R4=2, R5=8, %blockIdx=0, %threadIdx=0]
    thread 0.1: act=1 lsu=IDLE  RS=1 RT=0  [R0=1, R1=0, R2=8, R3=4, R4=5, R5=9, %blockIdx=0, %threadIdx=1]
    thread 0.2: act=0 lsu=IDLE  RS=2 RT=0  [R0=2, R1=0, R2=8, R3=4, R4=3, R5=10, %blockIdx=0, %threadIdx=2]
    thread 0.3: act=1 lsu=IDLE  RS=3 RT=0  [R0=3, R1=0, R2=8, R3=4, R4=7, R5=11, %blockIdx=0, %threadIdx=3]
================================== Cycle 230 ==================================   ← ZERO-PATH (mask flipped)
+--------------------- Core 0 ---------------------+
current_warp=0  core_state=FETCH  fetcher_state=IDLE  PC=12  instr=BRnzp NZP, #14  done=0
  warp 0: state=FETCH <== current
    thread 0.0: act=1 lsu=IDLE  RS=0 RT=0  [R0=0, R1=0, R2=8, R3=4, R4=2, R5=8, %blockIdx=0, %threadIdx=0]
    thread 0.1: act=0 lsu=IDLE  RS=1 RT=4  [R0=1, R1=0, R2=8, R3=4, R4=5, R5=9, %blockIdx=0, %threadIdx=1]
    thread 0.2: act=1 lsu=IDLE  RS=2 RT=0  [R0=2, R1=0, R2=8, R3=4, R4=3, R5=10, %blockIdx=0, %threadIdx=2]
    thread 0.3: act=0 lsu=IDLE  RS=3 RT=4  [R0=3, R1=0, R2=8, R3=4, R4=7, R5=11, %blockIdx=0, %threadIdx=3]
================================== Cycle 280 ==================================   ← RECONVERGED
+--------------------- Core 0 ---------------------+
current_warp=0  core_state=FETCH  fetcher_state=IDLE  PC=14  instr=STR R5, R6  done=0
  warp 0: state=FETCH <== current
    thread 0.0: act=1 lsu=IDLE  RS=8 RT=0  [R0=0, R1=0, R2=8, R3=4, R4=2, R5=8, %blockIdx=0, %threadIdx=0]
    thread 0.1: act=1 lsu=IDLE  RS=1 RT=4  [R0=1, R1=0, R2=8, R3=4, R4=5, R5=9, %blockIdx=0, %threadIdx=1]
    thread 0.2: act=1 lsu=IDLE  RS=10 RT=0  [R0=2, R1=0, R2=8, R3=4, R4=3, R5=10, %blockIdx=0, %threadIdx=2]
    thread 0.3: act=1 lsu=IDLE  RS=3 RT=4  [R0=3, R1=0, R2=8, R3=4, R4=7, R5=11, %blockIdx=0, %threadIdx=3]
```

**Reading the trace:**
- Cy 169: `UPDATE` of `BRn #12` at PC=9 — all four threads `act=1`.  The N-flag
  is about to be evaluated per-thread (R4 vs T=4).
- Cy 170: PC advances to **10** (else path, `STR R5 R4`).  Threads 0 and 2
  (`X=2,3 < T=4`) have `act=0`; threads 1 and 3 (`X=5,7 ≥ T=4`) have `act=1`.
  Divergence is live.
- Cy 230: After the else-arm completes its `BRnzp #14` (PC 11→14), the
  scheduler runs min-PC = 12 for the parked threads.  PC switches to **12**
  (`CONST R6,#0`), mask is inverted: threads 0,2 `act=1`, threads 1,3 `act=0`.
- Cy 280: After the ZERO-arm's `STR R5 R6` (PC 13) completes, min-PC = 14 for
  all threads.  PC = **14** (`RET`), all four threads `act=1` — **reconverged**.
  (The `instr=` label still shows the previous cycle's decoded instruction; the
  fetch pipeline will resolve `RET` by the FETCHED sub-state.)

## Why it composes with latency hiding

A parked warp's `active_mask` only changes in UPDATE, which a parked warp never
enters, so its active threads' LSUs keep draining while another warp runs — the
stage-6 latency-hiding property is untouched by divergence.

# Stage 7 — Intra-Warp Branch Divergence (recombined with warp scheduling)

> Design spec. Date: 2026-07-04. Builds on stage 6 (switch-on-stall warp scheduling).

## Goal

Restore per-thread branch divergence **inside each warp** of the stage-6
switch-on-stall warp scheduler, so branchy kernels are correct again while
multi-warp latency hiding is preserved. Must-pass: `test_relu`, `test_divloop`,
and a new combined test with multiple diverging warps per core. Existing
non-divergent regressions (`warpadd`/`matadd`/`tail`/`multibatch`) must stay
bit-identical.

## Background

- **Stage 6 (current master)** rewrote the scheduler as a per-warp 6-stage FSM
  with a scalar `current_warp` pointer (single active warp, park-on-stall, round-
  robin ready/live switch). To keep it simple it dropped divergence: each warp
  carries a single `warp_pc[w]` and blindly follows **thread 0's** `next_pc`.
  No `active_mask`, no per-thread PC, no reconvergence. Branchy kernels
  (`relu`, `divloop`) therefore produce wrong results on current master.
- **Pre-warp master (`bd18e12`)** had the mechanism we need: per-thread
  `thread_pc[i]`, an `active_mask[i]`, a `done_mask[i]`, and **min-PC
  reconvergence** at UPDATE (fetch the minimum PC among still-running threads;
  only threads at that PC execute; paths auto-reconverge when PCs coincide;
  a thread executing `RET` sets `done_mask[i]`; all-retired ⇒ block done).

The task is to fold that per-thread mechanism into **each warp** of the stage-6
scheduler. The switch-on-stall warp selection is **orthogonal** to what happens
*inside* a warp, so it is left untouched.

## Approach (chosen: A)

**A — Per-warp min-PC active-mask reconvergence.** Replicate the `bd18e12`
mechanism per warp, flattened to 1-D `[W*T]` arrays (sv2v-safe, index
`p = w*THREADS_PER_BLOCK + i`).

Rejected alternatives:
- **B — explicit SIMT reconvergence stack (IPDOM).** More faithful to real GPUs
  but needs post-dominator info and a per-warp stack; no correctness gain over
  min-PC on these kernels. Overkill.
- **C — predication only.** Sidesteps *branch* divergence entirely; cannot
  express divloop's data-dependent trip counts. Rejected.

## Component changes

### scheduler.sv — state & interface

Flatten per-thread divergence state into the warp scheduler (`p = w*T + i`).

Interface:

| Dir | Remove | Add / change |
|-----|--------|--------------|
| in  | `warp_next_pc [W-1:0]` | `next_pc [W*T-1:0]` — every thread's next-PC from pc.sv |
| out | — | `thread_pc [W*T-1:0]` — per-thread PC (drives each pc_instance.current_pc) |
| out | — | `active_mask [W*T-1:0]` (packed) — threads executing this cycle |
| out | `warp_pc[w]` meaning | now = min-PC of running threads of warp w (fetch PC); still drives current_pc |

New internal regs: `done_mask [W*T-1:0]` (packed), and module-level scan
temporaries `reg [7:0] min_pc, eff; reg found, eligible;`.

Reset: `thread_pc`, `active_mask`, `done_mask` all 0.

Start (IDLE, on `start`), for each warp w:
- `warp_pc[w] <= 0`
- `warp_state[w] <= (warp_thread_count[w] != 0) ? FETCH : DONE`
- for each thread i (p=w*T+i): `thread_pc[p]<=0`, `done_mask[p]<=0`,
  `active_mask[p] <= (i < warp_thread_count[w])`
- `current_warp <= 0`

### scheduler.sv — FSM

Unchanged: the per-cycle `lsu_busy[w]` computation, the round-robin ready/live
scan, IDLE, FETCH, DECODE, REQUEST (mem op ⇒ park to WAIT + switch), EXECUTE,
WAIT, DONE. Divergence changes only **UPDATE**.

**UPDATE** (cw = current_warp): replace the stage-6 body
(`if(decoded_ret)→DONE else warp_pc<=warp_next_pc→FETCH`) with per-warp min-PC
reconvergence:

```
// 1) commit active threads
for i: p=cw*T+i;
  if (active_mask[p])
    if (decoded_ret) done_mask[p] <= 1'b1;
    else             thread_pc[p] <= next_pc[p];

// 2) min-PC over still-running threads (exclude threads retiring THIS cycle)
min_pc = 8'hFF; found = 1'b0;
for i: p=cw*T+i;
  eligible = (i < warp_thread_count[cw]) && !done_mask[p]
             && !(decoded_ret && active_mask[p]);
  eff      = (active_mask[p] && !decoded_ret) ? next_pc[p] : thread_pc[p];
  if (eligible && (!found || eff < min_pc)) begin min_pc = eff; found = 1'b1; end

// 3a) no runners left -> warp retires; switch away (same ready/live switch as DONE)
if (!found) begin
  warp_state[cw] <= DONE;
  if (ready_found) begin current_warp<=ready_warp;
      if (warp_state[ready_warp]==WAIT) warp_state[ready_warp]<=EXECUTE; end
  else if (live_found) current_warp <= live_warp;
end
// 3b) runners remain -> re-activate threads at min_pc, refetch (greedy, stay on cw)
else begin
  warp_pc[cw] <= min_pc;
  for i: p=cw*T+i;
    eligible = (i<warp_thread_count[cw]) && !done_mask[p]
               && !(decoded_ret && active_mask[p]);
    eff      = (active_mask[p] && !decoded_ret) ? next_pc[p] : thread_pc[p];
    active_mask[p] <= (eligible && (eff == min_pc)) ? 1'b1 : 1'b0;
  warp_state[cw] <= FETCH;
end
```

All reads use the pre-update `active_mask`/`done_mask` (NBA semantics); the
`!(decoded_ret && active_mask[p])` guard excludes threads retiring this very
cycle. Faithful to `bd18e12`.

**Why it composes with warp scheduling:** while a warp is parked in WAIT its
`active_mask` is frozen (only UPDATE mutates it, and a parked warp isn't in
UPDATE), so its active threads' LSUs keep draining in the background — latency
hiding preserved. The "warp done" trigger correctly moves from *any RET* to
*no runners left*, which divloop's staggered per-thread exits require.

### core.sv — wiring

1. Drop the `warp_next_pc[w]` wire and its `assign ... = next_pc[w*T+0]`.
2. Add wires `thread_pc [W*T-1:0]`, `active_mask [W*T-1:0]`.
3. Scheduler port map: remove `.warp_next_pc`; add `.next_pc(next_pc)`,
   `.thread_pc(thread_pc)`, `.active_mask(active_mask)`.
4. `pc_instance`: `.current_pc(warp_pc[w])` -> `.current_pc(thread_pc[p])`.
5. enable on alu/lsu/registers/pc: `(i < warp_thread_count[w])`
   -> `(i < warp_thread_count[w]) && active_mask[p]`.

Fetcher unchanged (`current_pc = warp_pc[current_warp]` = min-PC).
`gpu.sv`, `dispatch.sv`, `pc.sv`, `lsu.sv`, `decoder.sv`: no change.

## Tests & verification

Regression (must stay green, non-divergent ⇒ uniform active_mask):
`test_warpadd`, `test_matadd`, `test_matadd_tail`, `test_matadd_multibatch`.

Divergence targets:
- `test_relu` — 8 threads ⇒ 2 blocks ⇒ 1 diverging warp/core. if/else BRn +
  reconvergence at shared STR/RET. Expected Y[i] = (X[i]<T)?0:X[i].
- `test_divloop` — 4 threads, per-thread trip counts; staggered RET retirement
  (partial done_mask) + min-PC reconvergence. Expected Y[i] = threadIdx.

New combined test — `test_relu_warpsched.py`:
- relu program variant with `baseY=16` (`CONST R2,#16`); 16 threads ⇒ 4 blocks
  ⇒ 2 diverging warps per core (dispatch gives core0 blocks 0-1, core1 blocks 2-3).
- `X = [2,5,3,7, 1,6,0,4, 3,8,2,5, 6,1,7,0]` (each 4-thread warp straddles T=4);
  Y at addr 16-31; assert Y[i] == relu(X[i]).
- Proves divergence AND switch-on-stall interleaving run together per core.

Trace helper: light `format.py` update to print each warp's active_mask /
per-thread thread_pc for legible divergence traces (debug aid, not correctness).

Run (remote, `source env.sh`): `make test_relu test_divloop test_relu_warpsched`
plus the 4 regressions. cycles = SIM_TIME / 25000 (25us clock).
Docs: results note in `docs/` + `LEARNING_GUIDE.md` update.

## Out of scope

Reconvergence-stack (IPDOM), nested/irreducible control flow beyond what min-PC
handles, per-warp scheduling-policy comparison (LRR vs GTO), dynamic warp
formation.

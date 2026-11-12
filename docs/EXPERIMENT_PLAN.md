# Pi0.5 Experiment Gates

Scope: pi0.5 only. No cross-architecture calibration, static DRAM bandwidth shares,
SRAM-bank model, or results from other models. Every CPU/memory-intensive task on
Deep-space goes through Slurm with explicit CPU, RAM and wall-time limits.

## Research Questions

1. When does timestep pipelining improve throughput after shared DRAM interference?
2. Where do batch weight reuse, batch-formation delay, KV traffic and SRAM capacity
   move the useful microbatch size?
3. Does the best prefix/DiT compute split change under joint memory scheduling?

## Gate A: Implementation

- Implement pinned-source pi0.5 shapes, per-request KV, sequential timesteps,
  merged-M linear microbatching, finite weight-tile residency and request slots.
- Verify tensor/view addresses, scratch liveness, cache backpressure, K accumulators,
  slot retirement, partial-batch draining and DRAM-delay feedback in unit tests.
- Run a tiny pi0.5 graph against real Ramulator, retaining old regression fixtures.
- Compare native idle batching against tick-by-tick execution; timings and counters
  must match exactly. Determinism checks apply to identical source/config inputs.
- Derive exact GEMM FLOPs and weight/KV footprints from the graph, distinguishing
  full vocabulary capacity from rows read per request.

## Gate B: Runtime And Model Boundaries

- Profile the full prefix and full DiT graphs separately with fixed tick budgets.
  Such profiles do not produce end-to-end performance claims.
- Complete a real-width one-layer/one-step DiT component test. It is reduced-depth
  validation, not a substitute for a full pi0.5 run.
- Complete one full-size pi0.5 request stream with a short window. Measure whole-run
  wall time, peak RSS, transaction count, and active/idle-tick costs before budgeting
  the candidate sweep. If too slow, optimize the simulator before launching many jobs.
- Pin the actual HBM3 organization/data rate/channel mapping. Exercise sequential
  reads, strided reads and read/write mixes as sanity checks, without tuning DRAM
  timing to force a target observed bandwidth. Add LPDDR only as a separately
  reviewed implementation, never relabel DDR/HBM as LPDDR5X.
- Document sensitivity to dense materialized attention, vector cost, loop order,
  private cache reservation and SRAM capacity. Do not hide these behind scale factors.

## Gate C: Fair Baselines

Before drawing pipeline speedup conclusions, add an unpipelined schedule that can
use the same entire core/SRAM/DRAM budget. Add end-to-end batching as appropriate;
do not compare a partitioned pipeline against a baseline artificially restricted
to only the prefix or DiT subset of cores. All candidates need the same request
supply/latency definition, precision, operator mapping, memory configuration and
in-flight budget (or a separately reported budget sweep).

## Gate D: Small Pilot

Start with 32 explicit 32x32 arrays at 1 GHz, 1 MiB private SRAM/array, no shared
SRAM, fixed HBM3 and one prefix stage. Choose one cache reservation/tiling policy
for the first sweep and record it; cache bytes are part of the 1 MiB total.

Explore DiT stages `{1,2,5,10}`, microbatch `{1,2,5,10}` and 2-3 legal compute splits
per pair, at most about 48 pipeline candidates plus fair baselines. Set the
in-flight limit at least as large as the microbatch; then sweep representative
limits so queue depth is not an unnoticed source of speedup.

Warmup 20 and measure 50 are initial choices, not guarantees. Verify the best
candidates with longer windows. Output L mean/p95, S, batch wait, accepted traffic
by role/stage, SRAM/outstanding peaks and actual DRAM controller counters. Stall
counters overlap; do not sum them into makespan.

## Gate E: Expansion And Search

Only after runtime and fair-baseline gates pass, expand core count, DRAM type,
prefix partitioning, replicas and local SRAM. Shared SRAM requires explicit reuse
and transfer semantics before becoming a sweep axis. Do not merely divide its
capacity among stages and describe that as a shared SRAM design.

An isolated per-layer table can rank candidates, but is not automatically an
admissible bound for the joint pipelined simulator. Initially use it as a heuristic
and retain evaluated/unexamined candidate logs. Claim a global optimum or a full
Pareto frontier only with proven bounds or exhaustive finite-grid simulation.

Operator overhead is charged once per mapped operator invocation, not once per
tile. It blocks that stage's operator launch, not unrelated stages' DRAM traffic.
The core clock and DMA issue width must not be silently equated to the DRAM clock.

## Records

Keep source commits plus a source digest, input configuration, resolved DRAM hash,
Slurm job ID/resources, completion/window boundaries and small validation summaries.
Raw per-transaction traces stay opt-in and out of Git. Do not commit intermediate
result directories. A stopped profile, a reduced-shape test, and a complete pi0.5
run must be clearly distinguishable in reports and plots.

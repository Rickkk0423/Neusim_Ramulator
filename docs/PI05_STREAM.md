# Pi0.5 Request Stream

## Scope

Only pi0.5 is being developed for the experiment. Schema 1 remains a regression
surface; it is not a second model in the experimental campaign.

The source of operator shapes is Physical-Intelligence/openpi revision
`215abfb217dbac7d5f1273282331b9b1866c0479`, inspected from the official repository:

```text
https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models
```

`gemma.py` supplies Gemma-2B and Gemma-300M dimensions, single-head KV, gated MLP,
and adaptive RMS modulation. `siglip.py` supplies So400m/14 dimensions.
`pi0.py` supplies prefix caching, action/time projections, and denoising order.
The experimental scenario uses three 224x224 images, 200 text tokens, an action
horizon of 10, and 10 denoising steps. Horizon 10 is a scenario choice, not a claim
about the upstream default. All sizes are explicit in `Shape`.

This is a **shape-derived timing/traffic model**, not a numerical port of OpenPI.
No weights are downloaded, no inference outputs are produced, and no hardware
calibration factors are fitted. It must not be described as full framework or RTL
fidelity.

## Execution

1. A saturated source admits requests only while a finite slot is free. Each slot
   owns its image/input-action storage and prefix K/V for every transformer layer.
   A slot is recycled only after that request's final stage write callbacks.
2. The prefix stage handles one request at a time. Image attention is independent
   per image; Gemma prefix attention spans the image/text prefix. Position-encoded
   prefix K and V are written to the request's own external KV addresses.
3. A DiT stage waits for its FIFO microbatch, merges M across requests for linear
   operators, and evaluates attention separately per request. Its K/V operand is a
   logical concatenation of that request's prefix KV and current suffix KV. No
   free copy of the prefix KV is inserted, and requests cannot attend to each other.
4. Each DiT stage owns an equal contiguous range of timesteps. Requests move to
   the next stage only after all operators and their writes finish. Timesteps of
   one request remain sequential; concurrency comes from different requests.
5. A stage has explicit SA cores, each containing one NeuSim compute model and
   private SRAM. Output tile groups are distributed N-major, round-robin across
   cores. All K slices of an output group stay on its assigned core. Different
   cores/stages share one live Ramulator instance; there are no bandwidth shares.
6. Tile inputs, compute, K-slice lookahead, and stores advance on DRAM callbacks
   and compute events. Only one output group/core is active at a time. Operators
   have barriers: the next operator waits for every core's writes. Stage handoff
   uses DRAM, not an unmodeled 256 GB/s link.

Request slots bound persistent storage. Stage scratch is allocated by last use
across its operator graph and reused for later jobs. Operator barriers are what
make that recycling legal; enabling cross-operator fusion requires new lifetime
and dependency handling, not merely removing the barrier.

## SRAM And Reuse

SRAM banks/ports/timing and shared SRAM are intentionally absent. Local SRAM is
partitioned into scratch and a configurable weight-tile LRU. The configured total
includes both partitions. This reservation is an explicit architecture policy.

Immutable GEMM weight regions use physical-transaction-sized cache allocations.
Loading entries can serve later readers only when their DMA has completed. Entries
with readers or incomplete fills cannot be evicted. Scratch includes ordinary
inputs and pinned output accumulators. Partial sums remain local across K slices;
only the final slice stores the output. There is no cross-core weight multicast.

Batch reuse arises from merged GEMM M tiles and actual cache hits. Weight traffic
is never calculated by dividing total parameter bytes by microbatch size. Cache
hit/miss/eviction counters and accepted weight/KV/activation transaction bytes are
reported separately. Transaction alignment can make these larger than logical
tensor bytes.

## Deliberate Approximations

- Attention is dense and materializes FP32 logits and BF16 probabilities in DRAM.
  It is not FlashAttention, and it can overstate traffic relative to fused kernels.
- Norm, activation, RoPE, softmax, modulation application, copies, and Euler updates
  have explicit read/write operands and configurable vector element-work costs.
  Those costs are engineering estimates, not official NeuSim implementations of
  every vector kernel. Reduction behavior is represented at row granularity.
- Small biases, norm scales and learned positional parameter loads are omitted.
  The vocabulary table is allocated in DRAM; only deterministic synthetic token
  rows are read. Input preprocessing and generation of the initial noise/action
  contents are outside the measured interval.
- DMA issue width is per Ramulator model tick; there is no independently clocked
  DMA model. Request injection is round-robin across cores and ordered within a
  core. No NoC, direct forwarding, sharing/coherence, or SRAM-bank contention.
- Upstream write callbacks occur at terminal command issue/coalescing, not after
  physical write recovery. Partial writes cost full transactions without masks
  or read-modify-write. Physical DQ utilization is not interface bytes/time.
- Prefix layer partitioning, replicas, shared SRAM, LPDDR5X, optimized baselines,
  arbitrary graph import and a search engine are not implemented in schema 2 yet.

## Measurement

Request latency runs from finite-slot admission through its final write callback;
it includes stage queueing and batch formation, but not unlimited external queueing.
After W warmup completions, the next N completions define the measurement window.
S is `(completion[W+N] - completion[W]) / N`; W=0 uses time zero and is explicitly
marked as including cold fill. All timestamps in this expression are completion
ranks, starting at one.

The source retains additional successor requests until the window closes, then
stops admissions and drains active requests, including a final partial batch.
Drain time is excluded from window throughput. Per-stage interface bytes count
accepted transactions during the same window; they are not attributed only to
the N completed requests. The report includes warmup/window boundaries, per-request
stage timestamps, batch-formation wait, total stage-queue wait, SRAM/outstanding
peaks, and source/config metadata. Batch formation is arrival-to-last-member-arrival;
total queue wait is arrival-to-stage-start and also includes a busy stage's delay.
Longer-window checks are still required before calling measurements steady state.

`pi05_dit` is an explicit component mode with prefix KV prepopulated. Its latency
does not include prefix execution and must never be plotted as end-to-end pi0.5.
Shape overrides used for tiny/one-block tests are recorded in every report.

## Cost Controls

Tiles are generated lazily, with at most `cores * prefetch_tiles` active tiles.
Latency statistics use a histogram instead of an unbounded list. Graph objects
scale with active stage jobs, not the full request history. Per-request/job records
still grow with admitted requests; very long runs need streaming of those records.

When no transactions or callback-dependent work are outstanding, the driver may
batch idle ticks into the native bridge until the next compute/overhead event.
Ramulator still executes every tick, including refresh. Tests compare both modes
for identical timestamps, counters, and request records. New admissions are an
event boundary and must never be skipped.

`simulation.profile_cycles` stops a bounded cost profile. Its result has
`complete=false` and **no latency/throughput measurement**. Native batching cannot
remove the cost of memory-active ticks, so a short profile is not a reliable
whole-run or whole-DSE time estimate.

## Running On Deep-space

```bash
cd /fact_home/renati/workspace/Neusim_Ramulator
mkdir -p results
sbatch scripts/slurm_validate.sh
sbatch --array=0-1 scripts/slurm_profile.sh
```

Validation requests 4 CPUs/8 GiB for at most 30 minutes; profiles request 1 CPU/8 GiB
per array task for at most 15 minutes. BLAS/OpenMP threads are restricted to one.
The 4-CPU validation reservation is a ceiling, not a claim that pytest uses four
workers. No large simulation should run directly in the SSH/login shell.

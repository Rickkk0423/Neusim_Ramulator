# Implementation Status

This document records the **schema-1 baseline audit**. The later pi0.5-only
schema-2 request-stream implementation and its different residency, address
lifetime, and operator support are described in [PI05_STREAM.md](PI05_STREAM.md).
Schema-1 limitations below remain applicable to the original GEMM-chain entry point.

Audit date: 2026-11-13. This is a software implementation audit, not calibration
against another accelerator, RTL, or silicon. No fitted correction factors were
introduced. The pre-audit published integration revision was `75c2165`.

## Execution Model

1. A JSON frontend expands dense GEMM chains into explicit M/N/K slices, addresses,
   and dependencies. All stages currently use one configured tile shape.
2. Each core uses the official NeuSim compute function with its own ChipConfig.
   Each slice pays its own startup/padding estimate. One aggregate compute resource
   per core serializes slices; SA/VU overlap follows NeuSim's max-time abstraction.
3. The online scheduler admits inputs and pins accumulators subject to finite SRAM,
   a maximum number of unfinished tiles, and per-core outstanding requests.
4. Accepted read/write transactions enter one live official Ramulator memory
   system. Channels have independent controllers within that shared system.
5. Read completions enable compute; final K writes enable downstream producer /
   consumer dependencies. Intermediate K slices keep their accumulator in SRAM.
6. Core/stream request injection uses round-robin arbitration. Compute, prefetch,
   and stores can overlap, including across cores; no static bandwidth quotas exist.

The global time unit is one Ramulator model tick, with compute nanoseconds rounded
up to ticks. HBM3 half-CK ticks are already reflected in the upstream returned tCK.
This is online timing feedback, not fixed-timestamp trace replay or two-pass fitting.

## Supported Surface

- Multiple cores, pipeline stages, microbatches, repeated steps, and tail tiles.
- BFLOAT16, FLOAT16, FLOAT32, INT8 operand sizes. NeuSim compute does not promise
  different arithmetic throughput by dtype. Accumulator storage is explicit and
  must be at least as wide as an output element.
- DDR4, HBM2, HBM3; power-of-two channel counts with CacheLineInterleave, Open row
  policy, RoBaRaCoCh address mapping, FRFCFS, and standard-specific refresh.
- Default addressable capacity per configured channel: DDR4 8 GiB, HBM2 512 MiB,
  HBM3 1 GiB. Other organizations/device overrides change these capacities.
  An HBM channel is not an entire HBM stack.
- JSON summaries, tile CSVs, optional request CSVs, expanded DRAM configuration,
  official DRAM statistics, source revisions, and tracked dirty-state flags.

## Audit Findings and Fixes

| Issue | Previous behavior | Current behavior |
| --- | --- | --- |
| Python/C integer boundary | A `2**64` address silently became zero | Rejects out-of-range request IDs, addresses, and core IDs before ctypes conversion |
| Out-of-capacity addresses | High address bits could be discarded by DRAM mapping | Generated DRAM capacity is enforced at request injection; CLI rejects an oversized plan |
| Large-address rounding | Floating-point ceil could generate zero transactions for a nonempty region at `2**60` | Address alignment, allocations, and batch counts use integer arithmetic |
| Use after close | Calling tick after close caused SIGSEGV | Python operations raise a closed-instance error; repeated close is safe |
| Output reuse | Old traces / success reports could survive a later run | Known report files are cleared; validation failures emit error.json without stale success |
| Undersized accumulators | An accumulator narrower than its output could undercount SRAM | Invalid widths fail during workload validation |
| Native API mismatch | A stale bridge could be loaded without a compatibility check | Explicit ABI version check asks the user to rebuild |
| Build invalidation | Every CMake configure rewrote the portability header | Generated header is updated only when its contents change |

The standard example timing results are regression anchors, not hardware truth.
No change to official NeuSim compute formulas or Ramulator controller/device
implementations was made. Upstream sources stay pinned and unmodified.

## Remaining Boundaries

1. **Workload coverage:** dense GEMM chains only. The DiT-like example is repeated
   expand/contract GEMMs, not full DiT. There is no attention, normalization,
   residual/conditioning graph, KV cache, or arbitrary NeuSim graph import.
2. **Residency and reuse:** inputs/weights are loaded for each K slice; partial sums
   alone persist across K slices. There is reuse within a tile, but no persistent
   cross-tile weight/input cache. Batch weight reuse must not be assumed.
3. **Address lifetimes:** every intermediate receives a fresh DRAM allocation;
   storage is not recycled between steps or microbatches. This may exceed capacity
   earlier than a design with explicit lifetime-based buffer reuse. It now fails
   explicitly rather than pretending the larger workload fits.
4. **Compute abstraction:** per-slice NeuSim startup can overestimate a continuous
   fused kernel. SRAM latency/banks/ports are absent by design. Data is available
   for compute only after the entire tile input arrives.
5. **Write semantics:** upstream callbacks occur at terminal WR command issue;
   duplicate buffered writes may be coalesced earlier. Neither means all physical
   write recovery has elapsed. Shared-transaction partial writes do not have byte
   masks or read-modify-write costs. This is timing-only, without numeric values.
6. **Interconnect and DMA:** cross-core data goes through DRAM. There is no NoC,
   direct forwarding, coherence, DMA descriptor setup, or explicit link latency.
   The global issue width is per DRAM model tick and is not an independently
   clocked accelerator/DMA front end.
7. **Scheduling policy:** first-ready work is considered in plan order with finite
   buffers and round-robin injection. This is a stated policy, not an optimized
   compiler schedule or a guarantee that increasing a resource improves every run.
8. **Statistics:** interface bytes/time is not necessarily physical DQ throughput,
   due to forwarding/coalescing. Request latency starts at acceptance, not the
   first rejected attempt. Stall counters overlap and do not decompose makespan.
9. **Simulation cost:** Python scans tiles every tick and stores request latencies.
   Planning intersects producer regions directly. Full-model workloads and large
   DSE sweeps can therefore require substantial host time and memory.

## Improvement Priorities

1. Add explicit weight/input residency, eviction, and loop-order policies with SRAM
   accounting. This is the first requirement for useful batch-reuse experiments.
2. Add lifetime-based DRAM buffer recycling for repeated steps and microbatches.
3. Expose stage-specific tile shapes and scheduling choices; verify region
   dependencies when producer and consumer tiles differ.
4. Introduce optional independently clocked request injection and DMA startup costs
   if the explored architecture needs them. Do not approximate this with DRAM
   timing correction factors.
5. Replace whole-plan polling with ready/active queues and bounded latency
   histograms; batch native ticks only where dependency/DRAM semantics permit.
6. Expand operator coverage only after residency and scheduling semantics are
   explicit. SRAM bank modeling is not required for the current scope.

## Cleanup Policy

Raw simulation results, transaction traces, temporary test output, Python/pytest
caches, and bootstrap transfer files are not research source artifacts. They are
removed after this audit. Source, example input configurations, documentation,
virtual environments, and the native build remain so the simulator is runnable.
CI still runs examples but no longer uploads their result files as artifacts.
Versioned validation summaries are retained, without raw traces or test dumps.

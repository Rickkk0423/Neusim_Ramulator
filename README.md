# NeuSim + Ramulator

An **online, tile-level, multi-core co-simulator** built on the official
[NeuSim](https://github.com/platformxlab/NeuSim) and
[Ramulator2](https://github.com/CMU-SAFARI/ramulator2) implementations.

This is executable integration code, not a trace-only proposal or a fixed-bandwidth
model. All cores share one live Ramulator memory system. Actual memory completions
change tile readiness, compute start times, subsequent requests, and pipeline timing.
No bandwidth is statically assigned to a core.

## Scope

- NeuSim's official compute functions estimate each GEMM K-slice's SA/VU time.
- A tile scheduler overlaps input prefetch, computation, and output writes.
- Each core has one compute resource, finite SRAM, a prefetch window, and an
  outstanding-request limit. SRAM banks and SRAM access latency are not modeled.
- Partial sums stay in an output accumulator across K slices; only the last slice
  writes the output. The accumulator remains allocated until write callbacks arrive.
- Producer/consumer dependencies are calculated by intersecting tensor regions,
  not by incorrectly pairing identical tile indices across stages.
- Multiple microbatches and sequential repeated steps run on the same timeline.
- DDR4, HBM2, and HBM3 use official Ramulator devices, controllers, and timings.

**Current workload frontend:** chains of dense GEMMs, including tail tiles, pipeline
stages, microbatches, and repeated steps. The DiT-like example is an expand/contract
GEMM workload for memory experiments, **not a complete DiT implementation**. Attention,
normalization, residuals, timestep conditioning, KV caches, and arbitrary NeuSim
model-graph import are not yet implemented. NeuSim itself supports more operators
than this first co-simulation frontend.

## Quick Start

Requirements: Linux or macOS, Python 3.12, CMake 3.20+, a C++20 compiler, Git, and
network access for Python/build dependencies. GCC 12+ on Linux or AppleClang 16 on
macOS are suitable. On macOS install Xcode Command Line Tools first.

```bash
git clone --recurse-submodules https://github.com/Rickkk0423/Neusim_Ramulator.git
cd Neusim_Ramulator
./scripts/setup.sh
./scripts/run.sh examples/two_core_pipeline.json -o results/pipeline --request-trace
./scripts/run.sh examples/four_core_hbm.json -o results/hbm
./scripts/run.sh examples/dit_like_batch.json -o results/dit
```

`PYTHON=/path/to/python3.12`, `GIT=/path/to/git`, `CXX=/path/to/compiler`, and
`BUILD_JOBS=4` can be passed to setup. An existing `.venv` is reused. The script
builds the real native backend and runs unit and DRAM integration tests.

An alternative when using `uv`:

```bash
uv venv --python 3.12
uv pip install -e '.[dev]'
.venv/bin/cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
.venv/bin/cmake --build build --parallel 4
.venv/bin/python -m pytest -q
```

The submodules must be initialized before either build method. NeuSim is loaded
directly from its pinned source tree. Only the dependencies used by its compute
path are installed: Ray, its distributed runtime, and NeuSim's power analysis
are intentionally not required. The full NeuSim package can be installed
separately when those upstream features are needed.

## Architecture

```text
JSON workload -> tensor layout / M-N-K tiles / dependencies
                              |
                   online shared scheduler
                  /                       \
        NeuSim compute time         Ramulator request queues
                  \                       /
                   compute / memory completions
                              |
               ready tiles, freed buffers, next requests
```

Every scheduler cycle is one DRAM clock period, obtained from Ramulator's `tCK`.
NeuSim returns nanoseconds; compute duration is rounded up to a whole DRAM cycle.
All controllers use the same clock and the native driver ticks the memory system
once per scheduler cycle. There is no fixed-timestamp replay or second-pass latency
table.

A tile transitions through `waiting -> reading -> ready -> computing -> writing
-> done`. Intermediate K slices skip writing. Their compute dependencies enforce
reduction order while permitting the next slice's input prefetch. Loads wait for
all intersecting upstream producer tiles' writes, and compute waits for all input
transactions. Cores may therefore overlap with each other and internally overlap
loads, compute, and stores. There is no within-tile streaming compute before the
whole input tile arrives.

Ready work is admitted in plan order subject to dependencies, SRAM capacity, and
the prefetch limit. The driver uses round-robin request injection across cores and
active transfers; rejection in one stream does not block every other stream. The
DRAM controller still applies its own FRFCFS scheduling. Injection arbitration is
an explicit modeled policy, not an assertion about a particular hardware DMA.

The bridge registers a `NeuSimExternal` frontend implementing the official external
request interface and reporting the actual core count. Upstream's `External`
frontend reports one core, which is insufficient for its controllers' per-source
statistics arrays when injecting multiple source IDs. Source IDs are bounds-checked
by the adapter; the official memory/controller implementations are unchanged.

## Configuration

See `examples/two_core_pipeline.json` for the complete schema.

| Field | Meaning |
| --- | --- |
| `dram.standard` | `DDR4`, `HBM2`, or `HBM3` |
| `dram.channels` | Number of controllers / JEDEC channels, **not HBM stacks** |
| `dram.organization`, `dram.timing` | Optional upstream preset overrides |
| `dram.device` | Optional keyword arguments to the upstream DRAM wrapper |
| `simulation.issue_width` | Global accepted DRAM requests per DRAM cycle |
| `simulation.max_cycles` | Fail rather than silently hang if the run does not finish |
| `cores[].sram_bytes` | Capacity shared by prefetched inputs and output accumulators |
| `cores[].prefetch_tiles` | Maximum admitted, unfinished tiles per core |
| `cores[].max_outstanding` | Accepted but not yet callback-completed requests per core |
| `cores[].chip` | Official NeuSim `ChipConfig` fields for that core's compute model |
| `workload.batch_size` | Total independent samples |
| `workload.microbatch_size` | Samples grouped into one GEMM's M dimension |
| `workload.rows_per_sample` | Token/row count per sample |
| `workload.steps` | Repeated sequential passes; a sample's next step consumes its previous output |
| `workload.tile` | Tile `[M, N, K]`; partial tail tiles are supported |
| `workload.dtype` | `BFLOAT16`, `FLOAT16`, `FLOAT32`, or `INT8` for byte sizes |
| `workload.accumulator_bytes` | Storage bytes per output partial sum (default 4) |
| `workload.stages` | Ordered GEMMs with `name`, `core`, `k`, `n` |

For each microbatch, M is `samples * rows_per_sample`. A final partial microbatch
is supported. Adjacent stages must satisfy `previous.n == next.k`; repeated steps
also require the final N to match the initial K. Different stages may share a core.
Weights use the same addresses across microbatches and steps; each activation
output has a distinct allocated address range.

An HBM channel count is not chosen to force a particular measured bandwidth.
Specify the actual intended organization and inspect the resulting service behavior.
The available dtype options change traffic/storage, **not** the upstream array's
precision-dependent throughput: NeuSim's compute abstraction does not model that.

## Reports

- `summary.json`: makespan, sample throughput, aggregate/per-core observed bandwidth,
  accepted-request latency, microbatch completions, SRAM peaks, and upstream revisions.
- `tiles.csv`: admission, input-ready, compute start/end, and write completion times
  for every tile. All timestamps are DRAM cycles.
- `requests.csv`: optional per-request addresses, IDs, core/tile tags, accepted and
  completion cycles. Enable with `--request-trace`.
- `dram_stats.yaml`: official Ramulator controller/device statistics.
- `config.json`, `dram.yaml`: the input and fully expanded DRAM configuration.
- `error.json`: a failure diagnostic if execution fails after output initialization.

Bandwidth is accepted interface-transaction bytes / whole-run makespan (decimal GB/s).
It is effective interface throughput, not necessarily physical DRAM-bus throughput:
the controller may forward reads or coalesce writes. Official command-service
counters in `dram_stats.yaml` distinguish these cases.
Logical tensor bytes are also reported separately. Request latency starts at
acceptance, not first attempted submission. `input_wait_cycles` counts idle-compute
cycles with an active input transfer; `capacity_blocked_cycles` counts cycles with
a dependency-ready admission blocked by SRAM. These counters can overlap and are
not an additive decomposition of makespan. `samples_per_second` counts completed
samples over the full multi-step run; repeated steps are not extra samples.
Microbatch completion timestamps support throughput and
latency analysis; there is no claim that a short run has reached steady state.

## Modeling Boundaries

1. This is a timing simulator; it does not calculate or check numeric tensor values.
2. There is no persistent input/weight cache in v0.1. Each K-slice loads its A/B
   operands; only output partial sums are retained across K slices. Do not infer
   ideal batch-level weight reuse from these examples. SRAM capacity and prefetch
   still affect overlap, but exploring persistent weight reuse requires adding an
   explicit residency policy.
3. Row-major byte addresses and cache-line channel interleaving are used. Transfers
   are aligned and coalesced within one tile. Repeated transfers across tiles remain.
   Partial writes cost full transactions; byte-mask/RMW behavior is not modeled.
4. Cross-core intermediates go through shared DRAM. There is no direct forwarding,
   NoC, PCIe, ICI, cache coherence, or explicit interconnect latency.
5. Writes follow the pinned upstream callback semantics: callback at final write
   command issue. This releases the producer's output buffer and permits dependent
   consumer requests; it is not a claim of physical write recovery completion.
   Makespan ends when task dependencies and all callbacks complete.
6. Each compute K-slice pays NeuSim's compute startup/padding estimate. Its original
   HBM latency, bandwidth-based time, SRAM time, and `max(compute,memory)` result
   are **not** added again. Per-slice startup can be conservative compared to a
   continuously executing large kernel.
7. Current execution is cycle-stepped on the DRAM side and scans planned tiles.
   This first release prioritizes correctness and inspectability over million-point
   DSE throughput. Small/representative workloads are the intended starting point.

## Tests and Reproducibility

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest -m integration -q
```

Unit tests use a deterministic memory stub **only to test scheduler invariants**.
Runtime/CLI always uses the real Ramulator bridge. Integration tests run DDR4,
HBM2, HBM3, and online two-core DDR4 / four-core HBM pipelines. Other tests exercise the actual NeuSim
compute path, tail shapes, K reduction, step/stage dependencies, resource limits,
queue rejection, out-of-order completions, and propagation of memory delay into
future request times. Integration tests skip if the native bridge is absent;
`scripts/setup.sh` builds it before testing.

The source dependencies are fixed by Git submodule commits:

| Upstream | Commit |
| --- | --- |
| NeuSim | `72056f1e173538a64ff36fa987c1eb1112a670d7` |
| Ramulator2 (upstream 2.1) | `7bead6cb7c4b03e095c6d4769ae3181a8e0bf6da` |

Neither source checkout is edited. CMake generates a build-local portability overlay
for `ramulator/base/param.h`, changing `config[name].as<T>()` to
`config[name].template as<T>()`, which AppleClang requires. The semantic model is
unchanged. Upstream FetchContent resolves fmt `10.2.1` and yaml-cpp `0.9.0`.

Integration code is Apache-2.0. Upstream sources retain their own licenses. See
`NOTICE` and each submodule's license/citation instructions. This project does not
claim upstream authorship, endorsement, RTL validation, or silicon calibration.

## Linux Servers

Use a project-local virtual environment; no root privileges are needed. On RHEL 9,
`scripts/setup.sh` automatically chooses GCC Toolset 14 if present. If Python 3.12
is not on the non-interactive SSH PATH, pass its absolute path:

```bash
PYTHON=/path/to/python3.12 BUILD_JOBS=4 ./scripts/setup.sh
```

Keep build directories and virtual environments local to each machine. Transfer
source with Git/submodules, not macOS binaries or `.venv` directories. Validation
results and their environment are recorded in [docs/VALIDATION.md](docs/VALIDATION.md).

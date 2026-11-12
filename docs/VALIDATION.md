# Linux Validation

Validation date: 2026-11-12. These results establish executable integration and
scheduler invariants, **not calibrated hardware accuracy** or expected peak DRAM
bandwidth. The examples are small correctness workloads, not complete model runs.

## Environment

- Linux x86_64, RHEL 9 family, kernel `5.14.0-503.33.1.el9_5.x86_64`, glibc 2.34.
- CPython 3.12.15 in a project-local virtual environment.
- GCC Toolset 14, G++ 14.2.1; CMake 3.31.6; Release native build, four build jobs.
- NeuSim: `72056f1e173538a64ff36fa987c1eb1112a670d7`.
- Ramulator2: `7bead6cb7c4b03e095c6d4769ae3181a8e0bf6da`.
- Official upstream source checkouts are unmodified. See the README for the
  build-local C++ portability overlay and the external multicore frontend adapter.

## Tests

```bash
.venv/bin/python -m pytest -q
```

Result: **27 passed**, including native integration tests (none skipped).

Coverage includes:

- Real DDR4, HBM2, and HBM3 read/write completions and statistics.
- Two-core DDR4 and four-core HBM pipelines, with per-core controller statistics.
- Every input completion precedes tile compute; output requests follow compute.
- Compute intervals on the same core do not overlap.
- SRAM/outstanding limits, K-slice accumulator lifetime, and tight-SRAM progress.
- Producer-write dependencies, queue backpressure, and out-of-order completions.
- Increased memory delay shifts future requests in scheduler unit tests.
- Real DRAM outstanding-window sensitivity and invalid-source rejection.
- Actual NeuSim compute calls, tail tiles, microbatches, and repeated-step plans.

The deterministic backend exists only in scheduler unit tests. All CLI examples
and native integration tests use the official Ramulator memory implementation.

## Example Results

```bash
./scripts/run.sh examples/two_core_pipeline.json -o results/pipeline --request-trace
./scripts/run.sh examples/four_core_hbm.json -o results/hbm --request-trace
./scripts/run.sh examples/dit_like_batch.json -o results/dit --request-trace
```

All three commands completed successfully:

| Example | Tiles | DRAM cycles | Makespan (ns) | Completed requests | Interface GB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Two-core DDR4 pipeline | 48 | 52,087 | 43,388.471 | 10,752 | 15.860 |
| Four-core HBM2 pipeline | 128 | 27,040 | 27,040.000 | 57,344 | 67.863 |
| DiT-like GEMM chain | 384 | 109,189 | 109,189.000 | 165,888 | 48.617 |

Interface GB/s is accepted transaction bytes divided by total simulated time,
not a measured physical bus rate. Forwarding/coalescing and workload shape matter.
The DiT-like example has repeated expand/contract GEMMs, not full DiT attention,
normalization, residuals, or conditioning. Persistent weight caching is absent.

Each run emits configuration snapshots, upstream/source revisions, tile timing,
optional transaction traces, and official DRAM statistics. Large generated traces
are deliberately excluded from Git. GitHub Actions independently builds/tests on
Ubuntu; the Linux results above do not imply a particular CI run has passed.

## Implementation Audit: 2026-11-13

The same Linux environment rebuilt the bridge after the implementation audit.
Result: **55 passed**, with no skipped tests; 22 tests use the real native backend.
The added scheduler regression also sweeps 50 seeded tight-SRAM GEMM-chain plans.
An exploratory 150-plan seeded sweep completed separately before that regression
was added. These are software consistency checks, not hardware calibration.

New coverage checks exact large-address arithmetic, DRAM capacity and integer
boundaries, safe use-after-close errors, stale-output removal, failed-rerun
diagnostics, and invalid accumulator/channel/core settings. See `STATUS.md` for
the full findings and remaining model limits. Raw reports and test output are
removed after validation; CI no longer persists example reports as artifacts.

## Pi0.5 Stream: 2026-11-13

Implementation revision: `e0c586078af3c6d9562d74859ee6ff3057ce4269`.
All new compute/test/experiment execution below was submitted through Slurm.
No other model is part of this experimental campaign. Older schema-1 fixtures
remain regression tests, not additional research workloads.

- Slurm validation job `2016053`: **80 passed**, no skipped tests, then a successful
  real-HBM3 tiny pi0.5 pipeline and full-shape graph description.
- GitHub Actions run `43291602700`: successful independent build, all tests, legacy
  regression examples, tiny pi0.5 stream and full-shape description.
- Coverage adds physical view/transpose/concatenated-KV addresses, scratch liveness,
  slot retirement, tail-batch drain, cache traffic, backpressure, microbatch/stage
  order, component scope tags, stale-output cleanup and whole-run accounting.
- Tick-by-tick/full-core polling and native-idle/active-core polling produce exactly
  equal timestamps, traffic counters and request records with real Ramulator.

### Complete Component Run

Job `2016078`, `dev` partition, node `deep-space`, 1 CPU, 8 GiB reservation,
15-minute limit. Whole process peak RSS was 70,576 KiB. Source and both upstream
checkouts were clean at run start.

Reproduction from the repo root, inside the same Slurm resource envelope:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m scripts.profile_pi05 --component dit --one-block --complete \
  --output results/dit-block-e0c5860
```

This is a **real-width, one-layer, one-step expert component**, including action/
time projections, adaptive conditioning, dense attention, final projection and
Euler update. Prefix KV is prepopulated. It has 32 SA cores, batch/in-flight limit
2, warmup 2 and measurement 2. It is **not complete pi0.5, not a steady-state
performance claim, and not calibration**.

| Item | Result |
| --- | ---: |
| Completed simulation ticks | 2,243,727 |
| Accepted/completed DRAM transactions | 5,303,040 |
| Maximum active tiles | 64 |
| Wall time | 192.674 s |
| Address-space footprint | 2,165,653,504 bytes |

Reproducibility hashes:

```text
integration source  9d2f222506577d14ba2130937383bf90e99715e163d12d34574fb0d1eb3ef5e1
input configuration 3595d0fa31b714b13e3f889dbbf92d971fd22c1ec484611f5672f257a6245b19
resolved DRAM       878b59ed68ef2e087e0d90c626e7fc586b75f2607a31cba20556daf58ddf8417
```

Raw output stays in the ignored server results directory. No transaction trace was
enabled, and no generated results or profiler dumps are committed.

### Cost Profiles And Readiness

Earlier development jobs `2015916_0` and `2015916_1` each stopped at 200,000 ticks,
on the full-size prefix/DiT graphs respectively. They were explicitly incomplete
profiles, not model latency measurements. A Slurm cProfile job (`2015980`) localized
substantial cost to Python/ctypes request injection, including rejected retries.
Short profiles must not be extrapolated directly to whole-model runtime.

Full-shape graph sums at the implementation revision are 4,635,481,079,808 prefix
GEMM FLOPs/request and 7,908,950,016 expert GEMM FLOPs/step/request. Vector estimates
are separate. KV storage is 17,842,176 bytes/request. The full vocabulary table
occupies capacity but only selected rows are read; total allocated model-weight
bytes must not be confused with per-request weight traffic.

Gate A is implemented and tested. Gate B has profiles and a complete reduced-depth
component, **but no complete full-size pi0.5 stream yet**. Fair resource-matched
baselines, the 48-candidate pilot and large DSE are not yet completed. See
`EXPERIMENT_PLAN.md`. The next runtime priority is native/batched request driving
with exact semantic equivalence, followed by full-stream and baseline gates.

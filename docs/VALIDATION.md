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

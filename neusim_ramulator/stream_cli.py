"""Separate schema-v2 entry point; the legacy GEMM-chain CLI stays unchanged."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import time

from .cli import revision, source_dirty
from .dram import Ramulator, export_config
from .pi05 import Pi05
from .stream import StreamSimulator, validate_stream
from .upstream import ROOT

ARTIFACTS = ("config.json", "summary.json", "error.json", "dram.yaml", "dram_stats.yaml", "requests.csv")


def source_digest():
    digest = hashlib.sha256()
    paths = sorted([*(ROOT / "neusim_ramulator").glob("*.py"), *(ROOT / "native").glob("*.cpp"),
                    ROOT / "CMakeLists.txt", ROOT / "pyproject.toml"])
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def run(config_path, output, trace=False, describe_only=False):
    config_text = Path(config_path).read_text()
    if describe_only:
        config = json.loads(config_text)
        shape = validate_stream(config)
        return Pi05(shape, config["workload"]["max_inflight"]).describe()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACTS:
        (output / name).unlink(missing_ok=True)
    start = time.monotonic()
    trace_file = None
    try:
        config = json.loads(config_text)
        validate_stream(config)
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        sources = {"integration": ROOT, "NeuSim": ROOT / "third_party/NeuSim",
                   "ramulator2": ROOT / "third_party/ramulator2"}
        revisions = {name: revision(path) for name, path in sources.items()}
        dirty = {name: source_dirty(path) for name, path in sources.items()}
        digest = source_digest()
        dram_path = output / "dram.yaml"
        export_config(config.get("dram", {}), dram_path, sum(config["workload"]["stage_cores"]))
        sink = None
        if trace:
            trace_file = (output / "requests.csv").open("w", newline="")
            writer = csv.DictWriter(trace_file, fieldnames=["id", "core", "stage", "write", "address",
                                                          "accepted_cycle", "completion_cycle"])
            writer.writeheader()
            sink = writer.writerow
        with Ramulator(dram_path) as memory:
            engine = StreamSimulator(config, memory, sink)
            try:
                report = engine.run()
            finally:
                memory.finalize(output / "dram_stats.yaml")
        if source_digest() != digest:
            raise RuntimeError("Integration sources changed during the run; discard mixed-version results")
        report["revisions"], report["source_dirty"] = revisions, dirty
        report["dram_config_sha256"] = hashlib.sha256(dram_path.read_bytes()).hexdigest()
        report["integration_source_sha256"] = digest
        report["input_config_sha256"] = hashlib.sha256((output / "config.json").read_bytes()).hexdigest()
        report["slurm"] = {key: os.environ[key] for key in (
            "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_CPUS_PER_TASK",
            "SLURM_MEM_PER_NODE", "SLURM_JOB_PARTITION", "SLURMD_NODENAME") if key in os.environ}
        report["wall_seconds"] = time.monotonic() - start
        report["host_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    except Exception as error:
        (output / "error.json").write_text(json.dumps({"error": str(error),
            "wall_seconds": time.monotonic() - start}, indent=2) + "\n")
        raise
    finally:
        if trace_file:
            trace_file.close()


def main():
    parser = argparse.ArgumentParser(description="Online pi0.5 shape/traffic request stream")
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", "-o", type=Path, default=Path("results/pi05"))
    parser.add_argument("--request-trace", action="store_true")
    parser.add_argument("--describe-only", action="store_true")
    args = parser.parse_args()
    report = run(args.config, args.output, args.request_trace, args.describe_only)
    print(json.dumps(report if args.describe_only or not report.get("complete", True) else {
        "measurement": report["measurement"], "wall_seconds": report["wall_seconds"],
        "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()

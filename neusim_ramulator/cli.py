"""Command-line interface, reproducibility metadata, and machine-readable reports."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

from .dram import Ramulator, export_config
from .scheduler import Scheduler
from .upstream import ROOT
from .workload import build_plan

ARTIFACTS = ("summary.json", "error.json", "config.json", "dram.yaml", "tiles.csv",
             "requests.csv", "dram_stats.yaml")


def revision(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def source_dirty(path: Path) -> bool | None:
    try:
        return bool(subprocess.check_output(
            ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no",
             "--ignore-submodules=untracked"], text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def run(config_path: Path, output: Path, request_trace: bool = False, library=None) -> dict:
    # Read before cleanup so the saved config can itself be used as the input.
    config_text = config_path.read_text()
    output.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACTS:
        (output / name).unlink(missing_ok=True)
    trace_file = None
    try:
        config = json.loads(config_text)
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        tiles, metadata = build_plan(config)
        dram_config = output / "dram.yaml"
        export_config(config.get("dram", {}), dram_config, num_cores=len(config["cores"]))
        sink = None
        if request_trace:
            trace_file = (output / "requests.csv").open("w", newline="")
            writer = csv.DictWriter(trace_file, fieldnames=[
                "id", "core", "tile", "write", "address", "accepted_cycle", "completion_cycle"])
            writer.writeheader()
            sink = writer.writerow
        with Ramulator(dram_config, library) as memory:
            if metadata["address_space_bytes"] > memory.capacity_bytes:
                raise ValueError(
                    f"Workload address space ({metadata['address_space_bytes']} bytes) exceeds "
                    f"DRAM capacity ({memory.capacity_bytes} bytes); address reuse is not modeled")
            simulation = config.get("simulation", {})
            scheduler = Scheduler(tiles, config["cores"], memory,
                                  issue_width=simulation.get("issue_width", 4),
                                  max_cycles=simulation.get("max_cycles", 10000000),
                                  request_sink=sink)
            try:
                report = scheduler.run()
            finally:
                memory.finalize(output / "dram_stats.yaml")
        with (output / "tiles.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(report["tiles"][0]))
            writer.writeheader()
            writer.writerows(report.pop("tiles"))
        report["workload"] = metadata
        report["dram_capacity_bytes"] = memory.capacity_bytes
        sources = {"integration": ROOT, "NeuSim": ROOT / "third_party/NeuSim",
                   "ramulator2": ROOT / "third_party/ramulator2"}
        report["revisions"] = {name: revision(path) for name, path in sources.items()}
        report["source_dirty"] = {name: source_dirty(path) for name, path in sources.items()}
        report["model"] = {
            "compute": "official NeuSim compute functions, per GEMM K slice",
            "memory": "one live official Ramulator instance shared by all cores",
            "sram": "capacity-only; input prefetch plus pinned partial-sum accumulators",
            "reuse": "partial sums across K slices; no persistent input/weight cache",
            "inter_core": "producer write callback before dependent consumer reads",
            "arbitration": "round-robin injection across cores and active transfer streams",
            "write_completion": "upstream callback at terminal write command issue",
        }
        work = config["workload"]
        report["samples_per_second"] = work["batch_size"] * 1e9 / report["makespan_ns"]
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    except Exception as error:
        (output / "error.json").write_text(json.dumps({"error": str(error)}, indent=2) + "\n")
        raise
    finally:
        if trace_file:
            trace_file.close()


def main():
    parser = argparse.ArgumentParser(description="Online NeuSim + Ramulator tile co-simulation")
    parser.add_argument("config", type=Path, help="JSON workload and hardware configuration")
    parser.add_argument("--output", "-o", type=Path, default=Path("results/run"))
    parser.add_argument("--request-trace", action="store_true", help="Save every DRAM completion")
    parser.add_argument("--library", type=Path, help="Override the native bridge library path")
    args = parser.parse_args()
    try:
        report = run(args.config, args.output, args.request_trace, args.library)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"Simulation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
    print(f"Completed {report['workload']['tiles']} tiles in {report['cycles']} DRAM cycles")
    print(f"Makespan: {report['makespan_ns']:.3f} ns; "
          f"interface throughput: {report['total_bandwidth_GBps']:.3f} GB/s")
    print(f"Reports: {args.output.resolve()}")


if __name__ == "__main__":
    main()

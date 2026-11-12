"""Generate a bounded profile from a source config without claiming full timing."""

import argparse
import json
from pathlib import Path

from neusim_ramulator.stream_cli import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("examples/pi05_pilot.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=200000)
    parser.add_argument("--component", choices=("prefix", "dit"), default="prefix")
    parser.add_argument("--one-block", action="store_true")
    parser.add_argument("--complete", action="store_true")
    args = parser.parse_args()
    if args.cycles <= 0:
        parser.error("--cycles must be positive")
    if args.one_block and args.component != "dit":
        parser.error("--one-block requires --component dit")
    config = json.loads(args.config.read_text())
    if args.component == "dit":
        config["workload"]["kind"] = "pi05_dit"
        config["workload"]["stage_cores"] = [sum(config["workload"]["stage_cores"])]
    if args.one_block:
        config["workload"]["shape"].update(layers=1, steps=1)
        config["workload"].update(max_inflight=2, microbatch=2, warmup_requests=2, measure_requests=2)
    config["simulation"]["profile_cycles"] = 0 if args.complete else args.cycles
    args.output.mkdir(parents=True, exist_ok=True)
    generated = args.output / "generated_config.json"
    generated.write_text(json.dumps(config, indent=2) + "\n")
    report = run(generated, args.output)
    print(json.dumps({key: report[key] for key in (
        "complete", "cycles", "dram_transactions", "wall_seconds", "host_peak_rss_kib")}, indent=2))


if __name__ == "__main__":
    main()

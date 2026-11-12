"""Real official Ramulator DRAM, not the deterministic unit-test memory."""

import csv
import sys

import pytest

from neusim_ramulator.cli import run
from neusim_ramulator.dram import Ramulator, export_config
from neusim_ramulator.upstream import ROOT

LIBRARY = ROOT / "build" / ("libneusim_ramulator_bridge.dylib" if sys.platform == "darwin"
                            else "libneusim_ramulator_bridge.so")
pytestmark = [pytest.mark.integration,
              pytest.mark.skipif(not LIBRARY.exists(), reason="Build the native bridge first")]


@pytest.mark.parametrize("standard", ["DDR4", "HBM2", "HBM3"])
def test_real_memory_completes_read_and_write(standard, tmp_path):
    config = tmp_path / "dram.yaml"
    export_config({"standard": standard, "channels": 2}, config, num_cores=2)
    with Ramulator(config) as memory:
        assert memory.tx_bytes > 0 and memory.tck_ns > 0
        assert memory.send(0, False, 0, 0)
        assert memory.send(1, True, 4096, 1)
        completed = set()
        for _ in range(20000):
            completed.update(rid for rid, _ in memory.completions())
            if completed == {0, 1}:
                break
            memory.tick()
        assert completed == {0, 1}
        memory.finalize(tmp_path / "stats.yaml")
    assert (tmp_path / "stats.yaml").stat().st_size > 0


@pytest.mark.parametrize("example", ["two_core_pipeline.json", "four_core_hbm.json"])
def test_real_online_pipeline_and_reports(tmp_path, example):
    report = run(ROOT / "examples" / example, tmp_path, request_trace=True)
    assert report["backend"] == "official Ramulator2 (online)"
    assert report["requests_completed"] > 0
    assert all(c["peak_sram_bytes"] <= c["sram_capacity_bytes"] for c in report["cores"])
    assert all(c["compute_busy_cycles"] > 0 for c in report["cores"])
    assert report["stages"][1]["first_admitted_cycle"] < report["stages"][0]["last_done_cycle"]
    for name in ("summary.json", "tiles.csv", "requests.csv", "dram_stats.yaml", "dram.yaml"):
        assert (tmp_path / name).stat().st_size > 0
    with (tmp_path / "tiles.csv").open() as file:
        tiles = {row["tile"]: row for row in csv.DictReader(file)}
    with (tmp_path / "requests.csv").open() as file:
        requests = list(csv.DictReader(file))
    assert len(requests) == report["requests_completed"]
    for request in requests:
        task = tiles[request["tile"]]
        if request["write"] == "False":
            assert int(request["completion_cycle"]) <= int(task["compute_start_cycle"])
        else:
            assert int(request["accepted_cycle"]) >= int(task["compute_end_cycle"])
    stats_text = (tmp_path / "dram_stats.yaml").read_text()
    assert f"read_row_hits_core_{len(report['cores']) - 1}:" in stats_text
    for core in range(len(report["cores"])):
        intervals = sorted((int(t["compute_start_cycle"]), int(t["compute_end_cycle"]))
                           for t in tiles.values() if int(t["core"]) == core)
        assert all(left[1] <= right[0] for left, right in zip(intervals, intervals[1:]))


def test_real_outstanding_window_changes_memory_service(tmp_path):
    config = tmp_path / "dram.yaml"
    export_config({"standard": "DDR4", "channels": 1}, config)

    def measure(window):
        with Ramulator(config) as memory:
            sent, finished = 0, 0
            for cycle in range(100000):
                finished += len(memory.completions())
                if finished == 128:
                    return cycle
                while sent < 128 and sent - finished < window:
                    if not memory.send(sent, False, sent * memory.tx_bytes, 0):
                        break
                    sent += 1
                memory.tick()
            pytest.fail("Memory streaming test timed out")

    assert measure(32) < measure(1)


def test_invalid_dram_config_raises_not_crashes(tmp_path):
    config = tmp_path / "invalid.yaml"
    config.write_text("frontend: {impl: DoesNotExist}\nmemory_system: {impl: DoesNotExist}\n")
    with pytest.raises(RuntimeError):
        Ramulator(config)


def test_source_ids_are_bounds_checked(tmp_path):
    config = tmp_path / "dram.yaml"
    export_config({"standard": "DDR4"}, config)
    with Ramulator(config) as memory:
        with pytest.raises(RuntimeError, match="Source ID"):
            memory.send(0, False, 0, 1)

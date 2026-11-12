"""Real official Ramulator DRAM, not the deterministic unit-test memory."""

import csv
import json
import sys

import pytest

from neusim_ramulator.cli import ARTIFACTS, run
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


@pytest.mark.parametrize("standard,capacity", [
    ("DDR4", 8 * 2**30), ("HBM2", 512 * 2**20), ("HBM3", 2**30),
])
def test_capacity_matches_address_mapper_and_rejects_overflow(standard, capacity, tmp_path):
    config = tmp_path / "dram.yaml"
    export_config({"standard": standard}, config)
    with Ramulator(config) as memory:
        assert memory.capacity_bytes == capacity
        assert memory.send(0, False, capacity - memory.tx_bytes, 0)
        with pytest.raises(RuntimeError, match="DRAM address capacity"):
            memory.send(1, False, capacity, 0)


@pytest.mark.parametrize("field,value", [
    ("address", 2**64), ("address", -1), ("address", True),
    ("request_id", -1), ("request_id", 2**64), ("core", 2**32),
])
def test_python_integers_do_not_silently_wrap(field, value, tmp_path):
    config = tmp_path / "dram.yaml"
    export_config({"standard": "DDR4"}, config)
    with Ramulator(config) as memory:
        request = dict(request_id=0, write=False, address=0, core=0)
        request[field] = value
        with pytest.raises(ValueError, match=field):
            memory.send(**request)


def test_closed_instance_raises_instead_of_crashing(tmp_path):
    config = tmp_path / "dram.yaml"
    export_config({"standard": "DDR4"}, config)
    memory = Ramulator(config)
    memory.close()
    memory.close()
    operations = [memory.tick, memory.completions, memory.__enter__,
                  lambda: memory.send(0, False, 0, 0),
                  lambda: memory.finalize(tmp_path / "stats.yaml")]
    for operation in operations:
        with pytest.raises(RuntimeError, match="closed"):
            operation()


def test_real_pi05_stream_and_native_idle_equivalence(tmp_path):
    from neusim_ramulator.stream import StreamSimulator
    config = json.loads((ROOT / "examples/pi05_tiny.json").read_text())
    config["simulation"]["operator_overhead_ns"] = 100
    dram_path = tmp_path / "dram.yaml"
    export_config(config["dram"], dram_path, sum(config["workload"]["stage_cores"]))
    reports = []
    for batching in (False, True):
        config["simulation"]["active_core_polling"] = batching
        with Ramulator(dram_path) as memory:
            if not batching:
                memory.advance = None
            reports.append(StreamSimulator(config, memory).run())
    assert reports[1]["native_idle_ticks"] > 0
    for report in reports:
        report.pop("native_idle_ticks")
    assert reports[0] == reports[1]


def test_native_advance_argument_validation(tmp_path):
    path = tmp_path / "dram.yaml"
    export_config({"standard": "HBM3"}, path)
    with Ramulator(path) as memory:
        for count in (0, -1, True, 2**64):
            with pytest.raises(ValueError):
                memory.advance(count)
        memory.advance(100)
        assert memory.send(0, False, 0, 0)
        memory.advance(2000)
        assert [request for request, _ in memory.completions()] == [0]


def test_rerun_removes_stale_trace_and_preserves_unrelated_files(tmp_path):
    for name in ARTIFACTS:
        (tmp_path / name).write_text("stale")
    (tmp_path / "notes.txt").write_text("keep")
    run(ROOT / "examples/two_core_pipeline.json", tmp_path)
    assert not (tmp_path / "requests.csv").exists()
    assert not (tmp_path / "error.json").exists()
    assert (tmp_path / "notes.txt").read_text() == "keep"
    assert (tmp_path / "tiles.csv").read_text() != "stale"
    # A saved input snapshot is also a valid rerun input.
    run(tmp_path / "config.json", tmp_path)


@pytest.mark.parametrize("invalid_json", [False, True])
def test_failed_rerun_does_not_leave_success_from_previous_run(invalid_json, tmp_path):
    for name in ARTIFACTS:
        (tmp_path / name).write_text("stale")
    source = tmp_path / "input.json"
    source.write_text("{" if invalid_json else '{"schema_version": 0}')
    with pytest.raises(ValueError):
        run(source, tmp_path)
    assert "error" in json.loads((tmp_path / "error.json").read_text())
    for name in ("summary.json", "tiles.csv", "requests.csv", "dram_stats.yaml", "dram.yaml"):
        assert not (tmp_path / name).exists()


def test_cli_rejects_workload_exceeding_memory_capacity(monkeypatch, tmp_path):
    from neusim_ramulator import cli

    build = cli.build_plan

    def oversized(config):
        tiles, metadata = build(config)
        metadata["address_space_bytes"] = 2**50
        return tiles, metadata

    monkeypatch.setattr(cli, "build_plan", oversized)
    with pytest.raises(ValueError, match="exceeds DRAM capacity"):
        run(ROOT / "examples/two_core_pipeline.json", tmp_path)
    assert not (tmp_path / "summary.json").exists()
    assert (tmp_path / "error.json").exists()

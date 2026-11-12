"""Real official Ramulator DRAM, not the deterministic unit-test memory."""

import json
from pathlib import Path
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
    export_config({"standard": standard, "channels": 2}, config)
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


def test_real_online_pipeline_and_reports(tmp_path):
    report = run(ROOT / "examples/two_core_pipeline.json", tmp_path, request_trace=True)
    assert report["backend"] == "official Ramulator2 (online)"
    assert report["requests_completed"] > 0
    assert all(c["peak_sram_bytes"] <= c["sram_capacity_bytes"] for c in report["cores"])
    assert all(c["compute_busy_cycles"] > 0 for c in report["cores"])
    assert report["stages"][1]["first_admitted_cycle"] < report["stages"][0]["last_done_cycle"]
    for name in ("summary.json", "tiles.csv", "requests.csv", "dram_stats.yaml", "dram.yaml"):
        assert (tmp_path / name).stat().st_size > 0


def test_invalid_dram_config_raises_not_crashes(tmp_path):
    config = tmp_path / "invalid.yaml"
    config.write_text("frontend: {impl: DoesNotExist}\nmemory_system: {impl: DoesNotExist}\n")
    with pytest.raises(RuntimeError):
        Ramulator(config)

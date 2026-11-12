from copy import deepcopy
import json

import pytest

from neusim_ramulator.upstream import ROOT
from neusim_ramulator.workload import Region, build_plan


def config():
    return json.loads((ROOT / "examples/two_core_pipeline.json").read_text())


def test_byte_ranges_and_transaction_coalescing():
    region = Region(0, 2, 3, 8, 2)
    assert region.size == 12
    assert region.transactions(8) == [0, 8]
    assert Region(63, 1, 2, 4, 2).transactions(64) == [0, 64]


def test_transaction_alignment_is_exact_above_float_precision():
    base = 2**60
    assert Region(base, 1, 1, 1, 1).transactions(64) == [base]
    assert Region(base + 63, 1, 2, 2, 1).transactions(64) == [base, base + 64]


def test_k_partial_sums_and_cross_stage_dependencies():
    tiles, metadata = build_plan(config())
    assert metadata["microbatches"] == 3
    first = tiles[0]
    second = tiles[1]
    assert not first.last_k and not first.writes
    assert second.last_k and second.compute_after == (first.id,)
    assert second.group == first.group and second.writes
    consumers = [t for t in tiles if t.stage == "ffn"]
    assert all(t.load_after for t in consumers)
    lookup = {t.id: t for t in tiles}
    assert all(lookup[d].last_k for t in consumers for d in t.load_after)


def test_tail_shapes_and_partial_microbatch():
    cfg = config()
    cfg["workload"].update(batch_size=3, microbatch_size=2, rows_per_sample=17, tile=[16, 30, 50])
    tiles, metadata = build_plan(cfg)
    assert metadata["microbatches"] == 2
    assert any(t.reads[0].rows == 1 for t in tiles)
    assert any(t.reads[0].columns == 28 for t in tiles)


def test_steps_depend_on_prior_step_outputs():
    cfg = config()
    cfg["workload"]["steps"] = 2
    tiles, _ = build_plan(cfg)
    lookup = {t.id: t for t in tiles}
    for tile in tiles:
        if tile.step == 1 and tile.stage == "projection":
            assert tile.load_after
            assert all(lookup[d].step == 0 and lookup[d].stage == "ffn" for d in tile.load_after)


@pytest.mark.parametrize("mutate", [
    lambda c: c["workload"].update(batch_size=0),
    lambda c: c["workload"].update(tile=[1, 0, 1]),
    lambda c: c["workload"]["stages"][1].update(k=127),
    lambda c: c["cores"][0].update(sram_bytes=1),
    lambda c: c["cores"][0].update(max_outstanding=-1),
    lambda c: c["workload"].update(accumulator_bytes=1),
])
def test_invalid_configs_fail(mutate):
    cfg = deepcopy(config())
    mutate(cfg)
    with pytest.raises(ValueError):
        build_plan(cfg)

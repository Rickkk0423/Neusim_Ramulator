import json

import pytest

from neusim_ramulator.pi05 import Allocator, Pi05, Rows, Tensor, View, Shape, place_scratch
from neusim_ramulator.stream import StreamSimulator, validate_stream
from neusim_ramulator.upstream import ROOT


def config():
    return json.loads((ROOT / "examples/pi05_tiny.json").read_text())


def addresses(regions, tx=1):
    return {a for r in regions for a in r.transactions(tx)}


def test_transposed_submatrix_and_concatenated_kv_addresses():
    a = Tensor("a", 3, 4, base=0)
    b = Tensor("b", 2, 4, base=128)
    joined = Rows((a.view(), b.view()))
    expected = {16, 17, 18, 19, 20, 21, 22, 23, 128, 129, 130, 131, 132, 133, 134, 135}
    assert addresses(joined.regions(2, 0, 2, 4)) == expected
    assert addresses(joined.transpose().regions(0, 2, 4, 2)) == expected
    v = View(a, 1, 1, 2, 2)
    assert addresses(v.regions(0, 0, 2, 1)) == addresses(v.transpose().regions(0, 0, 1, 2))
    with pytest.raises(ValueError, match="bounds"):
        joined.regions(4, 0, 2, 4)


def test_official_shape_counts_and_shared_weights():
    model = Pi05(Shape(), 2)
    desc = model.describe()
    before = dict(model.weights)
    model.graph([0, 1], first_step=5, steps=2)
    assert model.weights == before
    assert desc["prefix_tokens"] == 968
    assert desc["kv_bytes_per_request"] == 17842176
    assert desc["expert_weight_bytes"] > 800_000_000
    assert desc["prefix_gemm_flops"] > 3e12
    assert desc["expert_gemm_flops_per_step"] > 5e9


def test_scratch_last_use_is_bounded_and_preserves_live_tensors():
    model = Pi05(Shape.parse(config()["workload"]["shape"]), 4)
    model.describe()
    ops = model.graph([0, 1], steps=2)
    external = {id(t) for op in ops for view in (*op.inputs, op.output)
                for t in view.tensors if t.base is not None}
    base = model.allocator.end
    end = place_scratch(ops, base)
    uses = {}
    for i, op in enumerate(ops):
        for view in (*op.inputs, op.output):
            for t in view.tensors:
                if id(t) not in external:
                    uses.setdefault(id(t), [t, i, i])[2] = i
    tensors = list(uses.values())
    for index, (a, first, last) in enumerate(tensors):
        assert base <= a.base < a.base + a.size <= end
        for b, bfirst, blast in tensors[index + 1:]:
            if first <= blast and bfirst <= last:
                assert a.base + a.size <= b.base or b.base + b.size <= a.base
    assert end - base < sum(t.size + 4095 for t, _, _ in tensors)


@pytest.mark.parametrize("mutate", [
    lambda c: c["workload"].update(max_inflight=1),
    lambda c: c["workload"].update(stage_cores=[1, 1, 1, 1]),
    lambda c: c["hardware"].update(weight_cache_bytes_per_core=8192),
    lambda c: c["hardware"]["chip"].update(num_sa=2),
    lambda c: c["hardware"].update(tile_mnk=[0, 4, 4]),
    lambda c: c["simulation"].update(operator_overhead_ns=-1),
    lambda c: c["hardware"].update(shared_sram_bytes=1000),
    lambda c: c["simulation"].update(active_core_polling="false"),
])
def test_invalid_stream_config_rejected(mutate):
    c = config()
    mutate(c)
    with pytest.raises(ValueError):
        validate_stream(c)


class Memory:
    tck_ns = 1.0
    tx_bytes = 32
    capacity_bytes = 2**34

    def __init__(self, latency=2, slots=128):
        self.cycle = 0
        self.latency, self.slots = latency, slots
        self.pending, self.sent = [], []

    def send(self, request, write, address, core):
        assert 0 <= address < self.capacity_bytes
        if len(self.pending) >= self.slots:
            return False
        self.pending.append((self.cycle + self.latency, request))
        self.sent.append((self.cycle, request, write, address, core))
        return True

    def completions(self):
        done = [(r, t) for t, r in self.pending if t <= self.cycle]
        self.pending = [(t, r) for t, r in self.pending if t > self.cycle]
        return list(reversed(done))

    def tick(self):
        self.cycle += 1


def simulate(c=None, latency=2, slots=128):
    memory = Memory(latency, slots)
    engine = StreamSimulator(c or config(), memory)
    return engine.run(), engine, memory


def test_online_pipeline_request_order_and_slot_reuse():
    report, engine, _ = simulate()
    assert report["measurement"]["measured_requests"] == 4
    assert report["measurement"]["mean_batch_wait_ns"] <= report["measurement"]["mean_stage_queue_wait_ns"]
    assert report["peak_active_tiles"] <= 3 * 2
    assert all(c["peak_sram_bytes"] <= c["sram_capacity_bytes"] for c in report["cores"])
    assert report["dram_transaction_latency"]["transactions"] == report["dram_transactions"]
    assert all(s["transactions"] > 0 and s["p95_ns"] > 0 for s in report["stage_dram_read_latency"])
    assert all(sum(c.get(role + "_read_bytes", 0) for role in ("weight", "prefix_kv", "activation"))
               == c["read_bytes"] for c in report["cores"])
    by_slot = {}
    for r in report["request_records"]:
        assert len(r["stage_start_cycles"]) == len(r["stage_done_cycles"]) == 3
        assert r["admitted_cycle"] <= r["stage_start_cycles"][0]
        assert all(a <= b for a, b in zip(r["stage_done_cycles"], r["stage_start_cycles"][1:]))
        assert r["stage_done_cycles"][-1] == r["done_cycle"]
        previous = by_slot.get(r["slot"])
        if previous:
            assert previous["done_cycle"] <= r["admitted_cycle"]
        by_slot[r["slot"]] = r
    jobs = report["jobs"]
    assert any(a["start_cycle"] < b["done_cycle"] and b["start_cycle"] < a["done_cycle"]
               for a in jobs for b in jobs if a["stage"] != b["stage"])
    assert not engine.pending
    assert any(c["weight_hit_bytes"] > 0 for c in report["cores"])


def test_dram_delay_backpropagates_and_backpressure_is_bounded():
    fast, _, _ = simulate(latency=2)
    slow, _, _ = simulate(latency=20, slots=1)
    assert slow["measurement"]["mean_output_interval_ns"] > fast["measurement"]["mean_output_interval_ns"]
    assert sum(c.get("queue_retries", 0) for c in slow["cores"]) > 0
    assert all(c["peak_outstanding"] <= 16 for c in slow["cores"])


def test_no_cache_and_small_cache_complete_and_change_weight_traffic():
    c = config()
    c["hardware"]["weight_cache_bytes_per_core"] = 0
    uncached, _, _ = simulate(c)
    cached, _, _ = simulate()
    assert sum(x["read_bytes"] for x in cached["cores"]) < sum(x["read_bytes"] for x in uncached["cores"])


def test_longer_stream_reuses_same_address_space_and_tail_batch_drains():
    c = config()
    c["workload"].update(measure_requests=5, max_inflight=3)
    first, _, _ = simulate(c)
    c["workload"]["measure_requests"] = 9
    longer, _, _ = simulate(c)
    assert first["address_space_bytes"] == longer["address_space_bytes"]
    assert longer["measurement"]["measured_requests"] == 9
    assert any(len(job["requests"]) == 1 for job in first["jobs"] if job["stage"] != "prefix")


def test_operator_overhead_is_per_op_and_deterministic():
    c = config()
    original, _, _ = simulate(c)
    c["simulation"]["operator_overhead_ns"] = 100
    overhead, _, _ = simulate(c)
    repeat, _, _ = simulate(c)
    assert overhead == repeat
    assert overhead["cycles"] > original["cycles"]


def test_idle_native_batching_preserves_timing_and_counters():
    c = config()
    c["simulation"]["operator_overhead_ns"] = 100
    reference, _, _ = simulate(c)
    memory = Memory()
    memory.advance = lambda ticks: setattr(memory, "cycle", memory.cycle + ticks)
    optimized = StreamSimulator(c, memory).run()
    assert optimized["native_idle_ticks"] > 0
    reference.pop("native_idle_ticks")
    optimized.pop("native_idle_ticks")
    assert optimized == reference


def test_profile_never_reports_end_to_end_performance():
    c = config()
    c["simulation"]["profile_cycles"] = 100
    report, _, _ = simulate(c)
    assert report["complete"] is False
    assert "measurement" not in report
    assert report["cycles"] == 100


def test_active_core_polling_preserves_arbitration_and_counters():
    c = config()
    c["workload"]["stage_cores"] = [4, 8, 8]
    optimized, _, memory = simulate(c)
    c["simulation"]["active_core_polling"] = False
    reference, _, old_memory = simulate(c)
    assert optimized == reference
    assert memory.sent == old_memory.sent


def test_dit_component_marks_prepopulated_kv_and_omits_prefix():
    c = config()
    c["workload"].update(kind="pi05_dit", stage_cores=[2, 2])
    report, _, _ = simulate(c)
    assert report["scope"] == "pi05_dit"
    assert report["prefix_kv_prepopulated"] is True
    assert {job["stage"] for job in report["jobs"]} == {"dit0", "dit1"}


def test_invalid_dram_capacity_and_tile_sram_fail_early():
    memory = Memory()
    memory.capacity_bytes = 1024
    with pytest.raises(ValueError, match="capacity"):
        StreamSimulator(config(), memory)
    c = config()
    c["hardware"]["tile_mnk"] = [1024, 1024, 1024]
    c["hardware"]["sram_bytes_per_core"] = 64
    c["hardware"]["weight_cache_bytes_per_core"] = 0
    with pytest.raises(ValueError, match="SRAM"):
        simulate(c)


def test_stream_cli_invalid_rerun_clears_stale_success(tmp_path):
    from neusim_ramulator.stream_cli import run
    output = tmp_path / "run"
    output.mkdir()
    (output / "summary.json").write_text('{"complete": true}')
    (output / "requests.csv").write_text("old trace")
    path = tmp_path / "invalid.json"
    path.write_text('{"schema_version": 0}')
    with pytest.raises(ValueError):
        run(path, output)
    assert not (output / "summary.json").exists()
    assert not (output / "requests.csv").exists()
    assert (output / "error.json").exists()


def test_optional_zero_warmup_marks_cold_fill():
    c = config()
    del c["workload"]["warmup_requests"]
    report, _, _ = simulate(c)
    assert report["measurement"]["warmup_requests"] == 0
    assert report["measurement"]["includes_cold_fill"] is True

"""Deterministic unit backend for causality; real DRAM tests are separate."""

import pytest

from neusim_ramulator.scheduler import Scheduler
from neusim_ramulator.workload import Region, Tile


class UnitMemory:
    tck_ns = 1.0
    tx_bytes = 64

    def __init__(self, latency=5, slots=128):
        self.latency, self.slots = latency, slots
        self.cycle = 0
        self.pending = []
        self.sent = []

    def send(self, request_id, write, address, core):
        if len(self.pending) >= self.slots:
            return False
        self.pending.append((self.cycle + self.latency, request_id))
        self.sent.append((self.cycle, request_id, core, write))
        return True

    def tick(self):
        self.cycle += 1

    def completions(self):
        done = [(rid, cycle) for cycle, rid in self.pending if cycle <= self.cycle]
        self.pending = [(cycle, rid) for cycle, rid in self.pending if cycle > self.cycle]
        return list(reversed(done))


def tile(name, core=0, **kwargs):
    values = dict(id=name, core=core, stage=f"stage{core}", microbatch=0, step=0,
                  group=name, reads=(Region(0, 1, 64, 64, 1),),
                  writes=(Region(4096, 1, 64, 64, 1),), load_after=(),
                  compute_after=(), compute_ns=20, accumulator_bytes=64, last_k=True)
    values.update(kwargs)
    return Tile(**values)


def core(capacity=1024, prefetch=2, outstanding=8):
    return {"sram_bytes": capacity, "prefetch_tiles": prefetch, "max_outstanding": outstanding}


def test_shared_timeline_and_producer_write_dependency():
    first = tile("first")
    second = tile("second", 1, load_after=("first",))
    other = tile("other", 1)
    scheduler = Scheduler([first, second, other], [core(), core()], UnitMemory())
    report = scheduler.run()
    tasks = scheduler.lookup
    assert tasks["second"].admitted >= tasks["first"].done
    assert tasks["other"].start < tasks["first"].end
    assert report["requests_completed"] == 6


def test_compute_and_prefetch_overlap_and_bounded_resources():
    scheduler = Scheduler([tile("a"), tile("b")], [core(256)], UnitMemory())
    scheduler.run()
    a, b = scheduler.lookup["a"], scheduler.lookup["b"]
    assert b.ready < a.end
    assert b.start >= a.end
    assert all(c.used == 0 and c.peak <= c.capacity for c in scheduler.cores)


def test_backpressure_retries_and_out_of_order_completions():
    memory = UnitMemory(slots=1)
    scheduler = Scheduler([tile("a"), tile("b", 1)], [core(), core()], memory)
    report = scheduler.run()
    assert sum(c["queue_retries"] for c in report["cores"]) > 0
    assert report["requests_completed"] == len(memory.sent) == 4


def test_memory_delay_changes_future_request_timestamps():
    tiles = [tile("a"), tile("b", 1, load_after=("a",))]
    memories = [UnitMemory(latency=2), UnitMemory(latency=20)]
    runs = [Scheduler(tiles, [core(), core()], memory) for memory in memories]
    for run in runs:
        run.run()
    assert runs[1].lookup["b"].admitted > runs[0].lookup["b"].admitted
    assert memories[1].sent[2][0] > memories[0].sent[2][0]


def test_k_accumulator_retained_and_only_final_k_writes():
    first = tile("k0", group="output", writes=(), last_k=False)
    second = tile("k1", group="output", compute_after=("k0",))
    scheduler = Scheduler([first, second], [core(128)], UnitMemory())
    report = scheduler.run()
    assert scheduler.lookup["k1"].start >= scheduler.lookup["k0"].end
    assert report["requests_completed"] == 3
    assert report["cores"][0]["peak_sram_bytes"] == 128


def test_accumulator_admission_does_not_deadlock():
    # Multiple output groups, tight SRAM, and a smaller tail K slice.
    tasks = []
    for group in ("x", "y", "z"):
        tasks.extend([
            tile(group + "0", group=group, writes=(), last_k=False),
            tile(group + "1", group=group, compute_after=(group + "0",),
                 reads=(Region(0, 1, 32, 32, 1),)),
        ])
    scheduler = Scheduler(tasks, [core(192, prefetch=4)], UnitMemory(), max_cycles=1000)
    assert scheduler.run()["requests_completed"] == 9


def test_outstanding_cap_and_timeout():
    scheduler = Scheduler([tile("a"), tile("b")], [core(outstanding=1)], UnitMemory())
    assert scheduler.run()["cores"][0]["peak_outstanding"] == 1
    with pytest.raises(RuntimeError, match="max_cycles"):
        Scheduler([tile("a")], [core()], UnitMemory(latency=100), max_cycles=5).run()


def test_bad_dependencies_and_insufficient_sram_rejected():
    with pytest.raises(ValueError, match="dependency"):
        Scheduler([tile("a", load_after=("absent",))], [core()], UnitMemory())
    with pytest.raises(ValueError, match="SRAM"):
        Scheduler([tile("a")], [core(1)], UnitMemory())

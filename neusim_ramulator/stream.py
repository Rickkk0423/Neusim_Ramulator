"""Bounded online request/timestep pipeline, with lazy tiles and weight residency."""

from collections import OrderedDict, Counter, deque
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from math import ceil

from .compute import NeuSimCompute
from .pi05 import Pi05, Shape, place_scratch
from .workload import positive


def latency_summary(histogram, tck_ns):
    count = sum(histogram.values())
    target, cumulative, p95 = ceil(.95 * count), 0, 0
    for ticks, frequency in sorted(histogram.items()):
        cumulative += frequency
        if cumulative >= target:
            p95 = ticks
            break
    return {"transactions": count,
            "mean_ns": sum(t * n for t, n in histogram.items()) * tck_ns / count if count else 0,
            "p95_ns": p95 * tck_ns}


@dataclass
class Work:
    group: int
    reads: tuple
    writes: tuple
    weight: tuple
    compute_ns: int
    accumulator: int
    last: bool
    category: str


@dataclass
class Transfer:
    work: Work
    input_bytes: int
    reads: deque
    cache_key: tuple = ()
    fills_cache: bool = False
    pending: int = 0
    state: str = "reading"
    writes: deque = field(default_factory=deque)


@dataclass
class CacheEntry:
    size: int
    refs: int = 0
    ready: bool = False


@dataclass
class EngineCore:
    stage: int
    model: NeuSimCompute
    scratch_capacity: int
    cache_capacity: int
    prefetch: int
    max_outstanding: int
    source: object = None
    next_work: Work | None = None
    exhausted: bool = True
    tasks: list = field(default_factory=list)
    used: int = 0
    peak: int = 0
    outstanding: int = 0
    peak_outstanding: int = 0
    busy: Transfer | None = None
    compute_end: int = 0
    group: int | None = None
    accumulator: int = 0
    cache: OrderedDict = field(default_factory=OrderedDict)
    cache_used: int = 0
    counts: Counter = field(default_factory=Counter)


@dataclass
class Stage:
    name: str
    cores: tuple
    prefix: bool = False
    first_step: int = 0
    steps: int = 0
    queue: deque = field(default_factory=deque)
    job: tuple = ()
    ops: list = field(default_factory=list)
    op_index: int = -1
    arena: int = 0
    arena_size: int = 0
    ready_at: int = 0
    op_started: bool = False
    job_start: int = 0


def validate_stream(config):
    if config.get("schema_version") != 2 or config.get("workload", {}).get("kind") not in ("pi05", "pi05_dit"):
        raise ValueError("Streaming pi05 requires schema_version=2 and kind=pi05 or pi05_dit")
    hardware, work = config["hardware"], config["workload"]
    allowed = {
        "hardware": {"chip", "sram_bytes_per_core", "weight_cache_bytes_per_core", "prefetch_tiles",
                     "max_outstanding", "vector_elements_per_cycle", "tile_mnk"},
        "workload": {"kind", "stage_cores", "microbatch", "max_inflight", "warmup_requests",
                     "measure_requests", "shape"},
        "simulation": {"issue_width", "max_cycles", "operator_overhead_ns", "profile_cycles", "active_core_polling"},
    }
    for section, keys in allowed.items():
        unknown = set(config.get(section, {})) - keys
        if unknown:
            raise ValueError(f"Unsupported {section} options: {sorted(unknown)}")
    for key in ("sram_bytes_per_core", "prefetch_tiles", "max_outstanding", "vector_elements_per_cycle"):
        positive(hardware[key], "hardware." + key)
    cache = hardware.get("weight_cache_bytes_per_core", 0)
    if not isinstance(cache, int) or isinstance(cache, bool) or not 0 <= cache < hardware["sram_bytes_per_core"]:
        raise ValueError("weight cache must be smaller than local SRAM")
    tile = hardware["tile_mnk"]
    if len(tile) != 3:
        raise ValueError("tile_mnk must contain M, N, K")
    for dimension in tile:
        positive(dimension, "tile dimension")
    chip = hardware.get("chip", {})
    if chip.get("num_sa", 1) != 1:
        raise ValueError("Each streaming core represents exactly one SA; allocate explicit cores to stages")
    positive(chip.get("sa_dim", 32), "sa_dim")
    freq = chip.get("freq_GHz", 1.0)
    if isinstance(freq, bool) or not isinstance(freq, (int, float)) or not 0 < freq < float("inf"):
        raise ValueError("freq_GHz must be positive and finite")
    stages = work["stage_cores"]
    has_prefix = work["kind"] == "pi05"
    if len(stages) < 1 + has_prefix:
        raise ValueError("stage_cores requires at least one timestep stage and optionally one prefix stage")
    for cores in stages:
        positive(cores, "stage core count")
    shape = Shape.parse(work.get("shape", {}))
    if shape.steps % (len(stages) - has_prefix):
        raise ValueError("Timestep stage count must divide the denoising step count")
    for key in ("microbatch", "max_inflight", "measure_requests"):
        positive(work[key], "workload." + key)
    if work["max_inflight"] < work["microbatch"]:
        raise ValueError("max_inflight must be at least microbatch")
    warmup = work.get("warmup_requests", 0)
    if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 0:
        raise ValueError("warmup_requests must be a nonnegative integer")
    sim = config.get("simulation", {})
    for key, default in (("issue_width", 16), ("max_cycles", 10000000)):
        positive(sim.get(key, default), key)
    overhead = sim.get("operator_overhead_ns", 0)
    if not isinstance(overhead, int) or isinstance(overhead, bool) or overhead < 0:
        raise ValueError("operator_overhead_ns must be a nonnegative integer")
    profile = sim.get("profile_cycles", 0)
    if not isinstance(profile, int) or isinstance(profile, bool) or profile < 0:
        raise ValueError("profile_cycles must be a nonnegative integer")
    if not isinstance(sim.get("active_core_polling", True), bool):
        raise ValueError("active_core_polling must be boolean")
    return shape


class StreamSimulator:
    def __init__(self, config, memory, request_sink=None):
        self.shape = validate_stream(config)
        self.config, self.memory = config, memory
        self.hardware, self.work = config["hardware"], dict(config["workload"])
        self.work.setdefault("warmup_requests", 0)
        self.model = Pi05(self.shape, self.work["max_inflight"])
        self.description = self.model.describe()  # allocate the shared weights first
        self.memory_ranges = sorted(
            [(t.base, t.base + t.size, "weight") for t in self.model.weights.values()]
            + [(t.base, t.base + t.size, "prefix_kv") for slot in self.model.slots
               for kind in ("k", "v") for t in slot[kind]])
        self.range_starts = [address for address, _, _ in self.memory_ranges]
        self.stages, self.cores = [], []
        chip = {"num_sa": 1, "num_vu": 1, "sa_dim": 32, "freq_GHz": 1.0,
                **self.hardware.get("chip", {})}
        cache = self.hardware.get("weight_cache_bytes_per_core", 0)
        has_prefix = self.work["kind"] == "pi05"
        per_stage_steps = self.shape.steps // (len(self.work["stage_cores"]) - has_prefix)
        for index, count in enumerate(self.work["stage_cores"]):
            indices = tuple(range(len(self.cores), len(self.cores) + count))
            prefix = has_prefix and index == 0
            stage = Stage("prefix" if prefix else f"dit{index - has_prefix}", indices, prefix,
                          max(0, index - has_prefix) * per_stage_steps, 0 if prefix else per_stage_steps)
            self.stages.append(stage)
            for _ in indices:
                self.cores.append(EngineCore(index, NeuSimCompute(chip),
                    self.hardware["sram_bytes_per_core"] - cache, cache,
                    self.hardware["prefetch_tiles"], self.hardware["max_outstanding"]))
            slots = list(range(1 if prefix else self.work["microbatch"]))
            template = self.model.graph(slots, prefix, stage.first_step, stage.steps)
            stage.arena_size = place_scratch(template, 0)
            stage.arena = self.model.allocator.allocate(stage.arena_size)
        if self.model.allocator.end > memory.capacity_bytes:
            raise ValueError(f"Bounded workload requires {self.model.allocator.end} bytes; "
                             f"DRAM capacity is {memory.capacity_bytes}")
        self.chip = chip
        self.active_cores = []
        self.cycle, self.request_id, self.cursor = 0, 0, 0
        self.pending, self.latencies = {}, Counter()
        self.stage_read_latencies = [Counter() for _ in self.stages]
        self.free_slots = deque(range(self.work["max_inflight"]))
        self.requests, self.completions = {}, []
        self.next_request = 0
        self.target = self.work["warmup_requests"] + self.work["measure_requests"]
        self.total_budget = self.target + self.work["max_inflight"]
        self.window_start = 0 if self.work["warmup_requests"] == 0 else None
        self.window_end = None
        self.window_bytes, self.window_roles = Counter(), Counter()
        self.job_events = []
        self.request_sink = request_sink
        self.peak_active_tiles = 0
        self.native_idle_ticks = 0
        self.overhead_ticks = ceil(config.get("simulation", {}).get("operator_overhead_ns", 0) / memory.tck_ns)

    def _tiles(self, op, local_core, core_count):
        tm, tn, tk = self.hardware["tile_mnk"]
        model = self.cores[local_core].model
        rank = self.stages[self.cores[local_core].stage].cores.index(local_core)
        if op.kind == "gemm":
            a, b = op.inputs
            m_count, n_count = (a.rows + tm - 1) // tm, (b.cols + tn - 1) // tn
            for group in range(rank, m_count * n_count, core_count):
                # N-major ordering keeps the same weight panel across M tiles.
                n, m = group // m_count * tn, group % m_count * tm
                mm, nn = min(tm, a.rows - m), min(tn, b.cols - n)
                for k in range(0, a.cols, tk):
                    kk = min(tk, a.cols - k)
                    lhs, rhs = a.regions(m, k, mm, kk), b.regions(k, n, kk, nn)
                    weight = rhs if all(t.weight for t in b.tensors) else ()
                    last = k + kk == a.cols
                    yield Work(group, lhs + (() if weight else rhs),
                               op.output.regions(m, n, mm, nn) if last else (), weight,
                               model.gemm_ns(mm, nn, kk, "BFLOAT16"), mm * nn * 4, last, op.category)
        else:
            # Full rows preserve reduction/broadcast operands. Reduce row count
            # to fit SRAM rather than charging a norm once per column fragment.
            row_bytes = op.output.cols * op.output.tensor.element
            fixed_bytes = 0
            for value in op.inputs:
                if value.rows == op.output.rows:
                    row_bytes += sum(t.element for t in value.tensors[:1]) * value.cols
                else:
                    fixed_bytes += sum(t.size for t in value.tensors)
            available = self.cores[local_core].scratch_capacity - fixed_bytes
            rows = min(tm, available // row_bytes)
            if rows < 1:
                raise ValueError(f"Vector row cannot fit SRAM: {op.name}")
            for group, row in enumerate(range(0, op.output.rows, rows)):
                if group % core_count != rank:
                    continue
                count = min(rows, op.output.rows - row)
                reads = []
                for value in op.inputs:
                    if value.rows == op.output.rows:
                        reads.extend(value.regions(row, 0, count, value.cols))
                    else:
                        reads.extend(value.regions(0, 0, value.rows, value.cols))
                duration = max(1, ceil(count * op.output.cols * op.vector_ops /
                               self.hardware["vector_elements_per_cycle"] / self.chip["freq_GHz"]))
                yield Work(group, tuple(reads), op.output.regions(row, 0, count, op.output.cols),
                           (), duration, count * op.output.cols * op.output.tensor.element, True, "vector")

    def _start_op(self, stage):
        op = stage.ops[stage.op_index]
        for index in stage.cores:
            core = self.cores[index]
            core.source = iter(self._tiles(op, index, len(stage.cores)))
            core.exhausted = False
        self.active_cores = sorted(set(self.active_cores) | set(stage.cores))
        stage.op_started = True

    def _poll_cores(self):
        if self.config.get("simulation", {}).get("active_core_polling", True):
            return self.active_cores
        return range(len(self.cores))

    def _advance_stage(self, index):
        stage = self.stages[index]
        if stage.job:
            if not stage.op_started:
                if self.cycle >= stage.ready_at:
                    self._start_op(stage)
                return
            if any(c.tasks or not c.exhausted for c in (self.cores[i] for i in stage.cores)):
                return
            stage.op_index += 1
            if stage.op_index < len(stage.ops):
                stage.op_started = False
                stage.ready_at = self.cycle + self.overhead_ticks
                return
            for request in stage.job:
                record = self.requests[request]
                record["stage_done_cycles"].append(self.cycle)
                if index + 1 < len(self.stages):
                    self.stages[index + 1].queue.append((request, self.cycle))
                else:
                    record["done_cycle"] = self.cycle
                    self.completions.append(request)
                    self.free_slots.append(record["slot"])
                    rank = len(self.completions)
                    if rank == self.work["warmup_requests"]:
                        self.window_start = self.cycle
                    if rank == self.target:
                        self.window_end = self.cycle
            self.job_events.append({"stage": stage.name, "requests": list(stage.job),
                                    "start_cycle": stage.job_start, "done_cycle": self.cycle})
            stage.job, stage.ops, stage.op_index = (), [], -1
            stage.op_started = False
        needed = 1 if stage.prefix else self.work["microbatch"]
        upstream_empty = (self.next_request >= self.total_budget or self.window_end is not None) and all(
            not s.job and not s.queue for s in self.stages[:index])
        if len(stage.queue) < needed and not (stage.queue and upstream_empty):
            return
        items = [stage.queue.popleft() for _ in range(min(needed, len(stage.queue)))]
        stage.job = tuple(request for request, _ in items)
        stage.job_start = self.cycle
        formed_at = max(arrival for _, arrival in items)
        slots = []
        for request, arrival in items:
            self.requests[request]["stage_queue_wait_cycles"] += self.cycle - arrival
            if not stage.prefix:
                self.requests[request]["batch_wait_cycles"] += formed_at - arrival
            self.requests[request]["stage_start_cycles"].append(self.cycle)
            slots.append(self.requests[request]["slot"])
        stage.ops = self.model.graph(slots, stage.prefix, stage.first_step, stage.steps)
        end = place_scratch(stage.ops, stage.arena)
        if end > stage.arena + stage.arena_size:
            raise RuntimeError("Stage scratch exceeded its reserved arena")
        stage.op_index, stage.ready_at = 0, self.cycle + self.overhead_ticks

    def _feed(self):
        while self.free_slots and self.next_request < self.total_budget and self.window_end is None:
            request = self.next_request
            self.next_request += 1
            slot = self.free_slots.popleft()
            self.requests[request] = {"request": request, "slot": slot, "admitted_cycle": self.cycle,
                                      "batch_wait_cycles": 0, "stage_queue_wait_cycles": 0, "stage_start_cycles": [],
                                      "stage_done_cycles": []}
            self.stages[0].queue.append((request, self.cycle))

    def _drain(self):
        for request_id, completed in self.memory.completions():
            index, task, write, accepted, address = self.pending.pop(request_id)
            core = self.cores[index]
            core.outstanding -= 1
            task.pending -= 1
            self.latencies[completed - accepted] += 1
            if not write:
                self.stage_read_latencies[core.stage][completed - accepted] += 1
            if self.request_sink:
                self.request_sink({"id": request_id, "core": index, "stage": self.stages[core.stage].name,
                                   "write": write, "address": address,
                                   "accepted_cycle": accepted, "completion_cycle": completed})
        for index in self._poll_cores():
            core = self.cores[index]
            for task in core.tasks:
                if task.state == "reading" and not task.reads and not task.pending:
                    if task.fills_cache:
                        core.cache[task.cache_key].ready = True
                        task.fills_cache = False
                    if not task.cache_key or core.cache[task.cache_key].ready:
                        task.state = "ready"
                if task.state == "writing" and not task.writes and not task.pending:
                    task.state = "done"
                    core.used -= core.accumulator
                    core.accumulator, core.group = 0, None
            core.tasks[:] = [t for t in core.tasks if t.state != "done"]
            if core.busy is not None and core.compute_end <= self.cycle:
                task = core.busy
                core.used -= task.input_bytes
                if task.cache_key:
                    core.cache[task.cache_key].refs -= 1
                    core.cache.move_to_end(task.cache_key)
                core.busy = None
                if task.work.last:
                    task.state = "writing"
                    task.writes = deque(sorted({a for r in task.work.writes for a in r.transactions(self.memory.tx_bytes)}))
                else:
                    task.state = "done"
                core.counts["tiles"] += 1

    def _admit(self, index):
        core = self.cores[index]
        while not core.exhausted and len(core.tasks) < core.prefetch:
            if core.next_work is None:
                core.next_work = next(core.source, None)
                if core.next_work is None:
                    core.exhausted = True
                    break
            work = core.next_work
            if core.group is not None and work.group != core.group:
                break
            addresses = sorted({a for r in work.weight for a in r.transactions(self.memory.tx_bytes)})
            weight_size = len(addresses) * self.memory.tx_bytes
            cache_key = work.weight if weight_size and weight_size <= core.cache_capacity else ()
            ordinary = work.reads + (() if cache_key else work.weight)
            input_bytes = sum(r.size for r in ordinary)
            extra = input_bytes + (work.accumulator if core.group is None else 0)
            if input_bytes + work.accumulator > core.scratch_capacity:
                raise ValueError("GEMM tile cannot fit scratch SRAM; reduce tile_mnk or cache reservation")
            if core.used + extra > core.scratch_capacity:
                core.counts["sram_blocked_ticks"] += 1
                break
            fill = False
            if cache_key and cache_key not in core.cache:
                while core.cache_used + weight_size > core.cache_capacity:
                    victim = next((key for key, e in core.cache.items() if not e.refs and e.ready), None)
                    if victim is None:
                        core.counts["cache_blocked_ticks"] += 1
                        return
                    core.cache_used -= core.cache.pop(victim).size
                    core.counts["weight_evictions"] += 1
                core.cache[cache_key] = CacheEntry(weight_size)
                core.cache_used += weight_size
                core.counts["weight_miss_bytes"] += weight_size
                fill = True
            elif cache_key:
                core.counts["weight_hit_bytes"] += weight_size
                addresses = []
            if cache_key:
                core.cache[cache_key].refs += 1
            addresses = sorted(set(addresses) | {a for r in work.reads for a in r.transactions(self.memory.tx_bytes)})
            # With no cache, weight addresses remain ordinary DMA reads.
            core.tasks.append(Transfer(work, input_bytes, deque(addresses), cache_key, fill))
            if core.group is None:
                core.group, core.accumulator = work.group, work.accumulator
            core.used += extra
            core.peak = max(core.peak, core.used + core.cache_used)
            core.next_work = None

    def _compute(self, core):
        if core.busy is not None:
            return
        task = next((t for t in core.tasks if t.state != "done"), None)
        if task is not None and task.state == "ready":
            task.state = "computing"
            core.busy = task
            duration = max(1, ceil(task.work.compute_ns / self.memory.tck_ns))
            core.compute_end = self.cycle + duration
            core.counts["compute_ticks"] += duration
        elif task is not None:
            core.counts["input_wait_ticks"] += 1

    def _issue(self):
        width = self.config.get("simulation", {}).get("issue_width", 16)
        pool = self._poll_cores()
        for _ in range(width):
            sent = False
            start = bisect_left(pool, self.cursor)
            for offset in range(len(pool)):
                index = pool[(start + offset) % len(pool)]
                core = self.cores[index]
                if core.outstanding >= core.max_outstanding:
                    continue
                for task in core.tasks:
                    write = task.state == "writing"
                    queue = task.writes if write else task.reads
                    if task.state not in ("reading", "writing") or not queue:
                        continue
                    address = queue[0]
                    if not self.memory.send(self.request_id, write, address, index):
                        core.counts["queue_retries"] += 1
                        continue
                    queue.popleft()
                    task.pending += 1
                    core.outstanding += 1
                    core.peak_outstanding = max(core.peak_outstanding, core.outstanding)
                    self.pending[self.request_id] = (index, task, write, self.cycle, address)
                    self.request_id += 1
                    core.counts["write_bytes" if write else "read_bytes"] += self.memory.tx_bytes
                    role_index = bisect_right(self.range_starts, address) - 1
                    role = "activation"
                    if role_index >= 0 and address < self.memory_ranges[role_index][1]:
                        role = self.memory_ranges[role_index][2]
                    role += "_write" if write else "_read"
                    core.counts[role + "_bytes"] += self.memory.tx_bytes
                    if self.window_start is not None and self.window_end is None:
                        self.window_bytes[self.stages[core.stage].name] += self.memory.tx_bytes
                        self.window_roles[role] += self.memory.tx_bytes
                    self.cursor = (index + 1) % len(self.cores)
                    sent = True
                    break
                if sent:
                    break
            if not sent:
                break

    def run(self):
        maximum = self.config.get("simulation", {}).get("max_cycles", 10000000)
        profile = self.config.get("simulation", {}).get("profile_cycles", 0)
        while True:
            self._drain()
            self._feed()
            for index in range(len(self.stages)):
                self._advance_stage(index)
            for index in self._poll_cores():
                core = self.cores[index]
                self._admit(index)
                self._compute(core)
            self.active_cores = [i for i in self.active_cores
                                 if self.cores[i].tasks or not self.cores[i].exhausted]
            active = sum(len(self.cores[i].tasks) for i in self.active_cores)
            self.peak_active_tiles = max(self.peak_active_tiles, active)
            if self.window_end is not None and not any(s.job or s.queue for s in self.stages):
                break
            if profile and self.cycle >= profile:
                return self.progress()
            if self.cycle >= maximum:
                raise RuntimeError(f"Exceeded max_cycles={maximum}; completed={len(self.completions)}; "
                                   f"queues={[len(s.queue) for s in self.stages]}")
            self._issue()
            ticks = self._idle_ticks()
            ticks = min(ticks, maximum - self.cycle)
            if profile:
                ticks = min(ticks, profile - self.cycle)
            if ticks > 1:
                self.memory.advance(ticks)
                self.native_idle_ticks += ticks
            else:
                self.memory.tick()
            self.cycle += ticks
        if self.pending or any(c.used or c.outstanding or c.cache_used > c.cache_capacity for c in self.cores):
            raise RuntimeError("Simulation ended with live requests or invalid SRAM accounting")
        if sum(self.latencies.values()) != self.request_id:
            raise RuntimeError("DRAM acceptance/completion accounting mismatch")
        return self.report()

    def _idle_ticks(self):
        # Only skip Python polling when no callbacks or transfers can wake work.
        # Ramulator still executes every single tick, including refresh.
        if self.pending or not callable(getattr(self.memory, "advance", None)):
            return 1
        if self.free_slots and self.next_request < self.total_budget and self.window_end is None:
            return 1
        cores = [self.cores[i] for i in self._poll_cores()]
        if any(any(t.state not in ("computing", "ready") for t in c.tasks) for c in cores):
            return 1
        if any(c.next_work is not None and c.next_work.group == c.group
               and len(c.tasks) < c.prefetch for c in cores):
            return 1
        events = [c.compute_end for c in cores if c.busy is not None]
        for stage in self.stages:
            if not stage.job:
                continue
            if not stage.op_started:
                if stage.ready_at <= self.cycle:
                    return 1
                events.append(stage.ready_at)
            elif not any(self.cores[i].tasks for i in stage.cores):
                return 1
        return max(1, min(events) - self.cycle) if events else 1

    def progress(self):
        return {"complete": False, "mode": "pi05_profile_only", "cycles": self.cycle,
                "tck_ns": self.memory.tck_ns, "native_idle_ticks": self.native_idle_ticks,
                "workload": self.description, "address_space_bytes": self.model.allocator.end,
                "dram_capacity_bytes": self.memory.capacity_bytes,
                "peak_active_tiles": self.peak_active_tiles, "dram_transactions": self.request_id,
                "pending_transactions": len(self.pending),
                "stages": [{"stage": s.name, "op_index": s.op_index, "ops_in_job": len(s.ops),
                            "queued_requests": len(s.queue)} for s in self.stages],
                "completed_requests": len(self.completions),
                "warning": "Bounded runtime profile, not an end-to-end latency/throughput result"}

    def report(self):
        warmup = self.work["warmup_requests"]
        measured = [self.requests[r] for r in self.completions[warmup:self.target]]
        latencies = sorted((r["done_cycle"] - r["admitted_cycle"]) * self.memory.tck_ns for r in measured)
        duration = (self.window_end - self.window_start) * self.memory.tck_ns
        core_rows = [{"core": index, "stage": self.stages[c.stage].name,
                      "peak_sram_bytes": c.peak, "sram_capacity_bytes": c.scratch_capacity + c.cache_capacity,
                      "resident_weight_bytes_at_end": c.cache_used,
                      "peak_outstanding": c.peak_outstanding, **dict(c.counts)}
                     for index, c in enumerate(self.cores)]
        return {
            "complete": True,
            "backend": "official Ramulator2 (online)", "mode": "pi05_request_stream",
            "scope": self.work["kind"], "prefix_kv_prepopulated": self.work["kind"] == "pi05_dit",
            "cycles": self.cycle, "tck_ns": self.memory.tck_ns,
            "makespan_ns": self.cycle * self.memory.tck_ns,
            "workload": self.description,
            "address_space_bytes": self.model.allocator.end,
            "dram_capacity_bytes": self.memory.capacity_bytes,
            "peak_active_tiles": self.peak_active_tiles, "requests_completed": len(self.completions),
            "native_idle_ticks": self.native_idle_ticks,
            "dram_transactions": self.request_id,
            "dram_transaction_latency": {"scope": "whole run including warmup and drain; acceptance to callback",
                                         **latency_summary(self.latencies, self.memory.tck_ns)},
            "stage_dram_read_latency": [{"stage": s.name,
                "scope": "whole run including warmup and drain; acceptance to callback",
                **latency_summary(self.stage_read_latencies[i], self.memory.tck_ns)}
                for i, s in enumerate(self.stages)],
            "measurement": {"warmup_requests": warmup, "measured_requests": len(measured),
                            "start_cycle": self.window_start, "end_cycle": self.window_end,
                            "includes_cold_fill": warmup == 0,
                            "mean_output_interval_ns": duration / len(measured),
                            "throughput_requests_per_second": len(measured) * 1e9 / duration if duration else None,
                            "latency_mean_ns": sum(latencies) / len(latencies),
                            "latency_p95_ns": latencies[ceil(.95 * len(latencies)) - 1],
                            "mean_batch_wait_ns": sum(r["batch_wait_cycles"] for r in measured) * self.memory.tck_ns / len(measured),
                            "mean_stage_queue_wait_ns": sum(r["stage_queue_wait_cycles"] for r in measured) * self.memory.tck_ns / len(measured),
                            "stage_interface_bytes": dict(self.window_bytes),
                            "interface_bytes_by_role": dict(self.window_roles),
                            "stage_interface_GBps": {k: n / duration if duration else None for k, n in self.window_bytes.items()}},
            "cores": core_rows, "request_records": list(self.requests.values()), "jobs": self.job_events,
            "model": {"attention": "dense, materialized; independent images/requests; MQA prefix KV",
                      "sram": "private capacity-only scratch and explicit bounded LRU weight tiles",
                      "inter_stage": "DRAM, stage job waits for all producer write callbacks",
                      "compute": "one official NeuSim compute model per explicit SA core",
                      "vector": "configurable element work per cycle, not calibrated",
                      "scheduling": "FIFO stages; one job/stage; operator barriers; K-slice prefetch",
                      "input": "saturated closed system with bounded in-flight requests",
                      "window": "completion ranks; saturated successors retained until window closes",
                      "write_completion": "Ramulator write callback, not physical write recovery",
                      "issue_width": "global accepted requests per Ramulator model tick"},
        }

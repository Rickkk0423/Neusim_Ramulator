"""One shared timeline; compute events and DRAM completions drive tile progress."""

from collections import deque
from dataclasses import dataclass, field
import heapq
from math import ceil

from .workload import Tile


@dataclass
class Runtime:
    tile: Tile
    state: str = "waiting"
    read_queue: deque = field(default_factory=deque)
    write_queue: deque = field(default_factory=deque)
    pending: int = 0
    admitted: int | None = None
    ready: int | None = None
    start: int | None = None
    end: int | None = None
    done: int | None = None
    read_bytes: int = 0
    write_bytes: int = 0


@dataclass
class Core:
    capacity: int
    prefetch: int
    max_outstanding: int
    used: int = 0
    peak: int = 0
    outstanding: int = 0
    peak_outstanding: int = 0
    busy: str | None = None
    accumulators: dict = field(default_factory=dict)
    input_stall_cycles: int = 0
    capacity_blocked_cycles: int = 0
    queue_retries: int = 0
    stream_cursor: int = 0


class Scheduler:
    def __init__(self, tiles: list[Tile], cores: list[dict], memory,
                 issue_width: int = 4, max_cycles: int = 10000000, request_sink=None):
        self.memory = memory
        self.tasks = [Runtime(tile) for tile in tiles]
        self.lookup = {task.tile.id: task for task in self.tasks}
        if len(self.lookup) != len(tiles):
            raise ValueError("Tile IDs must be unique")
        self.cores = [Core(c["sram_bytes"], c["prefetch_tiles"], c["max_outstanding"])
                      for c in cores]
        self.by_core = [[task for task in self.tasks if task.tile.core == index]
                        for index in range(len(cores))]
        self.issue_width, self.max_cycles = issue_width, max_cycles
        self.cycle = 0
        self.next_id = 0
        self.requests = {}
        self.compute_events = []
        self.computed = set()
        self.completed = set()
        self.core_cursor = 0
        self.request_sink = request_sink
        self.request_latencies = []
        self.group_inputs = {}
        for tile in tiles:
            self.group_inputs[tile.group] = max(self.group_inputs.get(tile.group, 0), tile.input_bytes)
        self._check_plan()

    def _check_plan(self):
        # Plans are topological: rejecting cycles here avoids a silent max-cycle hang.
        seen = set()
        for task in self.tasks:
            tile = task.tile
            if not 0 <= tile.core < len(self.cores) or tile.compute_ns <= 0:
                raise ValueError(f"Invalid core/compute time for {tile.id}")
            if not set(tile.load_after + tile.compute_after) <= seen:
                raise ValueError(f"Missing, cyclic, or non-topological dependency for {tile.id}")
            if tile.input_bytes + tile.accumulator_bytes > self.cores[tile.core].capacity:
                raise ValueError(f"Insufficient SRAM for {tile.id}")
            seen.add(tile.id)

    def _complete(self, task: Runtime):
        task.state = "done"
        task.done = self.cycle
        self.completed.add(task.tile.id)
        if task.tile.last_k:
            core = self.cores[task.tile.core]
            core.used -= core.accumulators.pop(task.tile.group)

    def _drain(self):
        for request_id, completion_cycle in self.memory.completions():
            if request_id not in self.requests:
                raise RuntimeError(f"Unexpected or duplicate completion {request_id}")
            task, write, address, accepted = self.requests.pop(request_id)
            task.pending -= 1
            self.cores[task.tile.core].outstanding -= 1
            self.request_latencies.append(completion_cycle - accepted)
            if self.request_sink:
                self.request_sink({"id": request_id, "core": task.tile.core,
                                   "tile": task.tile.id, "write": write, "address": address,
                                   "accepted_cycle": accepted, "completion_cycle": completion_cycle})
        for task in self.tasks:
            if task.state == "reading" and not task.read_queue and not task.pending:
                task.state, task.ready = "ready", self.cycle
            elif task.state == "writing" and not task.write_queue and not task.pending:
                self._complete(task)
        while self.compute_events and self.compute_events[0][0] <= self.cycle:
            _, tile_id = heapq.heappop(self.compute_events)
            task = self.lookup[tile_id]
            task.end = self.cycle
            self.computed.add(tile_id)
            core = self.cores[task.tile.core]
            core.busy = None
            core.used -= task.tile.input_bytes
            if task.tile.writes:
                task.state = "writing"
                addresses = {a for r in task.tile.writes for a in r.transactions(self.memory.tx_bytes)}
                task.write_queue = deque(sorted(addresses))
            else:
                self._complete(task)

    def _admit_and_start(self):
        for core_index, core in enumerate(self.cores):
            tasks = self.by_core[core_index]
            active = sum(task.state not in ("waiting", "done") for task in tasks)
            capacity_blocked = False
            for task in tasks:
                if task.state != "waiting" or active >= core.prefetch:
                    continue
                if not set(task.tile.load_after) <= self.completed:
                    continue
                # K slices must acquire their output accumulator in order. Their
                # loads may overlap earlier K computation once that group exists.
                group_present = task.tile.group in core.accumulators
                if task.tile.compute_after and not group_present:
                    continue
                if any(self.lookup[dep].state == "waiting" for dep in task.tile.compute_after):
                    continue
                if not group_present:
                    # Do not strand all capacity in unfinished accumulators.
                    max_input = max([self.group_inputs[group] for group in core.accumulators]
                                    + [self.group_inputs[task.tile.group]])
                    reserved = sum(core.accumulators.values()) + task.tile.accumulator_bytes
                    if reserved + max_input > core.capacity:
                        capacity_blocked = True
                        continue
                extra = task.tile.input_bytes + (0 if group_present else task.tile.accumulator_bytes)
                if core.used + extra > core.capacity:
                    capacity_blocked = True
                    continue
                core.used += extra
                core.peak = max(core.peak, core.used)
                if not group_present:
                    core.accumulators[task.tile.group] = task.tile.accumulator_bytes
                addresses = {a for r in task.tile.reads for a in r.transactions(self.memory.tx_bytes)}
                task.read_queue = deque(sorted(addresses))
                task.admitted, task.state = self.cycle, "reading"
                if not task.read_queue:
                    task.ready, task.state = self.cycle, "ready"
                active += 1
            core.capacity_blocked_cycles += int(capacity_blocked)
            if core.busy is None:
                for task in tasks:
                    if task.state == "ready" and set(task.tile.compute_after) <= self.computed:
                        task.start, task.state = self.cycle, "computing"
                        core.busy = task.tile.id
                        duration = max(1, ceil(task.tile.compute_ns / self.memory.tck_ns))
                        heapq.heappush(self.compute_events, (self.cycle + duration, task.tile.id))
                        break
                if core.busy is None and any(task.state == "reading" for task in tasks):
                    core.input_stall_cycles += 1

    def _issue_core(self, index: int) -> bool:
        core = self.cores[index]
        if core.outstanding >= core.max_outstanding:
            return False
        streams = [task for task in self.by_core[index]
                   if (task.state == "reading" and task.read_queue)
                   or (task.state == "writing" and task.write_queue)]
        for offset in range(len(streams)):
            position = (core.stream_cursor + offset) % len(streams)
            task = streams[position]
            write = task.state == "writing"
            queue = task.write_queue if write else task.read_queue
            address = queue[0]
            if not self.memory.send(self.next_id, write, address, index):
                core.queue_retries += 1
                continue
            queue.popleft()
            task.pending += 1
            core.outstanding += 1
            core.peak_outstanding = max(core.peak_outstanding, core.outstanding)
            self.requests[self.next_id] = (task, write, address, self.cycle)
            self.next_id += 1
            if write:
                task.write_bytes += self.memory.tx_bytes
            else:
                task.read_bytes += self.memory.tx_bytes
            core.stream_cursor = position + 1
            return True
        return False

    def _issue(self):
        for _ in range(self.issue_width):
            for offset in range(len(self.cores)):
                index = (self.core_cursor + offset) % len(self.cores)
                if self._issue_core(index):
                    self.core_cursor = (index + 1) % len(self.cores)
                    break
            else:
                break

    def run(self) -> dict:
        while True:
            self._drain()
            if len(self.completed) == len(self.tasks):
                break
            if self.cycle >= self.max_cycles:
                unfinished = [(t.tile.id, t.state) for t in self.tasks if t.state != "done"][:8]
                raise RuntimeError(f"Exceeded max_cycles={self.max_cycles}; unfinished={unfinished}")
            self._admit_and_start()
            self._issue()
            self.memory.tick()
            self.cycle += 1
        if self.requests or any(core.used or core.outstanding for core in self.cores):
            raise RuntimeError("Simulation ended with live requests or reserved SRAM")
        return self.report()

    def report(self) -> dict:
        total_ns = self.cycle * self.memory.tck_ns
        tile_rows = [{"tile": t.tile.id, "core": t.tile.core, "stage": t.tile.stage,
                      "microbatch": t.tile.microbatch, "step": t.tile.step,
                      "admitted_cycle": t.admitted, "inputs_ready_cycle": t.ready,
                      "compute_start_cycle": t.start, "compute_end_cycle": t.end,
                      "done_cycle": t.done, "compute_model_ns": t.tile.compute_ns,
                      "dram_read_bytes": t.read_bytes, "dram_write_bytes": t.write_bytes}
                     for t in self.tasks]
        core_rows = []
        for index, core in enumerate(self.cores):
            tasks = self.by_core[index]
            traffic = sum(t.read_bytes + t.write_bytes for t in tasks)
            core_rows.append({
                "core": index, "tiles": len(tasks), "peak_sram_bytes": core.peak,
                "sram_capacity_bytes": core.capacity, "peak_outstanding": core.peak_outstanding,
                "input_wait_cycles": core.input_stall_cycles,
                "capacity_blocked_cycles": core.capacity_blocked_cycles,
                "queue_retries": core.queue_retries,
                "compute_busy_cycles": sum(t.end - t.start for t in tasks),
                "dram_bytes": traffic,
                "bandwidth_GBps": traffic / total_ns if total_ns else 0,
            })
        stages = []
        for name in dict.fromkeys(t.tile.stage for t in self.tasks):
            tasks = [t for t in self.tasks if t.tile.stage == name]
            stages.append({"stage": name, "first_admitted_cycle": min(t.admitted for t in tasks),
                           "last_done_cycle": max(t.done for t in tasks),
                           "dram_bytes": sum(t.read_bytes + t.write_bytes for t in tasks)})
        microbatches = []
        for batch in sorted({t.tile.microbatch for t in self.tasks}):
            tasks = [t for t in self.tasks if t.tile.microbatch == batch]
            microbatches.append({"microbatch": batch,
                                 "first_admitted_cycle": min(t.admitted for t in tasks),
                                 "done_cycle": max(t.done for t in tasks)})
        latencies = sorted(self.request_latencies)
        return {
            "backend": "official Ramulator2 (online)", "cycles": self.cycle,
            "tck_ns": self.memory.tck_ns, "transaction_bytes": self.memory.tx_bytes,
            "makespan_ns": total_ns, "requests_completed": len(latencies),
            "request_latency_cycles_mean": sum(latencies) / len(latencies) if latencies else 0,
            "request_latency_cycles_p95": latencies[min(len(latencies) - 1, ceil(len(latencies) * .95) - 1)] if latencies else 0,
            "total_bandwidth_GBps": sum(c["dram_bytes"] for c in core_rows) / total_ns if total_ns else 0,
            "cores": core_rows, "stages": stages, "microbatches": microbatches,
            "tiles": tile_rows,
        }

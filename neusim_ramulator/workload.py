"""Explicit GEMM tile plans, physical byte ranges, and producer dependencies."""

from dataclasses import dataclass
from math import ceil

from .compute import NeuSimCompute

DTYPES = {"BFLOAT16": 2, "FLOAT16": 2, "FLOAT32": 4, "INT8": 1}


def positive(value, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class Region:
    base: int
    rows: int
    columns: int
    stride_bytes: int
    element_bytes: int

    @property
    def size(self) -> int:
        return self.rows * self.columns * self.element_bytes

    def transactions(self, tx_bytes: int) -> list[int]:
        addresses = set()
        for row in range(self.rows):
            start = self.base + row * self.stride_bytes
            end = start + self.columns * self.element_bytes
            addresses.update(range(start // tx_bytes * tx_bytes,
                                   ceil(end / tx_bytes) * tx_bytes, tx_bytes))
        return sorted(addresses)


@dataclass(frozen=True)
class Tile:
    id: str
    core: int
    stage: str
    microbatch: int
    step: int
    group: str
    reads: tuple[Region, ...]
    writes: tuple[Region, ...]
    load_after: tuple[str, ...]
    compute_after: tuple[str, ...]
    compute_ns: int
    accumulator_bytes: int
    last_k: bool

    @property
    def input_bytes(self) -> int:
        return sum(region.size for region in self.reads)


@dataclass(frozen=True)
class Producer:
    row: int
    column: int
    rows: int
    columns: int
    tile: str

    def intersects(self, row: int, column: int, rows: int, columns: int) -> bool:
        return (self.row < row + rows and row < self.row + self.rows
                and self.column < column + columns and column < self.column + self.columns)


def validate(config: dict) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    cores = config.get("cores", [])
    if not cores:
        raise ValueError("At least one core is required")
    for index, core in enumerate(cores):
        for name in ("sram_bytes", "prefetch_tiles", "max_outstanding"):
            positive(core.get(name), f"cores[{index}].{name}")
        for name in ("num_sa", "num_vu", "sa_dim"):
            positive(core.get("chip", {}).get(name, 1 if name != "sa_dim" else 128), name)
        frequency = core.get("chip", {}).get("freq_GHz", 1.75)
        if not isinstance(frequency, (int, float)) or not 0 < frequency < float("inf"):
            raise ValueError("freq_GHz must be finite and positive")
    simulation = config.get("simulation", {})
    for name, default in (("issue_width", 4), ("max_cycles", 10000000)):
        positive(simulation.get(name, default), f"simulation.{name}")
    work = config["workload"]
    for name in ("batch_size", "microbatch_size", "rows_per_sample", "steps"):
        positive(work.get(name), f"workload.{name}")
    if work.get("dtype", "BFLOAT16") not in DTYPES:
        raise ValueError(f"dtype must be one of {list(DTYPES)}")
    positive(work.get("accumulator_bytes", 4), "accumulator_bytes")
    if len(work.get("tile", [])) != 3:
        raise ValueError("tile must be [M, N, K]")
    for value in work["tile"]:
        positive(value, "tile dimension")
    stages = work.get("stages", [])
    if not stages or len({stage["name"] for stage in stages}) != len(stages):
        raise ValueError("Stages must be nonempty and have unique names")
    for index, stage in enumerate(stages):
        positive(stage["k"], "stage.k")
        positive(stage["n"], "stage.n")
        core = stage.get("core")
        if not isinstance(core, int) or isinstance(core, bool) or not 0 <= core < len(cores):
            raise ValueError("stage.core must index cores")
        if index and stages[index - 1]["n"] != stage["k"]:
            raise ValueError("Adjacent stages must have matching output/input dimensions")
    if work["steps"] > 1 and stages[-1]["n"] != stages[0]["k"]:
        raise ValueError("Repeated steps require the last output dimension to match the first input")


def build_plan(config: dict) -> tuple[list[Tile], dict]:
    validate(config)
    work = config["workload"]
    dtype = work.get("dtype", "BFLOAT16")
    element_bytes = DTYPES[dtype]
    accumulator_bytes = work.get("accumulator_bytes", 4)
    tm, tn, tk = work["tile"]
    models = [NeuSimCompute(core.get("chip", {})) for core in config["cores"]]
    stages = work["stages"]
    next_address = 0

    def allocate(size: int) -> int:
        nonlocal next_address
        base = next_address
        next_address += ceil(size / 4096) * 4096
        if next_address >= 2**63:
            raise ValueError("Workload exceeds the signed 64-bit address space")
        return base

    weights = [allocate(stage["k"] * stage["n"] * element_bytes) for stage in stages]
    tiles = []
    batches = ceil(work["batch_size"] / work["microbatch_size"])
    for microbatch in range(batches):
        samples = min(work["microbatch_size"],
                      work["batch_size"] - microbatch * work["microbatch_size"])
        rows = samples * work["rows_per_sample"]
        input_base = allocate(rows * stages[0]["k"] * element_bytes)
        producers: list[Producer] = []
        for step in range(work["steps"]):
            for stage_index, stage in enumerate(stages):
                output_base = allocate(rows * stage["n"] * element_bytes)
                output_producers = []
                for m in range(0, rows, tm):
                    mm = min(tm, rows - m)
                    for n in range(0, stage["n"], tn):
                        nn = min(tn, stage["n"] - n)
                        group = f"b{microbatch}/s{step}/{stage['name']}/m{m}n{n}"
                        previous_k = None
                        for k in range(0, stage["k"], tk):
                            kk = min(tk, stage["k"] - k)
                            tile_id = f"{group}/k{k}"
                            a = Region(input_base + (m * stage["k"] + k) * element_bytes,
                                       mm, kk, stage["k"] * element_bytes, element_bytes)
                            b = Region(weights[stage_index] + (k * stage["n"] + n) * element_bytes,
                                       kk, nn, stage["n"] * element_bytes, element_bytes)
                            output = Region(output_base + (m * stage["n"] + n) * element_bytes,
                                            mm, nn, stage["n"] * element_bytes, element_bytes)
                            last = k + kk == stage["k"]
                            dependencies = tuple(p.tile for p in producers if p.intersects(m, k, mm, kk))
                            tiles.append(Tile(
                                id=tile_id, core=stage["core"], stage=stage["name"],
                                microbatch=microbatch, step=step, group=group,
                                reads=(a, b), writes=(output,) if last else (),
                                load_after=dependencies,
                                compute_after=(previous_k,) if previous_k else (),
                                compute_ns=models[stage["core"]].gemm_ns(mm, nn, kk, dtype),
                                accumulator_bytes=mm * nn * accumulator_bytes, last_k=last,
                            ))
                            previous_k = tile_id
                        output_producers.append(Producer(m, n, mm, nn, previous_k))
                input_base, producers = output_base, output_producers
    for tile in tiles:
        required = tile.input_bytes + tile.accumulator_bytes
        capacity = config["cores"][tile.core]["sram_bytes"]
        if required > capacity:
            raise ValueError(f"Tile {tile.id} requires {required} SRAM bytes, capacity is {capacity}")
    return tiles, {
        "tiles": len(tiles), "microbatches": batches, "address_space_bytes": next_address,
        "logical_read_bytes": sum(tile.input_bytes for tile in tiles),
        "logical_write_bytes": sum(r.size for tile in tiles for r in tile.writes),
        "neusim_compute_calls": sum(model.calls for model in models),
        "dtype": dtype, "tile_mnk": [tm, tn, tk],
    }

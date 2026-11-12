"""Incremental ctypes interface to a real official Ramulator memory system."""

import ctypes as ct
from math import prod
from pathlib import Path
import sys

import yaml

from .upstream import ROOT, add_upstream


def export_config(config: dict, output: Path, num_cores: int = 1) -> None:
    add_upstream("RAMULATOR_ROOT", "third_party/ramulator2/python")
    import ramulator as ram

    standard = config.get("standard", "DDR4")
    choices = {
        "DDR4": (ram.dram.DDR4, ram.controller.GenericDDR, "DDR4_8Gb_x8", "DDR4_2400R"),
        "HBM2": (ram.dram.HBM2, ram.controller.HBM12, "HBM2_2Gb", "HBM2_2000Mbps"),
        "HBM3": (ram.dram.HBM3, ram.controller.HBM34, "HBM3_16Gb_8hi", "HBM3_6400Mbps"),
    }
    if standard not in choices:
        raise ValueError(f"Unsupported DRAM standard {standard}; choose {list(choices)}")
    dram_class, controller_class, organization, timing = choices[standard]
    channels = config.get("channels", 1)
    if not isinstance(channels, int) or isinstance(channels, bool) or channels < 1:
        raise ValueError("dram.channels must be a positive integer")
    if channels & (channels - 1):
        raise ValueError("dram.channels must be a power of two for CacheLineInterleave")
    if not isinstance(num_cores, int) or isinstance(num_cores, bool) or not 0 < num_cores < 2**31:
        raise ValueError("num_cores must be a positive signed 32-bit integer")
    controllers = []
    for _ in range(channels):
        dram = dram_class(org_preset=config.get("organization", organization),
                          timing_preset=config.get("timing", timing), **config.get("device", {}))
        kwargs = dict(
            dram=dram, scheduler=ram.scheduler.FRFCFS(),
            row_policy=ram.row_policy.Open(), addr_mapper=ram.addr_mapper.RoBaRaCoCh(),
        )
        if standard == "DDR4":
            kwargs["refresh_manager"] = ram.refresh_manager.AllBank()
        elif standard == "HBM2":
            kwargs["refresh_manager"] = ram.refresh_manager.HBMPerBankRefresh()
        else:
            kwargs["refresh_manager"] = ram.refresh_manager.HBM34PerBankRefresh()
        controllers.append(controller_class(**kwargs))
    memory = ram.memory_system.GenericDRAM(
        clock_ratio=1, controllers=controllers,
        channel_mapper=ram.channel_mapper.CacheLineInterleave(),
    )
    expanded = memory.to_config()
    # Match the upstream mapper's addressable transaction space, including HBM
    # pseudochannels; channel width alone is not the HBM transaction payload.
    device = expanded["controllers"][0]["dram"]
    tx_bytes = device.get("data_payload_bytes",
                          dram_class.internal_prefetch_size * device["channel_width"] // 8)
    capacity = (channels * prod(device["org"]["count"])
                // dram_class.internal_prefetch_size * tx_bytes)
    if not 0 < capacity <= 2**63 - 1:
        raise ValueError("DRAM addressable capacity must fit the signed 64-bit address space")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump({
        "frontend": {"impl": "NeuSimExternal", "num_cores": num_cores,
                     "address_capacity_bytes": capacity},
        "memory_system": expanded,
    }, sort_keys=False))


class Ramulator:
    def __init__(self, config: Path, library: Path | None = None):
        suffix = ".dylib" if sys.platform == "darwin" else ".so"
        path = library or ROOT / "build" / f"libneusim_ramulator_bridge{suffix}"
        if not path.is_file():
            raise RuntimeError(f"Missing {path}; run ./scripts/setup.sh")
        self.lib = ct.CDLL(str(path))
        abi = getattr(self.lib, "nr_abi_version", None)
        if abi is None:
            raise RuntimeError("Outdated native bridge; run ./scripts/setup.sh")
        abi.argtypes, abi.restype = [], ct.c_int
        if abi() != 2:
            raise RuntimeError("Incompatible native bridge; run ./scripts/setup.sh")
        signatures = {
            "nr_error": ([], ct.c_char_p),
            "nr_create": ([ct.c_char_p], ct.c_void_p),
            "nr_geometry": ([ct.c_void_p, ct.POINTER(ct.c_double), ct.POINTER(ct.c_int),
                             ct.POINTER(ct.c_uint64)], ct.c_int),
            "nr_send": ([ct.c_void_p, ct.c_uint64, ct.c_int, ct.c_int64, ct.c_int], ct.c_int),
            "nr_tick": ([ct.c_void_p, ct.c_uint64], ct.c_int),
            "nr_pop": ([ct.c_void_p, ct.POINTER(ct.c_uint64), ct.POINTER(ct.c_uint64)], ct.c_int),
            "nr_finalize": ([ct.c_void_p, ct.c_char_p], ct.c_int),
            "nr_destroy": ([ct.c_void_p], None),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = args, result
        self.handle = self.lib.nr_create(str(config).encode())
        if not self.handle:
            self._check(-1)
        tck, tx, capacity = ct.c_double(), ct.c_int(), ct.c_uint64()
        try:
            self._check(self.lib.nr_geometry(self.handle, ct.byref(tck), ct.byref(tx),
                                             ct.byref(capacity)))
        except Exception:
            self.close()
            raise
        self.tck_ns, self.tx_bytes, self.capacity_bytes = tck.value, tx.value, capacity.value

    def _check(self, status: int) -> int:
        if status < 0:
            raise RuntimeError(self.lib.nr_error().decode())
        return status

    def _require_open(self) -> None:
        if not self.handle:
            raise RuntimeError("Ramulator instance is closed")

    @staticmethod
    def _integer(value: int, name: str, upper: int) -> None:
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < upper:
            raise ValueError(f"{name} must be an integer in [0, {upper})")

    def send(self, request_id: int, write: bool, address: int, core: int) -> bool:
        self._require_open()
        self._integer(request_id, "request_id", 2**64)
        self._integer(address, "address", 2**63)
        self._integer(core, "core", 2**31)
        if not isinstance(write, bool):
            raise ValueError("write must be a boolean")
        return bool(self._check(self.lib.nr_send(self.handle, request_id, int(write), address, core)))

    def tick(self) -> None:
        self._require_open()
        self._check(self.lib.nr_tick(self.handle, 1))

    def advance(self, cycles: int) -> None:
        """Run real DRAM ticks in native code; caller must preserve feedback events."""
        self._require_open()
        self._integer(cycles, "cycles", 2**64)
        if cycles == 0:
            raise ValueError("cycles must be positive")
        self._check(self.lib.nr_tick(self.handle, cycles))

    def completions(self) -> list[tuple[int, int]]:
        self._require_open()
        events = []
        request_id, cycle = ct.c_uint64(), ct.c_uint64()
        while self.lib.nr_pop(self.handle, ct.byref(request_id), ct.byref(cycle)):
            events.append((request_id.value, cycle.value))
        return events

    def finalize(self, path: Path) -> None:
        self._require_open()
        self._check(self.lib.nr_finalize(self.handle, str(path).encode()))

    def close(self) -> None:
        if self.handle:
            self.lib.nr_destroy(self.handle)
            self.handle = None

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *_):
        self.close()

"""Only compute is delegated to NeuSim: DRAM timing is never counted twice."""

from functools import lru_cache

from .upstream import add_upstream


class NeuSimCompute:
    def __init__(self, chip: dict):
        add_upstream("NEUSIM_ROOT", "third_party/NeuSim")
        from neusim.configs.chips.ChipConfig import ChipConfig

        self.config = ChipConfig(**chip)
        self.calls = 0

    @lru_cache(maxsize=4096)
    def gemm_ns(self, m: int, n: int, k: int, dtype: str) -> int:
        from neusim.npusim.backend import npusim_lib, util
        from neusim.npusim.frontend.llm_ops_lib import create_einsum_op

        op = create_einsum_op([m, k], [k, n], "MK;KN->MN", dtype=f"DT_{dtype}")
        module = util.construct_hlo_module_from_node_costs([op])
        instruction, op = npusim_lib.parse_tensor_shapes_for_node_cost(op, module)
        sa_ns, vu_ns = npusim_lib.compute_node_cost_compute_time(instruction, op, self.config)
        self.calls += 1
        return max(sa_ns, vu_ns, 1)

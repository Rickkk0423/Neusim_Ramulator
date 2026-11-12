from neusim_ramulator.compute import NeuSimCompute


def test_real_neusim_compute_and_cache():
    model = NeuSimCompute({"num_sa": 1, "num_vu": 1, "sa_dim": 16, "freq_GHz": 1.0})
    result = model.gemm_ns(32, 64, 64, "BFLOAT16")
    assert result > 0
    assert model.gemm_ns(32, 64, 64, "BFLOAT16") == result
    assert model.calls == 1


def test_frequency_changes_compute_not_memory():
    base = {"num_sa": 1, "num_vu": 1, "sa_dim": 16}
    slow = NeuSimCompute({**base, "freq_GHz": 0.5})
    fast = NeuSimCompute({**base, "freq_GHz": 1.0})
    assert slow.gemm_ns(32, 64, 64, "BFLOAT16") > fast.gemm_ns(32, 64, 64, "BFLOAT16")

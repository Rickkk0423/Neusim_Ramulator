import pytest
import yaml

from neusim_ramulator.dram import export_config


@pytest.mark.parametrize("channels", [0, -1, True, 1.5, 3])
def test_invalid_channel_counts_are_rejected(channels, tmp_path):
    with pytest.raises(ValueError, match="channels"):
        export_config({"channels": channels}, tmp_path / "dram.yaml")


@pytest.mark.parametrize("cores", [0, -1, True, 1.5, 2**31])
def test_invalid_core_counts_are_rejected(cores, tmp_path):
    with pytest.raises(ValueError, match="num_cores"):
        export_config({}, tmp_path / "dram.yaml", num_cores=cores)


def test_capacity_and_core_count_are_serialized(tmp_path):
    path = tmp_path / "dram.yaml"
    export_config({"standard": "HBM2", "channels": 4}, path, num_cores=4)
    config = yaml.safe_load(path.read_text())
    assert config["frontend"]["num_cores"] == 4
    assert config["frontend"]["address_capacity_bytes"] == 2**31

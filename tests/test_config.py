"""Tests for the config system."""

import pytest
from quantum_iql.config import IQLConfig, NetworkConfig, load_config


def test_defaults():
    from omegaconf import OmegaConf

    cfg = OmegaConf.to_object(OmegaConf.structured(IQLConfig))
    assert cfg.tau == 0.7
    assert cfg.beta == 3.0
    assert cfg.gamma == 0.99
    assert cfg.batch_size == 256
    assert cfg.value_net.hidden_dims == [256, 256]


def test_load_config_from_yaml(tmp_path):
    yaml_content = """
    tau: 0.9
    beta: 5.0
    seed: 42
    value_net:
      hidden_dims: [512, 512]
    """
    config_file = tmp_path / "test.yaml"
    config_file.write_text(yaml_content)

    cfg = load_config(str(config_file))
    assert cfg.tau == 0.9
    assert cfg.beta == 5.0
    assert cfg.seed == 42
    assert cfg.value_net.hidden_dims == [512, 512]
    # Unset keys should keep defaults
    assert cfg.gamma == 0.99


def test_load_config_with_overrides(tmp_path):
    yaml_content = "tau: 0.7\n"
    config_file = tmp_path / "test.yaml"
    config_file.write_text(yaml_content)

    cfg = load_config(str(config_file), overrides=["tau=0.95", "seed=7"])
    assert cfg.tau == 0.95
    assert cfg.seed == 7


def test_load_hopper_config():
    cfg = load_config("configs/iql_hopper.yaml")
    assert cfg.dataset_id == "mujoco/hopper/medium-v2"
    assert cfg.env_id == "Hopper-v4"
    assert cfg.use_twin_critic is True
    assert cfg.device == "auto"

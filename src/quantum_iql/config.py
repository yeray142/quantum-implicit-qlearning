"""Configuration dataclasses for IQL, loadable from YAML via OmegaConf."""

from __future__ import annotations

from dataclasses import dataclass, field

from omegaconf import DictConfig, OmegaConf


@dataclass
class NetworkConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"   # "relu" | "tanh" | "elu"
    layer_norm: bool = False


@dataclass
class IQLConfig:
    # --- Environment / Data ---
    dataset_id: str = "mujoco/hopper/medium-v2"
    env_id: str = "Hopper-v4"

    # --- IQL hyperparameters ---
    tau: float = 0.7       # expectile level (0.5 = MSE; higher = more optimistic)
    beta: float = 3.0      # inverse temperature for advantage weighting
    gamma: float = 0.99    # discount factor
    polyak: float = 0.005  # EMA rate for V-target

    # --- Optimization ---
    lr_v: float = 3e-4
    lr_q: float = 3e-4
    lr_actor: float = 3e-4
    batch_size: int = 256
    num_steps: int = 1_000_000
    warmup_steps: int = 0  # steps before actor updates begin

    # --- Networks ---
    value_net: NetworkConfig = field(default_factory=NetworkConfig)
    critic_net: NetworkConfig = field(default_factory=NetworkConfig)
    actor_net: NetworkConfig = field(default_factory=NetworkConfig)
    use_twin_critic: bool = True       # twin Q-heads to reduce overestimation
    advantage_clip: float = 100.0     # max value for exp(β·A)

    # --- Logging ---
    log_interval: int = 1_000
    eval_interval: int = 10_000
    eval_episodes: int = 10
    wandb_project: str = "quantum-iql"
    wandb_run_name: str | None = None
    wandb_offline: bool = False

    # --- Reproducibility ---
    seed: int = 0
    device: str = "auto"  # "auto" | "cpu" | "cuda"


def _parse_overrides(overrides: list[str]) -> dict:
    """Convert ["key=value", ...] strings into a nested dict via OmegaConf."""
    if not overrides:
        return {}
    return OmegaConf.to_container(OmegaConf.from_dotlist(overrides), resolve=True)


def load_config(path: str, overrides: list[str] | None = None) -> IQLConfig:
    """
    Load an IQLConfig from a YAML file, then apply optional CLI overrides.

    Args:
        path:      Path to a YAML config file.
        overrides: List of "key=value" or "nested.key=value" strings.

    Returns:
        A fully resolved IQLConfig instance.

    Example:
        cfg = load_config("configs/iql_hopper.yaml", ["tau=0.9", "seed=1"])
    """
    base: DictConfig = OmegaConf.structured(IQLConfig)
    from_file: DictConfig = OmegaConf.load(path)
    merged: DictConfig = OmegaConf.merge(base, from_file)

    if overrides:
        merged = OmegaConf.merge(merged, _parse_overrides(overrides))

    # Validate by converting back to the structured type
    cfg: IQLConfig = OmegaConf.to_object(merged)  # type: ignore[assignment]
    return cfg

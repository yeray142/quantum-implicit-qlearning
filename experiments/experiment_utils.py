"""Experiment utilities: runtime estimation and inline SPS measurement for ablation grid.

SPS is measured at the start of each run via a short warm-up benchmark, so results
reflect the actual hardware the script is running on — no hardcoded values.
"""

from __future__ import annotations

import importlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Process-level cache: mode -> measured SPS (persists across runs in same process)
_MEASURED_SPS: dict[str, float] = {}


@dataclass
class SPSResult:
    sps: float
    gpu_sps: float | None
    quantum_sps: float


def estimate_sps(mode: str, cfg: Any = None) -> SPSResult:
    """Return the best-known SPS for a mode.

    If a real benchmark has been run (``_MEASURED_SPS``), that measured value is used.
    Otherwise returns zeros — call ``measure_sps`` first to get real values.
    """
    if mode not in _MEASURED_SPS:
        return SPSResult(sps=0.0,
                         gpu_sps=0.0 if mode.startswith("quantum") else None,
                         quantum_sps=0.0 if mode.startswith("quantum") else 0.0)

    sps = _MEASURED_SPS[mode]
    return SPSResult(sps=sps,
                     gpu_sps=sps if mode.startswith("quantum") else None,
                     quantum_sps=sps if mode.startswith("quantum") else 0.0)


def measure_sps(
    mode: str,
    cfg: Any,
    steps: int = 50,
    env_cfg: dict | None = None,
) -> float:
    """Run a short warm-up benchmark and return measured SPS.

    Runs full IQL steps (V + Q + π updates) for ``steps`` iterations and measures
    wall-clock time.  No W&B logging, no eval, no checkpointing.

    Args:
        mode:     Experiment mode string (e.g. "quantum-fixed-c").
        cfg:      Fully constructed QuantumIQLConfig.
        steps:    Number of warm-up steps to run. Default 50.
        env_cfg:  Optional env registry dict (contains env_id for dimension lookup).

    Returns:
        Measured SPS (steps per second).
    """
    from quantum_iql.buffer import ReplayBuffer
    from quantum_iql.utils import make_env, set_seed

    # Determine observation and action dimensions from the environment
    if mode == "constant-v" or mode.startswith("classical"):
        obs_dim = 11   # mujoco hopper default
        act_dim = 3
    else:
        # Quantum modes — look up from env_cfg
        if env_cfg is not None:
            from gymnasium import make as gym_make
            _env = gym_make(env_cfg["env_id"])
            obs_dim = int(_env.observation_space.shape[0])
            act_dim = int(_env.action_space.shape[0])
            _env.close()
        else:
            obs_dim = 11
            act_dim = 3

    # Build a sufficiently large synthetic buffer for the configured batch size
    warmup_batch_size = getattr(cfg, "batch_size", 256)
    buffer_capacity = max(warmup_batch_size * 4, 2048)
    buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, capacity=buffer_capacity, device="cpu")
    rng = np.random.default_rng(seed=int(cfg.seed or 0))
    # Fill with enough transitions to support full-batch sampling without re-sampling
    num_warmup_samples = warmup_batch_size * 3
    all_obs = rng.standard_normal((num_warmup_samples + 1, obs_dim)).astype(np.float32)
    all_act = rng.standard_normal((num_warmup_samples, act_dim)).astype(np.float32)
    all_rew = rng.standard_normal((num_warmup_samples,)).astype(np.float32)
    all_term = np.zeros((num_warmup_samples,), dtype=np.float32)
    all_term[-1] = 1.0
    # Single episode containing all transitions
    episode = type("EpisodeData", (), {
        "observations": all_obs,
        "actions": all_act,
        "rewards": all_rew,
        "terminations": all_term,
        "truncations": np.zeros_like(all_term),
    })()
    buffer.add_from_episode(episode)

    # Add project root so top-level imports work
    _root = Path(__file__).resolve().parent.parent
    for _p in [str(_root / "src"), str(_root / "scripts"), str(_root)]:
        if _p not in sys.path:
            sys.path.insert(0, _p)

    if mode == "constant-v":
        from experiments.ablation_study import ConstantVIQLTrainer
        trainer_cls = ConstantVIQLTrainer
    else:
        from experiments.benchmark_comparison.benchmark_comparison import BenchmarkTrainer
        trainer_cls = BenchmarkTrainer

    env_id = env_cfg["env_id"] if env_cfg else "Hopper-v4"
    env = make_env(env_id, seed=int(cfg.seed or 0))
    set_seed(int(cfg.seed or 0), env=env)

    # Use the configured batch size (no override) for accurate SPS measurement
    trainer = trainer_cls(cfg, buffer, env, torch.zeros(1))

    # Warm-up: JIT-compile / allocate GPU memory
    for _ in range(3):
        trainer.train_step()

    # Timed measurement run
    t0 = time.time()
    for _ in range(steps):
        trainer.train_step()
    elapsed = time.time() - t0

    # Cleanup
    env.close()
    del trainer, env, buffer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    sps = steps / elapsed
    _MEASURED_SPS[mode] = sps
    return sps


def estimate_runtime(num_steps: int, mode: str, cfg: Any = None) -> tuple[float, float]:
    """Estimate wall-clock time for a single run in minutes."""
    result = estimate_sps(mode, cfg)
    sps = max(result.sps, 0.01)
    minutes = num_steps / sps / 60.0
    return round(minutes, 1), round(minutes / 60.0, 2)


def _build_config_for_mode(mode: str, seed: int, env_cfg: dict) -> Any:
    """Build a QuantumIQLConfig for a mode (standalone, no circular imports)."""
    from quantum_iql.quantum_config import (
        LayerwiseScheduleEntry,
        QuantumIQLConfig,
        QuantumNetConfig,
    )

    cfg = QuantumIQLConfig()
    cfg.dataset_id = env_cfg["dataset_id"]
    cfg.env_id = env_cfg["env_id"]
    cfg.tau = env_cfg.get("tau", 0.7)
    cfg.beta = env_cfg.get("beta", 3.0)
    cfg.gamma = 0.99
    cfg.polyak = 0.005
    cfg.lr_v = 3e-4
    cfg.lr_q = 3e-4
    cfg.lr_actor = 3e-4
    cfg.lr_quantum = 1e-3
    cfg.quantum_grad_clip = 1.0
    cfg.batch_size = 64  # smaller for warmup to fit small synthetic buffer
    cfg.num_steps = 100_000
    cfg.warmup_steps = 0
    cfg.use_twin_critic = True
    cfg.advantage_clip = 100.0
    cfg.critic_net.hidden_dims = [256, 256]
    cfg.critic_net.activation = "relu"
    cfg.critic_net.layer_norm = False
    cfg.actor_net.hidden_dims = [256, 256]
    cfg.actor_net.layer_norm = False
    cfg.log_interval = 1_000
    cfg.eval_interval = 5_000
    cfg.eval_episodes = 10
    cfg.seed = seed
    cfg.device = "auto"
    cfg.log_quantum_metrics = True
    cfg.stats_update_interval = 1_000

    if mode in ("constant-v", "classical"):
        cfg.mode = "classical"
        cfg.value_net.hidden_dims = [256, 256]
        cfg.value_net.activation = "relu"
        cfg.value_net.layer_norm = False
    elif mode == "classical-deep":
        cfg.mode = "classical"
        cfg.value_net.hidden_dims = [8, 8, 8]
        cfg.value_net.layer_norm = False
    else:
        # All quantum modes share the same quantum_value config
        cfg.mode = "quantum"
        cfg.quantum_value = QuantumNetConfig(
            n_qubits=8,
            n_layers=3,
            device_name="default.qubit",
            diff_method="backprop",
            running_stats=True,
            layerwise_schedule=[LayerwiseScheduleEntry(start_step=0, active_layers=3)],
        )
        cfg.quantum_batch_size = 256
        if mode == "quantum-fixed-c":
            cfg.fix_c_enabled = True
            cfg.v_freeze_steps = 1500

    return cfg


def print_grid_summary(
    grid: list[tuple[str, int]],
    num_steps: int,
    cfg_fn: callable,
    env_name: str,
    env_cfg: dict,
) -> None:
    """Print a formatted table of estimated runtimes for an ablation grid.

    Runs a 20-step benchmark for each unique mode not yet cached, then prints
    the table with real measured SPS values.  No hardcoded defaults.

    Args:
        grid:      List of (mode, seed) pairs.
        num_steps: Total steps per run.
        cfg_fn:    Callable(mode) -> QuantumIQLConfig | None (kept for compatibility).
        env_name:  Environment name (used in header).
        env_cfg:   Environment registry dict (for env_id, obs_dim, act_dim lookup).
    """
    # Deduplicate modes while preserving order
    seen: dict[str, None] = {}
    for m, _ in grid:
        if m not in seen:
            seen[m] = None
    unique_modes = list(seen.keys())

    # Measure SPS for any mode not yet cached
    for mode in unique_modes:
        if mode not in _MEASURED_SPS:
            cfg = _build_config_for_mode(mode, seed=0, env_cfg=env_cfg)
            print(f"  [SPS warmup {mode}] running 20-step benchmark...")
            sps = measure_sps(mode, cfg, steps=20, env_cfg=env_cfg)
            print(f"  [SPS warmup {mode}] → {sps:.1f} steps/s")

    print(f"\nQ-IQL Ablation Study  —  {len(grid)} run(s)  [{env_name}]\n")
    print(f"  {'#':>3}  {'mode':<24}  {'seed':>4}  {'~min':>7}  {'SPS':>8}")
    print(f"  {'─'*55}")

    total_min = 0.0
    for i, (mode, seed) in enumerate(grid, 1):
        cfg = cfg_fn(mode)
        result = estimate_sps(mode, cfg)
        est_min, _ = estimate_runtime(num_steps, mode, cfg)
        total_min += est_min
        marker = "*" if mode in _MEASURED_SPS else ""
        print(f"  {i:>3}  {mode:<24}  {seed:>4}  {est_min:>6.1f}m  {result.sps:>7.1f}{marker}")

    print(f"\n  Total ≈ {total_min:.0f} min ({total_min/60:.1f} h)")
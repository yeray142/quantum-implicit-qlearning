#!/usr/bin/env python3
"""
Ablation Study: Diagnosing and Fixing V(s) in Hybrid Q-IQL
===========================================================

Phase 1 (diagnostic):
  constant-v        -- V(s) ≡ 100 (frozen). Decisive test of V contribution.
  classical-deep    -- Depth-3 MLP [8,8,8], depth-matched to DRU.
  quantum-no-warmup -- All 3 DRU layers active from step 0.

Phase 2 (repair):
  quantum-fixed          -- Fix A (b_init=V_INIT, a_init=A_INIT) + Fix B (no warmup).
                            The primary repair experiment.
  quantum-fixed-warmup   -- Fix A only; keeps the Skolik warm-up schedule.
                            Isolates the effect of affine-head initialisation.
  quantum-fixed-c        -- Fix A + Fix B + Fix C: V-gradient freeze for first 1500 steps.
                            Allows Q to bootstrap toward r + γ·374 before V begins moving.

Usage
-----
  # All diagnostic modes (default):
  python experiments/ablation_study.py

  # Repair experiments on Hopper, seeds 0-7:
  python experiments/ablation_study.py --modes quantum-fixed quantum-fixed-c --seeds 0 1 2 3 4 5 6 7

  # Transfer: quantum-fixed + baseline on Walker2d:
  python experiments/ablation_study.py --modes quantum-fixed constant-v classical \\
      --env walker2d --seeds 0 1 2

  # Dry-run:
  python experiments/ablation_study.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import wandb

from quantum_iql.buffer import load_minari_dataset
from quantum_iql.networks import ActorNetwork, CriticNetwork, ValueNetwork
from quantum_iql.quantum_config import LayerwiseScheduleEntry, QuantumIQLConfig, QuantumNetConfig
from quantum_iql.trainer import IQLTrainer
from quantum_iql.utils import make_env, set_seed

from experiment_utils import estimate_sps, estimate_runtime, measure_sps, print_grid_summary

_EXPERIMENTS_DIR = Path(__file__).resolve().parent
_BC_PATH = _EXPERIMENTS_DIR / "benchmark_comparison" / "benchmark_comparison.py"
import importlib.util as _ilu
_bc_spec = _ilu.spec_from_file_location("benchmark_comparison", _BC_PATH)
_bc = _ilu.module_from_spec(_bc_spec)
sys.modules["benchmark_comparison"] = _bc
_bc_spec.loader.exec_module(_bc)
BenchmarkTrainer = _bc.BenchmarkTrainer
_count_params    = _bc._count_params

# ── Environment registry ──────────────────────────────────────────────────────
# V_INIT = mean_reward / (1 - gamma)  from the dataset statistics.
# A_INIT = target std(V) at correct scale, chosen to match classical V std ≈ 55.
_ENV_REGISTRY: dict[str, dict] = {
    "hopper": {
        "dataset_id": "mujoco/hopper/medium-v0",
        "env_id":     "Hopper-v4",
        "tau":   0.7,
        "beta":  3.0,
        "v_init": 374.0,   # 3.74 / 0.01
        "a_init":  55.0,
        "group":  "hopper-medium",
    },
    "walker2d": {
        "dataset_id": "mujoco/walker2d/medium-v0",
        "env_id":     "Walker2d-v4",
        "tau":   0.7,
        "beta":  3.0,
        "v_init": 595.5,   # 5.955 / 0.01
        "a_init":  55.0,
        "group":  "walker2d-medium",
    },
}

# ── Shared constants ──────────────────────────────────────────────────────────
V_CONSTANT   = 100.0          # constant used in constant-v ablation
NUM_STEPS    = 100_000
EVAL_INTERVAL= 5_000
LOG_INTERVAL = 1_000
WANDB_PROJECT= "quantum-iql"
DEEP_HIDDEN  = [8, 8, 8]

ALL_MODES = [
    "constant-v", "classical", "classical-deep",
    "quantum-no-warmup", "quantum-fixed", "quantum-fixed-warmup", "quantum-fixed-c",
]

class ConstantValueNetwork(nn.Module):
    """V(s) = constant for all s. No trainable parameters."""

    def __init__(self, constant: float = V_CONSTANT) -> None:
        super().__init__()
        self.register_buffer("constant",
                             torch.tensor([[constant]], dtype=torch.float32))
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.constant.expand(obs.shape[0], 1)


# ── ConstantV trainer ─────────────────────────────────────────────────────────

class ConstantVIQLTrainer(IQLTrainer):
    """IQL with V frozen at a constant. V update is a no-op."""

    def __init__(self, config, buffer, env, checkpoint_dir: Path) -> None:
        self._checkpoint_dir = checkpoint_dir
        self._v_constant = V_CONSTANT
        super().__init__(config, buffer, env)

    def _build_networks(self) -> None:
        obs_dim = self.buffer.obs_dim
        act_dim = self.buffer.act_dim
        qcfg = self.cfg.critic_net
        acfg = self.cfg.actor_net
        self.value_net = ConstantValueNetwork(self._v_constant).to(self.device)
        self.critic_net = CriticNetwork(
            obs_dim, act_dim,
            hidden_dims=qcfg.hidden_dims, activation=qcfg.activation,
            layer_norm=qcfg.layer_norm, use_twin=self.cfg.use_twin_critic,
        ).to(self.device)
        self.actor_net = ActorNetwork(
            obs_dim, act_dim,
            hidden_dims=acfg.hidden_dims, activation=acfg.activation,
            layer_norm=acfg.layer_norm,
        ).to(self.device)

    def _build_targets(self) -> None:
        self.value_target = ConstantValueNetwork(self._v_constant).to(self.device)

    def _build_optimizers(self) -> None:
        self.value_optimizer  = optim.Adam(self.value_net.parameters(), lr=self.cfg.lr_v)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.cfg.lr_q)
        self.actor_optimizer  = optim.Adam(self.actor_net.parameters(), lr=self.cfg.lr_actor)

    def update_value(self, batch) -> dict[str, float]:
        return {"loss/value": 0.0}

    def update_targets(self) -> None:
        pass

    def save_checkpoint(self, filename: str) -> None:
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "step": self._step, "mode": "constant-v",
            "v_constant": self._v_constant,
            "critic_net": self.critic_net.state_dict(),
            "actor_net":  self.actor_net.state_dict(),
        }, self._checkpoint_dir / filename)


# ── QuantumFixedTrainer: Fix A + (optional) Fix B ────────────────────────────

class QuantumFixedTrainer(BenchmarkTrainer):
    """Quantum IQL with corrected affine-head initialisation.

    Fix A: initialise b = v_init (geometric-sum return estimate for this dataset)
           so V starts at the correct value scale, removing the scale bottleneck.
    Fix B (optional): no layerwise warm-up — all 3 layers active from step 0.
           Controlled by the layerwise_schedule in the config.
    """

    def __init__(self, config, buffer, env, checkpoint_dir: Path,
                 v_init: float = 374.0, a_init: float = 55.0) -> None:
        self._v_init = v_init
        self._a_init = a_init
        super().__init__(config, buffer, env, checkpoint_dir)

    def _build_networks(self) -> None:
        super()._build_networks()
        # Reinitialise the affine output head of the quantum circuit.
        # This is called before _build_targets, so hard_update will propagate
        # the initialised values to value_target automatically.
        if hasattr(self.value_net, "b"):
            with torch.no_grad():
                self.value_net.b.fill_(self._v_init)
                self.value_net.a.fill_(self._a_init)
            print(f"  [QuantumFixed] affine head init: a={self._a_init:.1f}  "
                  f"b={self._v_init:.1f}  →  V ∈ [{self._v_init-self._a_init:.1f}, "
                  f"{self._v_init+self._a_init:.1f}]")


# ── Config builder ────────────────────────────────────────────────────────────

def _base_config(seed: int, env_cfg: dict) -> QuantumIQLConfig:
    cfg = QuantumIQLConfig()
    cfg.dataset_id = env_cfg["dataset_id"]
    cfg.env_id     = env_cfg["env_id"]
    cfg.tau    = env_cfg["tau"]
    cfg.beta   = env_cfg["beta"]
    cfg.gamma  = 0.99
    cfg.polyak = 0.005
    cfg.lr_v = 3e-4;  cfg.lr_q = 3e-4;  cfg.lr_actor = 3e-4
    cfg.lr_quantum    = 1e-3
    cfg.quantum_grad_clip = 1.0
    cfg.batch_size    = 256
    cfg.num_steps     = NUM_STEPS
    cfg.warmup_steps  = 0
    cfg.use_twin_critic  = True
    cfg.advantage_clip   = 100.0
    cfg.critic_net.hidden_dims = [256, 256]; cfg.critic_net.activation = "relu"
    cfg.critic_net.layer_norm  = False
    cfg.actor_net.hidden_dims  = [256, 256]; cfg.actor_net.activation = "relu"
    cfg.actor_net.layer_norm   = False
    cfg.log_interval  = LOG_INTERVAL
    cfg.eval_interval = EVAL_INTERVAL
    cfg.eval_episodes = 10
    cfg.wandb_project = WANDB_PROJECT
    cfg.log_quantum_metrics    = True
    cfg.stats_update_interval  = 1_000
    cfg.seed   = seed
    cfg.device = "auto"
    return cfg


def _warmup_schedule(total_steps: int) -> list[LayerwiseScheduleEntry]:
    """10%→L2, 30%→L3 proportional schedule (original benchmark config)."""
    l2 = max(1, round(total_steps * 0.10))
    l3 = max(l2 + 1, round(total_steps * 0.30))
    return [
        LayerwiseScheduleEntry(start_step=0,  active_layers=1),
        LayerwiseScheduleEntry(start_step=l2, active_layers=2),
        LayerwiseScheduleEntry(start_step=l3, active_layers=3),
    ]


def _quantum_cfg(use_warmup: bool, total_steps: int) -> QuantumNetConfig:
    schedule = _warmup_schedule(total_steps) if use_warmup else [
        LayerwiseScheduleEntry(start_step=0, active_layers=3),
    ]
    return QuantumNetConfig(
        n_qubits=8, n_layers=3,
        device_name="default.qubit", diff_method="backprop",
        running_stats=True, layerwise_schedule=schedule,
    )


def _build_config(mode: str, seed: int, env_cfg: dict) -> QuantumIQLConfig:
    cfg = _base_config(seed, env_cfg)

    if mode == "constant-v":
        cfg.mode = "classical"
        cfg.value_net.hidden_dims = [256, 256]

    elif mode == "classical":
        cfg.mode = "classical"
        cfg.value_net.hidden_dims = [256, 256]
        cfg.value_net.activation  = "relu"
        cfg.value_net.layer_norm  = False

    elif mode == "classical-deep":
        cfg.mode = "classical"
        cfg.value_net.hidden_dims = DEEP_HIDDEN
        cfg.value_net.activation  = "relu"
        cfg.value_net.layer_norm  = False

    elif mode == "quantum-no-warmup":
        cfg.mode = "quantum"
        cfg.quantum_value = _quantum_cfg(use_warmup=False, total_steps=NUM_STEPS)
        cfg.quantum_batch_size = 256

    elif mode == "quantum-fixed":
        cfg.mode = "quantum"
        cfg.quantum_value = _quantum_cfg(use_warmup=False, total_steps=NUM_STEPS)
        cfg.quantum_batch_size = 256

    elif mode == "quantum-fixed-warmup":
        cfg.mode = "quantum"
        cfg.quantum_value = _quantum_cfg(use_warmup=True, total_steps=NUM_STEPS)
        cfg.quantum_batch_size = 256

    elif mode == "quantum-fixed-c":
        cfg.mode = "quantum"
        cfg.quantum_value = _quantum_cfg(use_warmup=False, total_steps=NUM_STEPS)
        cfg.quantum_batch_size = 256
        # Fix C: freeze V gradient for first v_freeze_steps to mitigate cold-start overshoot
        cfg.fix_c_enabled  = True
        cfg.v_freeze_steps = 1500

    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    return cfg


# ── Experiment runner ─────────────────────────────────────────────────────────

def run_ablation(mode: str, seed: int, env_name: str,
                 wandb_offline: bool = False) -> None:
    env_cfg   = _ENV_REGISTRY[env_name]
    dataset_q = env_cfg["dataset_id"].split("/")[-1].replace("-v0", "")  # "medium"
    run_name  = f"{mode}-{env_name}-{dataset_q}-s{seed}"
    group     = env_cfg["group"]
    v_init    = env_cfg["v_init"]
    a_init    = env_cfg["a_init"]

    print(f"\n{'='*65}")
    print(f"  {run_name}  ({NUM_STEPS:,} steps)")
    print(f"{'='*65}")

    set_seed(seed)
    buffer = load_minari_dataset(env_cfg["dataset_id"], device="cpu")
    cfg    = _build_config(mode, seed, env_cfg)

    checkpoint_dir = (
        Path(__file__).parent / "checkpoints"
        / env_name / dataset_q / mode / f"seed_{seed}"
    )

    wandb.init(
        project=WANDB_PROJECT, name=run_name, group=group,
        tags=[mode, env_name, dataset_q, f"seed={seed}", "ablation"],
        config={
            "mode": mode, "env": env_name, "dataset_quality": dataset_q,
            "seed": seed,
            "v_init": v_init if "fixed" in mode else None,
            "a_init": a_init if "fixed" in mode else None,
            "v_constant": V_CONSTANT if mode == "constant-v" else None,
        },
        mode="offline" if wandb_offline else "online",
        reinit=True,
    )

    gymnasium_env = make_env(env_cfg["env_id"], seed=seed)
    set_seed(seed, env=gymnasium_env)

    if mode == "constant-v":
        trainer = ConstantVIQLTrainer(cfg, buffer, gymnasium_env, checkpoint_dir)
        v_params = 0

    elif mode in ("quantum-fixed", "quantum-fixed-warmup", "quantum-fixed-c"):
        trainer = QuantumFixedTrainer(
            cfg, buffer, gymnasium_env, checkpoint_dir,
            v_init=v_init, a_init=a_init,
        )
        v_params = _count_params(trainer.value_net)

    else:
        trainer = BenchmarkTrainer(cfg, buffer, gymnasium_env, checkpoint_dir)
        v_params = _count_params(trainer.value_net)

    wandb.config.update({
        "value_net_params": v_params,
        "v_init_actual": v_init if "fixed" in mode else None,
        "fix_c_enabled": cfg.fix_c_enabled if "fixed-c" in mode else None,
        "v_freeze_steps": cfg.v_freeze_steps if "fixed-c" in mode else None,
    }, allow_val_change=True)
    print(f"  value_net params : {v_params}")

    # ── Inline SPS measurement (first run of this mode in this process) ─────────
    # Re-use measured value if already cached; otherwise run 50-step warmup.
    from experiment_utils import _MEASURED_SPS as _sps_cache
    if mode not in _sps_cache:
        print(f"  [SPS benchmark] running 50-step warmup...")
        measured = measure_sps(mode, cfg, steps=50)
        print(f"  [SPS benchmark] measured = {measured:.1f} steps/s  (cached for remaining runs)")

    trainer.train()
    gymnasium_env.close()
    wandb.finish()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Q-IQL ablation study — diagnostic and repair experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--modes", nargs="+", default=["constant-v", "classical-deep", "quantum-no-warmup"],
        choices=ALL_MODES,
        help="Modes to run. Default: diagnostic trio.",
    )
    p.add_argument(
        "--env", default="hopper", choices=list(_ENV_REGISTRY),
        help="Environment (default: hopper). Use 'walker2d' for transfer test.",
    )
    p.add_argument("--seeds",         nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--wandb-offline", action="store_true")
    p.add_argument("--dry-run",       action="store_true")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    env_cfg = _ENV_REGISTRY[args.env]

    grid = [(mode, seed) for mode in args.modes for seed in args.seeds]

    def _cfg_for_mode(mode: str):
        return _build_config(mode, seed=0, env_cfg=env_cfg)

    print_grid_summary(grid, NUM_STEPS, _cfg_for_mode, args.env, env_cfg)

    if "quantum-fixed" in args.modes or "quantum-fixed-warmup" in args.modes:
        v_init = env_cfg["v_init"]
        a_init = env_cfg["a_init"]
        print(f"  Fix A: b_init={v_init:.1f}  a_init={a_init:.1f}  "
              f"→  V_init ∈ [{v_init-a_init:.1f}, {v_init+a_init:.1f}]")
        print(f"  Fix B: no-warmup (all 3 layers active from step 0)")

    if args.dry_run:
        print("\nDry-run: exiting.")
        return

    failed: list[tuple] = []
    for i, (mode, seed) in enumerate(grid, 1):
        print(f"\n[{i}/{len(grid)}] {mode} / {args.env} / seed={seed}")
        try:
            run_ablation(mode, seed, env_name=args.env,
                         wandb_offline=args.wandb_offline)
        except Exception as exc:
            import traceback
            print(f"  ERROR: {exc}")
            traceback.print_exc()
            failed.append((mode, seed, str(exc)))
            try:
                wandb.finish(exit_code=1)
            except Exception:
                pass

    print(f"\n{'='*65}")
    print(f"  Done: {len(grid)-len(failed)}/{len(grid)} succeeded")
    if failed:
        for mode, seed, msg in failed:
            print(f"    FAILED: {mode}/seed={seed}: {msg}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()

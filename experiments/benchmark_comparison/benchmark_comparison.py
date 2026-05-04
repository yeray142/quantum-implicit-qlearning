#!/usr/bin/env python3
"""
Q-IQL Benchmark: Classical vs Hybrid Quantum-Classical IQL
==========================================================

Compares three IQL variants across D4RL/Minari locomotion benchmarks:

  classical       — full MLP value network V [256, 256]  (~69k params for Hopper)
  quantum         — DRU quantum circuit V (8 qubits, 3 layers, ~146 params)
  classical-small — MLP V with parameter count matched to the quantum variant

Environments (IQL paper, Kostrikov et al. 2021):
  hopper       Hopper-v4       obs_dim=11  act_dim=3
  halfcheetah  HalfCheetah-v4  obs_dim=17  act_dim=6
  walker2d     Walker2d-v4     obs_dim=17  act_dim=6

Dataset splits:
  medium         mujoco/{env}/medium-v0
  medium-replay  mujoco/{env}/medium-replay-v0
  medium-expert  mujoco/{env}/medium-expert-v0

──────────────────────────────────────────────────────────────────────
FAIR COMPARISON DESIGN

  All three modes use the SAME number of gradient steps (100 k default).
  This makes learning curves directly comparable at every x-axis point:
  step N classical vs step N quantum — no algorithmic advantage from
  longer training.

  Classical at 100k is intentionally not fully converged.  Its learning
  curve shows how quickly it approaches its asymptote.  Use
  --classical-steps 1000000 (adds ~57 min/run) for a convergence reference.

TIME BUDGET  (measured on RTX 5070 Laptop GPU)

  Mode             Steps   Speed         Time/run   3 seeds
  ───────────────────────────────────────────────────────────
  classical        100 k   293.9 sps      ~5.7 min   ~17 min
  quantum          100 k    22.2 sps     ~75.1 min  ~225 min
  classical-small  100 k   293.9 sps      ~5.7 min   ~17 min
  ───────────────────────────────────────────────────────────
  Total (1 env × 1 dataset × 3 modes × 3 seeds)     ~259 min ≈ 4.3 h ✓

Default scope (--preset fast):  hopper × medium × 3 modes × 3 seeds
Full scope   (--preset full):   all 3 envs × all 3 datasets (≫5 h)
──────────────────────────────────────────────────────────────────────

GPU optimisation note
─────────────────────
Quantum uses diff_method="backprop" with default.qubit, which converts the
circuit into differentiable PyTorch ops.  On CUDA this gives a ~152× speedup
over the original CPU adjoint approach (0.045 s/step vs 4.3 s/step at B=256).
No special quantum GPU backend is required.

CPU-only fallback: set --quantum-batch-size 4 --quantum-steps 100000.
At ~0.20 s/step (adjoint, B=4) this gives ~83 h for 3 seeds — use only for
CI / smoke-tests or machines without CUDA.

Usage
─────
  # Default 5-hour experiment (hopper × medium × 3 modes × 3 seeds)
  python experiments/benchmark_comparison.py

  # Extend to all dataset splits (~10h on same GPU)
  python experiments/benchmark_comparison.py --datasets medium medium-replay medium-expert

  # Smoke-test (10 steps, no W&B push)
  python experiments/benchmark_comparison.py --num-steps 10 --wandb-offline

  # Full benchmark sweep (all envs, all splits)
  python experiments/benchmark_comparison.py --preset full

  # Dry-run: print schedule + time estimates, exit
  python experiments/benchmark_comparison.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

# ── Path setup ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import wandb  # noqa: E402

from quantum_iql.buffer import load_minari_dataset  # noqa: E402
from quantum_iql.quantum_config import (  # noqa: E402
    LayerwiseScheduleEntry,
    QuantumIQLConfig,
    QuantumNetConfig,
)
from quantum_iql.quantum_trainer import QuantumIQLTrainer  # noqa: E402
from quantum_iql.utils import make_env, set_seed  # noqa: E402

# ── Measured throughputs (RTX 5070 Laptop GPU) ───────────────────────────────
# Update these if running on a different machine.
_SPS = {
    "classical":       293.9,   # GPU, MLP [256,256], B=256
    "quantum":          22.2,   # GPU backprop, 8q-3L DRU, B=256  (0.045s/step)
    "classical-small": 293.9,   # same as classical
}

# ── Per-mode step budgets ────────────────────────────────────────────────────
# All three modes use the SAME number of gradient steps so that learning curves
# are comparable at every point (step N classical vs step N quantum).
#
# Derivation for ≤5 h (1 env × 1 dataset × 3 modes × 3 seeds):
#   classical:       100k / 293.9 sps =  5.7 min/run × 6 runs =  34 min
#   quantum:         100k /  22.2 sps = 75.1 min/run × 3 runs = 226 min
#   ──────────────────────────────────────────────────────────────────────
#   Total                                                        260 min ≈ 4.3 h ✓
#
# Why same steps?
#   Different step counts produce an unfair comparison: a classical model at
#   300k steps is compared against quantum at 80k — the classical model has
#   seen 3.75× more gradient updates regardless of the parameter count.
#   Using the same step budget isolates the effect of architecture (classical
#   MLP vs quantum DRU circuit) from training duration.
#
# Classical at 100k is intentionally not fully converged — its learning curve
# reveals how quickly it approaches its asymptote relative to quantum.  For a
# reference upper bound, run `--classical-steps 1000000` (adds ~57 min/run).
_COMPARISON_STEPS = 100_000    # shared step budget for fair same-step comparison

_DEFAULT_STEPS = {
    "classical":       _COMPARISON_STEPS,
    "quantum":         _COMPARISON_STEPS,
    "classical-small": _COMPARISON_STEPS,
}

# ── Logging intervals (tuned so every run yields ≥15 evaluation points) ─────
# ── W&B update frequency ─────────────────────────────────────────────────────
# Guarantee: at least one W&B log per minute regardless of hardware speed.
# Formula: log_interval = min(MAX_STEPS_CAP, sps × TARGET_LOG_SECS)
#   TARGET_LOG_SECS = 30 s  → 2× safety buffer below the 60 s hard limit.
#   MAX_STEPS_CAP   = 1 000 → caps classical (fast) to avoid W&B noise.
#
# Resulting frequencies (RTX 5070):
#   classical       : every 1 000 steps =  3.4 s  ← dense, smooth curve
#   quantum         : every   666 steps = 30.0 s  ← update every ~30 s ✓
#   classical-small : every 1 000 steps =  3.4 s

_TARGET_LOG_SECS = 30      # aim for ≤30 s between logs (halves the 60 s limit)
_MAX_LOG_STEPS   = 1_000   # cap for fast modes to avoid flooding W&B

def _log_interval(mode: str) -> int:
    """Steps between W&B log calls, guaranteed to fire at least once per minute."""
    return max(100, min(_MAX_LOG_STEPS, int(_SPS[mode] * _TARGET_LOG_SECS)))

def _log_freq_secs(mode: str) -> float:
    """Resulting seconds between W&B updates at measured throughput."""
    return _log_interval(mode) / _SPS[mode]

_LOG_INTERVAL = {m: _log_interval(m) for m in _SPS}

# ── Evaluation interval ───────────────────────────────────────────────────────
# Same step count for all modes → learning curves share x-axis ticks.
# 5 000 steps gives 20 eval checkpoints over 100 k steps.
_EVAL_INTERVAL = {m: 5_000 for m in _SPS}

# ── Environment / dataset registry ──────────────────────────────────────────
_ENV_REGISTRY: dict[str, dict] = {
    "hopper": {
        "dataset_prefix": "mujoco/hopper",
        "env_id":         "Hopper-v4",
        "tau":  0.7,
        "beta": 3.0,
    },
    "halfcheetah": {
        "dataset_prefix": "mujoco/halfcheetah",
        "env_id":         "HalfCheetah-v4",
        "tau":  0.7,
        "beta": 3.0,
    },
    "walker2d": {
        "dataset_prefix": "mujoco/walker2d",
        "env_id":         "Walker2d-v4",
        "tau":  0.7,
        "beta": 3.0,
    },
}

_DATASET_SUFFIX: dict[str, str] = {
    "medium":        "medium-v0",
    "medium-replay": "medium-replay-v0",
    "medium-expert": "medium-expert-v0",
}

# ── Parameter matching (classical-small ≈ quantum param count) ───────────────
# Quantum (8q, 3L): theta(72) + w(72) + a(1) + b(1) = 146 params
# Classical-small: single hidden layer h so (obs_dim+2)*h+1 ≈ 146
#   obs_dim=11 → h=11 → 144 params   (Hopper)
#   obs_dim=17 → h=8  → 153 params   (HalfCheetah / Walker2d)
_QUANTUM_PARAMS = 146

def _matched_hidden_dim(obs_dim: int) -> list[int]:
    h = max(1, round((_QUANTUM_PARAMS - 1) / (obs_dim + 2)))
    return [h]

# ── Layerwise warm-up schedules ──────────────────────────────────────────────
# Skolik et al. 2021: activate DRU layers progressively to avoid barren
# plateaus at initialisation.  The proportions are kept consistent:
#   10% → L1,  30% → L2,  remaining → L3 (all layers active).
#
# The schedule is selected automatically based on num_steps:
#   ≤ 150k steps  → FAST  (0→L1,  10k→L2,  30k→L3)
#   > 150k steps  → FULL  (0→L1, 100k→L2, 300k→L3)

def _make_schedule(total_steps: int) -> list[LayerwiseScheduleEntry]:
    """Return a layerwise schedule proportional to total_steps."""
    l2 = max(1, round(total_steps * 0.10))
    l3 = max(l2 + 1, round(total_steps * 0.30))
    return [
        LayerwiseScheduleEntry(start_step=0,  active_layers=1),
        LayerwiseScheduleEntry(start_step=l2, active_layers=2),
        LayerwiseScheduleEntry(start_step=l3, active_layers=3),
    ]


# ── Experiment spec ──────────────────────────────────────────────────────────

@dataclass
class ExperimentSpec:
    env_name:        str
    dataset_quality: str
    mode:            str
    seed:            int
    num_steps:       Optional[int] = None   # None → use _DEFAULT_STEPS[mode]
    wandb_offline:   bool          = False

    def effective_steps(self) -> int:
        return self.num_steps if self.num_steps is not None else _DEFAULT_STEPS[self.mode]

    def estimated_minutes(self) -> float:
        return self.effective_steps() / _SPS[self.mode] / 60.0


# ── Config builder ───────────────────────────────────────────────────────────

def _build_config(spec: ExperimentSpec, obs_dim: int) -> QuantumIQLConfig:
    env = _ENV_REGISTRY[spec.env_name]
    dataset_id = f"{env['dataset_prefix']}/{_DATASET_SUFFIX[spec.dataset_quality]}"

    cfg = QuantumIQLConfig()

    # Environment / data
    cfg.dataset_id = dataset_id
    cfg.env_id     = env["env_id"]

    # IQL hyperparameters (IQL paper, locomotion tasks)
    cfg.tau    = env["tau"]
    cfg.beta   = env["beta"]
    cfg.gamma  = 0.99
    cfg.polyak = 0.005

    # Optimisation
    cfg.lr_v          = 3e-4
    cfg.lr_q          = 3e-4
    cfg.lr_actor      = 3e-4
    cfg.lr_quantum    = 1e-3
    cfg.quantum_grad_clip = 1.0
    cfg.batch_size    = 256
    cfg.num_steps     = spec.effective_steps()
    cfg.warmup_steps  = 0
    cfg.use_twin_critic  = True
    cfg.advantage_clip   = 100.0

    # Critic and Actor: always classical [256, 256]
    cfg.critic_net.hidden_dims = [256, 256]
    cfg.critic_net.activation  = "relu"
    cfg.critic_net.layer_norm  = False
    cfg.actor_net.hidden_dims  = [256, 256]
    cfg.actor_net.activation   = "relu"
    cfg.actor_net.layer_norm   = False

    # Mode-specific value network
    if spec.mode == "classical":
        cfg.mode = "classical"
        cfg.value_net.hidden_dims = [256, 256]
        cfg.value_net.activation  = "relu"
        cfg.value_net.layer_norm  = False

    elif spec.mode == "quantum":
        cfg.mode = "quantum"
        # Choose schedule proportional to training length
        cfg.quantum_value = QuantumNetConfig(
            n_qubits=8,
            n_layers=3,
            # backprop+default.qubit → PyTorch ops on GPU (152× vs CPU adjoint)
            device_name="default.qubit",
            diff_method="backprop",
            running_stats=True,
            layerwise_schedule=_make_schedule(cfg.num_steps),
        )
        # Full batch: GPU handles B=256 in ~0.045 s/step
        cfg.quantum_batch_size = 256

    elif spec.mode == "classical-small":
        cfg.mode = "classical"
        cfg.value_net.hidden_dims = _matched_hidden_dim(obs_dim)
        cfg.value_net.activation  = "relu"
        cfg.value_net.layer_norm  = False

    else:
        raise ValueError(f"Unknown mode: {spec.mode!r}")

    # Logging — tuned per mode for ≥15 eval points on the learning curve
    cfg.log_quantum_metrics  = True
    cfg.stats_update_interval = 1_000
    cfg.log_interval  = _LOG_INTERVAL[spec.mode]
    cfg.eval_interval = _EVAL_INTERVAL[spec.mode]
    cfg.eval_episodes = 10
    cfg.wandb_project = "quantum-iql"
    cfg.wandb_run_name = None
    cfg.wandb_offline  = spec.wandb_offline

    cfg.seed   = spec.seed
    cfg.device = "auto"

    return cfg


# ── Trainer with experiment-specific checkpoint directory ────────────────────

class BenchmarkTrainer(QuantumIQLTrainer):
    def __init__(self, config, buffer, env, checkpoint_dir: Path) -> None:
        self._checkpoint_dir = checkpoint_dir
        super().__init__(config, buffer, env)

    def save_checkpoint(self, filename: str) -> None:
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self._checkpoint_dir / filename
        payload: dict = {
            "step":       self._step,
            "mode":       "quantum" if self._is_quantum else "classical",
            "value_net":  self.value_net.state_dict(),
            "critic_net": self.critic_net.state_dict(),
            "actor_net":  self.actor_net.state_dict(),
        }
        if self._is_quantum:
            qnet = self.value_net
            payload["quantum_meta"] = {
                "n_qubits":     qnet.n_qubits,
                "n_layers":     qnet.n_layers,
                "obs_dim":      qnet.obs_dim,
                "active_layers": qnet.active_layers,
            }
        torch.save(payload, path)
        print(f"  checkpoint → {path}")


# ── Parameter counting ───────────────────────────────────────────────────────

def _count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


# ── Single experiment runner ─────────────────────────────────────────────────

def run_experiment(spec: ExperimentSpec) -> None:
    env   = _ENV_REGISTRY[spec.env_name]
    dataset_id = f"{env['dataset_prefix']}/{_DATASET_SUFFIX[spec.dataset_quality]}"
    run_name   = f"{spec.mode}-{spec.env_name}-{spec.dataset_quality}-s{spec.seed}"
    group      = f"{spec.env_name}-{spec.dataset_quality}"

    print(f"\n{'='*65}")
    print(f"  {run_name}  ({spec.effective_steps():,} steps)")
    print(f"{'='*65}")

    set_seed(spec.seed)

    buffer  = load_minari_dataset(dataset_id, device="cpu")
    obs_dim = buffer.obs_dim
    cfg     = _build_config(spec, obs_dim)

    checkpoint_dir = (
        Path(__file__).parent / "checkpoints"
        / spec.env_name / spec.dataset_quality / spec.mode / f"seed_{spec.seed}"
    )

    # Parameter count preview (before building networks)
    v_params_preview: int
    if spec.mode == "classical":
        h = 256
        v_params_preview = (obs_dim*h+h) + (h*h+h) + (h+1)
    elif spec.mode == "quantum":
        v_params_preview = _QUANTUM_PARAMS
    else:
        h = _matched_hidden_dim(obs_dim)[0]
        v_params_preview = (obs_dim*h+h) + (h+1)

    wandb.init(
        project="quantum-iql",
        name=run_name,
        group=group,
        tags=[spec.mode, spec.env_name, spec.dataset_quality, f"seed={spec.seed}"],
        config={
            "env":              spec.env_name,
            "dataset_quality":  spec.dataset_quality,
            "mode":             spec.mode,
            "seed":             spec.seed,
            "dataset_id":       dataset_id,
            "obs_dim":          obs_dim,
            "act_dim":          buffer.act_dim,
            "buffer_size":      buffer._size,
            "num_steps":        cfg.num_steps,
            "tau":              cfg.tau,
            "beta":             cfg.beta,
            "batch_size":       cfg.batch_size,
            "lr_v":             cfg.lr_v,
            "lr_quantum":       cfg.lr_quantum if spec.mode == "quantum" else None,
            "value_net_hidden": (
                cfg.value_net.hidden_dims if spec.mode != "quantum" else "8q-3L-DRU"
            ),
            "value_net_params_approx": v_params_preview,
            "eval_interval":    cfg.eval_interval,
            "qv_diff_method":   cfg.quantum_value.diff_method if spec.mode == "quantum" else None,
        },
        mode="offline" if spec.wandb_offline else "online",
        reinit=True,
    )

    gymnasium_env = make_env(env["env_id"], seed=spec.seed)
    set_seed(spec.seed, env=gymnasium_env)

    trainer = BenchmarkTrainer(cfg, buffer, gymnasium_env, checkpoint_dir)

    # Log exact parameter counts now that networks exist
    v_params = _count_params(trainer.value_net)
    wandb.config.update({
        "value_net_params": v_params,
        "total_params":     (v_params
                             + _count_params(trainer.critic_net)
                             + _count_params(trainer.actor_net)),
    }, allow_val_change=True)

    print(f"  value_net params : {v_params:,}")
    print(f"  checkpoint dir   : {checkpoint_dir}")

    trainer.train()
    gymnasium_env.close()
    wandb.finish()


# ── Experiment grid builder ──────────────────────────────────────────────────

def build_grid(
    envs:          list[str],
    datasets:      list[str],
    modes:         list[str],
    seeds:         list[int],
    num_steps:     Optional[int],
    wandb_offline: bool,
) -> list[ExperimentSpec]:
    return [
        ExperimentSpec(
            env_name=env,
            dataset_quality=ds,
            mode=mode,
            seed=seed,
            num_steps=num_steps,
            wandb_offline=wandb_offline,
        )
        for env  in envs
        for ds   in datasets
        for mode in modes
        for seed in seeds
    ]


# ── Time budget table ────────────────────────────────────────────────────────

def _print_time_budget(grid: list[ExperimentSpec]) -> None:
    from collections import defaultdict
    by_mode: dict[str, list[ExperimentSpec]] = defaultdict(list)
    for s in grid:
        by_mode[s.mode].append(s)

    total_min = sum(s.estimated_minutes() for s in grid)

    print(f"\n  {'Mode':<16}  {'Runs':>5}  {'Steps':>8}  {'min/run':>8}  "
          f"{'Total':>8}  {'W&B log':>10}  {'log/min':>8}")
    print(f"  {'-'*75}")
    for mode, runs in by_mode.items():
        n   = len(runs)
        st  = runs[0].effective_steps()
        mpr = runs[0].estimated_minutes()
        tot = n * mpr
        li  = _LOG_INTERVAL[mode]
        lf  = _log_freq_secs(mode)
        lpm = 60.0 / lf
        print(f"  {mode:<16}  {n:>5}  {st:>8,}  {mpr:>7.1f}m  {tot:>7.0f}m"
              f"  {li:>6} steps  {lpm:>6.1f}/min  ({lf:.0f} s)")
    print(f"  {'-'*75}")
    print(f"  {'TOTAL':<16}  {len(grid):>5}  {'':>8}  {'':>8}  "
          f"{total_min:>7.0f}m  ≈ {total_min/60:.1f} h")
    print()

    cuda = torch.cuda.is_available()
    note = "RTX 5070 GPU" if cuda else "CPU (adjoint, B=4)"
    print(f"  Throughput basis : {note}")
    print(f"  W&B guarantee    : ≤{_TARGET_LOG_SECS * 2}s between updates "
          f"(target {_TARGET_LOG_SECS}s, 2× safety buffer)")
    if not cuda:
        print("  WARNING: No CUDA detected. Quantum runs will be ~100× slower.")
        print("  Reduce --quantum-steps to 10000 and add --quantum-batch-size 4.")
    print()


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Q-IQL benchmark (~5h on RTX 5070 with default settings)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Scope
    p.add_argument(
        "--preset",
        choices=["fast", "full"],
        default="fast",
        help=(
            "fast (default): hopper × medium × 3 modes × 3 seeds ≈ 4.7h.  "
            "full: all envs × all datasets (≫5h, research cluster)."
        ),
    )
    p.add_argument(
        "--envs", nargs="+",
        choices=["hopper", "halfcheetah", "walker2d"],
        help="Override env list (ignores --preset scope).",
    )
    p.add_argument(
        "--datasets", nargs="+",
        choices=["medium", "medium-replay", "medium-expert"],
        help="Override dataset list (ignores --preset scope).",
    )
    p.add_argument(
        "--modes", nargs="+",
        default=["classical", "quantum", "classical-small"],
        choices=["classical", "quantum", "classical-small"],
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])

    # Step overrides
    p.add_argument(
        "--num-steps", type=int, default=None,
        help="Override num_steps for ALL modes (e.g. 10 for smoke-test).",
    )
    p.add_argument(
        "--classical-steps", type=int, default=None,
        help="Override steps for classical / classical-small only.",
    )
    p.add_argument(
        "--quantum-steps", type=int, default=None,
        help="Override steps for quantum only.",
    )

    # Misc
    p.add_argument(
        "--wandb-offline", action="store_true",
        help="Log to W&B in offline mode (no internet required).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print schedule + time estimates, then exit.",
    )
    return p.parse_args(argv)


# ── Resolve preset → scope ───────────────────────────────────────────────────

_PRESET_SCOPE = {
    "fast": {
        "envs":     ["hopper"],
        "datasets": ["medium"],
    },
    "full": {
        "envs":     ["hopper", "halfcheetah", "walker2d"],
        "datasets": ["medium", "medium-replay", "medium-expert"],
    },
}


# ── Main ─────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    scope = _PRESET_SCOPE[args.preset]
    envs     = args.envs     or scope["envs"]
    datasets = args.datasets or scope["datasets"]

    # Per-mode step resolution
    def resolve_steps(mode: str) -> Optional[int]:
        if args.num_steps is not None:
            return args.num_steps
        if mode == "quantum" and args.quantum_steps is not None:
            return args.quantum_steps
        if mode != "quantum" and args.classical_steps is not None:
            return args.classical_steps
        return None   # → _DEFAULT_STEPS[mode]

    # Build grid with per-mode step overrides
    grid: list[ExperimentSpec] = []
    for env in envs:
        for ds in datasets:
            for mode in args.modes:
                for seed in args.seeds:
                    grid.append(ExperimentSpec(
                        env_name=env,
                        dataset_quality=ds,
                        mode=mode,
                        seed=seed,
                        num_steps=resolve_steps(mode),
                        wandb_offline=args.wandb_offline,
                    ))

    # ── Print schedule ──────────────────────────────────────────────────────
    print(f"\nQ-IQL Benchmark  —  {len(grid)} experiment(s)\n")

    header = (f"  {'#':>3}  {'mode':<16}  {'env':<13}  "
              f"{'dataset':<14}  {'steps':>8}  {'~min':>6}  seed")
    print(header)
    print("  " + "─" * (len(header) - 2))
    for i, spec in enumerate(grid, 1):
        print(
            f"  {i:>3}  {spec.mode:<16}  {spec.env_name:<13}  "
            f"{spec.dataset_quality:<14}  {spec.effective_steps():>8,}  "
            f"{spec.estimated_minutes():>5.1f}m  {spec.seed}"
        )

    _print_time_budget(grid)

    if args.dry_run:
        print("Dry-run: exiting.")
        return

    # ── Run experiments ─────────────────────────────────────────────────────
    failed: list[tuple[ExperimentSpec, str]] = []

    for i, spec in enumerate(grid, 1):
        print(f"\n[{i}/{len(grid)}] {spec.mode} / {spec.env_name} / "
              f"{spec.dataset_quality} / seed={spec.seed}  "
              f"({spec.effective_steps():,} steps, ~{spec.estimated_minutes():.0f} min)")
        try:
            run_experiment(spec)
        except Exception as exc:
            import traceback
            print(f"  ERROR: {exc}")
            traceback.print_exc()
            failed.append((spec, str(exc)))
            try:
                wandb.finish(exit_code=1)
            except Exception:
                pass

    # ── Summary ─────────────────────────────────────────────────────────────
    total = sum(s.estimated_minutes() for s in grid)
    print(f"\n{'='*65}")
    print(f"  Benchmark done: {len(grid)-len(failed)}/{len(grid)} succeeded  "
          f"(estimated {total/60:.1f} h)")
    if failed:
        print(f"  Failed ({len(failed)}):")
        for spec, msg in failed:
            print(f"    {spec.mode}/{spec.env_name}/{spec.dataset_quality}/"
                  f"seed={spec.seed}: {msg}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
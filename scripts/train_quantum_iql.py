"""CLI entry-point for the hybrid Quantum-IQL training pipeline (issue #9).

This script wires together every component of the Q-IQL stack:
  • QuantumIQLConfig  — unified config for classical and quantum runs
  • QuantumIQLTrainer — hybrid trainer (quantum V, classical Q + π)
  • W&B logging       — shared loss curves + quantum-specific diagnostics

Usage
-----
Classical baseline (ablation control arm):
    python scripts/train_quantum_iql.py \\
        --config configs/quantum_iql_hopper.yaml \\
        --overrides mode=classical

Quantum run with default schedule:
    python scripts/train_quantum_iql.py \\
        --config configs/quantum_iql_hopper.yaml \\
        --overrides mode=quantum

Quick smoke-test (5 steps, offline W&B, default.qubit backend):
    python scripts/train_quantum_iql.py \\
        --config configs/quantum_iql_hopper.yaml \\
        --overrides mode=quantum num_steps=5 wandb_offline=true \\
                    quantum_value.device_name=default.qubit

Ablation: vary expectile level across both arms:
    for MODE in classical quantum; do
      python scripts/train_quantum_iql.py \\
          --config configs/quantum_iql_hopper.yaml \\
          --overrides mode=$MODE tau=0.9 seed=1 wandb_offline=true
    done
"""

from __future__ import annotations

import argparse
import sys

import wandb

from src.quantum_iql.buffer import load_minari_dataset
from src.quantum_iql.quantum_config import QuantumIQLConfig, load_quantum_config
from src.quantum_iql.quantum_trainer import QuantumIQLTrainer
from src.quantum_iql.utils import get_device, make_env, set_seed


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a hybrid Quantum-IQL agent on a Minari offline dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to a YAML config file (e.g. configs/quantum_iql_hopper.yaml).",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Zero or more dot-notation overrides, e.g.\n"
            "  mode=quantum  tau=0.9  seed=1\n"
            "  quantum_value.n_qubits=4  quantum_value.device_name=default.qubit"
        ),
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Sanity-check helpers
# ---------------------------------------------------------------------------

def _check_mode(cfg: QuantumIQLConfig) -> None:
    if cfg.mode not in ("classical", "quantum"):
        raise ValueError(
            f"config.mode must be 'classical' or 'quantum', got '{cfg.mode}'."
        )


def _print_config_summary(cfg: QuantumIQLConfig) -> None:
    print("=" * 60)
    print("  Hybrid Q-IQL — configuration summary")
    print("=" * 60)
    print(f"  mode           : {cfg.mode}")
    print(f"  dataset        : {cfg.dataset_id}")
    print(f"  env            : {cfg.env_id}")
    print(f"  steps          : {cfg.num_steps:,}")
    print(f"  batch_size     : {cfg.batch_size}")
    print(f"  tau (expectile): {cfg.tau}")
    print(f"  beta (adv temp): {cfg.beta}")
    print(f"  gamma          : {cfg.gamma}")
    print(f"  device         : {cfg.device}")
    if cfg.mode == "quantum":
        qv = cfg.quantum_value
        print(f"  qubits         : {qv.n_qubits}")
        print(f"  DRU layers     : {qv.n_layers}")
        print(f"  PL device      : {qv.device_name}")
        print(f"  lr_quantum     : {cfg.lr_quantum}")
        print(f"  grad_clip_q    : {cfg.quantum_grad_clip}")
        print(f"  log_q_metrics  : {cfg.log_quantum_metrics}")
        sched = qv.layerwise_schedule
        if sched:
            print("  layerwise warmup:")
            for entry in sched:
                print(f"    step ≥ {entry.start_step:>8,}  →  {entry.active_layers} layers")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # ── Config ─────────────────────────────────────────────────────────────
    cfg = load_quantum_config(args.config, overrides=args.overrides)
    _check_mode(cfg)
    device = get_device(cfg.device)
    _print_config_summary(cfg)

    # ── Reproducibility ─────────────────────────────────────────────────────
    set_seed(cfg.seed)

    # ── W&B ────────────────────────────────────────────────────────────────
    run_name = cfg.wandb_run_name or f"{cfg.mode}-iql-{cfg.dataset_id.replace('/', '_')}-s{cfg.seed}"
    wandb.init(
        project=cfg.wandb_project,
        name=run_name,
        config={
            **{k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
            # Flatten quantum sub-config for easy W&B filter
            "qv_n_qubits":    cfg.quantum_value.n_qubits,
            "qv_n_layers":    cfg.quantum_value.n_layers,
            "qv_device":      cfg.quantum_value.device_name,
        },
        mode="offline" if cfg.wandb_offline else "online",
        tags=[cfg.mode, cfg.dataset_id, f"seed={cfg.seed}"],
    )

    # ── Data + env ─────────────────────────────────────────────────────────
    print(f"\nLoading dataset '{cfg.dataset_id}' …")
    buffer = load_minari_dataset(cfg.dataset_id, device="cpu")
    print(f"  buffer size    : {buffer._size:,} transitions")
    print(f"  obs_dim        : {buffer.obs_dim}")
    print(f"  act_dim        : {buffer.act_dim}\n")

    env = make_env(cfg.env_id, seed=cfg.seed)
    set_seed(cfg.seed, env=env)

    # ── Trainer ────────────────────────────────────────────────────────────
    trainer = QuantumIQLTrainer(cfg, buffer, env)
    trainer.train()

    # ── Cleanup ────────────────────────────────────────────────────────────
    env.close()
    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()

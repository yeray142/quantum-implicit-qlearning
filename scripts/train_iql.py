"""CLI entry point for classical IQL training.

Usage
-----
Basic run (downloads dataset on first use):
    python scripts/train_iql.py --config configs/iql_hopper.yaml

With key=value overrides:
    python scripts/train_iql.py --config configs/iql_hopper.yaml \\
        --overrides tau=0.9 seed=1 wandb_offline=true

Offline W&B logging (no internet required):
    python scripts/train_iql.py --config configs/iql_hopper.yaml \\
        --overrides wandb_offline=true num_steps=5
"""

from __future__ import annotations

import argparse

import wandb

from quantum_iql.buffer import load_minari_dataset
from quantum_iql.config import load_config
from quantum_iql.trainer import IQLTrainer
from quantum_iql.utils import get_device, make_env, set_seed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a classical IQL agent on a Minari offline dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to a YAML config file (e.g. configs/iql_hopper.yaml).",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Zero or more dot-notation overrides, e.g. tau=0.9 seed=1.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # ------------------------------------------------------------------ config
    cfg = load_config(args.config, overrides=args.overrides)
    device = get_device(cfg.device)
    print(f"Config loaded from '{args.config}'  (device={device})")

    # --------------------------------------------------------- reproducibility
    set_seed(cfg.seed)

    # ------------------------------------------------------------------ W&B
    wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_run_name,
        config=vars(cfg) if not hasattr(cfg, "__dataclass_fields__") else {
            k: getattr(cfg, k) for k in cfg.__dataclass_fields__
        },
        mode="offline" if cfg.wandb_offline else "online",
    )

    # -------------------------------------------------------------- data + env
    buffer = load_minari_dataset(cfg.dataset_id, device=str(device))
    env = make_env(cfg.env_id, seed=cfg.seed)
    set_seed(cfg.seed, env=env)

    # --------------------------------------------------------------- training
    trainer = IQLTrainer(cfg, buffer, env)
    trainer.train()

    # ---------------------------------------------------------------- cleanup
    env.close()
    wandb.finish()


if __name__ == "__main__":
    main()

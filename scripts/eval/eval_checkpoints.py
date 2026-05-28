#!/usr/bin/env python3
"""
Evaluate trained Q-IQL checkpoints and compare with W&B results.

Usage
-----
  python scripts/eval_checkpoints.py --mode quantum --seed 0
  python scripts/eval_checkpoints.py --mode quantum-fixed --seed 0 --episodes 20
  python scripts/eval_checkpoints.py --all  (evaluate all modes/seeds)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from quantum_iql import QuantumValueNetwork
from quantum_iql.networks import ActorNetwork, CriticNetwork, ValueNetwork
from quantum_iql.quantum_config import QuantumIQLConfig
from quantum_iql.quantum_trainer import QuantumIQLTrainer
from quantum_iql.utils import get_device, set_seed, make_env


def get_env_cfg(env_name: str, dataset: str = "medium"):
    """Return environment configuration for a given env/dataset."""
    configs = {
        "hopper": {
            "dataset_id": "mujoco/hopper/medium-v0",
            "env_id": "Hopper-v4",
            "obs_dim": 11,
            "act_dim": 3,
        },
        "walker2d": {
            "dataset_id": "mujoco/walker2d/medium-v0",
            "env_id": "Walker2d-v4",
            "obs_dim": 17,
            "act_dim": 6,
        },
    }
    return configs.get(env_name, configs["hopper"])


def load_vnet(mode: str, ckpt: dict, obs_dim: int) -> nn.Module | None:
    """Load value network from checkpoint based on mode. Returns None for constant-v."""
    if "quantum" in mode:
        multi_qubit = ckpt["value_net"]["a"].shape[0] > 1
        vnet = QuantumValueNetwork(
            n_qubits=8, n_layers=3, obs_dim=obs_dim,
            multi_qubit_readout=multi_qubit,
        )
        vnet.load_state_dict(ckpt["value_net"], strict=False)
    elif mode == "classical-deep":
        vnet = ValueNetwork(obs_dim, hidden_dims=[8, 8, 8])
        vnet.load_state_dict(ckpt["value_net"])
    elif mode == "classical-small":
        h = max(1, round((146 - 1) / (obs_dim + 2)))
        vnet = ValueNetwork(obs_dim, hidden_dims=[h])
        vnet.load_state_dict(ckpt["value_net"])
    elif mode == "constant-v":
        # constant-v doesn't have value_net - use standard ValueNetwork for actor update
        return None
    else:  # classical
        vnet = ValueNetwork(obs_dim, hidden_dims=[256, 256])
        vnet.load_state_dict(ckpt["value_net"])
    return vnet


def evaluate_checkpoint(
    mode: str,
    seed: int,
    env_name: str = "hopper",
    dataset: str = "medium",
    episodes: int = 10,
    checkpoint_dir: Path | None = None,
) -> dict | None:
    """Evaluate a single checkpoint and return metrics."""
    if checkpoint_dir is None:
        checkpoint_dir = (
            _PROJECT_ROOT / "experiments" / "checkpoints"
            / env_name / dataset / mode / f"seed_{seed}"
        )

    ckpt_path = checkpoint_dir / "checkpoint_final.pt"
    if not ckpt_path.exists():
        print(f"  [warn] No checkpoint: {ckpt_path}")
        return None

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        print(f"  [warn] Failed to load {ckpt_path}: {e}")
        return None

    env_cfg = get_env_cfg(env_name, dataset)
    obs_dim = env_cfg["obs_dim"]
    act_dim = env_cfg["act_dim"]

    device = get_device("auto")
    set_seed(seed)

    # Create environment
    env = make_env(env_cfg["env_id"], seed=seed)
    obs, _ = env.reset(seed=seed)

    # Load networks
    vnet = load_vnet(mode, ckpt, obs_dim)
    if vnet is not None:
        vnet = vnet.to(device)
        vnet.eval()

    # Actor always uses [256, 256] - only value network changes per mode
    actor = ActorNetwork(obs_dim, act_dim, hidden_dims=[256, 256]).to(device)
    actor.load_state_dict(ckpt["actor_net"])
    actor.eval()

    # Evaluate
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total_reward = 0.0
        steps = 0

        while not done and steps < 1000:
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                action = actor.get_action(obs_t, deterministic=True)
                action = action.cpu().numpy().squeeze()

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        returns.append(total_reward)
        print(f"  Episode {ep+1}/{episodes}: return = {total_reward:.2f}")

    env.close()

    return {
        "mode": mode,
        "seed": seed,
        "mean_return": np.mean(returns),
        "std_return": np.std(returns),
        "min_return": np.min(returns),
        "max_return": np.max(returns),
        "episodes": episodes,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Q-IQL checkpoints")
    parser.add_argument("--mode", default=None, help="Mode to evaluate (e.g. quantum, classical-deep)")
    parser.add_argument("--seed", type=int, default=None, help="Seed to evaluate")
    parser.add_argument("--env", default="hopper", choices=["hopper", "walker2d"])
    parser.add_argument("--dataset", default="medium")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--all", action="store_true", help="Evaluate all modes and seeds")
    args = parser.parse_args()

    if args.all:
        modes = ["classical", "classical-deep", "classical-small", "quantum",
                 "quantum-fixed", "quantum-fixed-warmup", "quantum-fixed-c",
                 "constant-v"]
        seeds = [0, 1, 2]
    else:
        if args.mode is None:
            print("Error: --mode required (or use --all)")
            return
        modes = [args.mode]
        seeds = [args.seed] if args.seed is not None else [0]

    print(f"\nEvaluating Q-IQL checkpoints")
    print(f"  env={args.env}, dataset={args.dataset}")
    print(f"  modes={modes}")
    print(f"  seeds={seeds}")
    print()

    results = []
    for mode in modes:
        for seed in seeds:
            print(f"{mode}/seed_{seed}:")
            result = evaluate_checkpoint(
                mode=mode,
                seed=seed,
                env_name=args.env,
                dataset=args.dataset,
                episodes=args.episodes,
            )
            if result:
                results.append(result)
                print(f"  => mean_return={result['mean_return']:.2f} ± {result['std_return']:.2f}")
            print()

    # Summary
    if results:
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        print(f"{'Mode':<25} {'Seed':>5} {'Mean Return':>12} {'Std':>8}")
        print("-" * 70)
        for r in results:
            print(f"{r['mode']:<25} {r['seed']:>5} {r['mean_return']:>12.2f} {r['std_return']:>8.2f}")
        print("=" * 70)


if __name__ == "__main__":
    main()
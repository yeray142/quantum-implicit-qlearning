"""Evaluate a trained IQL agent (or an untrained one as a baseline).

Usage
-----
# Evaluate random/untrained policy (sanity-check baseline):
    python scripts/eval_iql.py --config configs/iql_hopper.yaml

# Evaluate a saved checkpoint:
    python scripts/eval_iql.py --config configs/iql_hopper.yaml \
        --checkpoint path/to/checkpoint.pt

# More episodes for tighter confidence interval:
    python scripts/eval_iql.py --config configs/iql_hopper.yaml \
        --episodes 100

Checkpoint format (saved by this script or train_iql.py):
    {
        "step": int,
        "value_net": state_dict,
        "critic_net": state_dict,
        "actor_net": state_dict,
        "config": dict,
    }
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from quantum_iql.buffer import ReplayBuffer
from quantum_iql.config import load_config
from quantum_iql.trainer import IQLTrainer
from quantum_iql.utils import get_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a classical IQL agent.")
    parser.add_argument("--config", required=True, metavar="PATH")
    parser.add_argument("--checkpoint", default=None, metavar="PATH",
                        help="Path to a .pt checkpoint. "
                             "If omitted, evaluates the untrained (random) policy.")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes (default: 10).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override config seed for eval.")
    parser.add_argument("--render", action="store_true",
                        help="Render episodes (requires a display).")
    return parser.parse_args()


def make_dummy_buffer(obs_dim: int, act_dim: int) -> ReplayBuffer:
    """Minimal buffer with correct dims — no data needed for eval-only."""
    buf = ReplayBuffer(obs_dim, act_dim, capacity=1)
    # Write one fake transition so the buffer is valid
    buf._observations[0] = np.zeros(obs_dim, dtype=np.float32)
    buf._actions[0] = np.zeros(act_dim, dtype=np.float32)
    buf._rewards[0] = 0.0
    buf._next_observations[0] = np.zeros(obs_dim, dtype=np.float32)
    buf._dones[0] = 0.0
    buf._size = 1
    return buf


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.seed is not None:
        cfg.seed = args.seed
    if args.episodes != 10:
        cfg.eval_episodes = args.episodes

    device = get_device(cfg.device)
    set_seed(cfg.seed)

    render_mode = "human" if args.render else None
    import gymnasium as gym
    env = gym.make(cfg.env_id, render_mode=render_mode)
    env.reset(seed=cfg.seed)

    # Infer obs/act dims from the environment
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Build trainer (networks only — no real buffer needed for eval)
    buffer = make_dummy_buffer(obs_dim, act_dim)
    trainer = IQLTrainer(cfg, buffer, env)

    # Load checkpoint weights if provided
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location=device)
        trainer.value_net.load_state_dict(ckpt["value_net"])
        trainer.critic_net.load_state_dict(ckpt["critic_net"])
        trainer.actor_net.load_state_dict(ckpt["actor_net"])
        step = ckpt.get("step", "?")
        print(f"Loaded checkpoint from '{args.checkpoint}' (step={step})")
    else:
        print("No checkpoint provided — evaluating untrained (random-init) policy.")

    # Run evaluation
    metrics = trainer.evaluate()

    print(f"\nEvaluation over {cfg.eval_episodes} episodes on '{cfg.env_id}':")
    print(f"  mean return : {metrics['eval/mean_return']:.2f}")
    print(f"  std  return : {metrics['eval/std_return']:.2f}")
    print(f"  min  return : {metrics['eval/min_return']:.2f}")
    print(f"  max  return : {metrics['eval/max_return']:.2f}")

    env.close()


if __name__ == "__main__":
    main()

"""Checkpoint consistency test: verify seed 6 checkpoint loads and produces correct return."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from quantum_iql import ActorNetwork, QuantumValueNetwork


def test_checkpoint_consistency():
    """Load seed 6 quantum checkpoint, roll one Hopper episode, check return ~3626.9."""
    # Locate checkpoint — search experiments/ first
    checkpoint_paths = [
        Path("experiments/checkpoints/hopper/medium/quantum-fixed/seed_6/checkpoint_final.pt"),
        Path("experiments/checkpoints/hopper/medium/quantum-fixed-c/seed_6/checkpoint_final.pt"),
        Path("experiments/checkpoints/hopper/medium/quantum-fixed-warmup/seed_6/checkpoint_final.pt"),
        Path("experiments/checkpoints/hopper/medium/quantum-no-warmup/seed_6/checkpoint_final.pt"),
        Path("experiments/checkpoints/hopper/medium/quantum-multi-qubit-readout/seed_6/checkpoint_final.pt"),
        Path("experiments/checkpoints/hopper/medium/quantum/seed_6/checkpoint_final.pt"),
        Path("experiments/checkpoints/hopper/medium/quantum/seed_0/checkpoint_final.pt"),
    ]
    ckpt_path = None
    for p in checkpoint_paths:
        if p.exists():
            ckpt_path = p
            break

    if ckpt_path is None:
        pytest.skip("No trained quantum checkpoint found")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Read network dimensions from quantum_meta so we create networks with correct shapes
    quantum_meta = ckpt.get("quantum_meta", {})
    obs_dim = quantum_meta.get("obs_dim", 11)
    n_qubits = quantum_meta.get("n_qubits", 8)
    n_layers = quantum_meta.get("n_layers", 3)

    value_net = QuantumValueNetwork(
        n_qubits=n_qubits,
        n_layers=n_layers,
        obs_dim=obs_dim,
        device_name="default.qubit",
        diff_method="backprop",
        running_stats=True,
        use_pre_encoder=True,
    )
    actor = ActorNetwork(obs_dim=obs_dim, act_dim=3)

    value_net.load_state_dict(ckpt["value_net"], strict=False)
    actor.load_state_dict(ckpt["actor_net"])

    # Verify mu/sigma buffers are populated (from value_net state dict)
    mu = getattr(value_net, "mu", None)
    sigma = getattr(value_net, "sigma", None)
    assert mu is not None, "mu buffer missing"
    assert sigma is not None, "sigma buffer missing"
    assert torch.any(mu != 0), "mu buffer is all zeros — not populated from checkpoint"
    assert torch.any(sigma != 1), "sigma buffer is all ones — not populated from checkpoint"

    value_net.eval()
    actor.eval()

    # Roll out one deterministic Hopper-v4 episode
    try:
        import gymnasium as gym
    except ImportError:
        pytest.skip("gymnasium not available")

    env = gym.make("Hopper-v4")
    obs, _ = env.reset(seed=6)
    obs = np.array(obs, dtype=np.float32)
    total_reward = 0.0
    n_steps = 0
    max_steps = 1000

    while n_steps < max_steps:
        obs_t = torch.from_numpy(obs).unsqueeze(0)
        with torch.no_grad():
            action = actor.get_action(obs_t, deterministic=True)
        action_np = action.cpu().numpy()[0]
        obs_next, reward, term, trunc, _ = env.step(action_np)
        obs_next = np.array(obs_next, dtype=np.float32)
        total_reward += float(reward)
        obs = obs_next
        n_steps += 1
        if term or trunc:
            break

    env.close()

    # Expected return ± 5%
    expected = 3626.9
    lo = expected * 0.95
    hi = expected * 1.05
    assert lo <= total_reward <= hi, (
        f"Episode return {total_reward:.1f} outside [{lo:.1f}, {hi:.1f}] "
        f"(expected ~{expected:.1f})"
    )
    assert n_steps >= 100, f"Episode too short: {n_steps} steps"

    print(f"  Episode return: {total_reward:.1f} (expected ~{expected:.1f}), "
          f"{n_steps} steps")


if __name__ == "__main__":
    test_checkpoint_consistency()
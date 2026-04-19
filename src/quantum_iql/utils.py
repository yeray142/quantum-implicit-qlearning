"""Shared utility functions used across the quantum-iql codebase."""

from __future__ import annotations

import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak (EMA) update: target ← (1 - tau) * target + tau * source."""
    for t_param, s_param in zip(target.parameters(), source.parameters()):
        t_param.data.copy_((1.0 - tau) * t_param.data + tau * s_param.data)


def hard_update(target: nn.Module, source: nn.Module) -> None:
    """Exact parameter copy: target ← source."""
    target.load_state_dict(source.state_dict())


def set_seed(seed: int, env: gym.Env | None = None) -> None:
    """Set random seeds for Python, NumPy, and PyTorch (+ optional Gym env)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if env is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)


def get_device(device_str: str) -> torch.device:
    """Resolve 'auto' to 'cuda' if available, otherwise 'cpu'."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def make_env(env_id: str, seed: int) -> gym.Env:
    """Create and seed a Gymnasium environment."""
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env

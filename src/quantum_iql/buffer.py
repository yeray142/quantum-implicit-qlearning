"""Offline replay buffer and Minari dataset loader for IQL.

Design notes
------------
- All data is stored as float32 NumPy arrays pre-allocated at construction
  time (using the total step count from the dataset). This avoids repeated
  array growth and gives a single contiguous memory layout.
- Tensors are moved to the target device lazily at sample time, keeping the
  buffer itself on CPU.
- `done` flags are set from `episode.terminations` only — NOT `truncations`.
  Conflating them would cause the Bellman backup to treat time-limit cutoffs
  as absorbing states, corrupting Q-value targets for continuing episodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Protocol for episode objects (matches minari.EpisodeData duck-type)
# ---------------------------------------------------------------------------

class EpisodeData(Protocol):
    observations: Any   # array-like, shape (T+1, obs_dim)
    actions: Any        # array-like, shape (T, act_dim)
    rewards: Any        # array-like, shape (T,)
    terminations: Any   # array-like, shape (T,)
    truncations: Any    # array-like, shape (T,)


# Batch container
@dataclass
class Batch:
    observations: torch.Tensor       # (B, obs_dim)
    actions: torch.Tensor            # (B, act_dim)
    rewards: torch.Tensor            # (B, 1)
    next_observations: torch.Tensor  # (B, obs_dim)
    dones: torch.Tensor              # (B, 1)  float32, 1.0 = terminal


# Replay buffer
class ReplayBuffer:
    """Pre-allocated, CPU-resident offline replay buffer.

    Intended to be filled once from a dataset (not written online).
    Call `add_from_episode` for each episode, then `sample` during training.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        capacity: int,
        device: str | torch.device = "cpu",
    ) -> None:
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.capacity = capacity
        self.device = torch.device(device)

        self._observations = np.empty((capacity, obs_dim), dtype=np.float32)
        self._actions = np.empty((capacity, act_dim), dtype=np.float32)
        self._rewards = np.empty((capacity, 1), dtype=np.float32)
        self._next_observations = np.empty((capacity, obs_dim), dtype=np.float32)
        self._dones = np.empty((capacity, 1), dtype=np.float32)

        self._ptr = 0   # next write index
        self._size = 0  # number of valid transitions

    # Writing
    def add_from_episode(self, episode: EpisodeData) -> None:
        """Ingest one episode from a Minari dataset.

        Args:
            episode: A ``minari.EpisodeData`` object with attributes
                     ``observations``, ``actions``, ``rewards``,
                     ``terminations``, and ``truncations``.

        Notes:
            Minari stores T+1 observations per episode (including the
            final next-observation). We build T transitions by pairing
            obs[t] with obs[t+1].

            ``done`` is 1.0 only when the episode ended due to a true
            terminal condition (``terminations``), not a time-limit
            truncation (``truncations``).
        """
        obs = np.asarray(episode.observations, dtype=np.float32)       # (T+1, obs_dim)
        actions = np.asarray(episode.actions, dtype=np.float32)        # (T, act_dim)
        rewards = np.asarray(episode.rewards, dtype=np.float32)        # (T,)
        terminations = np.asarray(episode.terminations, dtype=np.float32)  # (T,)

        T = len(actions)
        obs_t = obs[:-1]       # (T, obs_dim)
        obs_tp1 = obs[1:]      # (T, obs_dim)

        end = self._ptr + T
        if end <= self.capacity:
            self._observations[self._ptr:end] = obs_t
            self._actions[self._ptr:end] = actions
            self._rewards[self._ptr:end] = rewards.reshape(-1, 1)
            self._next_observations[self._ptr:end] = obs_tp1
            self._dones[self._ptr:end] = terminations.reshape(-1, 1)
        else:
            # Episode wraps around — split into two slices
            first = self.capacity - self._ptr
            self._observations[self._ptr:] = obs_t[:first]
            self._observations[:T - first] = obs_t[first:]
            self._actions[self._ptr:] = actions[:first]
            self._actions[:T - first] = actions[first:]
            self._rewards[self._ptr:] = rewards[:first].reshape(-1, 1)
            self._rewards[:T - first] = rewards[first:].reshape(-1, 1)
            self._next_observations[self._ptr:] = obs_tp1[:first]
            self._next_observations[:T - first] = obs_tp1[first:]
            self._dones[self._ptr:] = terminations[:first].reshape(-1, 1)
            self._dones[:T - first] = terminations[first:].reshape(-1, 1)

        self._ptr = end % self.capacity
        self._size = min(self._size + T, self.capacity)

    # Sampling
    def sample(self, batch_size: int) -> Batch:
        """Uniformly sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            A :class:`Batch` with tensors on ``self.device``.
        """
        if batch_size > self._size:
            raise ValueError(
                f"Requested {batch_size} samples but buffer only has {self._size}."
            )
        idx = np.random.randint(0, self._size, size=batch_size)

        def to_tensor(arr: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(arr[idx]).to(self.device)

        return Batch(
            observations=to_tensor(self._observations),
            actions=to_tensor(self._actions),
            rewards=to_tensor(self._rewards),
            next_observations=to_tensor(self._next_observations),
            dones=to_tensor(self._dones),
        )

    # Dunder helpers
    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return (
            f"ReplayBuffer(size={self._size}/{self.capacity}, "
            f"obs_dim={self.obs_dim}, act_dim={self.act_dim}, device={self.device})"
        )


# Minari dataset loader
def load_minari_dataset(dataset_id: str, device: str | torch.device = "cpu") -> ReplayBuffer:
    """Download (if needed) and load a Minari dataset into a ReplayBuffer.

    Args:
        dataset_id: Minari dataset identifier, e.g. ``"mujoco/hopper/medium-v2"``.
        device:     Target device for sampled tensors.

    Returns:
        A fully populated :class:`ReplayBuffer`.

    Example::

        buffer = load_minari_dataset("mujoco/hopper/medium-v2", device="cuda")
        batch = buffer.sample(256)
    """
    import minari  # imported here to keep the module importable without minari installed

    dataset = minari.load_dataset(dataset_id, download=True)

    # Infer dimensions from the first episode
    first_ep = next(iter(dataset.iterate_episodes()))
    obs_dim = int(np.asarray(first_ep.observations).shape[-1])
    act_dim = int(np.asarray(first_ep.actions).shape[-1])
    total_steps = int(dataset.total_steps)

    print(
        f"Loading '{dataset_id}': {total_steps:,} steps, "
        f"obs_dim={obs_dim}, act_dim={act_dim}"
    )

    buffer = ReplayBuffer(obs_dim, act_dim, capacity=total_steps, device=device)

    for episode in dataset.iterate_episodes():
        buffer.add_from_episode(episode)

    print(f"Buffer ready: {len(buffer):,} transitions loaded.")
    return buffer

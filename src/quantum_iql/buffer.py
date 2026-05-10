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
- Goal-conditioned envs (PointMaze, AntMaze, Fetch, ...) emit Dict observations
  with `observation` and `desired_goal` keys. We concat both into a single flat
  vector at ingestion time so the policy is goal-aware. Older code that just
  pulled out `observation` and discarded `desired_goal` produced silently
  goal-blind policies that learn nothing on goal-reaching tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Protocol for episode objects (matches minari.EpisodeData duck-type)
# ---------------------------------------------------------------------------

class EpisodeData(Protocol):
    observations: Any   # array-like, shape (T+1, obs_dim) OR Dict[str, array]
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


# ---------------------------------------------------------------------------
# Goal-conditioned obs flattening
# ---------------------------------------------------------------------------

def _flatten_obs(raw_obs: Any) -> np.ndarray:
    """Flatten a Minari episode's observations to a 2D float32 array.

    Behavior:
      * If `raw_obs` is already array-like → returned as float32 ndarray.
      * If `raw_obs` is a Dict (goal-conditioned env) → concat `observation`
        with `desired_goal` along the last axis. Falls back to just
        `observation` if `desired_goal` is missing.

    The achieved_goal field is intentionally dropped because it equals (or is
    a subset of) the position component of `observation` in standard
    gymnasium-robotics envs and would just inflate `obs_dim` redundantly.
    """
    if isinstance(raw_obs, dict):
        obs_part = np.asarray(raw_obs["observation"], dtype=np.float32)
        goal = raw_obs.get("desired_goal", None)
        if goal is not None:
            goal_part = np.asarray(goal, dtype=np.float32)
            return np.concatenate([obs_part, goal_part], axis=-1)
        return obs_part
    return np.asarray(raw_obs, dtype=np.float32)


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

            For Dict observations (goal-conditioned envs), `observation` and
            `desired_goal` are concatenated into a single flat vector via
            ``_flatten_obs``.
        """
        obs = _flatten_obs(episode.observations)              # (T+1, obs_dim)
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


def load_custom_dataset(save_dir: Path | str, device: str | torch.device = "cpu") -> ReplayBuffer:
    """Load a custom dataset from npz episode files into a ReplayBuffer.

    The dataset should have been saved with save_dataset() from
    scripts/generate_pointmaze_dataset.py, containing:
    - metadata.json with dataset metadata
    - episode_XXXXX.npz files with observations, actions, rewards,
      terminations, truncations, and infos_success arrays.

    Args:
        save_dir: Path to the dataset directory.
        device: Target device for sampled tensors.

    Returns:
        A fully populated :class:`ReplayBuffer`.
    """
    import json

    save_dir = Path(save_dir)
    metadata_path = save_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {save_dir}")

    with open(metadata_path) as f:
        metadata = json.load(f)

    # Find episode files
    episode_files = sorted(save_dir.glob("episode_*.npz"))
    if not episode_files:
        raise FileNotFoundError(f"No episode_*.npz files found in {save_dir}")

    # Load first episode to get dimensions
    first_ep = dict(np.load(episode_files[0]))
    raw_obs = first_ep["observations"]
    if raw_obs.ndim == 2:
        obs_dim = raw_obs.shape[1]
    else:
        obs_dim = raw_obs.shape[0]
    act_dim = first_ep["actions"].shape[1] if first_ep["actions"].ndim > 1 else first_ep["actions"].shape[0]
    total_steps = metadata["total_steps"]

    print(
        f"Loading custom dataset from '{save_dir}': {total_steps:,} steps, "
        f"obs_dim={obs_dim}, act_dim={act_dim}"
    )

    buffer = ReplayBuffer(obs_dim, act_dim, capacity=total_steps, device=device)

    class EpisodeLike:
        def __init__(self, data: dict):
            self.observations = data["observations"]
            self.actions = data["actions"]
            self.rewards = data["rewards"]
            self.terminations = data["terminations"]
            self.truncations = data["truncations"]

    for ep_file in episode_files:
        ep_data = dict(np.load(ep_file))
        buffer.add_from_episode(EpisodeLike(ep_data))

    print(f"Buffer ready: {len(buffer):,} transitions loaded.")
    return buffer


# Minari dataset loader
def load_minari_dataset(
    dataset_id: str,
    device: str | torch.device = "cpu",
    reward_shift: float = 0.0,
) -> ReplayBuffer:
    """Download (if needed) and load a Minari dataset into a ReplayBuffer.

    Args:
        dataset_id:    Minari dataset identifier, e.g. ``"mujoco/hopper/medium-v2"``.
        device:        Target device for sampled tensors.
        reward_shift:  Constant added to every reward at ingestion time. The
                       canonical IQL recipe for AntMaze uses ``reward_shift=-1.0``
                       to remap sparse {0, 1} rewards to {-1, 0}, which keeps V
                       on a useful negative scale and prevents the advantage
                       signal from collapsing to zero outside the goal. Default
                       0.0 (no shift) — Hopper / Walker / HalfCheetah keep their
                       dense rewards untouched.

    Returns:
        A fully populated :class:`ReplayBuffer`.

    Example::

        buffer = load_minari_dataset("mujoco/hopper/medium-v2", device="cuda")
        antmaze = load_minari_dataset(
            "D4RL/antmaze/medium-play-v1", reward_shift=-1.0,
        )
    """
    import minari  # imported here to keep the module importable without minari installed

    dataset = minari.load_dataset(dataset_id, download=True)

    # Infer dimensions from the first episode (uses the same flatten as ingestion
    # so obs_dim accounts for goal concatenation on Dict-obs datasets).
    first_ep = next(iter(dataset.iterate_episodes()))
    obs_flat = _flatten_obs(first_ep.observations)
    obs_dim = int(obs_flat.shape[-1])
    act_dim = int(np.asarray(first_ep.actions).shape[-1])
    total_steps = int(dataset.total_steps)

    print(
        f"Loading '{dataset_id}': {total_steps:,} steps, "
        f"obs_dim={obs_dim}, act_dim={act_dim}"
        + (f", reward_shift={reward_shift:+g}" if reward_shift != 0.0 else "")
    )

    buffer = ReplayBuffer(obs_dim, act_dim, capacity=total_steps, device=device)

    for episode in dataset.iterate_episodes():
        buffer.add_from_episode(episode)

    if reward_shift != 0.0:
        # Apply shift to the in-use slice; pre-allocated unused rows stay zeroed.
        buffer._rewards[:buffer._size] += reward_shift

    print(f"Buffer ready: {len(buffer):,} transitions loaded.")
    return buffer

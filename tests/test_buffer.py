"""Tests for the offline replay buffer (no Minari download required)."""

import numpy as np
import pytest
import torch

from quantum_iql.buffer import Batch, ReplayBuffer


OBS_DIM, ACT_DIM = 11, 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeEpisode:
    """Minimal stand-in for minari.EpisodeData."""

    def __init__(self, T: int, obs_dim: int, act_dim: int, terminal: bool = False):
        rng = np.random.default_rng(0)
        self.observations = rng.standard_normal((T + 1, obs_dim)).astype(np.float32)
        self.actions = rng.standard_normal((T, act_dim)).astype(np.float32)
        self.rewards = rng.standard_normal(T).astype(np.float32)
        self.terminations = np.zeros(T, dtype=np.float32)
        self.truncations = np.zeros(T, dtype=np.float32)
        if terminal:
            self.terminations[-1] = 1.0
        else:
            self.truncations[-1] = 1.0  # time-limit cutoff


def make_buffer(n_episodes: int = 3, T: int = 50) -> ReplayBuffer:
    buf = ReplayBuffer(OBS_DIM, ACT_DIM, capacity=n_episodes * T, device="cpu")
    for i in range(n_episodes):
        ep = _FakeEpisode(T, OBS_DIM, ACT_DIM, terminal=(i == 0))
        buf.add_from_episode(ep)
    return buf


# ---------------------------------------------------------------------------
# Construction & sizing
# ---------------------------------------------------------------------------

def test_buffer_empty_on_init():
    buf = ReplayBuffer(OBS_DIM, ACT_DIM, capacity=100)
    assert len(buf) == 0


def test_buffer_size_after_adding():
    buf = make_buffer(n_episodes=3, T=50)
    assert len(buf) == 150


def test_buffer_repr():
    buf = make_buffer()
    assert "ReplayBuffer" in repr(buf)


# ---------------------------------------------------------------------------
# Batch shapes and dtypes
# ---------------------------------------------------------------------------

def test_sample_shapes():
    buf = make_buffer()
    batch = buf.sample(32)
    assert batch.observations.shape == (32, OBS_DIM)
    assert batch.actions.shape == (32, ACT_DIM)
    assert batch.rewards.shape == (32, 1)
    assert batch.next_observations.shape == (32, OBS_DIM)
    assert batch.dones.shape == (32, 1)


def test_sample_dtypes():
    buf = make_buffer()
    batch = buf.sample(32)
    for tensor in (
        batch.observations,
        batch.actions,
        batch.rewards,
        batch.next_observations,
        batch.dones,
    ):
        assert tensor.dtype == torch.float32


def test_sample_device_cpu():
    buf = make_buffer()
    batch = buf.sample(32)
    for tensor in (
        batch.observations, batch.actions, batch.rewards,
        batch.next_observations, batch.dones,
    ):
        assert tensor.device.type == "cpu"


# ---------------------------------------------------------------------------
# done flag correctness
# ---------------------------------------------------------------------------

def test_done_zero_for_truncated_episodes():
    """Truncated episodes (time-limit) must have done=0, not 1."""
    buf = ReplayBuffer(OBS_DIM, ACT_DIM, capacity=200)
    truncated_ep = _FakeEpisode(50, OBS_DIM, ACT_DIM, terminal=False)
    buf.add_from_episode(truncated_ep)
    # All transitions should have done=0
    batch = buf.sample(50)
    assert (batch.dones == 0.0).all()


def test_done_one_for_terminal_episode():
    """Only the last transition of a truly terminal episode has done=1."""
    buf = ReplayBuffer(OBS_DIM, ACT_DIM, capacity=200)
    terminal_ep = _FakeEpisode(50, OBS_DIM, ACT_DIM, terminal=True)
    buf.add_from_episode(terminal_ep)
    assert buf._dones[:50].sum() == 1.0
    assert buf._dones[49, 0] == 1.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_sample_raises_when_too_few_transitions():
    buf = ReplayBuffer(OBS_DIM, ACT_DIM, capacity=100)
    buf.add_from_episode(_FakeEpisode(10, OBS_DIM, ACT_DIM))
    with pytest.raises(ValueError, match="buffer only has"):
        buf.sample(100)


def test_wraparound_write():
    """Episode that straddles the buffer boundary should not lose transitions."""
    T = 30
    buf = ReplayBuffer(OBS_DIM, ACT_DIM, capacity=50)
    # Fill 40 transitions first so the next episode wraps around
    ep1 = _FakeEpisode(40, OBS_DIM, ACT_DIM)
    buf.add_from_episode(ep1)
    assert len(buf) == 40

    ep2 = _FakeEpisode(T, OBS_DIM, ACT_DIM)
    buf.add_from_episode(ep2)
    # capacity=50, so size stays capped at 50
    assert len(buf) == 50


def test_observations_and_next_observations_are_shifted():
    """obs[t+1] must equal next_obs[t] for each transition."""
    buf = ReplayBuffer(OBS_DIM, ACT_DIM, capacity=100)
    ep = _FakeEpisode(10, OBS_DIM, ACT_DIM)
    buf.add_from_episode(ep)
    np.testing.assert_array_equal(buf._observations[1:10], buf._next_observations[0:9])

"""Smoke tests for IQLTrainer — no Minari download or GPU required."""

import math

import numpy as np
import torch

from quantum_iql.buffer import ReplayBuffer
from quantum_iql.config import IQLConfig, NetworkConfig
from quantum_iql.trainer import IQLTrainer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OBS_DIM, ACT_DIM = 11, 3
TINY_NET = NetworkConfig(hidden_dims=[32, 32])


def make_tiny_config(**overrides) -> IQLConfig:
    base = dict(
        dataset_id="mujoco/hopper/medium-v2",  # not used in tests
        env_id="Hopper-v4",                    # not used in smoke tests
        batch_size=8,
        num_steps=1,
        warmup_steps=0,
        value_net=TINY_NET,
        critic_net=TINY_NET,
        actor_net=TINY_NET,
        device="cpu",
        wandb_offline=True,
        seed=0,
    )
    base.update(overrides)
    return IQLConfig(**base)


def make_synthetic_buffer(n: int = 200) -> ReplayBuffer:
    """Build a buffer filled with random transitions (no Minari needed)."""
    buf = ReplayBuffer(OBS_DIM, ACT_DIM, capacity=n, device="cpu")
    # Fake an episode by directly writing into the backing arrays
    rng = np.random.default_rng(42)
    buf._observations[:n] = rng.standard_normal((n, OBS_DIM)).astype("float32")
    buf._actions[:n] = rng.standard_normal((n, ACT_DIM)).astype("float32")
    buf._rewards[:n] = rng.standard_normal((n, 1)).astype("float32")
    buf._next_observations[:n] = rng.standard_normal((n, OBS_DIM)).astype("float32")
    buf._dones[:n] = np.zeros((n, 1), dtype="float32")
    buf._size = n
    buf._ptr = n % buf.capacity
    return buf


class _FakeEnv:
    """Minimal stand-in for a Gymnasium environment."""

    observation_space_shape = (OBS_DIM,)
    action_space_shape = (ACT_DIM,)

    def reset(self, seed=None):
        return np.zeros(OBS_DIM, dtype=np.float32), {}

    def step(self, action):
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        reward = 1.0
        terminated = True   # single-step episode for speed
        truncated = False
        return obs, reward, terminated, truncated, {}

    def action_space(self):
        pass


def make_trainer(**overrides) -> IQLTrainer:
    cfg = make_tiny_config(**overrides)
    buf = make_synthetic_buffer()
    env = _FakeEnv()
    return IQLTrainer(cfg, buf, env)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_trainer_builds_without_error():
    trainer = make_trainer()
    assert trainer.value_net is not None
    assert trainer.critic_net is not None
    assert trainer.actor_net is not None
    assert trainer.value_target is not None


def test_value_target_matches_value_net_at_init():
    trainer = make_trainer()
    for p_target, p_source in zip(
        trainer.value_target.parameters(), trainer.value_net.parameters()
    ):
        assert torch.allclose(p_target, p_source)


def test_value_target_frozen():
    trainer = make_trainer()
    for p in trainer.value_target.parameters():
        assert not p.requires_grad


# ---------------------------------------------------------------------------
# Individual update steps
# ---------------------------------------------------------------------------

def test_update_value_returns_finite_loss():
    trainer = make_trainer()
    batch = trainer.buffer.sample(8)
    metrics = trainer.update_value(batch)
    assert "loss/value" in metrics
    assert math.isfinite(metrics["loss/value"])


def test_update_critic_returns_finite_loss():
    trainer = make_trainer()
    batch = trainer.buffer.sample(8)
    metrics = trainer.update_critic(batch)
    assert "loss/critic" in metrics
    assert math.isfinite(metrics["loss/critic"])


def test_update_actor_returns_finite_loss():
    trainer = make_trainer()
    batch = trainer.buffer.sample(8)
    metrics = trainer.update_actor(batch)
    assert "loss/actor" in metrics
    assert math.isfinite(metrics["loss/actor"])
    assert "advantage_mean" in metrics


def test_update_targets_moves_value_target():
    trainer = make_trainer(polyak=1.0)   # polyak=1 → hard copy
    # Modify value_net parameters
    with torch.no_grad():
        for p in trainer.value_net.parameters():
            p.fill_(99.0)
    trainer.update_targets()
    for p_t, p_s in zip(trainer.value_target.parameters(), trainer.value_net.parameters()):
        assert torch.allclose(p_t, p_s)


# ---------------------------------------------------------------------------
# train_step
# ---------------------------------------------------------------------------

def test_train_step_returns_all_keys():
    trainer = make_trainer()
    metrics = trainer.train_step()
    assert "loss/value" in metrics
    assert "loss/critic" in metrics
    assert "loss/actor" in metrics


def test_train_step_all_finite():
    trainer = make_trainer()
    metrics = trainer.train_step()
    for key, val in metrics.items():
        assert math.isfinite(val), f"{key} = {val} is not finite"


def test_train_step_increments_step_counter():
    trainer = make_trainer()
    assert trainer._step == 0
    trainer.train_step()
    assert trainer._step == 1


def test_warmup_skips_actor():
    trainer = make_trainer(warmup_steps=10)
    metrics = trainer.train_step()
    assert "loss/actor" not in metrics


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def test_evaluate_returns_expected_keys():
    trainer = make_trainer(eval_episodes=2)
    metrics = trainer.evaluate()
    assert "eval/mean_return" in metrics
    assert "eval/std_return" in metrics
    assert "eval/min_return" in metrics
    assert "eval/max_return" in metrics


def test_evaluate_returns_finite_values():
    trainer = make_trainer(eval_episodes=2)
    metrics = trainer.evaluate()
    for val in metrics.values():
        assert math.isfinite(val)


def test_actor_back_to_train_after_eval():
    trainer = make_trainer(eval_episodes=1)
    trainer.evaluate()
    assert trainer.actor_net.training

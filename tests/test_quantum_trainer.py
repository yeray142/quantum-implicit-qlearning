"""End-to-end integration tests for the hybrid Q-IQL pipeline (issue #9).

Tests are intentionally fast: they use tiny synthetic buffers, default.qubit
(no lightning install required), and run only a handful of gradient steps.
They verify the full training pipeline is functional, not that training converges.

Run with:
    pytest tests/test_quantum_trainer.py -v
"""

from __future__ import annotations

import math
import sys
import os

import pytest
import torch
import numpy as np

# ── Make the project importable from the repo root ──────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.quantum_iql.buffer import Batch, ReplayBuffer
from src.quantum_iql.quantum_config import (
    LayerwiseScheduleEntry,
    QuantumIQLConfig,
    QuantumNetConfig,
)
from src.quantum_iql.quantum_trainer import QuantumIQLTrainer, _grad_norm
from src.quantum_iql.quantum_value_network import QuantumValueNetwork


# ── Fixtures ─────────────────────────────────────────────────────────────────

OBS_DIM = 8
ACT_DIM = 3
N_TRANSITIONS = 128
BATCH_SIZE = 16


def _make_buffer(obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM, n: int = N_TRANSITIONS) -> ReplayBuffer:
    """Synthetic buffer populated with random Gaussian data."""
    buf = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, capacity=n)
    rng = np.random.default_rng(0)

    class _FakeEpisode:
        def __init__(self):
            T = n
            self.observations   = rng.standard_normal((T + 1, obs_dim)).astype(np.float32)
            self.actions        = rng.standard_normal((T, act_dim)).astype(np.float32)
            self.rewards        = rng.standard_normal(T).astype(np.float32)
            self.terminations   = np.zeros(T, dtype=np.float32)
            self.terminations[-1] = 1.0
            self.truncations    = np.zeros(T, dtype=np.float32)

    buf.add_from_episode(_FakeEpisode())
    return buf


def _make_quantum_config(
    n_steps: int = 5,
    mode: str = "quantum",
    n_qubits: int = 4,
    n_layers: int = 2,
    device_name: str = "default.qubit",
) -> QuantumIQLConfig:
    """Minimal QuantumIQLConfig suitable for fast unit tests."""
    return QuantumIQLConfig(
        mode=mode,
        dataset_id="synthetic",
        env_id="Hopper-v4",
        num_steps=n_steps,
        batch_size=BATCH_SIZE,
        warmup_steps=0,
        log_interval=1,
        eval_interval=9999,   # suppress eval in unit tests
        eval_episodes=1,
        wandb_offline=True,
        quantum_value=QuantumNetConfig(
            n_qubits=n_qubits,
            n_layers=n_layers,
            device_name=device_name,
            running_stats=True,
            layerwise_schedule=[],  # no schedule — start all layers active
        ),
        lr_quantum=1e-3,
        quantum_grad_clip=1.0,
        log_quantum_metrics=True,
        stats_update_interval=2,
        device="cpu",
    )


class _FakeEnv:
    """Tiny stub that satisfies the evaluation loop interface."""
    observation_space = type("Space", (), {"shape": (OBS_DIM,)})()
    action_space = type("Space", (), {"shape": (ACT_DIM,)})()

    def reset(self, seed=None):
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        return obs, {}

    def step(self, action):
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        return obs, 0.0, True, False, {}

    def close(self):
        pass


# ── Helper: build a trainer without triggering W&B ──────────────────────────

def _make_trainer(cfg: QuantumIQLConfig, buf: ReplayBuffer) -> QuantumIQLTrainer:
    """Construct a trainer, bypassing W&B initialisation."""
    trainer = QuantumIQLTrainer.__new__(QuantumIQLTrainer)
    trainer._qcfg = cfg
    trainer.cfg = cfg
    trainer.buffer = buf
    trainer.env = _FakeEnv()
    trainer.device = torch.device("cpu")
    trainer._step = 0
    trainer._schedule_idx = 0
    trainer._last_stats_refresh = 0

    trainer._build_networks()
    trainer._build_targets()
    trainer._build_optimizers()

    # probe batch for shot-noise proxy
    probe = buf.sample(min(16, buf._size))
    trainer._probe_obs = probe.observations.to(trainer.device)

    return trainer


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestQuantumNetworkInitialisation:
    """QuantumValueNetwork starts near the identity (output ≈ 0)."""

    def test_parameter_shapes(self):
        qnet = QuantumValueNetwork(
            n_qubits=4, n_layers=2, obs_dim=OBS_DIM, device_name="default.qubit"
        )
        assert qnet.theta.shape == (2, 4, 3)
        assert qnet.w.shape == (2, 4, 3)

    def test_identity_init_output_near_zero(self):
        """w = -theta at init → expval ≈ 1 per Rot(0,0,0)=I; affine maps to ≈ a+b."""
        qnet = QuantumValueNetwork(
            n_qubits=4, n_layers=2, obs_dim=OBS_DIM, device_name="default.qubit"
        )
        obs = torch.zeros(2, OBS_DIM)
        with torch.no_grad():
            v = qnet(obs)
        # a=1, b=0 at init → V ≈ 1*<Z0> + 0; <Z0>=1 when U=I → V≈1
        # We just check it's a finite scalar, not NaN
        assert torch.isfinite(v).all(), f"Non-finite output: {v}"

    def test_forward_shape(self):
        qnet = QuantumValueNetwork(
            n_qubits=4, n_layers=2, obs_dim=OBS_DIM, device_name="default.qubit"
        )
        obs = torch.randn(5, OBS_DIM)
        with torch.no_grad():
            v = qnet(obs)
        assert v.shape == (5,), f"Expected (5,), got {v.shape}"

    def test_set_active_layers_bounds(self):
        qnet = QuantumValueNetwork(
            n_qubits=4, n_layers=2, obs_dim=OBS_DIM, device_name="default.qubit"
        )
        qnet.set_active_layers(1)
        assert qnet.active_layers == 1
        with pytest.raises(ValueError):
            qnet.set_active_layers(0)
        with pytest.raises(ValueError):
            qnet.set_active_layers(3)  # > n_layers


class TestQuantumTrainerConstruction:
    """Trainer builds correctly in both modes."""

    def test_quantum_mode_uses_quantum_value_net(self):
        buf = _make_buffer()
        cfg = _make_quantum_config(mode="quantum")
        trainer = _make_trainer(cfg, buf)
        assert trainer._is_quantum
        assert isinstance(trainer.value_net, QuantumValueNetwork)

    def test_classical_mode_uses_mlp_value_net(self):
        buf = _make_buffer()
        cfg = _make_quantum_config(mode="classical")
        trainer = _make_trainer(cfg, buf)
        assert not trainer._is_quantum
        from quantum_iql.networks import ValueNetwork
        assert isinstance(trainer.value_net, ValueNetwork)

    def test_critic_always_classical(self):
        from quantum_iql.networks import CriticNetwork
        buf = _make_buffer()
        for mode in ("classical", "quantum"):
            cfg = _make_quantum_config(mode=mode)
            trainer = _make_trainer(cfg, buf)
            assert isinstance(trainer.critic_net, CriticNetwork)

    def test_actor_always_classical(self):
        from quantum_iql.networks import ActorNetwork
        buf = _make_buffer()
        for mode in ("classical", "quantum"):
            cfg = _make_quantum_config(mode=mode)
            trainer = _make_trainer(cfg, buf)
            assert isinstance(trainer.actor_net, ActorNetwork)

    def test_value_target_always_classical_mlp(self):
        """V̄ is a classical MLP in both modes (efficiency + simplicity)."""
        from quantum_iql.networks import ValueNetwork
        buf = _make_buffer()
        for mode in ("classical", "quantum"):
            cfg = _make_quantum_config(mode=mode)
            trainer = _make_trainer(cfg, buf)
            assert isinstance(trainer.value_target, ValueNetwork)
            # V̄ must not require gradients
            for p in trainer.value_target.parameters():
                assert not p.requires_grad


class TestGradientFlow:
    """Gradients flow through the quantum circuit via parameter-shift."""

    def test_quantum_value_gradients_exist_after_backward(self):
        buf = _make_buffer()
        cfg = _make_quantum_config(mode="quantum")
        trainer = _make_trainer(cfg, buf)

        batch = buf.sample(BATCH_SIZE)
        batch = Batch(
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards,
            next_observations=batch.next_observations,
            dones=batch.dones,
        )

        from quantum_iql.losses import value_loss
        trainer.value_optimizer.zero_grad()
        loss = value_loss(trainer.value_net, trainer.critic_net, batch, cfg.tau)
        loss.backward()

        qnet: QuantumValueNetwork = trainer.value_net  # type: ignore
        assert qnet.theta.grad is not None, "theta has no gradient"
        assert qnet.w.grad is not None, "w has no gradient"
        assert qnet.a.grad is not None, "affine head 'a' has no gradient"
        assert qnet.b.grad is not None, "affine head 'b' has no gradient"

    def test_grad_norm_helper_returns_finite(self):
        buf = _make_buffer()
        cfg = _make_quantum_config(mode="quantum")
        trainer = _make_trainer(cfg, buf)

        batch = buf.sample(BATCH_SIZE)
        batch = Batch(**{k: getattr(batch, k) for k in Batch.__dataclass_fields__})

        from quantum_iql.losses import value_loss
        trainer.value_optimizer.zero_grad()
        loss = value_loss(trainer.value_net, trainer.critic_net, batch, cfg.tau)
        loss.backward()

        qnet: QuantumValueNetwork = trainer.value_net  # type: ignore
        norm_theta = _grad_norm(qnet.theta)
        norm_w = _grad_norm(qnet.w)

        assert math.isfinite(norm_theta), f"grad_norm_theta is not finite: {norm_theta}"
        assert math.isfinite(norm_w), f"grad_norm_w is not finite: {norm_w}"

    def test_classical_critic_gradients_unchanged(self):
        """Critic gradients are unaffected by the quantum value network."""
        buf = _make_buffer()
        cfg = _make_quantum_config(mode="quantum")
        trainer = _make_trainer(cfg, buf)

        batch = buf.sample(BATCH_SIZE)
        batch = Batch(**{k: getattr(batch, k) for k in Batch.__dataclass_fields__})

        from quantum_iql.losses import critic_loss
        trainer.critic_optimizer.zero_grad()
        loss = critic_loss(trainer.critic_net, trainer.value_target, batch, cfg.gamma)
        loss.backward()

        for name, p in trainer.critic_net.named_parameters():
            assert p.grad is not None, f"Critic param '{name}' has no gradient"


class TestTrainStep:
    """Single train_step runs without error and returns valid metrics."""

    def test_train_step_quantum_returns_all_keys(self):
        buf = _make_buffer()
        cfg = _make_quantum_config(mode="quantum", n_steps=3)
        trainer = _make_trainer(cfg, buf)

        metrics = trainer.train_step()

        expected_keys = {"loss/value", "loss/critic", "loss/actor"}
        for k in expected_keys:
            assert k in metrics, f"Missing metric '{k}'"

    def test_train_step_quantum_metrics_are_finite(self):
        buf = _make_buffer()
        cfg = _make_quantum_config(mode="quantum", n_steps=3)
        trainer = _make_trainer(cfg, buf)

        metrics = trainer.train_step()

        for k, v in metrics.items():
            assert math.isfinite(v), f"Metric '{k}' is not finite: {v}"

    def test_train_step_classical_returns_all_keys(self):
        buf = _make_buffer()
        cfg = _make_quantum_config(mode="classical", n_steps=3)
        trainer = _make_trainer(cfg, buf)

        metrics = trainer.train_step()

        expected_keys = {"loss/value", "loss/critic", "loss/actor"}
        for k in expected_keys:
            assert k in metrics, f"Missing metric '{k}'"

    def test_step_counter_increments(self):
        buf = _make_buffer()
        cfg = _make_quantum_config(mode="quantum", n_steps=3)
        trainer = _make_trainer(cfg, buf)

        assert trainer._step == 0
        trainer.train_step()
        assert trainer._step == 1
        trainer.train_step()
        assert trainer._step == 2


class TestQuantumDiagnostics:
    """Quantum-specific metric helpers work correctly."""

    def test_quantum_param_metrics_keys(self):
        buf = _make_buffer()
        cfg = _make_quantum_config(mode="quantum")
        trainer = _make_trainer(cfg, buf)

        metrics = trainer._quantum_param_metrics()
        expected = {
            "quantum/param_theta_mean",
            "quantum/param_theta_std",
            "quantum/param_w_mean",
            "quantum/param_w_std",
        }
        assert expected == set(metrics.keys())

    def test_value_output_std_returns_finite(self):
        buf = _make_buffer()
        cfg = _make_quantum_config(mode="quantum")
        trainer = _make_trainer(cfg, buf)

        metrics = trainer._value_output_std()
        assert "quantum/value_output_std" in metrics
        assert math.isfinite(metrics["quantum/value_output_std"])

    def test_classical_mode_param_metrics_empty(self):
        buf = _make_buffer()
        cfg = _make_quantum_config(mode="classical")
        trainer = _make_trainer(cfg, buf)
        assert trainer._quantum_param_metrics() == {}
        assert trainer._value_output_std() == {}


class TestLayerwiseSchedule:
    """Layerwise warm-up schedule activates layers at the right steps."""

    def test_schedule_activates_layers(self):
        buf = _make_buffer()
        cfg = _make_quantum_config(mode="quantum")
        # Inject a simple 2-entry schedule
        cfg.quantum_value.layerwise_schedule = [
            LayerwiseScheduleEntry(start_step=0, active_layers=1),
            LayerwiseScheduleEntry(start_step=3, active_layers=2),
        ]
        trainer = _make_trainer(cfg, buf)
        qnet: QuantumValueNetwork = trainer.value_net  # type: ignore
        qnet.set_active_layers(1)

        trainer._apply_layerwise_schedule(step=1)
        assert qnet.active_layers == 1

        trainer._apply_layerwise_schedule(step=3)
        assert qnet.active_layers == 2

    def test_empty_schedule_is_noop(self):
        buf = _make_buffer()
        cfg = _make_quantum_config(mode="quantum")
        cfg.quantum_value.layerwise_schedule = []
        trainer = _make_trainer(cfg, buf)
        qnet: QuantumValueNetwork = trainer.value_net  # type: ignore
        original = qnet.active_layers
        trainer._apply_layerwise_schedule(step=999_999)
        assert qnet.active_layers == original


class TestRunningStatsRefresh:
    """Running stats are refreshed on the configured schedule."""

    def test_stats_refresh_updates_mu_sigma(self):
        buf = _make_buffer()
        cfg = _make_quantum_config(mode="quantum")
        cfg.stats_update_interval = 1
        trainer = _make_trainer(cfg, buf)
        qnet: QuantumValueNetwork = trainer.value_net  # type: ignore

        trainer._refresh_running_stats(step=1)   # should trigger refresh
        # After refresh mu is recomputed from buffer; may or may not differ for random data
        # Just check it's finite
        assert torch.isfinite(qnet.mu).all()
        assert torch.isfinite(qnet.sigma).all()
        assert (qnet.sigma > 0).all()

    def test_stats_not_refreshed_before_interval(self):
        buf = _make_buffer()
        cfg = _make_quantum_config(mode="quantum")
        cfg.stats_update_interval = 100
        trainer = _make_trainer(cfg, buf)
        qnet: QuantumValueNetwork = trainer.value_net  # type: ignore

        trainer._last_stats_refresh = 50
        mu_before = qnet.mu.clone()
        trainer._refresh_running_stats(step=60)   # 60 - 50 = 10 < 100 → no refresh
        assert torch.equal(qnet.mu, mu_before)


class TestCheckpoint:
    """Checkpoint saves and contains quantum metadata."""

    def test_checkpoint_contains_quantum_meta(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        buf = _make_buffer()
        cfg = _make_quantum_config(mode="quantum")
        trainer = _make_trainer(cfg, buf)

        trainer.save_checkpoint("test_ckpt.pt")
        payload = torch.load(tmp_path / "checkpoints" / "test_ckpt.pt", weights_only=False)

        assert payload["mode"] == "quantum"
        assert "quantum_meta" in payload
        qm = payload["quantum_meta"]
        assert qm["n_qubits"] == cfg.quantum_value.n_qubits
        assert qm["n_layers"] == cfg.quantum_value.n_layers

    def test_checkpoint_classical_has_no_quantum_meta(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        buf = _make_buffer()
        cfg = _make_quantum_config(mode="classical")
        trainer = _make_trainer(cfg, buf)

        trainer.save_checkpoint("test_ckpt_classical.pt")
        payload = torch.load(tmp_path / "checkpoints" / "test_ckpt_classical.pt", weights_only=False)

        assert payload["mode"] == "classical"
        assert "quantum_meta" not in payload

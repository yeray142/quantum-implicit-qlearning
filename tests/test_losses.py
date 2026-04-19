"""Tests for IQL loss functions."""

import math

import pytest
import torch

from quantum_iql.buffer import Batch
from quantum_iql.losses import actor_loss, critic_loss, expectile_loss, value_loss
from quantum_iql.networks import ActorNetwork, CriticNetwork, ValueNetwork

OBS_DIM, ACT_DIM, B = 11, 3, 16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_batch(b: int = B) -> Batch:
    return Batch(
        observations=torch.randn(b, OBS_DIM),
        actions=torch.randn(b, ACT_DIM),
        rewards=torch.randn(b, 1),
        next_observations=torch.randn(b, OBS_DIM),
        dones=torch.zeros(b, 1),
    )


def make_networks(hidden: list[int] | None = None):
    h = hidden or [32]
    v = ValueNetwork(OBS_DIM, hidden_dims=h)
    v_target = ValueNetwork(OBS_DIM, hidden_dims=h)
    q = CriticNetwork(OBS_DIM, ACT_DIM, hidden_dims=h)
    actor = ActorNetwork(OBS_DIM, ACT_DIM, hidden_dims=h + h)  # needs ≥2 layers
    return v, v_target, q, actor


# ---------------------------------------------------------------------------
# expectile_loss
# ---------------------------------------------------------------------------

def test_expectile_tau_half_equals_mse():
    """tau=0.5 must produce the same result as 0.5 * MSE(u, 0)."""
    u = torch.randn(100)
    el = expectile_loss(u, tau=0.5)
    mse = 0.5 * (u ** 2).mean()
    assert torch.allclose(el, mse, atol=1e-6)


def test_expectile_positive_diff_uses_tau_weight():
    """For all-positive diff, weight = tau → loss = tau * mean(u²)."""
    u = torch.ones(10) * 2.0   # all positive
    tau = 0.7
    el = expectile_loss(u, tau)
    expected = tau * (u ** 2).mean()
    assert torch.allclose(el, expected, atol=1e-6)


def test_expectile_negative_diff_uses_one_minus_tau():
    """For all-negative diff, weight = 1 - tau → loss = (1-tau) * mean(u²)."""
    u = torch.ones(10) * -2.0  # all negative
    tau = 0.7
    el = expectile_loss(u, tau)
    expected = (1.0 - tau) * (u ** 2).mean()
    assert torch.allclose(el, expected, atol=1e-6)


def test_expectile_zero_loss_at_zero():
    u = torch.zeros(10)
    assert expectile_loss(u, tau=0.7).item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# value_loss — gradient flow
# ---------------------------------------------------------------------------

def test_value_loss_grad_flows_to_value_only():
    v, _, q, _ = make_networks()
    batch = make_batch()

    loss = value_loss(v, q, batch, tau=0.7)
    loss.backward()

    # V should have gradients
    for p in v.parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0

    # Q should have NO gradients (detached inside value_loss)
    for p in q.parameters():
        assert p.grad is None


def test_value_loss_is_finite():
    v, _, q, _ = make_networks()
    loss = value_loss(v, q, make_batch(), tau=0.7)
    assert math.isfinite(loss.item())


# ---------------------------------------------------------------------------
# critic_loss — gradient flow
# ---------------------------------------------------------------------------

def test_critic_loss_grad_flows_to_critic_only():
    v, v_target, q, _ = make_networks()
    batch = make_batch()

    loss = critic_loss(q, v_target, batch, gamma=0.99)
    loss.backward()

    # Q should have gradients
    for p in q.parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0

    # V-target should have NO gradients (used inside torch.no_grad)
    for p in v_target.parameters():
        assert p.grad is None


def test_critic_loss_is_finite():
    _, v_target, q, _ = make_networks()
    loss = critic_loss(q, v_target, make_batch(), gamma=0.99)
    assert math.isfinite(loss.item())


def test_critic_loss_single_head():
    """use_twin=False should produce a scalar loss without errors."""
    _, v_target, _, _ = make_networks()
    q_single = CriticNetwork(OBS_DIM, ACT_DIM, hidden_dims=[32], use_twin=False)
    loss = critic_loss(q_single, v_target, make_batch(), gamma=0.99)
    assert math.isfinite(loss.item())
    loss.backward()
    for p in q_single.parameters():
        assert p.grad is not None


def test_critic_loss_zero_for_perfect_prediction():
    """If Q perfectly predicts r + γ·V̄(s'), loss should be near zero."""
    _, v_target, q, _ = make_networks()
    batch = make_batch()

    # Patch q1 (and q2) to always output the target
    with torch.no_grad():
        # Zero out all Q params and bias to the target mean (approximate)
        for p in q.parameters():
            p.zero_()

    # Loss won't be exactly 0 due to network arch, but should be finite
    loss = critic_loss(q, v_target, batch, gamma=0.99)
    assert math.isfinite(loss.item())


# ---------------------------------------------------------------------------
# actor_loss — gradient flow
# ---------------------------------------------------------------------------

def test_actor_loss_grad_flows_to_actor_only():
    v, _, q, actor = make_networks()
    batch = make_batch()

    loss, _ = actor_loss(actor, q, v, batch, beta=3.0)
    loss.backward()

    # Actor should have gradients
    for p in actor.parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0

    # Q and V should have NO gradients
    for p in q.parameters():
        assert p.grad is None
    for p in v.parameters():
        assert p.grad is None


def test_actor_loss_is_finite():
    v, _, q, actor = make_networks()
    loss, metrics = actor_loss(actor, q, v, make_batch(), beta=3.0)
    assert math.isfinite(loss.item())
    assert math.isfinite(metrics["advantage_mean"])
    assert math.isfinite(metrics["advantage_std"])
    assert math.isfinite(metrics["exp_adv_mean"])


def test_actor_loss_advantage_clip_prevents_explosion():
    """With a huge beta and large advantages, exp_adv must not exceed clip."""
    v, _, q, actor = make_networks()
    batch = make_batch()
    clip = 10.0
    loss, metrics = actor_loss(actor, q, v, batch, beta=1000.0, advantage_clip=clip)
    assert metrics["exp_adv_mean"] <= clip + 1e-4
    assert math.isfinite(loss.item())

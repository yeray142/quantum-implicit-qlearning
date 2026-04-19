"""Tests for MLP networks: shapes, gradient flow, and correctness."""

import pytest
import torch
import torch.nn as nn

from quantum_iql.networks import (
    build_mlp,
    ValueNetwork,
    CriticNetwork,
    ActorNetwork,
)

B, OBS_DIM, ACT_DIM = 8, 11, 3  # Hopper-like dims


# ---------------------------------------------------------------------------
# build_mlp
# ---------------------------------------------------------------------------

def test_build_mlp_output_shape():
    net = build_mlp(OBS_DIM, 1, [64, 64])
    x = torch.randn(B, OBS_DIM)
    assert net(x).shape == (B, 1)


def test_build_mlp_depth():
    # hidden_dims=[64, 64] → 2 Linear + 1 output Linear = 3 Linear layers total
    net = build_mlp(OBS_DIM, 1, [64, 64])
    linears = [m for m in net if isinstance(m, nn.Linear)]
    assert len(linears) == 3


def test_build_mlp_layer_norm():
    net = build_mlp(OBS_DIM, 1, [64, 64], layer_norm=True)
    layer_norms = [m for m in net if isinstance(m, nn.LayerNorm)]
    assert len(layer_norms) == 2  # one per hidden layer


def test_build_mlp_unknown_activation():
    with pytest.raises(ValueError, match="Unknown activation"):
        build_mlp(OBS_DIM, 1, [64], activation="sigmoid")


# ---------------------------------------------------------------------------
# ValueNetwork
# ---------------------------------------------------------------------------

def test_value_network_output_shape():
    v = ValueNetwork(OBS_DIM)
    obs = torch.randn(B, OBS_DIM)
    out = v(obs)
    assert out.shape == (B, 1)


def test_value_network_grad_flow():
    v = ValueNetwork(OBS_DIM, hidden_dims=[32])
    obs = torch.randn(B, OBS_DIM)
    loss = v(obs).mean()
    loss.backward()
    for p in v.parameters():
        assert p.grad is not None


# ---------------------------------------------------------------------------
# CriticNetwork
# ---------------------------------------------------------------------------

def test_critic_twin_output_shapes():
    q = CriticNetwork(OBS_DIM, ACT_DIM, use_twin=True)
    obs = torch.randn(B, OBS_DIM)
    act = torch.randn(B, ACT_DIM)
    q1, q2 = q(obs, act)
    assert q1.shape == (B, 1)
    assert q2.shape == (B, 1)


def test_critic_single_output_shape():
    q = CriticNetwork(OBS_DIM, ACT_DIM, use_twin=False)
    obs = torch.randn(B, OBS_DIM)
    act = torch.randn(B, ACT_DIM)
    out = q(obs, act)
    assert out.shape == (B, 1)


def test_critic_q_min_leq_both():
    q = CriticNetwork(OBS_DIM, ACT_DIM, use_twin=True)
    obs = torch.randn(B, OBS_DIM)
    act = torch.randn(B, ACT_DIM)
    q1, q2 = q(obs, act)
    q_min = q.q_min(obs, act)
    assert (q_min <= q1 + 1e-6).all()
    assert (q_min <= q2 + 1e-6).all()


def test_critic_grad_flow():
    q = CriticNetwork(OBS_DIM, ACT_DIM, hidden_dims=[32])
    obs = torch.randn(B, OBS_DIM)
    act = torch.randn(B, ACT_DIM)
    q1, q2 = q(obs, act)
    (q1.mean() + q2.mean()).backward()
    for p in q.parameters():
        assert p.grad is not None


# ---------------------------------------------------------------------------
# ActorNetwork
# ---------------------------------------------------------------------------

def test_actor_forward_shapes():
    actor = ActorNetwork(OBS_DIM, ACT_DIM)
    obs = torch.randn(B, OBS_DIM)
    mean, log_std = actor(obs)
    assert mean.shape == (B, ACT_DIM)
    assert log_std.shape == (B, ACT_DIM)


def test_actor_log_std_clamped():
    actor = ActorNetwork(OBS_DIM, ACT_DIM, log_std_min=-5.0, log_std_max=2.0)
    obs = torch.randn(B, OBS_DIM)
    _, log_std = actor(obs)
    assert (log_std >= -5.0).all()
    assert (log_std <= 2.0).all()


def test_actor_get_action_shape():
    actor = ActorNetwork(OBS_DIM, ACT_DIM)
    obs = torch.randn(B, OBS_DIM)
    action = actor.get_action(obs)
    assert action.shape == (B, ACT_DIM)


def test_actor_get_action_squashed():
    actor = ActorNetwork(OBS_DIM, ACT_DIM, action_scale=1.0)
    obs = torch.randn(B, OBS_DIM)
    action = actor.get_action(obs)
    assert (action.abs() < 1.0).all()


def test_actor_log_prob_shape():
    actor = ActorNetwork(OBS_DIM, ACT_DIM)
    obs = torch.randn(B, OBS_DIM)
    action = actor.get_action(obs)
    lp = actor.log_prob(obs, action)
    assert lp.shape == (B, 1)


def test_actor_log_prob_finite():
    actor = ActorNetwork(OBS_DIM, ACT_DIM)
    obs = torch.randn(B, OBS_DIM)
    action = actor.get_action(obs)
    lp = actor.log_prob(obs, action)
    assert torch.isfinite(lp).all()


def test_critic_q_min_single_head():
    """q_min on a single-head critic should return the same value as forward."""
    q = CriticNetwork(OBS_DIM, ACT_DIM, use_twin=False)
    obs = torch.randn(B, OBS_DIM)
    act = torch.randn(B, ACT_DIM)
    assert torch.allclose(q.q_min(obs, act), q(obs, act))


def test_actor_grad_flow():
    actor = ActorNetwork(OBS_DIM, ACT_DIM, hidden_dims=[32, 32])
    obs = torch.randn(B, OBS_DIM)
    action = actor.get_action(obs)
    lp = actor.log_prob(obs, action)
    (-lp.mean()).backward()
    for p in actor.parameters():
        assert p.grad is not None

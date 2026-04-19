"""Loss functions for IQL: expectile regression, critic, and actor updates.

All four functions follow the same convention:
  - Networks that should NOT receive gradients are called with `.detach()` by
    the caller (value_loss) or internally (critic_loss, actor_loss).
  - Each function returns a scalar tensor so the caller can call `.backward()`
    directly and log the `.item()`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from quantum_iql.buffer import Batch
from quantum_iql.networks import ActorNetwork, CriticNetwork, ValueNetwork


# Primitive: expectile regression
def expectile_loss(diff: torch.Tensor, tau: float) -> torch.Tensor:
    """Asymmetric L2 (expectile) loss.

    .. math::
        L_{\\tau}(u) = |\\tau - \\mathbb{1}[u < 0]| \\cdot u^2

    When ``tau=0.5`` this is identical to MSE (both weights equal 0.5,
    so the result equals ``0.5 * mean(u²)``).  Larger ``tau`` pushes
    V(s) towards higher quantiles of Q(s, ·), which is the key mechanism
    of IQL's policy-free value estimation.

    Args:
        diff: Tensor of residuals ``Q(s,a) - V(s)``, any shape.
        tau:  Expectile level in (0, 1).

    Returns:
        Scalar mean loss.
    """
    weight = torch.where(diff < 0,
                         torch.full_like(diff, 1.0 - tau),
                         torch.full_like(diff, tau))
    return (weight * diff.pow(2)).mean()


# V-network loss
def value_loss(
    value_net: ValueNetwork,
    critic_net: CriticNetwork,
    batch: Batch,
    tau: float,
) -> torch.Tensor:
    """Expectile regression loss for the value network V_ψ(s).

    .. math::
        L_V(\\psi) = \\mathbb{E}_{(s,a)\\sim D}
                     \\bigl[L_{\\tau}\\bigl(Q_\\theta(s,a) - V_\\psi(s)\\bigr)\\bigr]

    Q is treated as a fixed target here (detached), so gradients flow only
    through V_ψ.

    Args:
        value_net:  V_ψ — parameters receive gradients.
        critic_net: Q_θ — used in inference mode (detached).
        batch:      Sampled transitions.
        tau:        Expectile level.

    Returns:
        Scalar loss tensor.
    """
    with torch.no_grad():
        q_target = critic_net.q_min(batch.observations, batch.actions)

    v = value_net(batch.observations)                   # (B, 1), has grad
    diff = q_target - v                                 # positive → V underestimates
    return expectile_loss(diff, tau)


# Q-network loss
def critic_loss(
    critic_net: CriticNetwork,
    value_target_net: ValueNetwork,
    batch: Batch,
    gamma: float,
) -> torch.Tensor:
    """Bellman MSE loss for the critic Q_θ(s, a).

    .. math::
        L_Q(\\theta) = \\mathbb{E}_{(s,a,r,s')\\sim D}
                       \\bigl[(r + \\gamma \\bar{V}(s') - Q_\\theta(s,a))^2\\bigr]

    The TD target uses ``value_target_net`` (the EMA copy V̄), which is
    always detached so gradients flow only through Q_θ.

    When ``use_twin=True`` the loss is the sum of both heads' MSEs, which
    is equivalent to training each head independently with the same target.

    Args:
        critic_net:        Q_θ — parameters receive gradients.
        value_target_net:  V̄  — EMA copy, used without gradients.
        batch:             Sampled transitions.
        gamma:             Discount factor.

    Returns:
        Scalar loss tensor.
    """
    with torch.no_grad():
        v_next = value_target_net(batch.next_observations)          # (B, 1)
        td_target = batch.rewards + gamma * (1.0 - batch.dones) * v_next  # (B, 1)

    out = critic_net(batch.observations, batch.actions)

    if isinstance(out, tuple):
        q1, q2 = out
        loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)
    else:
        loss = F.mse_loss(out, td_target)

    return loss


# Actor loss
def actor_loss(
    actor_net: ActorNetwork,
    critic_net: CriticNetwork,
    value_net: ValueNetwork,
    batch: Batch,
    beta: float,
    advantage_clip: float = 100.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Advantage-weighted behaviour cloning loss for the actor π_φ.

    .. math::
        L_\\pi(\\phi) = -\\mathbb{E}_{(s,a)\\sim D}
                        \\bigl[\\exp(\\beta A(s,a)) \\cdot \\log \\pi_\\phi(a|s)\\bigr]

    where :math:`A(s,a) = Q(s,a) - V(s)`.  Both Q and V are detached so
    gradients flow only through the log-probability term.

    The ``advantage_clip`` guard clamps ``exp(β·A)`` before it is used as
    a loss weight.  Without this, a handful of very high-advantage
    transitions can dominate the gradient and destabilise early training.

    Args:
        actor_net:       π_φ — parameters receive gradients.
        critic_net:      Q_θ — used without gradients.
        value_net:       V_ψ — used without gradients.
        batch:           Sampled transitions (actions are dataset actions).
        beta:            Inverse temperature; larger → sharper advantage weighting.
        advantage_clip:  Upper bound for ``exp(β·A)``; default 100.0.

    Returns:
        (loss, metrics) where metrics contains ``advantage_mean``,
        ``advantage_std``, and ``exp_adv_mean`` for diagnostic logging.
    """
    with torch.no_grad():
        q = critic_net.q_min(batch.observations, batch.actions)   # (B, 1)
        v = value_net(batch.observations)                          # (B, 1)
        advantage = q - v                                          # (B, 1)
        exp_adv = torch.clamp(torch.exp(beta * advantage), max=advantage_clip)

    log_prob = actor_net.log_prob(batch.observations, batch.actions)  # (B, 1), has grad
    loss = -(exp_adv * log_prob).mean()

    metrics = {
        "advantage_mean": advantage.mean().item(),
        "advantage_std": advantage.std().item(),
        "exp_adv_mean": exp_adv.mean().item(),
    }
    return loss, metrics

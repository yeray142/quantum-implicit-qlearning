"""MLP networks for IQL: value, critic (twin Q), and actor (squashed Gaussian).

QUANTUM EXTENSION HOOK
----------------------
`build_mlp` is the single factory used by all three network classes.
To inject quantum layers, subclass any network and override its constructor
to call a quantum-aware factory instead of `build_mlp`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn

# Numerically stable log-prob for squashed (tanh) Gaussians
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
_LOG_2PI = math.log(2 * math.pi)


# MLP factory
_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
}


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: Sequence[int],
    activation: str = "relu",
    layer_norm: bool = False,
    output_activation: nn.Module | None = None,
) -> nn.Sequential:
    """Build a fully-connected MLP.

    Args:
        input_dim:          Size of the input vector.
        output_dim:         Size of the output vector.
        hidden_dims:        Sizes of hidden layers.
        activation:         Activation name applied between hidden layers.
        layer_norm:         If True, insert LayerNorm after each activation.
        output_activation:  Optional module applied after the final linear layer.

    Returns:
        An nn.Sequential representing the MLP.
    """
    if activation not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation '{activation}'. Choose from {list(_ACTIVATIONS)}")

    act_cls = _ACTIVATIONS[activation]
    layers: list[nn.Module] = []
    in_dim = input_dim

    for h_dim in hidden_dims:
        layers.append(nn.Linear(in_dim, h_dim))
        layers.append(act_cls())
        if layer_norm:
            layers.append(nn.LayerNorm(h_dim))
        in_dim = h_dim

    layers.append(nn.Linear(in_dim, output_dim))
    if output_activation is not None:
        layers.append(output_activation)

    return nn.Sequential(*layers)


# Value network  V_ψ(s)
class ValueNetwork(nn.Module):
    """State-value function V_ψ(s) → scalar.

    Trained via expectile regression against Q(s, a).
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        activation: str = "relu",
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.net = build_mlp(obs_dim, 1, hidden_dims, activation, layer_norm)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (B, obs_dim)
        Returns:
            (B, 1) scalar value estimates
        """
        return self.net(obs)


# Critic network  Q_θ(s, a)  — optional twin heads
class CriticNetwork(nn.Module):
    """Action-value function Q_θ(s, a) → scalar.

    Optionally uses twin critics (two independent MLPs) to reduce
    overestimation bias. When `use_twin=True`, both `forward` and `q_min`
    operate on both heads; the V-update uses `q_min`.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        activation: str = "relu",
        layer_norm: bool = False,
        use_twin: bool = True,
    ) -> None:
        super().__init__()
        self.use_twin = use_twin
        in_dim = obs_dim + act_dim
        self.q1 = build_mlp(in_dim, 1, hidden_dims, activation, layer_norm)
        if use_twin:
            self.q2 = build_mlp(in_dim, 1, hidden_dims, activation, layer_norm)

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """
        Args:
            obs:    (B, obs_dim)
            action: (B, act_dim)
        Returns:
            (q1, q2) each (B, 1) when use_twin=True, else q1 (B, 1).
        """
        x = torch.cat([obs, action], dim=-1)
        q1 = self.q1(x)
        if self.use_twin:
            return q1, self.q2(x)
        return q1

    def q_min(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return min(q1, q2) for use in the value loss (reduces overestimation).

        Returns:
            (B, 1)
        """
        if self.use_twin:
            q1, q2 = self.forward(obs, action)  # type: ignore[misc]
            return torch.min(q1, q2)
        return self.forward(obs, action)  # type: ignore[return-value]


# Actor network  π_φ(a | s)  — squashed Gaussian
class ActorNetwork(nn.Module):
    """Gaussian policy with state-dependent mean and log_std.

    Actions are squashed through tanh so they lie in (-1, 1)^act_dim.
    The log-probability accounts for the tanh change-of-variables correction.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        activation: str = "relu",
        layer_norm: bool = False,
        log_std_min: float = LOG_STD_MIN,
        log_std_max: float = LOG_STD_MAX,
        action_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_scale = action_scale

        act_cls = _ACTIVATIONS[activation]
        self.trunk = build_mlp(
            obs_dim,
            hidden_dims[-1],
            hidden_dims[:-1],
            activation,
            layer_norm,
            output_activation=act_cls(),
        )
        # Separate linear heads for mean and log_std
        self.mean_head = nn.Linear(hidden_dims[-1], act_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], act_dim)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Gaussian parameters from observation.

        Args:
            obs: (B, obs_dim)
        Returns:
            mean:    (B, act_dim) — pre-squash mean
            log_std: (B, act_dim) — clamped log standard deviation
        """
        # Activate the last hidden layer manually (trunk ends before final activation)
        h = self.trunk(obs)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        """Sample (or take the mean of) the squashed Gaussian policy.

        Args:
            obs:           (B, obs_dim) or (obs_dim,)
            deterministic: If True, return tanh(mean) without sampling.
        Returns:
            action: (B, act_dim) scaled to [-action_scale, action_scale]
        """
        mean, log_std = self.forward(obs)
        if deterministic:
            return torch.tanh(mean) * self.action_scale
        std = log_std.exp()
        eps = torch.randn_like(mean)
        return torch.tanh(mean + std * eps) * self.action_scale

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Log probability of `action` under the current policy.

        Applies the tanh change-of-variables correction:
            log π(a|s) = log N(u; μ, σ) - Σ log(1 - tanh²(u))
        where u = arctanh(a / action_scale).

        Args:
            obs:    (B, obs_dim)
            action: (B, act_dim) — squashed actions in (-action_scale, action_scale)
        Returns:
            (B, 1) summed log-probability over action dimensions
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()

        # Inverse tanh to recover the pre-squash value u
        a = action / self.action_scale
        # Clamp to avoid NaN at the boundaries of tanh
        a = a.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        u = torch.atanh(a)

        # Gaussian log-prob in pre-squash space
        log_prob_gaussian = -0.5 * (((u - mean) / std) ** 2 + 2 * log_std + _LOG_2PI)

        # tanh change-of-variables correction: - log(1 - tanh²(u))
        log_det = torch.log(1.0 - a.pow(2) + 1e-6)

        log_prob = (log_prob_gaussian - log_det).sum(dim=-1, keepdim=True)
        return log_prob

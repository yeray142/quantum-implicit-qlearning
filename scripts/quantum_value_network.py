"""
Architecture:
  - 8-qubit Data Re-uploading (DRU) circuit (Pérez-Salinas et al., 2020)
  - 3 DRU layers  (L <= floor(log2(8)) = 3, polynomial BP guarantee)
  - Nearest-neighbour CZ entangling layers between re-uploading blocks
  - Fixed CZ preamble for initial entanglement (no BP depth contribution)
  - Identity-block initialisation: w^(i) = -theta^(i) so U = I at t=0
  - Local Pauli-Z readout on qubit 0 only (Cerezo et al., 2021 Thm 2(ii))
  - Trainable affine output head: V(s) = a * <Z_0> + b

Quantised network: V(s)   — classical Q(s,a) and pi(a|s) unchanged.
Rationale: V(s) is the central IQL object (feeds Bellman backup and actor
advantage); scalar state-only input minimises qubit requirements; local
observable + shallow DRU keeps gradient variance in poly(n) regime.
"""

from __future__ import annotations

import math

import pennylane as qml
import torch
import torch.nn as nn


# Helpers
def _arctan_encode(s: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Equation (4): xs = arctan((s - mu_D) / sigma_D)  in (-pi/2, pi/2)."""
    return torch.arctan((s - mu) / (sigma + 1e-8))


def _cz_preamble(n_qubits: int) -> None:
    """Fixed non-trained nearest-neighbour CZ layer (preamble B_pre).

    Introduces initial entanglement without adding to trainable BP depth.
    Called once at circuit start; gates are never updated.
    """
    for q in range(0, n_qubits - 1, 2):
        qml.CZ(wires=[q, q + 1])
    for q in range(1, n_qubits - 1, 2):
        qml.CZ(wires=[q, q + 1])


def _cz_entangler(n_qubits: int) -> None:
    """Nearest-neighbour CZ ladder B_CZ between DRU layers."""
    for q in range(n_qubits - 1):
        qml.CZ(wires=[q, q + 1])


# PennyLane QNode — batched via parameter broadcasting
def _build_qnode(
    n_qubits: int,
    n_layers: int,
    device_name: str = "default.qubit",
    diff_method: str = "backprop",
):
    """Construct a batched PennyLane QNode using parameter broadcasting.

    The circuit accepts xs of shape (B, n_qubits) and evaluates all B samples
    in a single QNode call via PennyLane parameter broadcasting (introduced in
    PennyLane 0.26, supported by lightning.qubit + adjoint since 0.36).

    For each qubit q in each DRU layer l the angle tensor has shape (B, 3):
        angles[k, j] = theta[l, q, j] + w[l, q, j] * xs[k, q]
    Passing (B,)-shaped tensors to qml.Rot triggers broadcasting over B circuits.

    Gradient flow:  angles = f(theta, w, xs)  is tracked by PyTorch autograd
    before the QNode sees it, so all three inputs receive gradients correctly
    regardless of the diff_method chosen.

    diff_method notes:
      "adjoint" (default) — computes all parameter gradients in one backward
        sweep; recommended for lightning.qubit.
      "parameter-shift"   — 2 evals per parameter; use only if adjoint is
        unavailable for the chosen device.
    """
    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(
        theta: torch.Tensor,   # (n_layers, n_qubits, 3)
        w: torch.Tensor,       # (n_layers, n_qubits, 3)
        xs: torch.Tensor,      # (B, n_qubits)  ← full batch
        active_layers: int,
    ) -> torch.Tensor:         # returns (B,) via parameter broadcasting
        _cz_preamble(n_qubits)

        for layer_idx in range(active_layers):
            th = theta[layer_idx]   # (n_qubits, 3)
            ww = w[layer_idx]       # (n_qubits, 3)
            for q in range(n_qubits):
                # xs[:, q]          → (B,)
                # th[q] / ww[q]     → (3,)
                # angles            → (B, 3) via broadcasting
                angles = th[q] + ww[q] * xs[:, q % xs.shape[1]].unsqueeze(-1)
                qml.Rot(angles[:, 0], angles[:, 1], angles[:, 2], wires=q)
            _cz_entangler(n_qubits)

        return qml.expval(qml.PauliZ(0))   # (B,) when xs is batched

    return circuit

# Identity-block initialisation 
def _identity_block_init(n_layers: int, n_qubits: int) -> tuple[nn.Parameter, nn.Parameter]:
    """Initialise theta randomly; set w = -theta so U = I at t=0."""
    theta_init = torch.empty(n_layers, n_qubits, 3).uniform_(0, 2 * math.pi)
    w_init = -theta_init.clone()
    return nn.Parameter(theta_init), nn.Parameter(w_init)

# Main module
class QuantumValueNetwork(nn.Module):
    """Hybrid quantum-classical value network V_psi(s) for Q-IQL."""

    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 3,
        obs_dim: int = 11,
        device_name: str = "default.qubit",
        diff_method: str = "backprop",
        running_stats: bool = True,
    ) -> None:
        super().__init__()

        assert n_layers <= math.floor(math.log2(n_qubits)), (
            f"n_layers={n_layers} violates L <= floor(log2({n_qubits}))="
            f"{math.floor(math.log2(n_qubits))}. Deeper circuits lose the polynomial gradient guarantee."
        )

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.obs_dim = obs_dim
        self._active_layers: int = n_layers

        self.theta, self.w = _identity_block_init(n_layers, n_qubits)
        self.a = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.zeros(1))

        self._running_stats = running_stats
        if running_stats:
            self.register_buffer("mu",    torch.zeros(obs_dim))
            self.register_buffer("sigma", torch.ones(obs_dim))
        else:
            self.mu = None
            self.sigma = None

        self._diff_method = diff_method
        self._circuit = _build_qnode(n_qubits, n_layers, device_name, diff_method)

    def set_active_layers(self, n: int) -> None:
        """Set number of active DRU layers for layerwise warm-up (Skolik et al., 2021).

        Schedule from architecture doc Table 3:
            Steps   0-100k  -> set_active_layers(1)
            Steps 100-300k  -> set_active_layers(2)
            Steps 300k+     -> set_active_layers(3)
        """
        if n < 1 or n > self.n_layers:
            raise ValueError(f"n must be in [1, {self.n_layers}], got {n}.")
        self._active_layers = n

    @property
    def active_layers(self) -> int:
        return self._active_layers

    def update_running_stats(self, mu: torch.Tensor, sigma: torch.Tensor) -> None:
        if not self._running_stats:
            raise RuntimeError("running_stats=False; pass mu/sigma explicitly to forward().")
        self.mu.copy_(mu)
        self.sigma.copy_(sigma)

    def forward(
        self,
        s: torch.Tensor,
        mu: torch.Tensor | None = None,
        sigma: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute V_psi(s) for a batch of states, returns shape (B,) in float32."""

        if s.shape[-1] != self.obs_dim:
            raise ValueError(
                f"Expected last dimension of `s` to be obs_dim={self.obs_dim}, "
                f"but got {s.shape[-1]} (full shape: {tuple(s.shape)})."
            )

        _device = s.device

        _mu    = mu    if mu    is not None else self.mu
        _sigma = sigma if sigma is not None else self.sigma

        xs = _arctan_encode(s, _mu, _sigma)   # (B, obs_dim)

        B = xs.shape[0]
        if self.obs_dim < self.n_qubits:
            pad = torch.zeros(B, self.n_qubits - self.obs_dim, device=xs.device, dtype=xs.dtype)
            xs = torch.cat([xs, pad], dim=-1)
        elif self.obs_dim > self.n_qubits:
            xs = xs[:, :self.n_qubits]

        # Device handling depends on the diff method:
        #   backprop  — converts the circuit to PyTorch ops, so it runs on
        #               whatever device the tensors live on (GPU if available).
        #               Keep tensors on _device; no CPU transfer needed.
        #   adjoint / parameter-shift — PennyLane simulates on CPU.
        #               Move inputs to CPU; move output back afterwards.
        if self._diff_method == "backprop":
            expvals = self._circuit(self.theta, self.w, xs, self._active_layers)
        else:
            expvals = self._circuit(
                self.theta.cpu(), self.w.cpu(), xs.cpu(), self._active_layers
            ).to(_device)

        expvals = expvals.float()

        return self.a * expvals + self.b  # shape (B,)

    def parameter_count(self) -> dict:
        quantum = self.theta.numel() + self.w.numel()
        classical = self.a.numel() + self.b.numel()
        return {"quantum": quantum, "classical_head": classical, "total": quantum + classical}

    def __repr__(self) -> str:
        pc = self.parameter_count()
        return (
            f"QuantumValueNetwork(n_qubits={self.n_qubits}, n_layers={self.n_layers}, "
            f"obs_dim={self.obs_dim}, active_layers={self._active_layers}, "
            f"params={pc['total']} [{pc['quantum']} quantum + {pc['classical_head']} head])"
        )
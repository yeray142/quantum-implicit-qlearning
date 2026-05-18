"""Smoke test for run_on_helios_emulator with DepolarizingErrorModel.

One state through the full selene_sim path, verifying:
  - Returns a finite V
  - Returns per_qubit_expvals of shape (n_qubits,)
  - V differs from V_pennylane by at most 5 × |Σ|aᵢ|| × 0.05

No HQC spent — uses default.qubit (PennyLane) as reference only.
"""

from __future__ import annotations

import pytest

guppylang_available = False
try:
    import guppylang  # noqa: F401
    guppylang_available = True
except ImportError:
    pass


def _find_checkpoint():
    from pathlib import Path

    checkpoint_paths = [
        Path("experiments/checkpoints/hopper/medium/quantum-multi-qubit-readout/seed_6/checkpoint_final.pt"),
        Path("experiments/checkpoints/hopper/medium/quantum-fixed/seed_6/checkpoint_final.pt"),
        Path("experiments/checkpoints/hopper/medium/quantum-fixed-c/seed_6/checkpoint_final.pt"),
        Path("experiments/checkpoints/hopper/medium/quantum-fixed-warmup/seed_6/checkpoint_final.pt"),
        Path("experiments/checkpoints/hopper/medium/quantum-no-warmup/seed_6/checkpoint_final.pt"),
        Path("experiments/checkpoints/hopper/medium/quantum/seed_6/checkpoint_final.pt"),
        Path("experiments/checkpoints/hopper/medium/quantum/seed_0/checkpoint_final.pt"),
    ]
    for p in checkpoint_paths:
        if p.exists():
            return p
    return None


def _load_network(ckpt_path):
    import torch

    from quantum_iql import QuantumValueNetwork

    device = torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location=device)

    quantum_meta = ckpt.get("quantum_meta", {})
    obs_dim = quantum_meta.get("obs_dim", 11)
    n_qubits = quantum_meta.get("n_qubits", 8)
    n_layers = quantum_meta.get("n_layers", 3)

    a_ckpt = ckpt["value_net"]["a"]
    multi_qubit_readout = a_ckpt.shape[0] > 1

    value_net = QuantumValueNetwork(
        n_qubits=n_qubits,
        n_layers=n_layers,
        obs_dim=obs_dim,
        device_name="default.qubit",
        diff_method="backprop",
        running_stats=True,
        use_pre_encoder=True,
        multi_qubit_readout=multi_qubit_readout,
    )
    value_net.load_state_dict(ckpt["value_net"], strict=False)
    if "mu" in ckpt and "sigma" in ckpt:
        value_net.update_running_stats(
            torch.as_tensor(ckpt["mu"]),
            torch.as_tensor(ckpt["sigma"]),
        )
    value_net.eval()
    return value_net, multi_qubit_readout


@pytest.mark.skipif(not guppylang_available, reason="Requires guppylang")
def test_helios_emulator_smoke():
    """Run one state through run_on_helios_emulator and validate output."""
    import numpy as np
    import torch
    from selene_sim import DepolarizingErrorModel

    from src.eval.run_quantum_batch import HELIOS_ERROR_MODEL, run_on_helios_emulator

    ckpt_path = _find_checkpoint()
    if ckpt_path is None:
        pytest.skip("No trained quantum checkpoint found")

    value_net, _ = _load_network(ckpt_path)
    n_qubits = value_net.n_qubits

    # Build error model from config
    em = HELIOS_ERROR_MODEL
    error_model = DepolarizingErrorModel(
        p_1q=em["p_1q"],
        p_2q=em["p_2q"],
        p_init=em["p_init"],
        p_meas=em["p_meas"],
    )

    # One random-ish state (use deterministic seed for reproducibility)
    rng = np.random.default_rng(seed=42)
    state = rng.normal(size=(value_net.obs_dim,)).astype(np.float32)

    # V_pennylane reference (canonical forward pass)
    with torch.no_grad():
        s_t = torch.from_numpy(state).unsqueeze(0)
        v_pennylane = float(value_net(s_t).cpu().item())

    # V_helios_emulator
    v_vals, raw_expvals = run_on_helios_emulator(
        states=state,
        value_net=value_net,
        mu=value_net.mu,
        sigma=value_net.sigma,
        shots=100,
        error_model=error_model,
    )

    v_helios = v_vals[0]
    per_qubit_expvals = raw_expvals[0]

    # 1. Finite V
    assert np.isfinite(v_helios), f"V_helios is not finite: {v_helios}"

    # 2. Shape (n_qubits,)
    assert isinstance(per_qubit_expvals, np.ndarray), f"Expected np.ndarray, got {type(per_qubit_expvals)}"
    assert per_qubit_expvals.shape == (n_qubits,), (
        f"Expected per_qubit_expvals shape ({n_qubits},), got {per_qubit_expvals.shape}"
    )

    # 3. V differs from V_pennylane by at most 5 × |Σ|aᵢ|| × 0.05
    a = value_net.a.detach().cpu().numpy()
    abs_a_sum = float(np.sum(np.abs(a)))
    # Shot noise budget: 0.05 × |Σ|aᵢ| per shot at 100 shots
    # Scale factor 5× accounts for device noise + statisticalfluctuation
    tol = 5 * abs_a_sum * 0.05

    diff = abs(v_pennylane - v_helios)
    print(f"\n  V_pennylane={v_pennylane:.6f}, V_helios={v_helios:.6f}, "
          f"diff={diff:.6f}, tol={tol:.6f}, |Σ|aᵢ||={abs_a_sum:.4f}")

    assert diff <= tol, (
        f"V difference {diff:.6f} exceeds tolerance {tol:.6f} "
        f"(5 × |Σ|aᵢ|| × 0.05 = {tol:.6f})"
    )


if __name__ == "__main__":
    test_helios_emulator_smoke()
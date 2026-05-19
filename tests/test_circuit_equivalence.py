"""Circuit equivalence test: PennyLane vs guppylang HUGR for V(s).

Verifies that the guppylang DRU circuit produces the same V(s) as
QuantumValueNetwork.forward() for the same trained weights and state.
"""

from __future__ import annotations

import pytest

# Check dependencies before importing
guppylang_available = False
try:
    import guppylang  # noqa: F401
    guppylang_available = True
except ImportError:
    pass

pytorch_available = False
try:
    import torch  # noqa: F401
    pytorch_available = True
except ImportError:
    pass


def _find_checkpoint():
    """Locate a trained quantum checkpoint. Returns None if none found."""
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
    """Load and return (value_net, mu, sigma, a, b)."""
    import numpy as np
    import torch

    from quantum_iql import QuantumValueNetwork

    device = torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location=device)

    quantum_meta = ckpt.get("quantum_meta", {})
    obs_dim = quantum_meta.get("obs_dim", 11)
    n_qubits = quantum_meta.get("n_qubits", 8)
    n_layers = quantum_meta.get("n_layers", 3)

    # Detect multi_qubit_readout from checkpoint a shape
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

    mu = value_net.mu
    sigma = value_net.sigma
    a = value_net.a.detach().cpu().numpy()
    b = float(value_net.b.detach().cpu().item())

    return value_net, mu, sigma, a, b, multi_qubit_readout


@pytest.mark.skipif(
    not (guppylang_available and pytorch_available),
    reason="Requires guppylang and torch",
)
def test_pennylane_vs_guppylang_high_shots():
    """For 5 D4RL states, V_pennylane and V_guppylang agree at high shots (50k).

    At shots=50000 the standard error on ⟨Z₀⟩ is ~0.0045. Tolerance is
    0.05 * |a| (residual shot noise + numerical roundoff) with a floor of 2.0.
    """
    from pathlib import Path

    import numpy as np
    import torch

    from quantum_iql import load_minari_dataset
    from src.eval.utils import build_dru_hugr, run_hugr_and_collect_expvals, compute_v_from_expvals

    ckpt_path = _find_checkpoint()
    if ckpt_path is None:
        pytest.skip("No trained quantum checkpoint found")

    value_net, mu, sigma, a, b, multi_qubit_readout = _load_network(ckpt_path)

    # Load 5 random states from D4RL
    buffer = load_minari_dataset("mujoco/hopper/medium-v0", device="cpu")
    batch = buffer.sample(5)
    states = batch.observations.numpy()  # (5, 11)

    failures = []
    n_qubits = value_net.n_qubits
    for i, state in enumerate(states):
        # PennyLane path
        s_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0)
        v_pennylane = float(value_net(s_t).cpu().item())

        # guppylang path — 50k shots for low noise
        compiled_hugr = build_dru_hugr(value_net, mu, sigma, state, shots=50000)
        raw_expvals = run_hugr_and_collect_expvals(compiled_hugr, n_qubits, shots=50000)

        if multi_qubit_readout:
            v_guppylang = compute_v_from_expvals(raw_expvals, value_net)
            abs_a = float(np.sum(np.abs(a)))
        else:
            v_guppylang = float(a * raw_expvals[0] + b)
            abs_a = float(np.abs(a))

        diff = abs(v_pennylane - v_guppylang)
        raw_expval = raw_expvals[0] if not multi_qubit_readout else np.mean(raw_expvals)
        expected_se = np.sqrt((1 - raw_expval ** 2) / 50000)  # SE of raw ⟨Z₀⟩
        tol = max(0.05 * abs_a, 2.0)

        print(
            f"  State {i}: V_pennylane={v_pennylane:.6f}, V_guppylang={v_guppylang:.6f}, "
            f"diff={diff:.6f}, |a|={abs_a:.4f}, expected_SE={expected_se:.6f}, "
            f"ratio={diff / max(expected_se, 1e-9):.2f}"
        )

        if diff > tol:
            failures.append((i, v_pennylane, v_guppylang, diff, tol))

    if failures:
        # Print per-qubit diagnostic before failing
        for i, v_pennylane, v_guppylang, diff, tol in failures:
            print(f"\n  FAILURE state {i}: diff={diff:.6f} > tol={tol:.6f}")
        pytest.fail(
            f"{len(failures)}/5 states exceeded tolerance. "
            "If residual is NOT shot noise, there is still a circuit bug."
        )


@pytest.mark.skipif(
    not (guppylang_available and pytorch_available),
    reason="Requires guppylang and torch",
)
def test_shot_noise_statistics():
    """For 2 states, empirical mean of V_guppylang converges to V_pennylane.

    Runs guppylang at shots=100, N=30 times. Checks that the empirical mean
    is within 3 * (empirical_SE / sqrt(30)) of V_pennylane.
    """
    import numpy as np
    import torch

    from quantum_iql import load_minari_dataset
    from src.eval.utils import build_dru_hugr, run_hugr_and_collect_expvals, compute_v_from_expvals

    ckpt_path = _find_checkpoint()
    if ckpt_path is None:
        pytest.skip("No trained quantum checkpoint found")

    value_net, mu, sigma, a, b, multi_qubit_readout = _load_network(ckpt_path)

    # Load 2 random states
    buffer = load_minari_dataset("mujoco/hopper/medium-v0", device="cpu")
    batch = buffer.sample(2)
    states = batch.observations.numpy()

    N = 30
    shots = 100

    for idx, state in enumerate(states):
        s_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0)
        v_pennylane = float(value_net(s_t).cpu().item())

        v_samples = []
        compiled_hugr = build_dru_hugr(value_net, mu, sigma, state, shots=shots)
        n_qubits = value_net.n_qubits

        for _ in range(N):
            raw_expvals = run_hugr_and_collect_expvals(compiled_hugr, n_qubits, shots=shots)
            v_samples.append(compute_v_from_expvals(raw_expvals, value_net))

        empirical_mean = np.mean(v_samples)
        empirical_se = np.std(v_samples, ddof=1)  # sample std dev

        # 3-sigma check on the mean
        mean_se = empirical_se / np.sqrt(N)
        z_score = abs(empirical_mean - v_pennylane) / max(mean_se, 1e-12)

        print(
            f"  State {idx}: V_pennylane={v_pennylane:.6f}, "
            f"empirical_mean={empirical_mean:.6f}, empirical_SE={empirical_se:.6f}, "
            f"mean_SE={mean_se:.6f}, z_score={z_score:.2f}"
        )

        assert z_score < 3.0, (
            f"State {idx}: |empirical_mean - V_pennylane| = {z_score:.2f} * mean_SE "
            f"(expected < 3). empirical_SE={empirical_se:.6f}. "
            "The empirical distribution does not centre on V_pennylane."
        )


if __name__ == "__main__":
    test_pennylane_vs_guppylang_high_shots()
    test_shot_noise_statistics()
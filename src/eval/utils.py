"""Shared utilities for offline hardware evaluation.

Extracts guppylang DRU circuit builder and related helpers so they can be
imported by both phase2_hardware.py and the test suite.
"""

from __future__ import annotations

import importlib.util
import os
import tempfile
from typing import Any

import numpy as np
import torch

from quantum_iql import QuantumValueNetwork


def build_dru_hugr(
    value_net: QuantumValueNetwork,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    state: np.ndarray,
    shots: int = 100,
) -> Any:
    """Build a guppylang HUGR program encoding the trained DRU circuit.

    Generates guppylang source as a string with hardcoded parameter values
    (theta, w, x_encodings), then dynamically imports and compiles it.
    This avoids guppylang's scoping limitations with Python runtime values.

    Args:
        value_net: trained QuantumValueNetwork with theta, w, a, b weights
        mu, sigma: running stats for arctan encoding
        state: (obs_dim,) single state vector
        shots: number of shots (used for documentation; not encoded in HUGR)

    Returns:
        compiled guppylang HUGR program
    """
    try:
        from guppylang import guppy  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("guppylang not installed") from exc

    n_qubits = value_net.n_qubits
    active_layers = value_net._active_layers

    theta_np = value_net.theta.detach().cpu().numpy()
    w_np = value_net.w.detach().cpu().numpy()

    if not isinstance(state, np.ndarray):
        state = np.asarray(state, dtype=np.float32)
    s_t = torch.from_numpy(state.astype(np.float32)).to(mu.device)
    xs = torch.arctan((s_t - mu) / (sigma + 1e-8))
    xs_np = xs.cpu().numpy()

    obs_dim = value_net.obs_dim
    if obs_dim < n_qubits:
        xs_np = np.concatenate([xs_np, np.zeros(n_qubits - obs_dim)])
    elif obs_dim > n_qubits:
        if value_net.use_pre_encoder and value_net.pre_encode is not None:
            with torch.no_grad():
                xs_np = value_net.pre_encode(xs.unsqueeze(0)).cpu().numpy()[0]
        else:
            xs_np = xs_np[:n_qubits]

    src = _generate_dru_source(n_qubits, active_layers, theta_np, w_np, xs_np)

    with tempfile.NamedTemporaryFile(
        suffix=".py", delete=False, mode="w",
        prefix=f"dru_n{n_qubits}_L{active_layers}_"
    ) as f:
        f.write(src)
        tmpfile = f.name

    try:
        spec = importlib.util.spec_from_file_location("dru_circuit", tmpfile)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.hugr_pkg
    finally:
        os.unlink(tmpfile)


def _generate_dru_source(
    n_qubits: int,
    active_layers: int,
    theta_np: np.ndarray,
    w_np: np.ndarray,
    xs_np: np.ndarray,
) -> str:
    """Generate guppylang source for a DRU circuit with hardcoded parameter values."""
    qv = [f"q{i}" for i in range(n_qubits)]
    alloc = ", ".join(qv)
    rhs = ", ".join(["qubit()"] * n_qubits)

    body: list[str] = [
        "from guppylang import guppy",
        "from guppylang.std.quantum import cz, ry, rz, measure, qubit",
        "from guppylang.std.angles import angle",
        "",
        "@guppy",
        "def main() -> None:",
        f"    {alloc} = {rhs}",
        "",
        "    # CZ preamble — even pairs",
    ]

    for q in range(0, n_qubits - 1, 2):
        body.append(f"    cz(q{q}, q{q + 1})")
    body.append("    # CZ preamble — odd pairs")
    for q in range(1, n_qubits - 1, 2):
        body.append(f"    cz(q{q}, q{q + 1})")

    for layer in range(active_layers):
        body.append("")
        body.append(f"    # DRU layer {layer + 1}")
        for q in range(n_qubits):
            th = theta_np[layer, q]
            ww = w_np[layer, q]
            x_val = xs_np[q]

            r0 = float(th[0] + ww[0] * x_val)
            r1 = float(th[1] + ww[1] * x_val)
            r2 = float(th[2] + ww[2] * x_val)

            # Normalize to half-turns (guppy's angle type is half-turns: angle(1.0) = π radians)
            a0 = r0 / np.pi
            a1 = r1 / np.pi
            a2 = r2 / np.pi

            body.append(f"    # qubit {q}: r0={r0:.6f}, r1={r1:.6f}, r2={r2:.6f}")
            body.append(f"    rz(q{q}, angle({a0:.10f}))")
            body.append(f"    ry(q{q}, angle({a1:.10f}))")
            body.append(f"    rz(q{q}, angle({a2:.10f}))")

        body.append(f"    # DRU layer {layer + 1}: CZ entangler")
        for q in range(n_qubits - 1):
            body.append(f"    cz(q{q}, q{q + 1})")

    body.append("    # Measure ALL qubits — collect per-qubit ⟨Zᵢ⟩ for weighted sum V(s)")
    for q in range(n_qubits):
        bit_q = f"bit{q}"
        body.append(f"    {bit_q} = measure(q{q})")
        body.append(f"    result('Z{q}', {bit_q})")

    body.append("")
    body.append("hugr_pkg = main.compile()")
    return "\n".join(body)


def run_hugr_and_collect_expvals(
    compiled_hugr,
    n_qubits: int,
    shots: int = 100,
) -> np.ndarray:
    """Run compiled HUGR on default.qubit and return per-qubit ⟨Zᵢ⟩ values.

    Args:
        compiled_hugr: result of build_dru_hugr()
        n_qubits: number of qubits in the circuit
        shots: number of measurement shots

    Returns:
        np.ndarray of shape (n_qubits,) with ⟨Zᵢ⟩ = (n0 - n1) / shots for each qubit i
    """
    try:
        from hugr.qsystem.result import QsysResult
        from selene_sim import Quest
        from selene_sim import build as build_runner
    except ImportError as exc:
        raise RuntimeError("selene-sim not installed") from exc

    runner = build_runner(compiled_hugr)
    raw_result = runner.run_shots(
        simulator=Quest(),
        n_qubits=n_qubits,
        n_shots=shots,
    )
    result_obj = QsysResult(raw_result)
    counts = result_obj.collated_counts()
    return expvals_from_counts(counts, n_qubits, shots)


def expvals_from_counts(counts: dict, n_qubits: int, n_shots: int) -> np.ndarray:
    """Compute per-qubit ⟨Zᵢ⟩ array from collated counts.

    The counts dict uses multi-qubit tuple keys: each key is a tuple of
    (register_name, bit) pairs, one per qubit, e.g.:
        (('Z0', '0'), ('Z1', '0'), ('Z2', '0'), ('Z3', '1'), ...)
    for a shot where qubits 0,1,2 measured 0 and qubit 3 measured 1.

    Args:
        counts: collated_counts() dict
        n_qubits: number of qubits
        n_shots: total shot count

    Returns:
        np.ndarray of shape (n_qubits,)
    """
    expvals = np.zeros(n_qubits)
    for q in range(n_qubits):
        n0 = 0
        n1 = 0
        for key, count in counts.items():
            # key is e.g. (('Z0', '0'), ('Z1', '1'), ...)
            # The qubit q result is the q-th element: key[q] = ('Zq', '0'/'1')
            qubit_result = key[q]
            if qubit_result[1] == "0":
                n0 += count
            else:
                n1 += count
        expvals[q] = (n0 - n1) / n_shots
    return expvals


def compute_v_from_expvals(expvals: np.ndarray, value_net: QuantumValueNetwork) -> float:
    """Canonical V(s) reconstruction from per-qubit expectation values.

    Uses the value_net's _multi_qubit_readout flag to determine the correct
    readout formula. This is the ONLY place V is reconstructed from guppy output.

    Args:
        expvals: (n_qubits,) array of ⟨Zᵢ⟩ values
        value_net: QuantumValueNetwork with trained a, b weights

    Returns:
        V(s) = Σᵢ aᵢ⟨Zᵢ⟩ + b  (MQR) or a⟨Z₀⟩ + b (single-qubit)
    """
    a = value_net.a.detach().cpu().numpy()
    b = float(value_net.b.detach().cpu().item())
    if value_net._multi_qubit_readout:
        return float(np.dot(a, expvals) + b)
    else:
        return float(a.item() * expvals[0] + b)


def arctan_encode(
    state: np.ndarray,
    value_net: QuantumValueNetwork,
    mu: torch.Tensor,
    sigma: torch.Tensor,
) -> np.ndarray:
    """Arctan-encode a single state to (n_qubits,) xs array."""
    if not isinstance(state, np.ndarray):
        state = np.asarray(state, dtype=np.float32)
    s_t = torch.from_numpy(state.astype(np.float32)).to(mu.device)
    xs = torch.arctan((s_t - mu) / (sigma + 1e-8))
    xs_np = xs.cpu().numpy()

    obs_dim = value_net.obs_dim
    n_qubits = value_net.n_qubits

    if obs_dim < n_qubits:
        xs_np = np.concatenate([xs_np, np.zeros(n_qubits - obs_dim)])
    elif obs_dim > n_qubits:
        if value_net.use_pre_encoder and value_net.pre_encode is not None:
            with torch.no_grad():
                xs_np = value_net.pre_encode(xs.unsqueeze(0)).cpu().numpy()[0]
        else:
            xs_np = xs_np[:n_qubits]
    return xs_np
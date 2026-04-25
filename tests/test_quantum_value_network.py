"""
test_quantum_value_network.py
=============================
Pytest test suite for QuantumValueNetwork (Issue #7).

Place this file at: tests/test_quantum_value_network.py
The module must be at: src/quantum_iql/quantum_value_network.py

Run:
    pytest tests/test_quantum_value_network.py -v
"""

import math

import pytest
import torch
import torch.nn as nn

from scripts.quantum_value_network import (
    QuantumValueNetwork,
    _arctan_encode,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_QUBITS = 8
N_LAYERS = 3
OBS_DIM  = 11
BATCH    = 4
DEVICE   = "default.qubit"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def net():
    return QuantumValueNetwork(
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        obs_dim=OBS_DIM,
        device_name=DEVICE,
        running_stats=True,
    )


@pytest.fixture
def net_1layer():
    m = QuantumValueNetwork(
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        obs_dim=OBS_DIM,
        device_name=DEVICE,
        running_stats=True,
    )
    m.set_active_layers(1)
    return m


@pytest.fixture
def states():
    torch.manual_seed(0)
    return torch.randn(BATCH, OBS_DIM)


# ---------------------------------------------------------------------------
# T1 — Output shape
# ---------------------------------------------------------------------------

def test_output_shape(net, states):
    with torch.no_grad():
        out = net(states)
    assert out.shape == (BATCH,), f"Expected ({BATCH},), got {out.shape}"


# ---------------------------------------------------------------------------
# T2 — Raw circuit output in [-1, 1]
# ---------------------------------------------------------------------------

def test_output_range_raw(net, states):
    nn.init.ones_(net.a)
    nn.init.zeros_(net.b)
    with torch.no_grad():
        out = net(states)
    assert out.min() >= -1.0 - 1e-5, f"Below -1: {out.min().item():.6f}"
    assert out.max() <=  1.0 + 1e-5, f"Above +1: {out.max().item():.6f}"


# ---------------------------------------------------------------------------
# T3 — Parameter-shift gradients
# ---------------------------------------------------------------------------

def test_parameter_shift_gradients(net_1layer, states):
    out  = net_1layer(states)
    loss = out.sum()
    loss.backward()

    for name, param in [
        ("theta", net_1layer.theta),
        ("w",     net_1layer.w),
        ("a",     net_1layer.a),
        ("b",     net_1layer.b),
    ]:
        assert param.grad is not None, f"No gradient for '{name}'"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient for '{name}'"


# ---------------------------------------------------------------------------
# T4 — Identity-block initialisation
# ---------------------------------------------------------------------------

def test_identity_block_init(net):
    max_diff = (net.w + net.theta).abs().max().item()
    assert max_diff < 1e-7, f"Identity-block init violated: max |w + theta| = {max_diff:.2e}"


# ---------------------------------------------------------------------------
# T5 — Layerwise interface
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 3])
def test_set_active_layers_valid(net, n):
    net.set_active_layers(n)
    assert net.active_layers == n


@pytest.mark.parametrize("n", [0, 4, -1, 10])
def test_set_active_layers_invalid(net, n):
    with pytest.raises(ValueError):
        net.set_active_layers(n)


# ---------------------------------------------------------------------------
# T6 — Arctan encoding range
# ---------------------------------------------------------------------------

def test_arctan_encoding_range():
    torch.manual_seed(1)
    s     = torch.randn(20, OBS_DIM) * 10
    mu    = s.mean(0)
    sigma = s.std(0) + 1e-8
    xs    = _arctan_encode(s, mu, sigma)

    HALF_PI = math.pi / 2
    assert xs.min() > -HALF_PI - 1e-5, f"Below -pi/2: {xs.min().item():.6f}"
    assert xs.max() <  HALF_PI + 1e-5, f"Above +pi/2: {xs.max().item():.6f}"


# ---------------------------------------------------------------------------
# T7 — Parameter count
# ---------------------------------------------------------------------------

def test_parameter_count(net):
    pc = net.parameter_count()

    expected_quantum = 2 * N_LAYERS * N_QUBITS * 3   # 144
    expected_head    = 2
    expected_total   = expected_quantum + expected_head

    assert pc["quantum"]        == expected_quantum, \
        f"Expected {expected_quantum} quantum params, got {pc['quantum']}"
    assert pc["classical_head"] == expected_head, \
        f"Expected {expected_head} head params, got {pc['classical_head']}"
    assert pc["total"]          == expected_total, \
        f"Expected {expected_total} total, got {pc['total']}"


# ---------------------------------------------------------------------------
# T8 — Deterministic finite outputs
# ---------------------------------------------------------------------------

def test_deterministic_finite(net, states):
    with torch.no_grad():
        out1 = net(states)
        out2 = net(states)
    assert out1.shape == out2.shape == (BATCH,)
    assert torch.isfinite(out1).all(), f"NaN/Inf in first pass: {out1}"
    assert torch.isfinite(out2).all(), f"NaN/Inf in second pass: {out2}"


# ---------------------------------------------------------------------------
# T9 — Zero-padding when obs_dim < n_qubits
# ---------------------------------------------------------------------------

def test_zero_padding_small_obs():
    net_small = QuantumValueNetwork(
        n_qubits=N_QUBITS,
        n_layers=1,
        obs_dim=4,
        device_name=DEVICE,
        running_stats=True,
    )
    s = torch.randn(2, 4)
    with torch.no_grad():
        out = net_small(s)
    assert out.shape == (2,), f"Expected (2,), got {out.shape}"
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# T10 — Gradient through affine output head
# ---------------------------------------------------------------------------

def test_affine_head_gradients(net_1layer, states):
    out  = net_1layer(states)
    loss = out.mean()
    loss.backward()

    assert net_1layer.a.grad is not None, "No gradient for 'a'"
    assert net_1layer.b.grad is not None, "No gradient for 'b'"
    assert torch.isfinite(net_1layer.a.grad).all(), "Non-finite grad for 'a'"
    assert torch.isfinite(net_1layer.b.grad).all(), "Non-finite grad for 'b'"
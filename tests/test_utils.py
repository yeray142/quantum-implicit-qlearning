"""Tests for utility functions."""

import torch
import torch.nn as nn

from quantum_iql.utils import get_device, hard_update, set_seed, soft_update


def _simple_net(val: float) -> nn.Linear:
    net = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        net.weight.fill_(val)
    return net


# ---------------------------------------------------------------------------
# soft_update
# ---------------------------------------------------------------------------

def test_soft_update_tau_zero_leaves_target_unchanged():
    target = _simple_net(0.0)
    source = _simple_net(1.0)
    soft_update(target, source, tau=0.0)
    assert (target.weight == 0.0).all()


def test_soft_update_tau_one_copies_source():
    target = _simple_net(0.0)
    source = _simple_net(1.0)
    soft_update(target, source, tau=1.0)
    assert torch.allclose(target.weight, source.weight)


def test_soft_update_interpolates():
    target = _simple_net(0.0)
    source = _simple_net(1.0)
    soft_update(target, source, tau=0.1)
    assert torch.allclose(target.weight, torch.full_like(target.weight, 0.1))


# ---------------------------------------------------------------------------
# hard_update
# ---------------------------------------------------------------------------

def test_hard_update_copies_exactly():
    target = _simple_net(0.0)
    source = _simple_net(3.14)
    hard_update(target, source)
    assert torch.allclose(target.weight, source.weight)


def test_hard_update_is_copy_not_reference():
    target = _simple_net(0.0)
    source = _simple_net(1.0)
    hard_update(target, source)
    with torch.no_grad():
        source.weight.fill_(99.0)
    # target should still hold the old value
    assert (target.weight == 1.0).all()


# ---------------------------------------------------------------------------
# set_seed
# ---------------------------------------------------------------------------

def test_set_seed_makes_torch_deterministic():
    set_seed(42)
    a = torch.randn(10)
    set_seed(42)
    b = torch.randn(10)
    assert torch.allclose(a, b)


def test_set_seed_different_seeds_differ():
    set_seed(0)
    a = torch.randn(10)
    set_seed(1)
    b = torch.randn(10)
    assert not torch.allclose(a, b)


# ---------------------------------------------------------------------------
# get_device
# ---------------------------------------------------------------------------

def test_get_device_cpu():
    d = get_device("cpu")
    assert d.type == "cpu"


def test_get_device_auto_returns_device():
    d = get_device("auto")
    assert d.type in ("cpu", "cuda")

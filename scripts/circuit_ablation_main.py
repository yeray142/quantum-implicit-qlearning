#!/usr/bin/env python3
"""
Circuit Design Ablation Study — Topology, Measurement, Dataset Size
====================================================================
Covers: entanglement topology, measurement schemes, dataset size sensitivity,
plus Fourier spectrum and expressibility analysis.

NOT included here (separate scripts):
  layers ablation     → circuit_ablation_layers.py
  qubit×layer grid    → circuit_ablation_qubit_layer.py

Usage
-----
  python scripts/circuit_ablation_main.py --ablation topology measurement
  python scripts/circuit_ablation_main.py --ablation datasize --seeds 0 1 2 3 4
  python scripts/circuit_ablation_main.py --ablation fourier expr
  python scripts/circuit_ablation_main.py --ablation topology measurement datasize fourier expr --dry-run
"""
from __future__ import annotations

import argparse
import itertools
import math
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

print("[BOOT] Importing pennylane...", flush=True)
import pennylane as qml
print("[BOOT] Importing wandb...", flush=True)
import wandb
print("[BOOT] Importing scipy...", flush=True)
from scipy import stats as scipy_stats

print("[BOOT] Importing quantum_iql modules...", flush=True)
from quantum_iql.buffer import Batch, ReplayBuffer, load_minari_dataset
from quantum_iql.losses import critic_loss as _critic_loss, value_loss
from quantum_iql.networks import CriticNetwork, ValueNetwork
# FIX: QuantumIQLConfig/QuantumNetConfig don't exist → use IQLConfig/NetworkConfig
from quantum_iql.config import IQLConfig, NetworkConfig
# FIX: QuantumIQLTrainer doesn't exist → use IQLTrainer
from quantum_iql.trainer import IQLTrainer
from quantum_iql.utils import set_seed
print("[BOOT] All imports OK.", flush=True)

# ── Environment / dataset registry ───────────────────────────────────────────

_ENV_REGISTRY: Dict[str, dict] = {
    "hopper": {
        "dataset_id": "mujoco/hopper/medium-v0",
        "env_id":     "Hopper-v4",
        "obs_dim":    11,
        "act_dim":    3,
        "tau":        0.7,
        "beta":       3.0,
        "group":      "hopper-medium",
    },
    "walker2d": {
        "dataset_id": "mujoco/walker2d/medium-v0",
        "env_id":     "Walker2d-v4",
        "obs_dim":    17,
        "act_dim":    6,
        "tau":        0.7,
        "beta":       3.0,
        "group":      "walker2d-medium",
    },
}

# ── Shared training constants ─────────────────────────────────────────────────

NUM_STEPS       = 100_000
LOG_INTERVAL    = 500
WANDB_PROJECT   = "quantum-iql-circuit-ablation"

# Base circuit config (held fixed unless the ablation varies it)
BASE_N_QUBITS     = 4
BASE_N_LAYERS     = 2
BASE_ENTANGLEMENT = "linear"
BASE_MEASUREMENT  = "pauli_z0"
BASE_LR_V         = 1e-2
BASE_LR_Q         = 3e-4
BASE_TAU          = 0.7
BASE_POLYAK       = 0.005
BASE_GAMMA        = 0.99
BASE_BATCH        = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
QUANTUM_DEVICE      = "default.qubit"
QUANTUM_DIFF_METHOD = "backprop" if torch.cuda.is_available() else "adjoint"

RESULTS_DIR = Path("results/circuit_ablations")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Ablation axis definitions ─────────────────────────────────────────────────

LAYERS_CONDITIONS = {
    "n_layers=1(q=4)": dict(n_qubits=4, n_layers=1),
    "n_layers=2(q=4)": dict(n_qubits=4, n_layers=2),   # base
    "n_layers=3(q=8)": dict(n_qubits=8, n_layers=3),
}

TOPOLOGY_CONDITIONS = ["none", "linear", "circular", "all_to_all"]

QUBIT_CONDITIONS = {
    "n_qubits=2": dict(n_qubits=2, n_layers=1),
    "n_qubits=4": dict(n_qubits=4, n_layers=2),   # base
    "n_qubits=6": dict(n_qubits=6, n_layers=2),
    "n_qubits=8": dict(n_qubits=8, n_layers=2),
}

QUBIT_LAYER_GRID = [
    (2, 1),
    (4, 1), (4, 2),   # (4,2) = base ★
    (8, 1), (8, 2), (8, 3),
]

MEASUREMENT_CONDITIONS = ["pauli_z0", "mean_pauli_z", "zz_tensor", "learned"]

DATA_FRACTIONS = [0.10, 0.25, 0.50, 1.00]

ALL_ABLATIONS = ["layers", "topology", "qubits", "qubit_layer",
                 "measurement", "datasize", "fourier", "expr"]

# ── Utilities ─────────────────────────────────────────────────────────────────

def ci95(data: np.ndarray) -> np.ndarray:
    n = data.shape[0]
    se = data.std(axis=0) / np.sqrt(n)
    return scipy_stats.t.ppf(0.975, df=n - 1) * se


def to_device(batch: Batch) -> Batch:
    return Batch(
        observations=batch.observations.to(DEVICE),
        actions=batch.actions.to(DEVICE),
        rewards=batch.rewards.to(DEVICE),
        next_observations=batch.next_observations.to(DEVICE),
        dones=batch.dones.to(DEVICE),
    )


def _make_critic(obs_dim: int, act_dim: int, seed: int = 0) -> CriticNetwork:
    torch.manual_seed(seed)
    return CriticNetwork(obs_dim, act_dim, hidden_dims=(256, 256), use_twin=True).to(DEVICE)


def _n_params(n_qubits: int, n_layers: int) -> int:
    return n_qubits * n_layers * 3 * 2   # theta + w, each (n_layers, n_qubits, 3)


# ── Trainer config builder ────────────────────────────────────────────────────
# FIX: removed quantum-specific fields (lr_quantum, fix_c_enabled, etc.) that
#      don't exist in IQLConfig. Only fields present in IQLConfig are passed.

def _build_trainer_config(
    n_steps:    int   = NUM_STEPS,
    seed:       int   = 0,
    dataset_id: str   = _ENV_REGISTRY["hopper"]["dataset_id"],
    env_id:     str   = _ENV_REGISTRY["hopper"]["env_id"],
    tau:        float = BASE_TAU,
    # n_qubits / n_layers accepted but ignored here (used by FlexQVN path)
    n_qubits:   int   = BASE_N_QUBITS,
    n_layers:   int   = BASE_N_LAYERS,
) -> IQLConfig:
    print(f"    [config] building IQLConfig  dataset={dataset_id}  "
          f"steps={n_steps}  seed={seed}  tau={tau}", flush=True)
    return IQLConfig(
        dataset_id  = dataset_id,
        env_id      = env_id,
        tau         = tau,
        gamma       = BASE_GAMMA,
        polyak      = BASE_POLYAK,
        lr_v        = 3e-4,
        lr_q        = 3e-4,
        lr_actor    = 3e-4,
        batch_size  = BASE_BATCH,
        num_steps   = n_steps,
        log_interval= LOG_INTERVAL,
        eval_interval= n_steps + 1,   # skip mid-run evals for speed
        wandb_project= WANDB_PROJECT,
        wandb_offline= True,
        seed        = seed,
        device      = "auto",
    )


# ── FlexQuantumValueNetwork (topology + measurement ablations) ────────────────

def _apply_entanglement(n_qubits: int, topology: str) -> None:
    if topology == "none":
        return
    elif topology == "linear":
        for q in range(n_qubits - 1):
            qml.CZ(wires=[q, q + 1])
    elif topology == "circular":
        for q in range(n_qubits - 1):
            qml.CZ(wires=[q, q + 1])
        if n_qubits > 2:
            qml.CZ(wires=[n_qubits - 1, 0])
    elif topology == "all_to_all":
        for q1, q2 in itertools.combinations(range(n_qubits), 2):
            qml.CZ(wires=[q1, q2])
    else:
        raise ValueError(f"Unknown topology: {topology!r}")


class FlexQuantumValueNetwork(nn.Module):
    """Flexible QVN for topology and measurement ablations.

    Uses torch.vmap for batched circuit evaluation — no per-sample loop.
    Identity-block init (w = -theta) as per training_dynamics_final.ipynb.
    """

    def __init__(
        self,
        n_qubits:     int = BASE_N_QUBITS,
        n_layers:     int = BASE_N_LAYERS,
        obs_dim:      int = 11,
        entanglement: str = BASE_ENTANGLEMENT,
        measurement:  str = BASE_MEASUREMENT,
        device_name:  str = QUANTUM_DEVICE,
        diff_method:  str = QUANTUM_DIFF_METHOD,
    ) -> None:
        super().__init__()
        self.n_qubits     = n_qubits
        self.n_layers     = n_layers
        self.obs_dim      = obs_dim
        self.entanglement = entanglement
        self.measurement  = measurement

        print(f"    [FlexQVN] init  n_qubits={n_qubits}  n_layers={n_layers}  "
              f"entanglement={entanglement}  measurement={measurement}", flush=True)

        self.register_buffer("obs_mean",  torch.zeros(obs_dim))
        self.register_buffer("obs_std",   torch.ones(obs_dim))
        self.register_buffer("obs_count", torch.tensor(0.0))

        self.theta = nn.Parameter(
            torch.zeros(n_layers, n_qubits, 3, dtype=torch.float32))
        self.w = nn.Parameter(
            torch.zeros(n_layers, n_qubits, 3, dtype=torch.float32))
        with torch.no_grad():
            nn.init.uniform_(self.theta, 0, 2 * math.pi)
            self.w.copy_(-self.theta)   # identity-block init

        self.out_scale = nn.Parameter(torch.ones(1))
        self.out_bias  = nn.Parameter(torch.zeros(1))

        if measurement == "learned":
            self.meas_weights = nn.Parameter(torch.ones(n_qubits) / n_qubits)
        else:
            self.meas_weights = None

        dev = qml.device(device_name, wires=n_qubits)
        scalar_meas = measurement in ("pauli_z0", "zz_tensor")

        if scalar_meas:
            @qml.qnode(dev, interface="torch", diff_method=diff_method)
            def _circuit(theta, w, xs):
                _apply_entanglement(n_qubits, entanglement)
                for layer in range(n_layers):
                    for q in range(n_qubits):
                        angles = theta[layer, q] + w[layer, q] * xs[q]
                        qml.Rot(angles[0], angles[1], angles[2], wires=q)
                    _apply_entanglement(n_qubits, entanglement)
                if measurement == "pauli_z0":
                    return qml.expval(qml.PauliZ(0))
                else:   # zz_tensor
                    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
        else:
            @qml.qnode(dev, interface="torch", diff_method=diff_method)
            def _circuit(theta, w, xs):
                _apply_entanglement(n_qubits, entanglement)
                for layer in range(n_layers):
                    for q in range(n_qubits):
                        angles = theta[layer, q] + w[layer, q] * xs[q]
                        qml.Rot(angles[0], angles[1], angles[2], wires=q)
                    _apply_entanglement(n_qubits, entanglement)
                return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

        self._circuit      = _circuit
        self._scalar_meas  = scalar_meas
        print(f"    [FlexQVN] circuit built OK  params={_n_params(n_qubits, n_layers)}", flush=True)

    @torch.no_grad()
    def _update_running_stats(self, obs: torch.Tensor) -> None:
        n = obs.shape[0]
        self.obs_count += n
        delta = obs.mean(0) - self.obs_mean
        self.obs_mean += delta * n / self.obs_count
        self.obs_std = (
            self.obs_std ** 2 + delta ** 2 * n / self.obs_count
        ).sqrt().clamp(min=1e-6)

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        xs = torch.arctan((obs - self.obs_mean) / (self.obs_std + 1e-8))
        if xs.shape[1] > self.n_qubits:
            xs = xs[:, :self.n_qubits]
        elif xs.shape[1] < self.n_qubits:
            pad = torch.zeros(xs.shape[0], self.n_qubits - xs.shape[1],
                              device=obs.device)
            xs = torch.cat([xs, pad], dim=1)
        return xs

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.training:
            self._update_running_stats(obs.detach())
        xs = self._encode(obs).requires_grad_(True)   # (B, n_qubits)

        if self._scalar_meas:
            out = torch.vmap(
                lambda x: self._circuit(self.theta, self.w, x).to(torch.float32)
            )(xs)
        elif self.measurement == "mean_pauli_z":
            def _eval(x):
                raw = self._circuit(self.theta, self.w, x)
                return torch.stack([r.to(torch.float32) for r in raw]).mean()
            out = torch.vmap(_eval)(xs)
        else:   # learned
            w_norm = torch.softmax(self.meas_weights, dim=0)
            def _eval(x):
                raw = self._circuit(self.theta, self.w, x)
                return (torch.stack([r.to(torch.float32) for r in raw]) * w_norm).sum()
            out = torch.vmap(_eval)(xs)

        return out * self.out_scale + self.out_bias


class _FlexWrap(nn.Module):
    """Wrap FlexQVN so its output is (B,1) as expected by critic_loss/value_loss."""
    def __init__(self, net: FlexQuantumValueNetwork) -> None:
        super().__init__()
        self.net = net

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        out = self.net(obs)
        return out.to(torch.float32).unsqueeze(-1)


# ── Core training loops ───────────────────────────────────────────────────────

def _run_trainer_seed(
    buffer:  ReplayBuffer,
    cfg:     IQLConfig,        # FIX: was QuantumIQLConfig
    n_steps: int,
    seed:    int,
) -> dict:
    """One seed via IQLTrainer (layers / qubits / qubit_layer / datasize)."""
    print(f"    [trainer] seed={seed}  n_steps={n_steps}  "
          f"buffer_size={len(buffer)}", flush=True)
    set_seed(seed)
    trainer = IQLTrainer(cfg, buffer, env=None)   # FIX: was QuantumIQLTrainer
    losses, v_stds, adv_means, adv_stds, grad_norms, step_ms = [], [], [], [], [], []

    t_start = time.perf_counter()
    for step in range(1, n_steps + 1):
        t0 = time.perf_counter()
        metrics = trainer.train_step()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        lv  = metrics.get("loss/value",           float("nan"))
        adv = metrics.get("advantage_mean",        float("nan"))
        ads = metrics.get("advantage_std",         float("nan"))
        gn  = (metrics.get("quantum/grad_norm_theta", 0.0) +
               metrics.get("quantum/grad_norm_w",     0.0))

        losses.append(lv);        v_stds.append(ads)
        adv_means.append(adv);    adv_stds.append(ads)
        grad_norms.append(gn);    step_ms.append(elapsed_ms)

        if step % LOG_INTERVAL == 0:
            elapsed_total = time.perf_counter() - t_start
            eta_s = elapsed_total / step * (n_steps - step)
            wandb.log({
                "loss/value": lv, "advantage_mean": adv,
                "grad_norm": gn, "ms_per_step": elapsed_ms,
            }, step=step)
            print(f"    [trainer] step {step:>6}/{n_steps}  "
                  f"loss={lv:.4f}  adv={adv:.3f}  ms={elapsed_ms:.1f}  "
                  f"ETA={eta_s:.0f}s", end="\r", flush=True)

    print(flush=True)
    total_s = time.perf_counter() - t_start
    print(f"    [trainer] done  total={total_s:.1f}s  "
          f"final_loss={losses[-1]:.4f}", flush=True)
    return {
        "loss": losses, "v_std": v_stds, "adv_mean": adv_means,
        "adv_std": adv_stds, "grad_norm": grad_norms, "ms_per_step": step_ms,
    }


def _run_flex_seed(
    buffer:       ReplayBuffer,
    entanglement: str = BASE_ENTANGLEMENT,
    measurement:  str = BASE_MEASUREMENT,
    n_qubits:     int = BASE_N_QUBITS,
    n_layers:     int = BASE_N_LAYERS,
    obs_dim:      int = None,    # inferred from buffer if None
    seed:         int = 0,
    n_steps:      int = NUM_STEPS,
) -> dict:
    """One seed via FlexQuantumValueNetwork (topology / measurement ablations)."""
    _obs_dim = obs_dim if obs_dim is not None else buffer.obs_dim
    print(f"    [flex] seed={seed}  entanglement={entanglement}  "
          f"measurement={measurement}  n_qubits={n_qubits}  "
          f"n_layers={n_layers}  obs_dim={_obs_dim}", flush=True)
    set_seed(seed)
    qvn  = FlexQuantumValueNetwork(
        n_qubits=n_qubits, n_layers=n_layers,
        obs_dim=_obs_dim,
        entanglement=entanglement, measurement=measurement,
    ).to(DEVICE)
    wrapped       = _FlexWrap(qvn)
    critic        = _make_critic(buffer.obs_dim, buffer.act_dim, seed)
    critic_target = _make_critic(buffer.obs_dim, buffer.act_dim, seed)
    with torch.no_grad():
        for p, pt in zip(critic.parameters(), critic_target.parameters()):
            pt.data.copy_(p.data);  pt.requires_grad_(False)

    opt_v = optim.Adam(qvn.parameters(),    lr=BASE_LR_V)
    opt_q = optim.Adam(critic.parameters(), lr=BASE_LR_Q)
    losses, v_stds, adv_means, adv_stds, grad_norms, step_ms = [], [], [], [], [], []

    t_start = time.perf_counter()
    for step in range(1, n_steps + 1):
        t0 = time.perf_counter()
        b  = to_device(buffer.sample(BASE_BATCH))

        opt_q.zero_grad()
        _critic_loss(critic, wrapped, b, gamma=BASE_GAMMA).backward()
        opt_q.step()
        with torch.no_grad():
            for p, pt in zip(critic.parameters(), critic_target.parameters()):
                pt.data.mul_(1 - BASE_POLYAK).add_(p.data, alpha=BASE_POLYAK)

        opt_v.zero_grad()
        lv = value_loss(wrapped, critic_target, b, BASE_TAU)
        lv.backward()
        gn = sum(
            p.grad.norm(2).item() ** 2
            for p in qvn.parameters() if p.grad is not None
        ) ** 0.5
        opt_v.step()

        with torch.no_grad():
            v     = qvn(b.observations)
            q_min = critic_target.q_min(b.observations, b.actions).squeeze()
            adv   = q_min - v

        ms = (time.perf_counter() - t0) * 1000
        losses.append(lv.item());    v_stds.append(v.std().item())
        adv_means.append(adv.mean().item()); adv_stds.append(adv.std().item())
        grad_norms.append(gn);       step_ms.append(ms)

        if step % LOG_INTERVAL == 0:
            elapsed_total = time.perf_counter() - t_start
            eta_s = elapsed_total / step * (n_steps - step)
            wandb.log({
                "loss/value": lv.item(), "advantage_mean": adv.mean().item(),
                "grad_norm": gn, "ms_per_step": ms,
            }, step=step)
            print(f"    [flex] step {step:>6}/{n_steps}  "
                  f"loss={lv.item():.4f}  adv={adv.mean().item():.3f}  "
                  f"ms={ms:.1f}  ETA={eta_s:.0f}s", end="\r", flush=True)

    print(flush=True)
    total_s = time.perf_counter() - t_start
    print(f"    [flex] done  total={total_s:.1f}s  "
          f"final_loss={losses[-1]:.4f}", flush=True)
    return {
        "loss": losses, "v_std": v_stds, "adv_mean": adv_means,
        "adv_std": adv_stds, "grad_norm": grad_norms, "ms_per_step": step_ms,
    }


def _run_classical_seed(
    buffer:  ReplayBuffer,
    seed:    int = 0,
    n_steps: int = NUM_STEPS,
) -> dict:
    """Classical MLP V-network baseline (used in datasize ablation)."""
    print(f"    [classical] seed={seed}  n_steps={n_steps}", flush=True)
    set_seed(seed)
    net           = ValueNetwork(buffer.obs_dim, hidden_dims=(256, 256)).to(DEVICE)
    critic        = _make_critic(buffer.obs_dim, buffer.act_dim, seed)
    critic_target = _make_critic(buffer.obs_dim, buffer.act_dim, seed)
    with torch.no_grad():
        for p, pt in zip(critic.parameters(), critic_target.parameters()):
            pt.data.copy_(p.data);  pt.requires_grad_(False)

    class _Wrap(nn.Module):
        def forward(self_, obs): return net(obs)

    wrapped = _Wrap()
    opt_v = optim.Adam(net.parameters(), lr=3e-3)
    opt_q = optim.Adam(critic.parameters(), lr=BASE_LR_Q)
    losses, adv_means, step_ms = [], [], []

    t_start = time.perf_counter()
    for step in range(1, n_steps + 1):
        t0 = time.perf_counter()
        b  = to_device(buffer.sample(min(BASE_BATCH, len(buffer))))

        opt_q.zero_grad()
        _critic_loss(critic, wrapped, b, gamma=BASE_GAMMA).backward()
        opt_q.step()
        with torch.no_grad():
            for p, pt in zip(critic.parameters(), critic_target.parameters()):
                pt.data.mul_(1 - BASE_POLYAK).add_(p.data, alpha=BASE_POLYAK)

        opt_v.zero_grad()
        lv = value_loss(wrapped, critic_target, b, BASE_TAU)
        lv.backward();  opt_v.step()

        with torch.no_grad():
            v     = net(b.observations).squeeze()
            q_min = critic_target.q_min(b.observations, b.actions).squeeze()
            adv   = q_min - v

        ms = (time.perf_counter() - t0) * 1000
        losses.append(lv.item());  adv_means.append(adv.mean().item())
        step_ms.append(ms)

        if step % LOG_INTERVAL == 0:
            elapsed_total = time.perf_counter() - t_start
            eta_s = elapsed_total / step * (n_steps - step)
            wandb.log({"loss/value": lv.item(), "advantage_mean": adv.mean().item()},
                      step=step)
            print(f"    [classical] step {step:>6}/{n_steps}  "
                  f"loss={lv.item():.4f}  adv={adv.mean().item():.3f}  "
                  f"ETA={eta_s:.0f}s", end="\r", flush=True)

    print(flush=True)
    total_s = time.perf_counter() - t_start
    print(f"    [classical] done  total={total_s:.1f}s  "
          f"final_loss={losses[-1]:.4f}", flush=True)
    return {"loss": losses, "adv_mean": adv_means, "ms_per_step": step_ms}


# ── Per-ablation runners ──────────────────────────────────────────────────────

def _wandb_init(offline: bool = False, **kwargs) -> None:
    """wandb.init wrapper: handles new reinit API and falls back to offline on auth errors."""
    mode = "offline" if offline else "online"
    print(f"    [wandb] init  mode={mode}  name={kwargs.get('name','?')}", flush=True)
    try:
        wandb.init(mode=mode, reinit="finish_previous", **kwargs)
    except TypeError:
        wandb.init(mode=mode, reinit=True, **kwargs)
    except wandb.errors.CommError as e:
        print(f"  [wandb] online init failed ({e}), switching to offline.", flush=True)
        wandb.init(mode="offline", reinit="finish_previous", **kwargs)
    print(f"    [wandb] init OK", flush=True)


def run_condition(
    condition_key:  str,
    seed:           int,
    buffer:         ReplayBuffer,
    env_cfg:        dict,
    use_flex:       bool = False,
    flex_kwargs:    dict = None,
    trainer_kwargs: dict = None,
    n_steps:        int  = NUM_STEPS,
    wandb_tags:     List[str] = None,
    wandb_offline:  bool = False,
) -> dict:
    """Run one condition × seed, with W&B init/finish bracketing the run."""
    run_name = f"{condition_key}__seed{seed}"
    tags     = (wandb_tags or []) + [f"seed={seed}"]

    print(f"\n  [run_condition] {run_name}", flush=True)

    _wandb_init(
        project = WANDB_PROJECT,
        name    = run_name,
        group   = condition_key,
        tags    = tags,
        config  = {
            "condition": condition_key,
            "seed": seed,
            "n_steps": n_steps,
            "use_flex": use_flex,
            **(flex_kwargs or {}),
            **(trainer_kwargs or {}),
        },
        offline = wandb_offline,
    )

    try:
        if use_flex:
            result = _run_flex_seed(buffer, seed=seed, n_steps=n_steps,
                                    **(flex_kwargs or {}))
        else:
            cfg = _build_trainer_config(
                seed=seed, n_steps=n_steps,
                dataset_id=env_cfg["dataset_id"],
                env_id=env_cfg["env_id"],
                tau=env_cfg["tau"],
                **(trainer_kwargs or {}),
            )
            cfg.wandb_run_name = run_name
            result = _run_trainer_seed(buffer, cfg, n_steps=n_steps, seed=seed)

        final_loss = result["loss"][-1]
        mean_adv   = float(np.mean(result["adv_mean"][-50:]))
        mean_ms    = float(np.mean(result["ms_per_step"]))
        wandb.summary.update({
            "final_loss": final_loss,
            "mean_adv_last50": mean_adv,
            "mean_ms_per_step": mean_ms,
        })
        print(f"  [run_condition] {run_name} DONE  "
              f"final_loss={final_loss:.4f}  mean_adv={mean_adv:.3f}", flush=True)
    finally:
        wandb.finish()
        print(f"  [wandb] run finished", flush=True)

    return result


# ── Ablation entry points ─────────────────────────────────────────────────────

def ablation_topology(seeds: List[int], env_cfg: dict, n_steps: int,
                      wandb_offline: bool) -> None:
    print("\n" + "=" * 65)
    print("  ABLATION: Entanglement Topology")
    print("=" * 65)
    for ds_name, ds_id in [
        (env_cfg["group"], env_cfg["dataset_id"]),
        ("walker2d-medium", _ENV_REGISTRY["walker2d"]["dataset_id"]),
    ]:
        print(f"\n  [topology] loading dataset: {ds_id}", flush=True)
        buffer = load_minari_dataset(ds_id, device="cpu")
        print(f"  [topology] dataset loaded  size={len(buffer)}", flush=True)
        ds_results = {}
        for topo in TOPOLOGY_CONDITIONS:
            cond_key = f"topology={topo}"
            print(f"\n  === {cond_key} ===", flush=True)
            seed_results = []
            for seed in seeds:
                print(f"  seed={seed}", end="  ", flush=True)
                r = run_condition(
                    f"{cond_key}__{ds_name}", seed, buffer, env_cfg,
                    use_flex=True,
                    flex_kwargs=dict(
                        entanglement=topo,
                        measurement=BASE_MEASUREMENT,
                        n_qubits=BASE_N_QUBITS,
                        n_layers=BASE_N_LAYERS,
                        obs_dim=buffer.obs_dim,
                    ),
                    n_steps=n_steps,
                    wandb_tags=["ablation:topology", ds_name],
                    wandb_offline=wandb_offline,
                )
                seed_results.append(r)
                print(f"loss={r['loss'][-1]:.4f}  "
                      f"adv={np.mean(r['adv_mean'][-50:]):.3f}", flush=True)
            ds_results[cond_key] = seed_results
        _print_summary(f"TOPOLOGY ({ds_name})", ds_results)


def ablation_measurement(seeds: List[int], env_cfg: dict, n_steps: int,
                         wandb_offline: bool) -> None:
    print("\n" + "=" * 65)
    print("  ABLATION: Measurement Scheme")
    print("=" * 65)
    print(f"  [measurement] loading dataset: {env_cfg['dataset_id']}", flush=True)
    buffer = load_minari_dataset(env_cfg["dataset_id"], device="cpu")
    print(f"  [measurement] dataset loaded  size={len(buffer)}", flush=True)
    all_results = {}

    for scheme in MEASUREMENT_CONDITIONS:
        cond_key = f"meas={scheme}"
        print(f"\n  === {cond_key} ===", flush=True)
        seed_results = []
        for seed in seeds:
            print(f"  seed={seed}", end="  ", flush=True)
            r = run_condition(
                cond_key, seed, buffer, env_cfg,
                use_flex=True,
                flex_kwargs=dict(
                    measurement=scheme,
                    entanglement=BASE_ENTANGLEMENT,
                    n_qubits=BASE_N_QUBITS,
                    n_layers=BASE_N_LAYERS,
                    obs_dim=buffer.obs_dim,
                ),
                n_steps=n_steps,
                wandb_tags=["ablation:measurement"],
                wandb_offline=wandb_offline,
            )
            seed_results.append(r)
            print(f"loss={r['loss'][-1]:.4f}  "
                  f"adv={np.mean(r['adv_mean'][-50:]):.3f}", flush=True)
        all_results[cond_key] = seed_results

    _print_summary("MEASUREMENT", all_results)


def ablation_datasize(seeds: List[int], env_cfg: dict, n_steps: int,
                      wandb_offline: bool) -> None:
    print("\n" + "=" * 65)
    print("  ABLATION: Dataset Size Sensitivity")
    print("=" * 65)
    print(f"  [datasize] loading full dataset: {env_cfg['dataset_id']}", flush=True)
    buf_full = load_minari_dataset(env_cfg["dataset_id"], device="cpu")
    print(f"  [datasize] full dataset size={len(buf_full)}", flush=True)

    for frac in DATA_FRACTIONS:
        frac_key = f"data={int(frac*100)}%"
        sub_buf  = _subsample_buffer(buf_full, frac)
        print(f"\n  [{frac_key}]  {len(sub_buf):,} transitions", flush=True)

        # Quantum
        for seed in seeds:
            print(f"  [datasize] quantum seed={seed}", end="  ", flush=True)
            _wandb_init(
                project=WANDB_PROJECT,
                name=f"quantum__{frac_key}__seed{seed}",
                group=f"datasize_quantum_{frac_key}",
                tags=["ablation:datasize", "quantum", frac_key, f"seed={seed}"],
                config={"condition": frac_key, "network": "quantum",
                        "fraction": frac, "seed": seed},
                offline=wandb_offline,
            )
            try:
                cfg = _build_trainer_config(
                    seed=seed, n_steps=n_steps,
                    dataset_id=env_cfg["dataset_id"],
                    env_id=env_cfg["env_id"],
                    tau=env_cfg["tau"],
                )
                cfg.wandb_run_name = f"quantum__{frac_key}__seed{seed}"
                r = _run_trainer_seed(sub_buf, cfg, n_steps=n_steps, seed=seed)
                print(f"loss={r['loss'][-1]:.4f}  "
                      f"adv={np.mean(r['adv_mean'][-50:]):.3f}", flush=True)
            finally:
                wandb.finish()

        # Classical baseline
        for seed in seeds:
            print(f"  [datasize] classical seed={seed}", end="  ", flush=True)
            _wandb_init(
                project=WANDB_PROJECT,
                name=f"classical__{frac_key}__seed{seed}",
                group=f"datasize_classical_{frac_key}",
                tags=["ablation:datasize", "classical", frac_key, f"seed={seed}"],
                config={"condition": frac_key, "network": "classical",
                        "fraction": frac, "seed": seed},
                offline=wandb_offline,
            )
            try:
                r = _run_classical_seed(sub_buf, seed=seed, n_steps=n_steps)
                print(f"loss={r['loss'][-1]:.4f}  "
                      f"adv={np.mean(r['adv_mean'][-50:]):.3f}", flush=True)
            finally:
                wandb.finish()


def _subsample_buffer(buf: ReplayBuffer, fraction: float,
                      seed: int = 42) -> ReplayBuffer:
    import random
    rng     = random.Random(seed)
    n_total = len(buf)
    n_keep  = max(1, int(n_total * fraction))
    print(f"  [subsample] fraction={fraction}  keeping {n_keep}/{n_total}", flush=True)
    idx     = np.array(sorted(rng.sample(range(n_total), n_keep)), dtype=np.intp)

    new = ReplayBuffer(
        obs_dim  = buf.obs_dim,
        act_dim  = buf.act_dim,
        capacity = n_keep,
        device   = buf.device,
    )
    new._observations[:]      = buf._observations[idx]
    new._actions[:]           = buf._actions[idx]
    new._rewards[:]           = buf._rewards[idx]
    new._next_observations[:] = buf._next_observations[idx]
    new._dones[:]             = buf._dones[idx]
    new._ptr  = 0
    new._size = n_keep
    return new


# ── Analysis: Fourier spectrum ────────────────────────────────────────────────

def analysis_fourier(env_cfg: dict, n_steps: int) -> None:
    """Fourier spectrum of V(s) throughout training — no W&B, seed 0."""
    print("\n" + "=" * 65)
    print("  ANALYSIS: Fourier Spectrum Characterisation")
    print("=" * 65)
    N_POINTS   = 256
    FEATURE    = 0
    SNAPSHOTS  = [0, n_steps // 4, n_steps // 2, n_steps]
    print(f"  [fourier] loading dataset: {env_cfg['dataset_id']}", flush=True)
    buffer     = load_minari_dataset(env_cfg["dataset_id"], device="cpu")
    print(f"  [fourier] dataset loaded  size={len(buffer)}", flush=True)

    for label, (n_qubits, n_layers) in [
        ("base(4q-2L)", (4, 2)),
        ("deep(4q-3L-needs8q)", (8, 3)),
    ]:
        print(f"\n  [fourier] config: {label}", flush=True)
        qvn     = FlexQuantumValueNetwork(
            n_qubits=n_qubits, n_layers=n_layers,
            obs_dim=buffer.obs_dim).to(DEVICE)
        wrapped = _FlexWrap(qvn)
        critic        = _make_critic(buffer.obs_dim, buffer.act_dim, 0)
        critic_target = _make_critic(buffer.obs_dim, buffer.act_dim, 0)
        with torch.no_grad():
            for p, pt in zip(critic.parameters(), critic_target.parameters()):
                pt.data.copy_(p.data);  pt.requires_grad_(False)
        opt_v = optim.Adam(qvn.parameters(), lr=BASE_LR_V)
        opt_q = optim.Adam(critic.parameters(), lr=BASE_LR_Q)
        snapshots = {}

        if 0 in SNAPSHOTS:
            snapshots[0] = _fourier_snapshot(qvn, buffer.obs_dim, N_POINTS, FEATURE)
            print(f"  [fourier] snapshot at step=0 OK", flush=True)

        for step in range(1, n_steps + 1):
            b = to_device(buffer.sample(BASE_BATCH))
            opt_q.zero_grad()
            _critic_loss(critic, wrapped, b, gamma=BASE_GAMMA).backward()
            opt_q.step()
            with torch.no_grad():
                for p, pt in zip(critic.parameters(), critic_target.parameters()):
                    pt.data.mul_(1 - BASE_POLYAK).add_(p.data, alpha=BASE_POLYAK)

            opt_v.zero_grad()
            value_loss(wrapped, critic_target, b, BASE_TAU).backward()
            opt_v.step()

            if step in SNAPSHOTS:
                snapshots[step] = _fourier_snapshot(qvn, buffer.obs_dim, N_POINTS, FEATURE)
                freqs, mags = snapshots[step]
                dom = freqs[np.argmax(mags)]
                print(f"  [fourier] step {step:>6}  dominant_freq={dom:.1f}  "
                      f"max_mag={mags.max():.4f}", flush=True)

            if step % LOG_INTERVAL == 0:
                print(f"  [fourier] training step {step}/{n_steps}", end="\r", flush=True)

        print(flush=True)
        out_path = RESULTS_DIR / f"fourier_{label.replace('(','').replace(')','').replace('-','_')}.npz"
        np.savez(out_path, **{
            f"step_{s}": np.stack(v) for s, v in snapshots.items()
        })
        print(f"  [fourier] saved → {out_path}", flush=True)


def _fourier_snapshot(
    qvn: FlexQuantumValueNetwork,
    obs_dim: int,
    n_points: int,
    feature: int,
) -> Tuple[np.ndarray, np.ndarray]:
    xs = torch.zeros(n_points, obs_dim, device=DEVICE)
    xs[:, feature] = torch.linspace(-math.pi, math.pi, n_points, device=DEVICE)
    with torch.no_grad():
        ys = qvn(xs).cpu().numpy()
    freqs = np.fft.rfftfreq(n_points, d=1.0 / n_points)
    mags  = np.abs(np.fft.rfft(ys))
    return freqs, mags


# ── Analysis: Expressibility ──────────────────────────────────────────────────

def analysis_expr(env_cfg: dict) -> None:
    """KL(circuit || Haar) expressibility metric — no W&B."""
    print("\n" + "=" * 65)
    print("  ANALYSIS: Expressibility (KL divergence from Haar)")
    print("=" * 65)
    N_SAMPLES = 200
    N_BINS    = 75

    configs = {
        "2q-1L-linear":   dict(n_qubits=2, n_layers=1, entanglement="linear"),
        "4q-1L-linear":   dict(n_qubits=4, n_layers=1, entanglement="linear"),
        "4q-2L-linear":   dict(n_qubits=4, n_layers=2, entanglement="linear"),   # base
        "4q-2L-circular": dict(n_qubits=4, n_layers=2, entanglement="circular"),
        "4q-2L-all2all":  dict(n_qubits=4, n_layers=2, entanglement="all_to_all"),
        "8q-2L-linear":   dict(n_qubits=8, n_layers=2, entanglement="linear"),
        "8q-3L-linear":   dict(n_qubits=8, n_layers=3, entanglement="linear"),
    }

    results = {}
    for cfg_label, kwargs in configs.items():
        print(f"  [expr] computing KL for {cfg_label}...", flush=True)
        kl = _expressibility_kl(seed=0, n_samples=N_SAMPLES, n_bins=N_BINS, **kwargs)
        results[cfg_label] = kl
        marker = " ← base" if cfg_label == "4q-2L-linear" else ""
        print(f"  {cfg_label:<22}  KL = {kl:.4f}{marker}", flush=True)

    out_path = RESULTS_DIR / "expressibility.json"
    import json
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  [expr] saved → {out_path}", flush=True)


def _expressibility_kl(
    n_qubits:     int,
    n_layers:     int,
    entanglement: str,
    n_samples:    int,
    n_bins:       int,
    seed:         int = 0,
) -> float:
    np.random.seed(seed)
    dev = qml.device(QUANTUM_DEVICE, wires=n_qubits)
    dim = 2 ** n_qubits

    @qml.qnode(dev, interface="numpy")
    def _state(theta, w, xs):
        _apply_entanglement(n_qubits, entanglement)
        for layer in range(n_layers):
            for q in range(n_qubits):
                angles = theta[layer, q] + w[layer, q] * xs[q]
                qml.Rot(angles[0], angles[1], angles[2], wires=q)
            _apply_entanglement(n_qubits, entanglement)
        return qml.state()

    fidelities = []
    for i in range(n_samples):
        if i % 50 == 0:
            print(f"    [expr] sample {i}/{n_samples}", end="\r", flush=True)
        t1 = np.random.uniform(0, 2 * np.pi, (n_layers, n_qubits, 3))
        w1 = np.random.uniform(0, 2 * np.pi, (n_layers, n_qubits, 3))
        t2 = np.random.uniform(0, 2 * np.pi, (n_layers, n_qubits, 3))
        w2 = np.random.uniform(0, 2 * np.pi, (n_layers, n_qubits, 3))
        xs = np.random.uniform(-np.pi, np.pi, n_qubits)
        s1 = _state(t1, w1, xs)
        s2 = _state(t2, w2, xs)
        fidelities.append(float(abs(np.dot(s1.conj(), s2)) ** 2))
    print(flush=True)

    bins      = np.linspace(0, 1, n_bins + 1)
    f_mid     = (bins[:-1] + bins[1:]) / 2
    hist_c, _ = np.histogram(fidelities, bins=bins, density=True)
    p_haar    = (dim - 1) * (1 - f_mid) ** (dim - 2)
    p_haar   /= p_haar.sum()
    p_circ    = hist_c / (hist_c.sum() + 1e-12)
    return float(np.sum(
        np.where(p_circ > 0, p_circ * np.log(p_circ / (p_haar + 1e-12)), 0)
    ))


# ── Summary table ─────────────────────────────────────────────────────────────

def _print_summary(title: str, results: Dict[str, List[dict]]) -> None:
    print(f"\n{'─'*95}")
    print(f"  {title}")
    print(f"{'─'*95}")
    print(f"  {'Condition':<30} {'Loss (mean±CI)':>20} {'mean Ā':>10} "
          f"{'‖∇θ‖':>9} {'ms/step':>9}")
    print(f"{'─'*95}")
    for cond_label, seed_results in results.items():
        def tm(key, n=50):
            return np.mean([np.mean(s[key][-n:]) for s in seed_results
                            if key in s])
        final = np.array([s["loss"][-1] for s in seed_results])
        ci_v  = final.std() / max(np.sqrt(len(final)), 1) * 1.96
        gn    = tm("grad_norm") if "grad_norm" in seed_results[0] else float("nan")
        print(
            f"  {cond_label:<30} {final.mean():>9.4f}±{ci_v:<8.4f}"
            f"  {tm('adv_mean'):>9.3f}  {gn:>9.4f}  {tm('ms_per_step'):>9.1f}"
        )
    print(f"{'─'*95}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quantum-IQL Circuit Ablation — Topology / Measurement / Datasize",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--ablation", nargs="+",
        default=['topology', 'measurement', 'datasize'],
        choices=['topology', 'measurement', 'datasize', 'fourier', 'expr'],
        help="Which ablations to run.",
    )
    p.add_argument(
        "--env", default="hopper", choices=list(_ENV_REGISTRY),
        help="Primary environment (default: hopper).",
    )
    p.add_argument("--seeds",         nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--steps",         type=int,  default=NUM_STEPS,
                   help=f"Training steps per condition (default: {NUM_STEPS:,}).")
    p.add_argument("--wandb-offline", action="store_true")
    p.add_argument("--dry-run",       action="store_true")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    env_cfg = _ENV_REGISTRY[args.env]

    hw = (f"GPU · {torch.cuda.get_device_name(0)}"
          if torch.cuda.is_available() else "CPU only")
    print(f"\n{'='*65}")
    print(f"  Quantum-IQL Circuit Ablation — Topology / Measurement / Datasize")
    print(f"  Hardware : {hw}")
    print(f"  Env      : {args.env}  ({env_cfg['dataset_id']})")
    print(f"  Ablations: {', '.join(args.ablation)}")
    print(f"  Seeds    : {args.seeds}")
    print(f"  Steps    : {args.steps:,}")
    print(f"  Diff     : {QUANTUM_DIFF_METHOD}  |  Device: {QUANTUM_DEVICE}")
    print(f"{'='*65}\n")

    if args.dry_run:
        print("Dry-run: exiting without training.")
        return

    failed: List[tuple] = []

    # FIX: added 'fourier' and 'expr' which were missing from the dispatcher
    dispatcher = {
        "topology":    lambda: ablation_topology(args.seeds, env_cfg, args.steps, args.wandb_offline),
        "measurement": lambda: ablation_measurement(args.seeds, env_cfg, args.steps, args.wandb_offline),
        "datasize":    lambda: ablation_datasize(args.seeds, env_cfg, args.steps, args.wandb_offline),
        "fourier":     lambda: analysis_fourier(env_cfg, args.steps),   # FIX: was missing
        "expr":        lambda: analysis_expr(env_cfg),                   # FIX: was missing
    }

    for ablation_name in args.ablation:
        print(f"\n[RUN] {ablation_name}", flush=True)
        t0 = time.perf_counter()
        try:
            dispatcher[ablation_name]()
            elapsed = time.perf_counter() - t0
            print(f"[RUN] {ablation_name} finished in {elapsed:.1f}s", flush=True)
        except Exception as exc:
            import traceback
            print(f"  ERROR in {ablation_name}: {exc}", flush=True)
            traceback.print_exc()
            failed.append((ablation_name, str(exc)))
            try:
                wandb.finish(exit_code=1)
            except Exception:
                pass

    print(f"\n{'='*65}")
    print(f"  Done: {len(args.ablation)-len(failed)}/{len(args.ablation)} succeeded")
    if failed:
        for name, msg in failed:
            print(f"    FAILED: {name}: {msg}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()

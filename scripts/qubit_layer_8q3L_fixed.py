#!/usr/bin/env python3
"""
Circuit Design Ablation — Qubit × Layer Joint Grid
===================================================
Runs a 2D grid of (n_qubits, n_layers) pairs satisfying
L ≤ floor(log2(n_qubits)):

  (2,1)  (4,1)  (4,2)★  (8,1)  (8,2)  (8,3)

Results are summarised as a table showing final loss, mean advantage,
gradient norm and wall-clock cost per cell, so the optimal
cost/quality operating point is easy to identify.

Related scripts:
  layers sweep alone                 → circuit_ablation_layers.py
  topology / measurement / datasize  → circuit_ablation_main.py

Usage
-----
  python experiments/circuit_ablation_qubit_layer.py
  python experiments/circuit_ablation_qubit_layer.py --seeds 0 1 2 3 4 --wandb-offline
  python experiments/circuit_ablation_qubit_layer.py --steps 50000 --dry-run
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pennylane as qml
import wandb
from scipy import stats as scipy_stats

from quantum_iql.buffer import Batch, ReplayBuffer, load_minari_dataset
from quantum_iql.losses import critic_loss as _critic_loss, value_loss
from quantum_iql.networks import CriticNetwork, ValueNetwork
from quantum_iql.config import IQLConfig, NetworkConfig
from quantum_iql.trainer import IQLTrainer
from quantum_iql.utils import set_seed

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
WANDB_PROJECT   = "quantum-iql-circuit-ablation-8q3L"

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

RESULTS_DIR = Path("results/circuit_ablations_8q3L")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Ablation axis definitions ─────────────────────────────────────────────────
# Each entry: condition_key -> dict of kwargs forwarded to _build_trainer_config
# or to FlexQuantumValueNetwork.

LAYERS_CONDITIONS = {
    "n_layers=1(q=4)": dict(n_qubits=4, n_layers=1),
    "n_layers=2(q=4)": dict(n_qubits=4, n_layers=2),   # base
    "n_layers=3(q=8)": dict(n_qubits=8, n_layers=3),
}

TOPOLOGY_CONDITIONS = ["none", "linear", "circular", "all_to_all"]

QUBIT_CONDITIONS = {
    "n_qubits=2": dict(n_qubits=2, n_layers=1),   # 1 layer max for 2 qubits
    "n_qubits=4": dict(n_qubits=4, n_layers=2),   # base
    "n_qubits=6": dict(n_qubits=6, n_layers=2),
    "n_qubits=8": dict(n_qubits=8, n_layers=2),
}

# (n_qubits, n_layers) pairs satisfying L <= floor(log2(n_qubits))
QUBIT_LAYER_GRID = [
    (8, 3),   # only the slow one
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


# ── Trainer config builder (for QuantumIQLTrainer-based ablations) ────────────

def _build_trainer_config(
    n_qubits: int = BASE_N_QUBITS,
    n_layers: int = BASE_N_LAYERS,
    n_steps:  int = NUM_STEPS,
    seed:     int = 0,
    dataset_id: str = _ENV_REGISTRY["hopper"]["dataset_id"],
    env_id:     str = _ENV_REGISTRY["hopper"]["env_id"],
    tau:        float = BASE_TAU,
) -> IQLConfig:
    return IQLConfig(
        dataset_id   = dataset_id,
        env_id       = env_id,
        tau          = tau,
        gamma        = BASE_GAMMA,
        polyak       = BASE_POLYAK,
        lr_v         = 3e-4,
        lr_q         = 3e-4,
        lr_actor     = 3e-4,
        batch_size   = BASE_BATCH,
        num_steps    = n_steps,
        log_interval = LOG_INTERVAL,
        eval_interval= n_steps + 1,
        wandb_project= WANDB_PROJECT,
        wandb_offline= False,
        seed         = seed,
        device       = "auto",
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
            out = torch.stack([
                self._circuit(self.theta, self.w, xs[i]).to(torch.float32)
                for i in range(xs.shape[0])
            ])
        elif self.measurement == "mean_pauli_z":
            out = torch.stack([
                torch.stack([r.to(torch.float32)
                             for r in self._circuit(self.theta, self.w, xs[i])]).mean()
                for i in range(xs.shape[0])
            ])
        else:   # learned
            w_norm = torch.softmax(self.meas_weights, dim=0)
            out = torch.stack([
                (torch.stack([r.to(torch.float32)
                              for r in self._circuit(self.theta, self.w, xs[i])]) * w_norm).sum()
                for i in range(xs.shape[0])
            ])

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
    buffer: ReplayBuffer,
    cfg:    IQLConfig,
    n_steps: int,
    seed:   int,
) -> dict:
    """One seed via IQLTrainer."""
    set_seed(seed)
    trainer = IQLTrainer(cfg, buffer, env=None)
    losses, v_stds, adv_means, adv_stds, grad_norms, step_ms = [], [], [], [], [], []

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
            wandb.log({
                "loss/value": lv, "advantage_mean": adv,
                "grad_norm": gn, "ms_per_step": elapsed_ms,
            }, step=step)
            print(f"    step {step:>6}/{n_steps}  loss={lv:.4f}  adv={adv:.3f}  ms={elapsed_ms:.1f}", end="\r", flush=True)

    print()
    return {
        "loss": losses, "v_std": v_stds, "adv_mean": adv_means,
        "adv_std": adv_stds, "grad_norm": grad_norms, "ms_per_step": step_ms,
    }


def _run_flex_seed(
    buffer:      ReplayBuffer,
    entanglement: str = BASE_ENTANGLEMENT,
    measurement:  str = BASE_MEASUREMENT,
    n_qubits:     int = BASE_N_QUBITS,
    n_layers:     int = BASE_N_LAYERS,
    seed:         int = 0,
    n_steps:      int = NUM_STEPS,
) -> dict:
    """One seed via FlexQuantumValueNetwork (topology / measurement ablations)."""
    set_seed(seed)
    qvn  = FlexQuantumValueNetwork(
        n_qubits=n_qubits, n_layers=n_layers,
        obs_dim=buffer.obs_dim,
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
            wandb.log({
                "loss/value": lv.item(), "advantage_mean": adv.mean().item(),
                "grad_norm": gn, "ms_per_step": ms,
            }, step=step)
            print(f"    step {step}/{n_steps}  "
                  f"loss={lv.item():.4f}  adv={adv.mean().item():.3f}  ms={ms:.1f}", end="\r")

    print()
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
    set_seed(seed)
    net     = ValueNetwork(buffer.obs_dim, hidden_dims=(256, 256)).to(DEVICE)
    critic        = _make_critic(buffer.obs_dim, buffer.act_dim, seed)
    critic_target = _make_critic(buffer.obs_dim, buffer.act_dim, seed)
    with torch.no_grad():
        for p, pt in zip(critic.parameters(), critic_target.parameters()):
            pt.data.copy_(p.data);  pt.requires_grad_(False)

    # Wrap net so value_loss gets (B,1)
    class _Wrap(nn.Module):
        def forward(self_, obs): return net(obs)   # ValueNetwork already returns (B,1)

    wrapped = _Wrap()
    opt_v = optim.Adam(net.parameters(), lr=3e-3)
    opt_q = optim.Adam(critic.parameters(), lr=BASE_LR_Q)
    losses, adv_means, step_ms = [], [], []

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
            wandb.log({"loss/value": lv.item(), "advantage_mean": adv.mean().item()},
                      step=step)

    return {"loss": losses, "adv_mean": adv_means, "ms_per_step": step_ms}


# ── Per-ablation runners ──────────────────────────────────────────────────────

def run_condition(
    condition_key: str,
    seed:          int,
    buffer:        ReplayBuffer,
    env_cfg:       dict,
    use_flex:      bool = False,
    flex_kwargs:   dict = None,
    trainer_kwargs: dict = None,
    n_steps:       int = NUM_STEPS,
    wandb_tags:    List[str] = None,
    wandb_offline: bool = False,
) -> dict:
    """Run one condition × seed, with W&B init/finish bracketing the run."""
    run_name = f"{condition_key}__seed{seed}"
    tags     = (wandb_tags or []) + [f"seed={seed}"]

    mode = "offline" if wandb_offline else "online"
    print(f"  [wandb] init  run={run_name}", flush=True)
    try:
        wandb.init(
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
            mode    = mode,
            reinit  = True,
        )
    except Exception as e:
        print(f"  [wandb] init failed ({e}), switching to offline.", flush=True)
        wandb.init(project=WANDB_PROJECT, name=run_name, mode="offline", reinit=True)

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
    finally:
        wandb.finish()

    return result



# ── Ablation entry points ─────────────────────────────────────────────────────

def ablation_qubit_layer(seeds: List[int], env_cfg: dict, n_steps: int,
                         wandb_offline: bool) -> None:
    print("\n" + "=" * 65)
    print("  ABLATION 3b: Qubit × Layer Joint Grid")
    print("=" * 65)
    buffer = load_minari_dataset(env_cfg["dataset_id"], device="cpu")
    all_results = {}

    for n_qubits, n_layers in QUBIT_LAYER_GRID:
        cond_key = f"q{n_qubits}_L{n_layers}"
        marker   = " ★ base" if (n_qubits, n_layers) == (BASE_N_QUBITS, BASE_N_LAYERS) else ""
        print(f"\n  === {cond_key}{marker}  params={_n_params(n_qubits, n_layers)} ===")
        seed_results = []
        for seed in seeds:
            print(f"  seed={seed}", end="  ")
            r = run_condition(
                cond_key, seed, buffer, env_cfg,
                use_flex=False,
                trainer_kwargs=dict(n_qubits=n_qubits, n_layers=n_layers),
                n_steps=n_steps,
                wandb_tags=["ablation:qubit_layer"],
                wandb_offline=wandb_offline,
            )
            seed_results.append(r)
            print(f"loss={r['loss'][-1]:.4f}  ms={np.mean(r['ms_per_step']):.1f}")
        all_results[cond_key] = seed_results

    _print_summary("QUBIT×LAYER", all_results)
    _plot_qubit_layer_heatmap(all_results, env_cfg["group"])


# ── Summary table ─────────────────────────────────────────────────────────────

# ── Figures ───────────────────────────────────────────────────────────────────

def _plot_qubit_layer_heatmap(all_results: Dict[str, List[dict]], ds_name: str) -> None:
    QUBITS_GRID = sorted(set(q for q, _ in QUBIT_LAYER_GRID))
    LAYERS_GRID = sorted(set(l for _, l in QUBIT_LAYER_GRID))
    base_qi = QUBITS_GRID.index(BASE_N_QUBITS)
    base_li = LAYERS_GRID.index(BASE_N_LAYERS)

    def grid_matrix(metric, n_tail=50):
        mat = np.full((len(QUBITS_GRID), len(LAYERS_GRID)), np.nan)
        for qi, nq in enumerate(QUBITS_GRID):
            for li, nl in enumerate(LAYERS_GRID):
                key = f"q{nq}_L{nl}"
                if key in all_results:
                    vals = [np.mean(s[metric][-n_tail:]) for s in all_results[key] if metric in s]
                    if vals:
                        mat[qi, li] = np.mean(vals)
        return mat

    loss_mat = grid_matrix("loss")
    adv_mat  = grid_matrix("adv_mean")
    gn_mat   = grid_matrix("grad_norm")
    ms_mat   = grid_matrix("ms_per_step")

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.suptitle(
        f"Joint Qubits × Layers ablation — {ds_name}  (last-50-step mean)",
        fontsize=12, fontweight="bold"
    )
    panels = [
        (loss_mat, "Expectile Loss\n(lower = better)",  "YlOrRd", False),
        (adv_mat,  "mean(A = Q-V)\n(higher = better)",  "RdYlGn", True),
        (gn_mat,   "grad norm\n(higher = healthier)",    "Blues",  True),
        (ms_mat,   "ms / step\n(lower = cheaper)",       "YlOrRd", False),
    ]
    for ax, (mat, title, cmap, _) in zip(axes, panels):
        masked = np.ma.masked_invalid(mat)
        im = ax.imshow(masked, cmap=cmap, aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.85)
        for qi in range(len(QUBITS_GRID)):
            for li in range(len(LAYERS_GRID)):
                val = mat[qi, li]
                if not np.isnan(val):
                    ax.text(li, qi, f"{val:.3f}", ha="center", va="center", fontsize=8,
                            color="white" if abs(val) > (np.nanmax(np.abs(mat)+1e-9) * 0.6) else "black")
        if not np.isnan(mat[base_qi, base_li]):
            ax.plot(base_li, base_qi, "w*", markersize=14, markeredgecolor="black", markeredgewidth=0.8)
        ax.set_xticks(range(len(LAYERS_GRID)))
        ax.set_xticklabels([f"L={l}" for l in LAYERS_GRID])
        ax.set_yticks(range(len(QUBITS_GRID)))
        ax.set_yticklabels([f"q={q}" for q in QUBITS_GRID])
        ax.set_xlabel("n_layers"); ax.set_ylabel("n_qubits")
        ax.set_title(title, fontsize=10)
    plt.tight_layout()
    path = RESULTS_DIR / f"ablation_qubit_layer_heatmap_{ds_name.replace('-','_')}.png"
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"  [fig] saved → {path}", flush=True)

    # Wall-clock scatter
    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4))
    fig2.suptitle("Qubit×Layer: performance vs cost", fontsize=12, fontweight="bold")
    keys   = list(all_results.keys())
    f_loss = [np.mean([s["loss"][-1] for s in all_results[k]]) for k in keys]
    f_ms   = [np.mean([np.mean(s["ms_per_step"]) for s in all_results[k]]) for k in keys]
    cmap_q = plt.cm.cool
    colors = [cmap_q(i / max(len(keys)-1, 1)) for i in range(len(keys))]
    for ax2, (vals, ylabel, title2) in zip(axes2, [
        (f_loss, "Final expectile loss",   "Convergence quality"),
        (f_ms,   "ms / step",              "Wall-clock cost"),
    ]):
        bars = ax2.bar(keys, vals, color=colors, edgecolor="white")
        ax2.set_xticklabels(keys, rotation=30, ha="right", fontsize=8)
        ax2.set_ylabel(ylabel); ax2.set_title(title2)
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f"{v:.3f}" if ylabel.startswith("Final") else f"{v:.1f}ms",
                     ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    path2 = RESULTS_DIR / f"ablation_qubit_layer_cost_{ds_name.replace('-','_')}.png"
    fig2.savefig(path2, bbox_inches="tight", dpi=130)
    plt.close(fig2)
    print(f"  [fig] saved → {path2}", flush=True)



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
        description="Quantum-IQL Circuit Ablation — Qubit × Layer Grid",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--env",           default="hopper", choices=list(_ENV_REGISTRY))
    p.add_argument("--seeds",         nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--steps",         type=int,  default=NUM_STEPS)
    p.add_argument("--wandb-offline", action="store_true")
    p.add_argument("--dry-run",       action="store_true")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    env_cfg = _ENV_REGISTRY[args.env]

    hw = (f"GPU · {torch.cuda.get_device_name(0)}"
          if torch.cuda.is_available() else "CPU only")
    grid_str = "  ".join(f"q{q}_L{l}" for q, l in QUBIT_LAYER_GRID)
    print(f"\n{'='*65}")
    print(f"  Quantum-IQL Circuit Ablation — Qubit × Layer Grid")
    print(f"  Hardware : {hw}")
    print(f"  Env      : {args.env}  ({env_cfg['dataset_id']})")
    print(f"  Grid     : {grid_str}")
    print(f"  Seeds    : {args.seeds}")
    print(f"  Steps    : {args.steps:,}")
    print(f"  Diff     : {QUANTUM_DIFF_METHOD}  |  Device: {QUANTUM_DEVICE}")
    n_runs = len(QUBIT_LAYER_GRID) * len(args.seeds)
    worst_ms = 126   # 8q-3L estimate
    est_h = n_runs * args.steps * worst_ms / 1000 / 3600
    print(f"  Est. wall-clock (worst case): {est_h:.0f}h  "
          f"({n_runs} runs × {args.steps:,} steps)")
    print(f"{'='*65}\n")

    if args.dry_run:
        print("Dry-run: exiting without training.")
        return

    try:
        ablation_qubit_layer(args.seeds, env_cfg, args.steps, args.wandb_offline)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        try:
            wandb.finish(exit_code=1)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()


# # Quantum-IQL: Circuit Design Ablation Study
# 
# **Scope:** Systematic ablation of the quantum `V(s)` circuit design choices.  
# This notebook continues from `training_dynamics_final.ipynb`, which established the optimal base config:  
# identity-block init · Adam lr=1e-2 · τ=0.7 · joint V+Q training.
# 
# **Five ablation axes:**
# 1. **Data-reuploading layers sweep** — n_layers ∈ {1, 2, 3, 4, 6}
# 2. **Entanglement topology** — linear vs circular vs all-to-all vs none
# 3. **Qubit count scaling** — n_qubits ∈ {2, 4, 6, 8}
# 4. **Measurement scheme** — local PauliZ(0) vs mean(PauliZ) vs ZZ tensor vs learned combo
# 5. **Dataset size sensitivity** — 10%, 25%, 50%, 100% of hopper-medium
# 
# **Two bonus analyses:**
# 6. **Fourier spectrum characterisation** — frequency components of V(s) throughout training
# 7. **Expressibility** — Haar-fidelity metric across circuit configurations
# 
# **Hardware:** Auto-detected (GPU: backprop · CPU: adjoint)  
# **Datasets:** D4RL via Minari — `hopper-medium` (primary) · `walker2d-medium` (validation)  
# **Seeds:** 5 per condition; mean ± 95% CI  
# **Base config (held fixed unless ablated):** n_qubits=4 · n_layers=2 · linear entanglement · PauliZ(0)


# ## 0. Setup
# 
# Shared infrastructure: imports, device detection, helpers, dataset loading, critic factory.


# ──────────────────────────────────────────────────────────────────────
# Cell 2
# ──────────────────────────────────────────────────────────────────────
import sys, math, time, json, warnings, itertools
from pathlib import Path
from copy import deepcopy
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats as scipy_stats
import wandb

# ── quantum-iql package imports (same as training_dynamics_final.ipynb) ──────
from quantum_iql.losses import critic_loss as _critic_loss, value_loss
from quantum_iql.networks import ValueNetwork, CriticNetwork
from quantum_iql.buffer import Batch, ReplayBuffer, load_minari_dataset
from quantum_iql.quantum_config import QuantumIQLConfig, QuantumNetConfig
from quantum_iql.quantum_trainer import QuantumIQLTrainer
from quantum_iql.utils import set_seed

warnings.filterwarnings("ignore")

# ── Project paths ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/work/quantum-implicit-qlearning")
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

RESULTS_DIR = Path("results/ablations")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Device detection ──────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE              = torch.device("cuda")
    QUANTUM_DEVICE      = "default.qubit"
    QUANTUM_DIFF_METHOD = "backprop"
    QUANTUM_BATCH_SIZE  = 256
    _hw = f"GPU · {torch.cuda.get_device_name(0)} · {torch.cuda.get_device_properties(0).total_memory // 2**20:,} MB VRAM"
else:
    DEVICE              = torch.device("cpu")
    QUANTUM_DEVICE      = "default.qubit"
    QUANTUM_DIFF_METHOD = "adjoint"
    QUANTUM_BATCH_SIZE  = 4
    _hw = "CPU only"

print(f"Hardware : {_hw}")
print(f"Quantum  : device={QUANTUM_DEVICE}  diff_method={QUANTUM_DIFF_METHOD}  batch={QUANTUM_BATCH_SIZE}")
print(f"PennyLane {qml.__version__}  |  PyTorch {torch.__version__}")

# ── Fixed base config ─────────────────────────────────────────────────────
BASE_N_QUBITS     = 4
BASE_N_LAYERS     = 2
BASE_ENTANGLEMENT = "linear"
BASE_MEASUREMENT  = "pauli_z0"
BASE_LR_V         = 1e-2
BASE_LR_Q         = 3e-4
BASE_TAU          = 0.7
BASE_POLYAK       = 0.005
BASE_GAMMA        = 0.99
BASE_BATCH        = QUANTUM_BATCH_SIZE

N_STEPS_ABLATION  = 100_000  # 10^5 steps per condition
SEEDS             = [0, 1, 2, 3, 4]
PRIMARY_DS        = "hopper-medium"
SECONDARY_DS      = "walker2d-medium"

WANDB_PROJECT     = "quantum-iql-ablation"
WANDB_ENTITY      = None   # set to your username/team if needed

# ── Plot style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

# ── Utilities ─────────────────────────────────────────────────────────────
def ci95(data: np.ndarray) -> np.ndarray:
    n = data.shape[0]
    se = data.std(axis=0) / np.sqrt(n)
    return scipy_stats.t.ppf(0.975, df=n - 1) * se

def save_json(data, path: Path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
    print(f"  Saved → {path}")

def to_device(batch) -> Batch:
    return Batch(
        observations=batch.observations.to(DEVICE),
        actions=batch.actions.to(DEVICE),
        rewards=batch.rewards.to(DEVICE),
        next_observations=batch.next_observations.to(DEVICE),
        dones=batch.dones.to(DEVICE),
    )


# ──────────────────────────────────────────────────────────────────────
# Cell 3
# ──────────────────────────────────────────────────────────────────────
# ── W&B login ─────────────────────────────────────────────────────────────
# Run this cell once — it will prompt you for your API key.
# Get your key from: https://wandb.ai/authorize
# After the first login the key is cached and this becomes a no-op.
import wandb
wandb.login()


# ──────────────────────────────────────────────────────────────────────
# Cell 4
# ──────────────────────────────────────────────────────────────────────
# ── Patch QuantumValueNetwork.forward at class level ─────────────────────
# Must run BEFORE any QuantumIQLTrainer is created.
# Root cause: PennyLane backprop needs at least one input with
# requires_grad=True to build the autograd graph. Observations from the
# replay buffer never have grad, so without this patch the circuit returns
# requires_grad=False → v_loss has no grad_fn → backward() raises RuntimeError
# and the loss becomes NaN.

import torch
import quantum_iql.quantum_value_network as _qvn_module
from quantum_iql.quantum_value_network import _arctan_encode

def _fixed_forward(self, s, mu=None, sigma=None):
    _device = s.device
    _mu    = mu    if mu    is not None else self.mu
    _sigma = sigma if sigma is not None else self.sigma
    xs = _arctan_encode(s, _mu, _sigma)
    B = xs.shape[0]
    if self.obs_dim < self.n_qubits:
        pad = torch.zeros(B, self.n_qubits - self.obs_dim,
                          device=xs.device, dtype=xs.dtype)
        xs = torch.cat([xs, pad], dim=-1)
    elif self.obs_dim > self.n_qubits:
        if self.use_pre_encoder:
            xs = self.pre_encode(xs)
        else:
            xs = xs[:, :self.n_qubits]
    # ← FIX: enable grad so PennyLane backprop builds the autograd graph
    xs = xs.requires_grad_(True)
    if self._diff_method == "backprop":
        expvals = self._circuit(self.theta, self.w, xs, self._active_layers)
    else:
        expvals = self._circuit(
            self.theta.cpu(), self.w.cpu(), xs.cpu(), self._active_layers
        ).to(_device)
    return self.a * expvals.float() + self.b  # (B,) — matches original contract

_qvn_module.QuantumValueNetwork.forward = _fixed_forward
print("QuantumValueNetwork.forward patched ✓")

# ── Patch actor_loss to handle (B,) value_net output ─────────────────────
# actor_loss calls value_net(obs) directly and gets (B,), but q is (B,1).
# The subtraction q - v broadcasts to (B,B) instead of (B,1) → NaN.
# Fix: unsqueeze v inside actor_loss before the subtraction.
import quantum_iql.losses as _losses_module

_orig_actor_loss = _losses_module.actor_loss

def _fixed_actor_loss(actor_net, critic_net, value_net, batch, beta, advantage_clip=100.0):
    import torch
    with torch.no_grad():
        q = critic_net.q_min(batch.observations, batch.actions)   # (B,1)
        v = value_net(batch.observations)                          # (B,) or (B,1)
        if v.dim() == 1:
            v = v.unsqueeze(-1)                                    # → (B,1)
        advantage = q - v                                          # (B,1)
        exp_adv = torch.clamp(torch.exp(beta * advantage), max=advantage_clip)
    log_prob = actor_net.log_prob(batch.observations, batch.actions)  # (B,1)
    loss = -(exp_adv * log_prob).mean()
    metrics = {
        "advantage_mean": advantage.mean().item(),
        "advantage_std":  advantage.std().item(),
        "exp_adv_mean":   exp_adv.mean().item(),
    }
    return loss, metrics

_losses_module.actor_loss = _fixed_actor_loss
import quantum_iql.quantum_trainer as _qt_module
_qt_module.actor_loss = _fixed_actor_loss

# update_actor captured actor_loss at import time via closure — patch the method directly
import types as _types

def _fixed_update_actor(self, batch):
    import torch
    self.actor_optimizer.zero_grad()
    loss, adv_metrics = _fixed_actor_loss(
        self.actor_net,
        self.critic_net,
        self.value_net,
        batch,
        self.cfg.beta,
        self.cfg.advantage_clip,
    )
    loss.backward()
    self.actor_optimizer.step()
    return {"loss/actor": loss.item(), **adv_metrics}

_qt_module.QuantumIQLTrainer.update_actor = _fixed_update_actor
print("actor_loss + update_actor patched ✓")


# <!-- wandb is now initialised inside run_ablation_condition via QuantumIQLTrainer config -->


# ### Dataset Loading
# 
# `hopper-medium` is the primary ablation dataset. `walker2d-medium` is used for cross-environment validation in the τ / topology ablations. Both share the same arctan encoding truncation to `n_qubits` features.


# ──────────────────────────────────────────────────────────────────────
# Cell 7
# ──────────────────────────────────────────────────────────────────────
DATASETS_TO_LOAD = {
    PRIMARY_DS:   "mujoco/hopper/medium-v0",
    SECONDARY_DS: "mujoco/walker2d/medium-v0",
}

buffers = {}
for name, dataset_id in DATASETS_TO_LOAD.items():
    print(f"Loading {name}...")
    buffers[name] = load_minari_dataset(dataset_id, device="cpu")

for name, buf in buffers.items():
    print(f"  {name}: obs_dim={buf.obs_dim}, act_dim={buf.act_dim}, size={len(buf):,}")


# ### FlexQuantumValueNetwork (circuit analysis only)
# 
# The `FlexQuantumValueNetwork` below is kept **only** for Fourier spectrum and
# expressibility analysis (Sections 6–7), which require direct circuit introspection.
# 
# All training ablations (Sections 1–5) use `QuantumIQLTrainer` via
# `run_ablation_condition`, exactly as `training_dynamics_final.ipynb` does.


# ──────────────────────────────────────────────────────────────────────
# Cell 9
# ──────────────────────────────────────────────────────────────────────
# FlexQuantumValueNetwork — used ONLY for Fourier spectrum (§6) and
# expressibility analysis (§7). Training ablations use QuantumIQLTrainer.

# ── Entanglement topology helpers ─────────────────────────────────────────

def _apply_entanglement(n_qubits: int, topology: str):
    """Apply a CZ entanglement layer for the given topology."""
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


# ── Measurement scheme helpers ────────────────────────────────────────────

def _get_measurement(n_qubits: int, scheme: str, learned_weights=None):
    """Return a PennyLane measurement (or list of measurements for learned)."""
    if scheme == "pauli_z0":
        return qml.expval(qml.PauliZ(0))
    elif scheme == "mean_pauli_z":
        # Average expectation over all qubits — not native qml, computed post-measurement
        return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]
    elif scheme == "zz_tensor":
        # ZZ correlator on adjacent pairs (first pair only for simplicity)
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
    elif scheme == "learned":
        # Weighted sum of individual Pauli-Z expectations
        return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]
    else:
        raise ValueError(f"Unknown measurement scheme: {scheme!r}")


class FlexQuantumValueNetwork(nn.Module):
    """
    Flexible quantum value network for ablation studies.
    Supports variable n_qubits, n_layers, entanglement topology, and measurement scheme.
    Uses identity-block init (w = -theta) as established optimal in training_dynamics.

    Args:
        n_qubits:      number of qubits
        n_layers:      DRU reuploading layers
        obs_dim:       observation dimensionality (truncated/padded to n_qubits)
        entanglement:  'linear' | 'circular' | 'all_to_all' | 'none'
        measurement:   'pauli_z0' | 'mean_pauli_z' | 'zz_tensor' | 'learned'
        device_name:   PennyLane device
        diff_method:   'backprop' | 'adjoint' | 'parameter-shift'
    """

    def __init__(
        self,
        n_qubits:     int    = 4,
        n_layers:     int    = 2,
        obs_dim:      int    = 11,
        entanglement: str    = "linear",
        measurement:  str    = "pauli_z0",
        device_name:  str    = "default.qubit",
        diff_method:  str    = "backprop",
    ):
        super().__init__()
        self.n_qubits     = n_qubits
        self.n_layers     = n_layers
        self.obs_dim      = obs_dim
        self.entanglement = entanglement
        self.measurement  = measurement

        # Running stats for arctan encoding normalisation
        self.register_buffer("obs_mean",  torch.zeros(obs_dim))
        self.register_buffer("obs_std",   torch.ones(obs_dim))
        self.register_buffer("obs_count", torch.tensor(0.0))

        # Circuit parameters: theta (encoding), w (reuploading weights)
        # Shape: (n_layers, n_qubits, 3)  — Rot gate has 3 Euler angles
        # FIX 2/3 — initialise as float32 so PennyLane receives float32 inputs
        # and cannot promote internal computation to float64.
        self.theta = nn.Parameter(torch.zeros(n_layers, n_qubits, 3, dtype=torch.float32))
        self.w     = nn.Parameter(torch.zeros(n_layers, n_qubits, 3, dtype=torch.float32))
        # Identity-block init: w = -theta so circuit collapses to identity at step 0
        with torch.no_grad():
            nn.init.uniform_(self.theta, 0, 2 * math.pi)
            self.w.copy_(-self.theta)

        # Learned measurement weights (used only when measurement='learned')
        if measurement == "learned":
            self.meas_weights = nn.Parameter(torch.ones(n_qubits) / n_qubits)
        else:
            self.meas_weights = None

        # Affine output head: scalar → scalar (bias + scale)
        self.out_scale = nn.Parameter(torch.ones(1))
        self.out_bias  = nn.Parameter(torch.zeros(1))

        # Build QNode
        dev = qml.device(device_name, wires=n_qubits)

        if measurement in ("pauli_z0", "zz_tensor"):
            @qml.qnode(dev, interface="torch", diff_method=diff_method)
            def circuit(theta, w, xs):
                _apply_entanglement(n_qubits, entanglement)
                for layer in range(n_layers):
                    for q in range(n_qubits):
                        angles = theta[layer, q] + w[layer, q] * xs[q % xs.shape[0]]
                        qml.Rot(angles[0], angles[1], angles[2], wires=q)
                    _apply_entanglement(n_qubits, entanglement)
                return _get_measurement(n_qubits, measurement)
        else:
            # Returns list of expectations → stack in forward()
            @qml.qnode(dev, interface="torch", diff_method=diff_method)
            def circuit(theta, w, xs):
                _apply_entanglement(n_qubits, entanglement)
                for layer in range(n_layers):
                    for q in range(n_qubits):
                        angles = theta[layer, q] + w[layer, q] * xs[q % xs.shape[0]]
                        qml.Rot(angles[0], angles[1], angles[2], wires=q)
                    _apply_entanglement(n_qubits, entanglement)
                return _get_measurement(n_qubits, measurement)
        self._circuit = circuit

    @torch.no_grad()
    def _update_running_stats(self, obs: torch.Tensor):
        """Online mean/variance update (Welford)."""
        n = obs.shape[0]
        self.obs_count += n
        delta = obs.mean(0) - self.obs_mean
        self.obs_mean += delta * n / self.obs_count
        self.obs_std = (self.obs_std ** 2 + delta ** 2 * n / self.obs_count).sqrt().clamp(min=1e-6)

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Arctan normalisation + truncate/pad to n_qubits features."""
        xs = torch.arctan((obs - self.obs_mean) / (self.obs_std + 1e-8))
        if xs.shape[1] > self.n_qubits:
            xs = xs[:, :self.n_qubits]
        elif xs.shape[1] < self.n_qubits:
            pad = torch.zeros(xs.shape[0], self.n_qubits - xs.shape[1], device=obs.device)
            xs = torch.cat([xs, pad], dim=1)
        return xs  # (B, n_qubits)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass → scalar V(s) per sample, shape (B,)."""
        if self.training:
            self._update_running_stats(obs.detach())
        xs = self._encode(obs)  # (B, n_qubits)

        outs = []
        for i in range(xs.shape[0]):
            raw = self._circuit(self.theta, self.w, xs[i])
            # FIX 3/3 — cast QNode output to float32 immediately after every
            # circuit evaluation.  PennyLane may still return float64 even with
            # set_default_dtype("float32") when using the torch interface; this
            # ensures the autograd graph stays in float32 throughout.
            if self.measurement == "pauli_z0":
                v = raw.to(torch.float32)
            elif self.measurement == "zz_tensor":
                v = raw.to(torch.float32)
            elif self.measurement == "mean_pauli_z":
                v = torch.stack([r.to(torch.float32) for r in raw]).mean()
            elif self.measurement == "learned":
                w_norm = torch.softmax(self.meas_weights, dim=0)
                v = (torch.stack([r.to(torch.float32) for r in raw]) * w_norm).sum()
            outs.append(v)

        out = torch.stack(outs)  # (B,) float32
        return out * self.out_scale + self.out_bias


def make_fqvn(
    obs_dim:      int,
    n_qubits:     int   = BASE_N_QUBITS,
    n_layers:     int   = BASE_N_LAYERS,
    entanglement: str   = BASE_ENTANGLEMENT,
    measurement:  str   = BASE_MEASUREMENT,
    seed:         int   = 0,
) -> FlexQuantumValueNetwork:
    torch.manual_seed(seed)
    return FlexQuantumValueNetwork(
        n_qubits=n_qubits, n_layers=n_layers,
        obs_dim=obs_dim, entanglement=entanglement,
        measurement=measurement,
        device_name=QUANTUM_DEVICE, diff_method=QUANTUM_DIFF_METHOD,
    ).float().to(DEVICE)  # .float() ensures all buffers/params are float32

print("FlexQuantumValueNetwork defined.")
print("Supported entanglement topologies:", ["none", "linear", "circular", "all_to_all"])
print("Supported measurement schemes:    ", ["pauli_z0", "mean_pauli_z", "zz_tensor", "learned"])


# ### Shared Training Loop — via QuantumIQLTrainer
# 
# All ablations use `QuantumIQLTrainer` (same as `training_dynamics_final.ipynb`).
# `run_ablation_condition` builds a `QuantumIQLConfig` per condition/seed, trains
# with `trainer.train_step()`, and logs everything to W&B automatically.


# ──────────────────────────────────────────────────────────────────────
# Cell 11
# ──────────────────────────────────────────────────────────────────────
def make_ablation_config(
    n_qubits:     int   = BASE_N_QUBITS,
    n_layers:     int   = BASE_N_LAYERS,
    n_steps:      int   = N_STEPS_ABLATION,
    seed:         int   = 0,
    wandb_offline: bool = False,
) -> QuantumIQLConfig:
    """Build a QuantumIQLConfig for one ablation condition.

    Note: entanglement topology is NOT a QuantumNetConfig field — the Trainer
    uses the circuit defined in QuantumValueNetwork. Topology ablations are
    handled by FlexQuantumValueNetwork in the Fourier/expressibility sections.
    """
    return QuantumIQLConfig(
        dataset_id     = "mujoco/hopper/medium-v0",
        env_id         = "Hopper-v4",
        mode           = "quantum",
        tau            = BASE_TAU,
        gamma          = BASE_GAMMA,
        polyak         = BASE_POLYAK,
        lr_v           = 3e-4,
        lr_q           = 3e-4,
        lr_quantum     = 1e-3,
        lr_actor       = 3e-4,
        beta           = 3.0,
        warmup_steps   = 1_000,        # freeze actor for first 1k steps
        fix_c_enabled  = True,
        v_freeze_steps = 3_000,        # freeze V while Q bootstraps to real scale
        quantum_grad_clip  = 1.0,
        batch_size     = BASE_BATCH,
        num_steps      = n_steps,
        log_interval   = 100,
        eval_interval  = n_steps + 1,
        wandb_project  = WANDB_PROJECT,
        wandb_offline  = wandb_offline,
        wandb_run_name = "placeholder",
        seed           = seed,
        quantum_batch_size = BASE_BATCH,
        device         = "auto",
        quantum_value  = QuantumNetConfig(
            n_qubits     = n_qubits,
            n_layers     = n_layers,
            device_name  = QUANTUM_DEVICE,
            diff_method  = QUANTUM_DIFF_METHOD,
            running_stats= True,
            layerwise_schedule = [],
        ),
        log_quantum_metrics = False,
    )


def run_ablation_seed(
    buffer,
    cfg: QuantumIQLConfig,
    n_steps: int,
    seed:    int,
) -> dict:
    """
    Train one seed with QuantumIQLTrainer.
    Returns dict of per-step metric lists compatible with plot_ablation_curves.
    """
    set_seed(seed)
    trainer = QuantumIQLTrainer(cfg, buffer, env=None)

    losses, v_stds, adv_means, adv_stds, grad_norms, step_ms = [], [], [], [], [], []

    for step in range(1, n_steps + 1):
        t0 = time.perf_counter()
        metrics = trainer.train_step()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Trainer metric keys: "loss/value", "advantage_mean", "advantage_std",
        # "quantum/grad_norm_theta", "quantum/grad_norm_w"
        lv  = metrics.get("loss/value", float("nan"))
        adv = metrics.get("advantage_mean", float("nan"))
        ads = metrics.get("advantage_std",  float("nan"))
        gn  = (metrics.get("quantum/grad_norm_theta", 0.0) +
               metrics.get("quantum/grad_norm_w",     0.0))  # combined grad norm
        # v_std not directly logged by trainer — use advantage_std as proxy
        losses.append(lv)
        v_stds.append(ads)
        adv_means.append(adv)
        adv_stds.append(ads)
        grad_norms.append(gn)
        step_ms.append(elapsed_ms)

        if step % 500 == 0:
            print(f"    step {step}/{n_steps}  loss={lv:.4f}  adv={adv:.3f}", end="\r")

    print()
    return {
        "loss":        losses,
        "v_std":       v_stds,
        "adv_mean":    adv_means,
        "adv_std":     adv_stds,
        "grad_norm":   grad_norms,
        "ms_per_step": step_ms,
    }


def run_ablation_condition(
    buffer,
    condition_label: str,
    n_steps: int = N_STEPS_ABLATION,
    wandb_offline: bool = False,
    **cfg_kwargs,
) -> List[dict]:
    """Run one ablation condition across all seeds using QuantumIQLTrainer.
    W&B runs are grouped under condition_label automatically via the config.
    """
    results = []
    for seed in SEEDS:
        print(f"  seed={seed}", end="  ")
        cfg = make_ablation_config(
            n_steps=n_steps, seed=seed,
            wandb_offline=wandb_offline,
            **cfg_kwargs,
        )
        # W&B run name and group set via config so the Trainer handles init/finish
        # Override the run name to include condition + seed for clarity in the dashboard
        cfg.wandb_run_name  = f"{condition_label}__seed{seed}"
        # wandb_group is not an IQLConfig field; W&B grouping is handled via
        # wandb_run_name prefix — all runs for a condition share the prefix.

        r = run_ablation_seed(buffer, cfg, n_steps=n_steps, seed=seed)
        results.append(r)

        final = r["loss"][-1]
        adv   = np.mean(r["adv_mean"][-50:])
        ms    = np.mean(r["ms_per_step"])
        print(f"final_loss={final:.4f}  mean_adv={adv:.3f}  ms/step={ms:.1f}")

    return results


def plot_ablation_curves(
    conditions: Dict[str, List[dict]],
    title:      str,
    filename:   str,
    metrics:    List[Tuple[str, str]] = None,
    colors:     List[str] = None,
    n_steps:    int = N_STEPS_ABLATION,
):
    """Plot mean±CI curves for an ablation across metrics."""
    if metrics is None:
        metrics = [
            ("loss",      "Expectile Loss (τ=0.7)"),
            ("v_std",     "std(V(s))  [expressibility proxy]"),
            ("adv_mean",  "mean(A = Q−V)  [actor signal]"),
            ("grad_norm", "‖∇θ‖₂  [gradient health]"),
        ]
    if colors is None:
        cmap   = plt.cm.tab10
        colors = [cmap(i / max(len(conditions) - 1, 1)) for i in range(len(conditions))]

    steps = np.arange(1, n_steps + 1)
    n_m   = len(metrics)
    fig, axes = plt.subplots(1, n_m, figsize=(4.5 * n_m, 4.5))
    if n_m == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=12, fontweight="bold")

    for col, (metric, ylabel) in enumerate(metrics):
        ax = axes[col]
        for (label, seed_results), color in zip(conditions.items(), colors):
            arr = np.array([s[metric] for s in seed_results])
            mu  = arr.mean(0)
            ci  = ci95(arr)
            ax.plot(steps, mu, color=color, label=label, lw=1.5)
            ax.fill_between(steps, mu - ci, mu + ci, color=color, alpha=0.18)
        ax.set_title(ylabel, fontsize=10)
        ax.set_xlabel("Gradient step")
        if col == 0:
            ax.set_ylabel("Value")
        if col == n_m - 1:
            ax.legend(fontsize=8)

    plt.tight_layout()
    path = RESULTS_DIR / filename
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.show()
    print(f"Saved → {path}")


def summary_table(conditions: Dict[str, List[dict]], label: str = ""):
    """Print a final-step summary table for an ablation."""
    print(f"\n{'─'*95}")
    print(f"{'Condition':<30} {'Loss (mean±CI)':>20} {'mean Ā':>10} {'std Ā':>9} {'‖∇θ‖':>9} {'ms/step':>9}")
    print(f"{'─'*95}")
    for cond_label, seed_results in conditions.items():
        def tail(key, n=50):
            return np.mean([np.mean(s[key][-n:]) for s in seed_results])
        final = np.array([s["loss"][-1] for s in seed_results])
        ci_v  = final.std() / np.sqrt(len(final)) * 1.96
        print(
            f"{cond_label:<30} {final.mean():>9.4f}±{ci_v:<8.4f}"
            f"  {tail('adv_mean'):>9.3f}  {tail('adv_std'):>9.3f}"
            f"  {tail('grad_norm'):>9.4f}  {tail('ms_per_step'):>9.1f}"
        )
    print(f"{'─'*95}\n")

print("QuantumIQLTrainer-based training loop defined.")


# ---
# ## 1. Data-Reuploading Layers Sweep
# 
# Vary `n_layers ∈ {1, 2, 3, 4, 6}` with all other parameters fixed at the base config.
# 
# **Expected:** convergence improves up to ~3 layers, then diminishing returns.  
# Beyond 4 layers, barren-plateau effects may cause gradient norms to collapse.  
# Wall-clock cost scales linearly with n_layers.


# ──────────────────────────────────────────────────────────────────────
# Cell 13
# ──────────────────────────────────────────────────────────────────────
# ── DRU layers sweep ──────────────────────────────────────────────────────
# Constraint from QuantumValueNetwork: n_layers <= floor(log2(n_qubits))
#   n_layers=1 → min n_qubits=2  (use 4 to match base config)
#   n_layers=2 → min n_qubits=4  (base config)
#   n_layers=3 → min n_qubits=8
# n_layers=4+ requires n_qubits>=16 which is computationally infeasible.
# Each condition uses the minimum n_qubits that satisfies the constraint
# while keeping n_qubits as close to base (4) as possible.

import math
LAYERS_SWEEP = [
    (1, 4),   # n_layers=1, n_qubits=4
    (2, 4),   # n_layers=2, n_qubits=4  ← base config
    (3, 8),   # n_layers=3, n_qubits=8  (min qubits for 3 layers)
]
N_STEPS_LAYERS = N_STEPS_ABLATION
buf = buffers[PRIMARY_DS]

layers_results = {}
for n_layers, n_qubits in LAYERS_SWEEP:
    cond_key = f"n_layers={n_layers}(q={n_qubits})"
    print(f"\n=== {cond_key} ===")
    assert n_layers <= math.floor(math.log2(n_qubits)), f"Invalid: {n_layers} layers, {n_qubits} qubits"
    layers_results[cond_key] = run_ablation_condition(
        buf, cond_key,
        n_steps=N_STEPS_LAYERS,
        n_qubits=n_qubits,
        n_layers=n_layers,
    )

save_json(
    {k: [dict(seed=i, **{m: v[m] for m in ["loss","v_std","adv_mean","adv_std","grad_norm","ms_per_step"]})
         for i, v in enumerate(vs)]
     for k, vs in layers_results.items()},
    RESULTS_DIR / "ablation_layers.json"
)


# ──────────────────────────────────────────────────────────────────────
# Cell 14
# ──────────────────────────────────────────────────────────────────────
plot_ablation_curves(
    layers_results,
    title=f"Ablation: DRU Reuploading Layers  ({PRIMARY_DS}, {len(SEEDS)} seeds)",
    filename="ablation_layers.png",
    n_steps=N_STEPS_LAYERS,
)
summary_table(layers_results)


# ──────────────────────────────────────────────────────────────────────
# Cell 15
# ──────────────────────────────────────────────────────────────────────
# ── Wall-clock cost vs n_layers ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
n_layers_vals = [int(k.split("=")[1]) for k in layers_results]
ms_vals  = [np.mean([np.mean(s["ms_per_step"]) for s in vs]) for vs in layers_results.values()]
ms_ci    = [scipy_stats.t.ppf(0.975, df=len(SEEDS)-1)
            * np.std([np.mean(s["ms_per_step"]) for s in vs]) / np.sqrt(len(SEEDS))
            for vs in layers_results.values()]

ax.plot(n_layers_vals, ms_vals, marker="o", color="steelblue", lw=2)
ax.errorbar(n_layers_vals, ms_vals, yerr=ms_ci, fmt="none", color="steelblue", capsize=4)
ax.set_xlabel("n_layers")
ax.set_ylabel("ms / gradient step")
ax.set_title(f"Wall-clock cost vs reuploading layers\n({PRIMARY_DS}, GPU backprop)")
plt.tight_layout()
fig.savefig(RESULTS_DIR / "ablation_layers_wallclock.png", bbox_inches="tight", dpi=130)
plt.show()
print("Saved → results/ablations/ablation_layers_wallclock.png")


# ---
# ## 2. Entanglement Topology Comparison
# 
# The `QuantumIQLTrainer` uses a fixed `linear` entanglement circuit defined
# inside `QuantumValueNetwork`. The topology ablation therefore runs via
# `FlexQuantumValueNetwork` + the manual loop (same approach as Sections 6–7),
# which gives full topology control.
# 
# Compare four entanglement patterns at fixed `n_qubits=4, n_layers=2`:
# 
# | Topology | CZ gates / layer | 2-qubit depth |
# |----------|-----------------|---------------|
# | none     | 0               | 0             |
# | linear   | n−1             | n−1           |
# | circular | n               | n             |
# | all_to_all | n(n−1)/2      | 1             |


# ──────────────────────────────────────────────────────────────────────
# Cell 17
# ──────────────────────────────────────────────────────────────────────
TOPOLOGIES   = ["none", "linear", "circular", "all_to_all"]
N_STEPS_TOPO = N_STEPS_ABLATION

# Topology ablation uses FlexQuantumValueNetwork directly (the Trainer circuit
# has a fixed linear topology; FlexQVN is the right tool for this ablation).
from quantum_iql.losses import critic_loss as _critic_loss, value_loss
import torch.optim as optim

def _run_topo_seed(buffer, topology, seed, n_steps):
    set_seed(seed)
    qvn = make_fqvn(
        buffer.obs_dim, seed=seed,
        n_qubits=BASE_N_QUBITS, n_layers=BASE_N_LAYERS,
        entanglement=topology, measurement=BASE_MEASUREMENT,
    )
    wrapped = FlexWrap(qvn)
    critic        = _make_critic(buffer.obs_dim, buffer.act_dim, seed)
    critic_target = _make_critic(buffer.obs_dim, buffer.act_dim, seed)
    with torch.no_grad():
        for p, pt in zip(critic.parameters(), critic_target.parameters()):
            pt.data.copy_(p.data); pt.requires_grad_(False)
    opt_v = optim.Adam(qvn.parameters(),    lr=BASE_LR_V)
    opt_q = optim.Adam(critic.parameters(), lr=BASE_LR_Q)
    losses, v_stds, adv_means, adv_stds, grad_norms, step_ms = [], [], [], [], [], []
    for _ in range(n_steps):
        t0 = time.perf_counter()
        b  = to_device(buffer.sample(BASE_BATCH))
        opt_q.zero_grad()
        lq = _critic_loss(critic, wrapped, b, gamma=BASE_GAMMA)
        lq.backward(); opt_q.step()
        with torch.no_grad():
            for p, pt in zip(critic.parameters(), critic_target.parameters()):
                pt.data.mul_(1 - BASE_POLYAK).add_(p.data, alpha=BASE_POLYAK)
        opt_v.zero_grad()
        lv = value_loss(wrapped, critic_target, b, BASE_TAU)
        lv.backward()
        gn = sum(p.grad.norm(2).item()**2 for p in qvn.parameters() if p.grad is not None)**0.5
        opt_v.step()
        with torch.no_grad():
            v     = qvn(b.observations)
            q_min = critic_target.q_min(b.observations, b.actions).squeeze()
            adv   = q_min - v
        losses.append(lv.item()); v_stds.append(v.std().item())
        adv_means.append(adv.mean().item()); adv_stds.append(adv.std().item())
        grad_norms.append(gn); step_ms.append((time.perf_counter()-t0)*1000)
    return {"loss": losses, "v_std": v_stds, "adv_mean": adv_means,
            "adv_std": adv_stds, "grad_norm": grad_norms, "ms_per_step": step_ms}

def _make_critic(obs_dim, act_dim, seed=0):
    torch.manual_seed(seed)
    return CriticNetwork(obs_dim, act_dim, hidden_dims=(256, 256), use_twin=True).to(DEVICE)

class FlexWrap(nn.Module):
    def __init__(self, net): super().__init__(); self.net = net
    def forward(self, obs):
        out = self.net(obs)
        if isinstance(out, (list, tuple)):
            out = torch.stack([o.to(torch.float32) for o in out]).mean(dim=0)
        return out.to(torch.float32).unsqueeze(-1)

topo_results = {}
for ds_name in [PRIMARY_DS, SECONDARY_DS]:
    buf = buffers[ds_name]
    topo_results[ds_name] = {}
    print(f"\n{'='*55}\nDataset: {ds_name}")
    for topo in TOPOLOGIES:
        print(f"\n  === topology={topo} ===")
        seeds_out = []
        for seed in SEEDS:
            print(f"  seed={seed}", end="  ")
            r = _run_topo_seed(buf, topo, seed, N_STEPS_TOPO)
            seeds_out.append(r)
            print(f"final_loss={r['loss'][-1]:.4f}  mean_adv={np.mean(r['adv_mean'][-50:]):.3f}")
        topo_results[ds_name][topo] = seeds_out

save_json(
    {ds: {t: [{m: v[m] for m in ["loss","v_std","adv_mean","grad_norm","ms_per_step"]}
              for v in vs]
           for t, vs in conds.items()}
     for ds, conds in topo_results.items()},
    RESULTS_DIR / "ablation_topology.json"
)


# ──────────────────────────────────────────────────────────────────────
# Cell 18
# ──────────────────────────────────────────────────────────────────────
# ── Plot: one figure per dataset ─────────────────────────────────────────
topo_colors = {"none": "#aaaaaa", "linear": "#2166ac", "circular": "#f4a582", "all_to_all": "#d6604d"}

for ds_name in [PRIMARY_DS, SECONDARY_DS]:
    plot_ablation_curves(
        topo_results[ds_name],
        title=f"Ablation: Entanglement Topology — {ds_name}  ({len(SEEDS)} seeds)",
        filename=f"ablation_topology_{ds_name.replace('-','_')}.png",
        colors=[topo_colors[t] for t in TOPOLOGIES],
        n_steps=N_STEPS_TOPO,
    )
    summary_table(topo_results[ds_name], label=ds_name)


# ──────────────────────────────────────────────────────────────────────
# Cell 19
# ──────────────────────────────────────────────────────────────────────
# ── Radar / spider chart: topology comparison across metrics ──────────────
import matplotlib.patches as mpatches

metric_keys   = ["loss", "adv_mean", "v_std", "grad_norm"]
metric_labels = ["Final Loss
(lower=better)", "mean Ā
(higher=better)",
                 "std(V)
(higher=more expressive)", "‖∇θ‖
(higher=less BP)"]

def tail_mean(results, key, n=50):
    return np.mean([np.mean(s[key][-n:]) for s in results])

fig, axes = plt.subplots(1, len(TOPOLOGIES), figsize=(16, 4), subplot_kw={"polar": True})
fig.suptitle(f"Topology comparison — radar (hopper-medium, normalised per metric)", fontsize=12, fontweight="bold")

# Normalise each metric across topologies to [0,1]
metric_raw = {}
for mk in metric_keys:
    vals = {t: tail_mean(topo_results[PRIMARY_DS][t], mk) for t in TOPOLOGIES}
    vmin, vmax = min(vals.values()), max(vals.values())
    rng = vmax - vmin if vmax != vmin else 1.0
    if mk == "loss":  # invert so higher = better
        metric_raw[mk] = {t: 1 - (v - vmin) / rng for t, v in vals.items()}
    else:
        metric_raw[mk] = {t: (v - vmin) / rng for t, v in vals.items()}

N = len(metric_keys)
angles = [n / float(N) * 2 * math.pi for n in range(N)]
angles += angles[:1]

for ax, topo in zip(axes, TOPOLOGIES):
    vals = [metric_raw[mk][topo] for mk in metric_keys]
    vals += vals[:1]
    ax.plot(angles, vals, color=topo_colors[topo], lw=2)
    ax.fill(angles, vals, color=topo_colors[topo], alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, size=8)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "", "", ""], size=7)
    ax.set_title(topo, size=11, fontweight="bold", pad=14, color=topo_colors[topo])
    ax.set_ylim(0, 1)

plt.tight_layout()
fig.savefig(RESULTS_DIR / "ablation_topology_radar.png", bbox_inches="tight", dpi=130)
plt.show()
print("Saved → results/ablations/ablation_topology_radar.png")


# ---
# ## 3. Qubit Count Scaling
# 
# Scale `n_qubits ∈ {2, 4, 6, 8}` at fixed `n_layers=2, linear entanglement`.
# 
# - **2 qubits:** very limited Fourier spectrum (frequencies {0, ±1}), fast
# - **4 qubits:** baseline; arctan encoding covers 4 of 11 hopper features
# - **6 qubits:** covers more features; heavier circuit (hopper has 11 obs dims)
# - **8 qubits:** expressible but approaching barren-plateau regime for 2 layers
# 
# Wall-clock cost is tracked — qubit count affects both circuit depth and PennyLane simulation overhead.


# ──────────────────────────────────────────────────────────────────────
# Cell 21
# ──────────────────────────────────────────────────────────────────────
QUBIT_SWEEP    = [2, 4, 6, 8]
N_STEPS_QUBITS = N_STEPS_ABLATION
buf = buffers[PRIMARY_DS]

qubit_results = {}
for n_qubits in QUBIT_SWEEP:
    cond_key = f"n_qubits={n_qubits}"
    print(f"\n=== {cond_key} ===")
    qubit_results[cond_key] = run_ablation_condition(
        buf, cond_key,
        n_steps=N_STEPS_QUBITS,
        n_qubits=n_qubits,
        n_layers=BASE_N_LAYERS,
    )

save_json(
    {k: [{m: v[m] for m in ["loss","v_std","adv_mean","grad_norm","ms_per_step"]} for v in vs]
     for k, vs in qubit_results.items()},
    RESULTS_DIR / "ablation_qubits.json"
)


# ──────────────────────────────────────────────────────────────────────
# Cell 22
# ──────────────────────────────────────────────────────────────────────
plot_ablation_curves(
    qubit_results,
    title=f"Ablation: Qubit Count  ({PRIMARY_DS}, {len(SEEDS)} seeds)",
    filename="ablation_qubits.png",
    n_steps=N_STEPS_QUBITS,
)
summary_table(qubit_results)


# ──────────────────────────────────────────────────────────────────────
# Cell 23
# ──────────────────────────────────────────────────────────────────────
# ── Qubit count vs wall-clock + final loss scatter ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle("Qubit count: performance vs cost", fontsize=12, fontweight="bold")

q_vals      = [int(k.split("=")[1]) for k in qubit_results]
final_loss  = [np.mean([s["loss"][-1] for s in vs]) for vs in qubit_results.values()]
ms_per_step = [np.mean([np.mean(s["ms_per_step"]) for s in vs]) for vs in qubit_results.values()]
cmap_q      = plt.cm.cool

colors_q = [cmap_q(i / max(len(q_vals)-1, 1)) for i in range(len(q_vals))]

# Loss
ax = axes[0]
bars = ax.bar([str(q) for q in q_vals], final_loss, color=colors_q, edgecolor="white")
ax.set_xlabel("n_qubits"); ax.set_ylabel("Final expectile loss")
ax.set_title("Convergence quality vs qubit count")
for bar, v in zip(bars, final_loss):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{v:.3f}", ha="center", va="bottom", fontsize=9)

# Wall-clock
ax = axes[1]
bars = ax.bar([str(q) for q in q_vals], ms_per_step, color=colors_q, edgecolor="white")
ax.set_xlabel("n_qubits"); ax.set_ylabel("ms / step")
ax.set_title("Wall-clock cost vs qubit count")
for bar, v in zip(bars, ms_per_step):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f"{v:.1f}ms", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
fig.savefig(RESULTS_DIR / "ablation_qubits_cost.png", bbox_inches="tight", dpi=130)
plt.show()


# ---
# ## 3b. Qubit × Layer Joint Ablation
# 
# Section 3 sweeps qubits with `n_layers` fixed, and Section 1 sweeps layers with `n_qubits` fixed.
# Neither captures their **interaction**: a wide-shallow circuit behaves very differently from a narrow-deep one.
# 
# | | Few layers | Many layers |
# |---|---|---|
# | **Few qubits** | Low capacity, fast | Barren-plateau risk, many params/qubit |
# | **Many qubits** | Wide but shallow, undertrained entanglement | High expressibility, expensive |
# 
# **Grid:** `n_qubits ∈ {2, 4, 6, 8}` × `n_layers ∈ {1, 2, 4}` — 12 conditions.  
# Results visualised as a 2D heatmap (final loss and mean advantage) so optimal operating points jump out.
# 
# > **Cost note:** On CPU (`adjoint`, batch=4) this is slow — consider reducing to 3 seeds and 2 000 steps
# > (`N_STEPS_QL = 2_000`, `SEEDS_QL = [0, 1, 2]`) for a quick exploration run.
# > On GPU (`backprop`, batch=256) the full grid at 5 000 steps is fine.


# ──────────────────────────────────────────────────────────────────────
# Cell 25
# ──────────────────────────────────────────────────────────────────────
# ── Qubit × Layer grid config ─────────────────────────────────────────────
N_STEPS_QL = N_STEPS_ABLATION
SEEDS_QL   = SEEDS

# Only include (n_qubits, n_layers) pairs satisfying n_layers <= floor(log2(n_qubits))
# (2,1): floor(log2(2))=1 ✓   (4,1): floor(log2(4))=2 ✓   (4,2): base ✓
# (8,2): ✓   (8,3): ✓
# Removed: (2,2),(2,4) floor(log2(2))=1; (4,4),(6,4) floor(log2(4/6))=2
QUBIT_LAYER_GRID = [
    (2, 1),          # minimal circuit
    (4, 1), (4, 2),  # base n_qubits; (4,2) = base config ★
    (8, 1), (8, 2), (8, 3),  # more qubits, up to 3 layers
]

buf = buffers[PRIMARY_DS]
ql_results = {}

for n_qubits, n_layers in QUBIT_LAYER_GRID:
    cond_key = f"q{n_qubits}_L{n_layers}"
    print(f"\n=== {cond_key} (params={n_qubits * n_layers * 3 * 2}) ===")
    results = []
    for seed in SEEDS_QL:
        print(f"  seed={seed}", end="  ")
        cfg = make_ablation_config(
            n_qubits=n_qubits, n_layers=n_layers,
            n_steps=N_STEPS_QL, seed=seed,
        )
        cfg.wandb_run_name = f"{cond_key}__seed{seed}"
        r = run_ablation_seed(buf, cfg, n_steps=N_STEPS_QL, seed=seed)
        results.append(r)
        print(f"final_loss={r['loss'][-1]:.4f}  "
              f"mean_adv={np.mean(r['adv_mean'][-50:]):.3f}  "
              f"ms/step={np.mean(r['ms_per_step']):.1f}")
    ql_results[cond_key] = results

save_json(
    {k: [{m: v[m] for m in ["loss", "v_std", "adv_mean", "adv_std", "grad_norm", "ms_per_step"]}
         for v in vs]
     for k, vs in ql_results.items()},
    RESULTS_DIR / "ablation_qubit_layer.json"
)


# ──────────────────────────────────────────────────────────────────────
# Cell 26
# ──────────────────────────────────────────────────────────────────────
# ── Heatmap: final loss and mean advantage on the qubits × layers grid ───
import matplotlib.colors as mcolors

QUBITS_GRID  = sorted(set(q for q, _ in QUBIT_LAYER_GRID))
LAYERS_GRID  = sorted(set(l for _, l in QUBIT_LAYER_GRID))

def grid_matrix(metric, n_tail=50, higher_better=False):
    """Build a (n_qubits × n_layers) matrix of tail-averaged metric values.
    Missing grid points are filled with NaN."""
    mat = np.full((len(QUBITS_GRID), len(LAYERS_GRID)), np.nan)
    for qi, nq in enumerate(QUBITS_GRID):
        for li, nl in enumerate(LAYERS_GRID):
            key = f"q{nq}_L{nl}"
            if key in ql_results:
                mat[qi, li] = np.mean(
                    [np.mean(s[metric][-n_tail:]) for s in ql_results[key]]
                )
    return mat

loss_mat = grid_matrix("loss")
adv_mat  = grid_matrix("adv_mean", higher_better=True)
gn_mat   = grid_matrix("grad_norm")
ms_mat   = grid_matrix("ms_per_step")

# Base config position for annotation
base_qi = QUBITS_GRID.index(BASE_N_QUBITS)
base_li = LAYERS_GRID.index(BASE_N_LAYERS)

fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
fig.suptitle(
    f"Joint Qubits × Layers ablation — {PRIMARY_DS}  ({len(SEEDS_QL)} seeds, last-50-step mean)",
    fontsize=12, fontweight="bold"
)

panels = [
    (loss_mat,  "Expectile Loss\n(lower = better)",   "YlOrRd",   False),
    (adv_mat,   "mean(A = Q−V)\n(higher = better)",   "RdYlGn",   True),
    (gn_mat,    "‖∇θ‖₂\n(higher = healthier)",        "Blues",    True),
    (ms_mat,    "ms / step\n(lower = cheaper)",        "YlOrRd",   False),
]

for ax, (mat, title, cmap, higher_better) in zip(axes, panels):
    # Mask NaN for display
    masked = np.ma.masked_invalid(mat)
    im = ax.imshow(masked, cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.85)

    # Annotate cells with value
    for qi in range(len(QUBITS_GRID)):
        for li in range(len(LAYERS_GRID)):
            val = mat[qi, li]
            if not np.isnan(val):
                ax.text(li, qi, f"{val:.3f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if abs(val) > (np.nanmax(np.abs(mat)) * 0.6) else "black")

    # Mark base config with a white star
    if not np.isnan(mat[base_qi, base_li]):
        ax.plot(base_li, base_qi, "w*", markersize=14, markeredgecolor="black", markeredgewidth=0.8)

    ax.set_xticks(range(len(LAYERS_GRID)))
    ax.set_xticklabels([f"L={l}" for l in LAYERS_GRID])
    ax.set_yticks(range(len(QUBITS_GRID)))
    ax.set_yticklabels([f"q={q}" for q in QUBITS_GRID])
    ax.set_xlabel("n_layers")
    ax.set_ylabel("n_qubits")
    ax.set_title(title, fontsize=10)

plt.tight_layout()
fig.savefig(RESULTS_DIR / "ablation_qubit_layer_heatmap.png", bbox_inches="tight", dpi=130)
plt.show()
print("Saved → results/ablations/ablation_qubit_layer_heatmap.png")


# ──────────────────────────────────────────────────────────────────────
# Cell 27
# ──────────────────────────────────────────────────────────────────────
# ── Learning curves: base config vs best grid point vs budget-matched alts ─
# Select configs to highlight: base + the cheapest config with loss within 5% of best
best_key_ql = min(
    ql_results,
    key=lambda k: np.mean([np.mean(s["loss"][-50:]) for s in ql_results[k]])
)
best_loss = np.mean([np.mean(s["loss"][-50:]) for s in ql_results[best_key_ql]])
threshold = best_loss * 1.05  # within 5%

# Cheapest = fewest total params among configs within threshold
def n_params(key):
    nq, nl = int(key.split("_")[0][1:]), int(key.split("_")[1][1:])
    return nq * nl * 3 * 2  # theta + w, each (nl, nq, 3)

candidates = [
    k for k in ql_results
    if np.mean([np.mean(s["loss"][-50:]) for s in ql_results[k]]) <= threshold
]
cheapest_key = min(candidates, key=n_params)

highlight_keys = list(dict.fromkeys(["q4_L2", best_key_ql, cheapest_key]))  # deduplicated
highlight_colors = ["#2166ac", "#d01c8b", "#4dac26"][:len(highlight_keys)]
highlight_labels = {
    "q4_L2": "base (q=4, L=2)",
    best_key_ql:    f"best: {best_key_ql}",
    cheapest_key:   f"cheapest-within-5%: {cheapest_key}",
}

plot_ablation_curves(
    {highlight_labels.get(k, k): ql_results[k] for k in highlight_keys},
    title=f"Qubit×Layer: base vs best vs cheapest-within-5%  ({PRIMARY_DS})",
    filename="ablation_qubit_layer_curves.png",
    colors=highlight_colors,
    n_steps=N_STEPS_QL,
)

# ── Summary table ──────────────────────────────────────────────────────────
print("\nFull grid summary:")
print(f"{'Config':<14} {'n_params':>9} {'Loss (mean)':>13} {'mean Ā':>9} {'‖∇θ‖':>9} {'ms/step':>9}")
print("─" * 65)
for k in sorted(ql_results, key=n_params):
    seeds = ql_results[k]
    def tm(metric): return np.mean([np.mean(s[metric][-50:]) for s in seeds])
    marker = " ← base" if k == "q4_L2" else (" ← best" if k == best_key_ql else "")
    print(f"{k:<14} {n_params(k):>9}  {tm('loss'):>11.4f}  {tm('adv_mean'):>9.3f}"
          f"  {tm('grad_norm'):>9.4f}  {tm('ms_per_step'):>7.1f}{marker}")
print("─" * 65)
print(f"\nBest config:              {best_key_ql}  (loss={best_loss:.4f})")
print(f"Cheapest within 5% best:  {cheapest_key}  (params={n_params(cheapest_key)})")


# ---
# ## 4. Measurement Scheme: Local vs Adaptive Non-local
# 
# `QuantumNetConfig` does not expose a measurement scheme parameter — the
# `QuantumValueNetwork` uses a fixed `PauliZ(0)` observable. The measurement
# ablation therefore uses `FlexQuantumValueNetwork` directly (same approach as
# the topology ablation), which supports:
# 
# | Scheme | Description |
# |--------|-------------|
# | `pauli_z0` | Local: single qubit PauliZ (baseline) |
# | `mean_pauli_z` | Non-local average over all qubits |
# | `zz_tensor` | 2-qubit ZZ correlator ⟨Z₀⊗Z₁⟩ |
# | `learned` | Trainable weighted sum of all qubit expectations |


# ──────────────────────────────────────────────────────────────────────
# Cell 29
# ──────────────────────────────────────────────────────────────────────
MEASUREMENT_SCHEMES = ["pauli_z0", "mean_pauli_z", "zz_tensor", "learned"]
N_STEPS_MEAS = N_STEPS_ABLATION
buf = buffers[PRIMARY_DS]

# Measurement ablation uses FlexQuantumValueNetwork directly because
# QuantumNetConfig only supports the fixed PauliZ(0) measurement.
def _run_meas_seed(buffer, measurement, seed, n_steps):
    import torch.optim as optim
    set_seed(seed)
    qvn = make_fqvn(
        buffer.obs_dim, seed=seed,
        n_qubits=BASE_N_QUBITS, n_layers=BASE_N_LAYERS,
        entanglement=BASE_ENTANGLEMENT, measurement=measurement,
    )
    wrapped = FlexWrap(qvn)
    critic        = _make_critic(buffer.obs_dim, buffer.act_dim, seed)
    critic_target = _make_critic(buffer.obs_dim, buffer.act_dim, seed)
    with torch.no_grad():
        for p, pt in zip(critic.parameters(), critic_target.parameters()):
            pt.data.copy_(p.data); pt.requires_grad_(False)
    opt_v = optim.Adam(qvn.parameters(),    lr=BASE_LR_V)
    opt_q = optim.Adam(critic.parameters(), lr=BASE_LR_Q)
    from quantum_iql.losses import critic_loss as _critic_loss, value_loss
    losses, v_stds, adv_means, adv_stds, grad_norms, step_ms = [], [], [], [], [], []
    for _ in range(n_steps):
        t0 = time.perf_counter()
        b  = to_device(buffer.sample(BASE_BATCH))
        opt_q.zero_grad()
        lq = _critic_loss(critic, wrapped, b, gamma=BASE_GAMMA)
        lq.backward(); opt_q.step()
        with torch.no_grad():
            for p, pt in zip(critic.parameters(), critic_target.parameters()):
                pt.data.mul_(1 - BASE_POLYAK).add_(p.data, alpha=BASE_POLYAK)
        opt_v.zero_grad()
        lv = value_loss(wrapped, critic_target, b, BASE_TAU)
        lv.backward()
        gn = sum(p.grad.norm(2).item()**2 for p in qvn.parameters() if p.grad is not None)**0.5
        opt_v.step()
        with torch.no_grad():
            v     = qvn(b.observations)
            q_min = critic_target.q_min(b.observations, b.actions).squeeze()
            adv   = q_min - v
        losses.append(lv.item()); v_stds.append(v.std().item())
        adv_means.append(adv.mean().item()); adv_stds.append(adv.std().item())
        grad_norms.append(gn); step_ms.append((time.perf_counter()-t0)*1000)
    return {"loss": losses, "v_std": v_stds, "adv_mean": adv_means,
            "adv_std": adv_stds, "grad_norm": grad_norms, "ms_per_step": step_ms}

meas_results = {}
for scheme in MEASUREMENT_SCHEMES:
    print(f"\n=== measurement={scheme} ===")
    seeds_out = []
    for seed in SEEDS:
        print(f"  seed={seed}", end="  ")
        r = _run_meas_seed(buf, scheme, seed, N_STEPS_MEAS)
        seeds_out.append(r)
        print(f"final_loss={r['loss'][-1]:.4f}  mean_adv={np.mean(r['adv_mean'][-50:]):.3f}")
    meas_results[scheme] = seeds_out

save_json(
    {k: [{m: v[m] for m in ["loss","v_std","adv_mean","grad_norm","ms_per_step"]} for v in vs]
     for k, vs in meas_results.items()},
    RESULTS_DIR / "ablation_measurement.json"
)


# ──────────────────────────────────────────────────────────────────────
# Cell 30
# ──────────────────────────────────────────────────────────────────────
meas_colors = {
    "pauli_z0":    "#2166ac",
    "mean_pauli_z": "#4dac26",
    "zz_tensor":   "#d01c8b",
    "learned":     "#f1a340",
}

plot_ablation_curves(
    meas_results,
    title=f"Ablation: Measurement Scheme  ({PRIMARY_DS}, {len(SEEDS)} seeds)",
    filename="ablation_measurement.png",
    colors=[meas_colors[s] for s in MEASUREMENT_SCHEMES],
    n_steps=N_STEPS_MEAS,
)
summary_table(meas_results)


# ──────────────────────────────────────────────────────────────────────
# Cell 31
# ──────────────────────────────────────────────────────────────────────
# ── Learned measurement weight evolution ──────────────────────────────────
# Snapshot learned weights at init vs after N_STEPS_MEAS for one seed
print("Learned measurement weight analysis (1 representative seed):")

torch.manual_seed(0)
qvn_learn = make_fqvn(buffers[PRIMARY_DS].obs_dim, measurement="learned", seed=0)
weights_init = torch.softmax(qvn_learn.meas_weights.detach(), dim=0).cpu().numpy().copy()

_ = run_ablation_seed(buffers[PRIMARY_DS], qvn_learn, n_steps=N_STEPS_MEAS // 2, seed=0)
weights_final = torch.softmax(qvn_learn.meas_weights.detach(), dim=0).cpu().numpy()

fig, ax = plt.subplots(figsize=(7, 3.5))
x = np.arange(BASE_N_QUBITS)
w = 0.35
ax.bar(x - w/2, weights_init,  width=w, label="Init",  color="#aaaaaa", edgecolor="white")
ax.bar(x + w/2, weights_final, width=w, label="Trained", color="#f1a340", edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels([f"qubit {q}" for q in range(BASE_N_QUBITS)])
ax.set_ylabel("Softmax weight"); ax.set_title("Learned measurement weight evolution (seed 0)")
ax.legend()
plt.tight_layout()
fig.savefig(RESULTS_DIR / "ablation_measurement_weights.png", bbox_inches="tight", dpi=130)
plt.show()
print("Saved → results/ablations/ablation_measurement_weights.png")


# ---
# ## 5. Dataset Size Sensitivity
# 
# Subsample `hopper-medium` at 10%, 25%, 50%, and 100% of transitions to probe data efficiency.  
# This tests whether the quantum circuit's inductive bias (limited Fourier spectrum) confers
# advantages in low-data offline RL regimes compared to a classical MLP baseline.
# 
# Both quantum (base config) and classical (256×256 MLP) V-networks are evaluated.


# ──────────────────────────────────────────────────────────────────────
# Cell 33
# ──────────────────────────────────────────────────────────────────────
DATA_FRACTIONS   = [0.10, 0.25, 0.50, 1.00]
N_STEPS_DATASIZE = N_STEPS_ABLATION
buf_full = buffers[PRIMARY_DS]

def make_subsampled_buffer(buf: ReplayBuffer, fraction: float, seed: int = 42) -> ReplayBuffer:
    """Return a ReplayBuffer containing `fraction` of transitions sampled without replacement."""
    import random
    rng = random.Random(seed)
    n_total   = len(buf)
    n_keep    = max(1, int(n_total * fraction))
    indices   = sorted(rng.sample(range(n_total), n_keep))
    idx_t     = torch.tensor(indices, dtype=torch.long)
    new_buf   = ReplayBuffer.__new__(ReplayBuffer)
    new_buf.obs        = buf.obs[idx_t]
    new_buf.actions    = buf.actions[idx_t]
    new_buf.rewards    = buf.rewards[idx_t]
    new_buf.next_obs   = buf.next_obs[idx_t]
    new_buf.dones      = buf.dones[idx_t]
    new_buf._size      = n_keep
    new_buf.obs_dim    = buf.obs_dim
    new_buf.act_dim    = buf.act_dim
    return new_buf


# ── Classical baseline (no Trainer — ValueNetwork has no QuantumNetConfig) ─
class _ClassicalWrap(nn.Module):
    def __init__(self, net): super().__init__(); self.net = net
    def forward(self, obs): return self.net(obs)

def make_critic(obs_dim, act_dim, seed=0):
    torch.manual_seed(seed)
    return CriticNetwork(obs_dim, act_dim, hidden_dims=(256, 256), use_twin=True).to(DEVICE)

def run_classical_seed(buffer, seed, n_steps=N_STEPS_DATASIZE):
    torch.manual_seed(seed)
    net     = ValueNetwork(buffer.obs_dim, hidden_dims=(256, 256)).to(DEVICE)
    wrapped = _ClassicalWrap(net)
    critic        = make_critic(buffer.obs_dim, buffer.act_dim, seed=seed)
    critic_target = make_critic(buffer.obs_dim, buffer.act_dim, seed=seed)
    with torch.no_grad():
        for p, pt in zip(critic.parameters(), critic_target.parameters()):
            pt.data.copy_(p.data); pt.requires_grad_(False)
    opt_v = optim.Adam(net.parameters(), lr=3e-3)
    opt_q = optim.Adam(critic.parameters(), lr=3e-4)
    losses, v_means, adv_means = [], [], []
    for _ in range(n_steps):
        b = to_device(buffer.sample(min(BASE_BATCH, len(buffer))))
        opt_q.zero_grad()
        loss_q = _critic_loss(critic, wrapped, b, gamma=BASE_GAMMA)
        loss_q.backward(); opt_q.step()
        with torch.no_grad():
            for p, pt in zip(critic.parameters(), critic_target.parameters()):
                pt.data.mul_(1 - BASE_POLYAK).add_(p.data, alpha=BASE_POLYAK)
        opt_v.zero_grad()
        loss_v = value_loss(wrapped, critic_target, b, BASE_TAU)
        loss_v.backward(); opt_v.step()
        with torch.no_grad():
            v     = net(b.observations).squeeze()
            q_min = critic_target.q_min(b.observations, b.actions).squeeze()
            adv   = q_min - v
        losses.append(loss_v.item())
        v_means.append(v.mean().item())
        adv_means.append(adv.mean().item())
    return {"loss": losses, "v_mean": v_means, "adv_mean": adv_means}


datasize_results = {}

for frac in DATA_FRACTIONS:
    frac_key = f"{int(frac*100)}%"
    print(f"\n{'='*55}")
    print(f"Data fraction: {frac_key}  ({int(len(buf_full)*frac):,} transitions)")
    sub_buf = make_subsampled_buffer(buf_full, frac)
    datasize_results[frac_key] = {"quantum": [], "classical": []}

    print("  Quantum (via QuantumIQLTrainer):")
    for seed in SEEDS:
        print(f"    seed={seed}", end="  ")
        cfg = make_ablation_config(
            n_qubits=BASE_N_QUBITS, n_layers=BASE_N_LAYERS,
            n_steps=N_STEPS_DATASIZE, seed=seed,
            wandb_offline=True,
        )
        cfg.wandb_run_name = f"datasize_{frac_key}__seed{seed}"
        r = run_ablation_seed(sub_buf, cfg, n_steps=N_STEPS_DATASIZE, seed=seed)
        datasize_results[frac_key]["quantum"].append(r)
        print(f"final_loss={r['loss'][-1]:.4f}  mean_adv={np.mean(r['adv_mean'][-50:]):.3f}")

    print("  Classical (ValueNetwork baseline):")
    for seed in SEEDS:
        print(f"    seed={seed}", end="  ")
        r = run_classical_seed(sub_buf, seed=seed, n_steps=N_STEPS_DATASIZE)
        datasize_results[frac_key]["classical"].append(r)
        print(f"final_loss={r['loss'][-1]:.4f}  mean_adv={np.mean(r['adv_mean'][-50:]):.3f}")

save_json(
    {frac: {net: [{k: v[k] for k in ["loss","adv_mean"]} for v in seeds]
            for net, seeds in conds.items()}
     for frac, conds in datasize_results.items()},
    RESULTS_DIR / "ablation_datasize.json"
)


# ──────────────────────────────────────────────────────────────────────
# Cell 34
# ──────────────────────────────────────────────────────────────────────
# ── Dataset-size: learning curves ────────────────────────────────────────
steps_ds = np.arange(1, N_STEPS_DATASIZE + 1)
fracs     = list(datasize_results.keys())
cmap_ds   = plt.cm.viridis
frac_colors = {f: cmap_ds(i / max(len(fracs)-1, 1)) for i, f in enumerate(fracs)}

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
fig.suptitle(f"Dataset Size Sensitivity — Quantum vs Classical  ({PRIMARY_DS}, {len(SEEDS)} seeds)",
             fontsize=12, fontweight="bold")

for col, (net_label, ls) in enumerate([("quantum", "-"), ("classical", "--")]):
    ax = axes[col]
    for frac_key in fracs:
        arr = np.array([s["loss"] for s in datasize_results[frac_key][net_label]])
        mu, ci = arr.mean(0), ci95(arr)
        ax.plot(steps_ds, mu, color=frac_colors[frac_key], label=frac_key, lw=1.5, ls=ls)
        ax.fill_between(steps_ds, mu - ci, mu + ci, color=frac_colors[frac_key], alpha=0.15)
    ax.set_title(f"{net_label.capitalize()} V-network", fontsize=11)
    ax.set_xlabel("Step"); ax.set_ylabel("Expectile loss")
    ax.legend(title="Data fraction", fontsize=8)

plt.tight_layout()
fig.savefig(RESULTS_DIR / "ablation_datasize_curves.png", bbox_inches="tight", dpi=130)
plt.show()


# ──────────────────────────────────────────────────────────────────────
# Cell 35
# ──────────────────────────────────────────────────────────────────────
# ── Dataset-size: advantage signal comparison ─────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
x  = np.arange(len(fracs))
w  = 0.35
q_adv = [np.mean([np.mean(s["adv_mean"][-50:]) for s in datasize_results[f]["quantum"]]) for f in fracs]
c_adv = [np.mean([np.mean(s["adv_mean"][-50:]) for s in datasize_results[f]["classical"]]) for f in fracs]

ax.bar(x - w/2, q_adv, width=w, label="Quantum V",   color="#2166ac", edgecolor="white")
ax.bar(x + w/2, c_adv, width=w, label="Classical V", color="#d6604d", edgecolor="white")
ax.axhline(0, color="black", lw=0.8, ls="--")
ax.set_xticks(x); ax.set_xticklabels(fracs)
ax.set_xlabel("Dataset size (fraction of hopper-medium)")
ax.set_ylabel("mean(A = Q−V)  [last 50 steps]")
ax.set_title("Advantage signal vs dataset size\n(positive = actor receives useful signal)")
ax.legend()
plt.tight_layout()
fig.savefig(RESULTS_DIR / "ablation_datasize_advantage.png", bbox_inches="tight", dpi=130)
plt.show()
print("Saved → results/ablations/ablation_datasize_advantage.png")


# ---
# ## 6. Fourier Spectrum Characterisation
# 
# The DRU circuit implements a truncated Fourier series in the input features.
# Here we characterise the frequency components accessible to the quantum V-network
# as a function of circuit depth (n_layers) and qubit count (n_qubits).
# 
# **Method:**
# 1. Generate a 1D sweep of a single input feature across [-π, π]
# 2. Evaluate V(s) for each point with all other features fixed at 0
# 3. Take the FFT to extract frequency components
# 4. Track how the spectrum evolves over the course of training
# 
# This maps directly onto the Fourier expressibility analysis in the Q-IQL paper.


# ──────────────────────────────────────────────────────────────────────
# Cell 37
# ──────────────────────────────────────────────────────────────────────
N_FOURIER_POINTS = 256   # resolution of the input sweep
N_FOURIER_STEPS  = [0, N_STEPS_ABLATION // 4, N_STEPS_ABLATION // 2, N_STEPS_ABLATION]
FOURIER_FEATURE  = 0     # which input feature to sweep (0 = first arctan component)

def compute_fourier_spectrum(
    qvn:       FlexQuantumValueNetwork,
    obs_dim:   int,
    n_points:  int = N_FOURIER_POINTS,
    feature:   int = FOURIER_FEATURE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sweep feature `feature` over [-π, π], evaluate V, return (frequencies, magnitudes).
    All other features are held at 0.
    """
    xs = torch.zeros(n_points, obs_dim, device=DEVICE)
    xs[:, feature] = torch.linspace(-math.pi, math.pi, n_points, device=DEVICE)
    with torch.no_grad():
        ys = qvn(xs).cpu().numpy()   # (n_points,)
    freqs = np.fft.rfftfreq(n_points, d=1.0/n_points)
    mags  = np.abs(np.fft.rfft(ys))
    return freqs, mags


def fourier_ablation(
    buffer,
    n_qubits:     int,
    n_layers:     int,
    entanglement: str = BASE_ENTANGLEMENT,
    measurement:  str = BASE_MEASUREMENT,
    seed:         int = 0,
    n_steps:      int = N_STEPS_ABLATION,
    snapshot_steps: list = None,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Train QVN and capture Fourier spectrum at specified step checkpoints."""
    if snapshot_steps is None:
        snapshot_steps = N_FOURIER_STEPS

    qvn     = make_fqvn(buffer.obs_dim, n_qubits=n_qubits, n_layers=n_layers,
                        entanglement=entanglement, measurement=measurement, seed=seed)
    wrapped = _Wrap(qvn)
    critic        = make_critic(buffer.obs_dim, buffer.act_dim, seed=seed)
    critic_target = make_critic(buffer.obs_dim, buffer.act_dim, seed=seed)
    with torch.no_grad():
        for p, pt in zip(critic.parameters(), critic_target.parameters()):
            pt.data.copy_(p.data); pt.requires_grad_(False)

    opt_v = optim.Adam(qvn.parameters(), lr=BASE_LR_V)
    opt_q = optim.Adam(critic.parameters(), lr=BASE_LR_Q)
    snapshots = {}

    if 0 in snapshot_steps:
        snapshots[0] = compute_fourier_spectrum(qvn, buffer.obs_dim)

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
        if step in snapshot_steps:
            snapshots[step] = compute_fourier_spectrum(qvn, buffer.obs_dim)

    return snapshots


buf = buffers[PRIMARY_DS]
print("Computing Fourier spectrum: base config (n_qubits=4, n_layers=2)...")
fourier_base = fourier_ablation(buf, n_qubits=4, n_layers=2, seed=0)
print("Done.  Snapshots at steps:", sorted(fourier_base.keys()))


# ──────────────────────────────────────────────────────────────────────
# Cell 38
# ──────────────────────────────────────────────────────────────────────
# ── Fourier spectrum: evolution over training ─────────────────────────────
fig, axes = plt.subplots(1, len(fourier_base), figsize=(14, 4))
fig.suptitle("Fourier Spectrum of V(s) — base config (4 qubits, 2 layers, linear)\n"
             f"Feature {FOURIER_FEATURE} sweep; seed 0", fontsize=12, fontweight="bold")

cmap_f = plt.cm.plasma
for ax, (step, (freqs, mags)) in zip(axes, sorted(fourier_base.items())):
    color = cmap_f(step / max(N_FOURIER_STEPS[-1], 1))
    ax.bar(freqs[:N_FOURIER_POINTS//8], mags[:N_FOURIER_POINTS//8],
           width=0.4, color=color, edgecolor="none")
    ax.set_title(f"Step {step:,}", fontsize=10)
    ax.set_xlabel("Frequency")
    if ax is axes[0]:
        ax.set_ylabel("|FFT(V)|")

plt.tight_layout()
fig.savefig(RESULTS_DIR / "fourier_spectrum_evolution.png", bbox_inches="tight", dpi=130)
plt.show()
print("Saved → results/ablations/fourier_spectrum_evolution.png")


# ──────────────────────────────────────────────────────────────────────
# Cell 39
# ──────────────────────────────────────────────────────────────────────
# ── Fourier spectrum: n_layers comparison (at final step) ─────────────────
print("Computing Fourier spectra across n_layers (seed=0, final step only)...")
fourier_by_layers = {}
for n_layers in [1, 2, 3, 4]:
    print(f"  n_layers={n_layers}", end="  ")
    snaps = fourier_ablation(buf, n_qubits=4, n_layers=n_layers, seed=0,
                             snapshot_steps=[N_STEPS_ABLATION])
    fourier_by_layers[n_layers] = snaps[N_STEPS_ABLATION]
    print("done")

fig, ax = plt.subplots(figsize=(9, 4))
cmap_l = plt.cm.cool
ax.set_title("Fourier Spectrum vs n_layers  (4 qubits, final step, seed 0)", fontsize=11)
for i, (n_l, (freqs, mags)) in enumerate(fourier_by_layers.items()):
    color = cmap_l(i / max(len(fourier_by_layers)-1, 1))
    ax.plot(freqs[:N_FOURIER_POINTS//8], mags[:N_FOURIER_POINTS//8],
            color=color, label=f"n_layers={n_l}", lw=1.8)
ax.set_xlabel("Frequency"); ax.set_ylabel("|FFT(V)|")
ax.legend()
plt.tight_layout()
fig.savefig(RESULTS_DIR / "fourier_spectrum_by_layers.png", bbox_inches="tight", dpi=130)
plt.show()


# ---
# ## 7. Expressibility Analysis
# 
# Expressibility quantifies how uniformly a parametrised circuit can sample the Haar measure
# over the unitary group. We use the frame-potential fidelity metric from Sim et al. (2019):
# 
# $$\mathcal{E} = \left\| \hat{F}^{(t)} - F^{(t)}_{\text{Haar}} \right\|_2$$
# 
# Lower $\mathcal{E}$ = more expressive (closer to Haar random).
# 
# **Procedure (approximate):**
# 1. Sample N_SAMPLES random parameter sets uniformly from [0, 2π]
# 2. Compute the circuit output fidelity distribution: |⟨ψ(θ₁)|ψ(θ₂)⟩|²
# 3. Compare to the Haar fidelity CDF via KL-divergence (tractable classical proxy)


# ──────────────────────────────────────────────────────────────────────
# Cell 41
# ──────────────────────────────────────────────────────────────────────
N_EXPR_SAMPLES = 200    # number of random parameter pairs
N_EXPR_BINS    = 75     # histogram bins for fidelity distribution

def expressibility_metric(
    n_qubits:     int,
    n_layers:     int,
    entanglement: str = BASE_ENTANGLEMENT,
    n_samples:    int = N_EXPR_SAMPLES,
    seed:         int = 0,
) -> float:
    """
    Estimate expressibility as KL divergence between circuit fidelity distribution
    and the Haar fidelity distribution: P_Haar(F) = (2^n - 1)(1 - F)^(2^n - 2).
    Lower = more expressive.
    """
    np.random.seed(seed)
    dev = qml.device(QUANTUM_DEVICE, wires=n_qubits)

    @qml.qnode(dev, interface="numpy")
    def circuit_state(theta, w, xs):
        _apply_entanglement(n_qubits, entanglement)
        for layer in range(n_layers):
            for q in range(n_qubits):
                angles = theta[layer, q] + w[layer, q] * xs[q]
                qml.Rot(angles[0], angles[1], angles[2], wires=q)
            _apply_entanglement(n_qubits, entanglement)
        return qml.state()

    dim = 2 ** n_qubits
    fidelities = []
    for _ in range(n_samples):
        t1 = np.random.uniform(0, 2*np.pi, (n_layers, n_qubits, 3))
        w1 = np.random.uniform(0, 2*np.pi, (n_layers, n_qubits, 3))
        t2 = np.random.uniform(0, 2*np.pi, (n_layers, n_qubits, 3))
        w2 = np.random.uniform(0, 2*np.pi, (n_layers, n_qubits, 3))
        xs = np.random.uniform(-np.pi, np.pi, n_qubits)
        s1 = circuit_state(t1, w1, xs)
        s2 = circuit_state(t2, w2, xs)
        F  = abs(np.dot(s1.conj(), s2)) ** 2
        fidelities.append(float(F.real))

    # Haar CDF: P_Haar(F) = 1 - (1-F)^(dim-1)
    fid_arr = np.array(fidelities)
    bins    = np.linspace(0, 1, N_EXPR_BINS + 1)
    hist_c, _ = np.histogram(fid_arr, bins=bins, density=True)
    f_mid     = (bins[:-1] + bins[1:]) / 2
    p_haar    = (dim - 1) * (1 - f_mid) ** (dim - 2)
    p_haar   /= p_haar.sum()
    p_circ    = hist_c / (hist_c.sum() + 1e-12)
    # KL(circuit || Haar)
    kl = float(np.sum(np.where(p_circ > 0, p_circ * np.log(p_circ / (p_haar + 1e-12)), 0)))
    return kl


print("Computing expressibility across circuit configurations...")
print("(This runs N_EXPR_SAMPLES=200 random circuit evaluations per config — may take a few minutes)")

expr_configs = {
    "2q-1L-linear": dict(n_qubits=2, n_layers=1, entanglement="linear"),
    "2q-2L-linear": dict(n_qubits=2, n_layers=2, entanglement="linear"),
    "4q-1L-linear": dict(n_qubits=4, n_layers=1, entanglement="linear"),
    "4q-2L-linear": dict(n_qubits=4, n_layers=2, entanglement="linear"),   # base config
    "4q-2L-circular": dict(n_qubits=4, n_layers=2, entanglement="circular"),
    "4q-2L-all2all": dict(n_qubits=4, n_layers=2, entanglement="all_to_all"),
    "4q-4L-linear": dict(n_qubits=4, n_layers=4, entanglement="linear"),
    "6q-2L-linear": dict(n_qubits=6, n_layers=2, entanglement="linear"),
}

expr_results = {}
for cfg_label, kwargs in expr_configs.items():
    kl = expressibility_metric(**kwargs)
    expr_results[cfg_label] = kl
    print(f"  {cfg_label:<20}  KL(circuit||Haar) = {kl:.4f}")

save_json(expr_results, RESULTS_DIR / "expressibility.json")


# ──────────────────────────────────────────────────────────────────────
# Cell 42
# ──────────────────────────────────────────────────────────────────────
# ── Expressibility bar chart ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 4.5))
labels  = list(expr_results.keys())
kls     = list(expr_results.values())
colors  = ["#2166ac" if "base" in l or "4q-2L-linear" in l else "#aaaaaa" for l in labels]
colors  = ["#d01c8b" if "all2all" in l else c for l, c in zip(labels, colors)]
colors  = ["#4dac26" if "4L" in l or "6q" in l else c for l, c in zip(labels, colors)]

bars = ax.bar(labels, kls, color=colors, edgecolor="white")
ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("KL(circuit || Haar)  [lower = more expressive]")
ax.set_title("Expressibility across circuit configurations\n(base config highlighted in blue)", fontsize=11)
for bar, v in zip(bars, kls):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f"{v:.3f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
fig.savefig(RESULTS_DIR / "expressibility.png", bbox_inches="tight", dpi=130)
plt.show()
print("Saved → results/ablations/expressibility.png")


# ---
# ## 8. Ablation Summary
# 
# Consolidated results table and best-configuration derivation.


# ──────────────────────────────────────────────────────────────────────
# Cell 44
# ──────────────────────────────────────────────────────────────────────
print("=" * 100)
print("ABLATION STUDY — CONSOLIDATED SUMMARY")
print("=" * 100)

# ── 1. Layers ───────────────────────────────────────────────────────────────
print("\n§1 DRU LAYERS SWEEP")
summary_table(layers_results)

# ── 2. Topology ─────────────────────────────────────────────────────────────
print("\n§2 ENTANGLEMENT TOPOLOGY (hopper-medium)")
summary_table(topo_results[PRIMARY_DS])

# ── 3. Qubits ───────────────────────────────────────────────────────────────
print("\n§3 QUBIT COUNT SCALING")
summary_table(qubit_results)

# ── 4. Measurement ──────────────────────────────────────────────────────────
print("\n§4 MEASUREMENT SCHEME")
summary_table(meas_results)

# ── 5. Dataset size ─────────────────────────────────────────────────────────
print("\n§5 DATASET SIZE SENSITIVITY")
print(f"{'Fraction':<10} {'Q loss':>10} {'Q Ā':>10} {'C loss':>10} {'C Ā':>10}")
print("─" * 55)
for frac_key in DATA_FRACTIONS:
    fk = f"{int(frac_key*100)}%"
    def tm(net, key): return np.mean([np.mean(s[key][-50:]) for s in datasize_results[fk][net]])
    print(f"{fk:<10} {tm('quantum','loss'):>10.4f} {tm('quantum','adv_mean'):>10.3f} "
          f"{tm('classical','loss'):>10.4f} {tm('classical','adv_mean'):>10.3f}")

# ── 6. Expressibility ────────────────────────────────────────────────────────
print("\n§6/7 EXPRESSIBILITY (KL divergence from Haar — lower = more expressive)")
for cfg_label, kl in sorted(expr_results.items(), key=lambda x: x[1]):
    marker = " ← base config" if "4q-2L-linear" == cfg_label else ""
    print(f"  {cfg_label:<22}  {kl:.4f}{marker}")

print("\n" + "=" * 100)
print("All results serialised to results/ablations/")


# ──────────────────────────────────────────────────────────────────────
# Cell 45
# ──────────────────────────────────────────────────────────────────────
# ── Best configuration derivation ────────────────────────────────────────
print("\nDeriving best configuration from ablation results...")

def best_key(results_dict, metric="loss", n_tail=50, lower_better=True):
    scores = {}
    for k, seed_results in results_dict.items():
        scores[k] = np.mean([np.mean(s[metric][-n_tail:]) for s in seed_results])
    return min(scores, key=scores.get) if lower_better else max(scores, key=scores.get)

best_layers = best_key(layers_results, "loss")
best_topo   = best_key(topo_results[PRIMARY_DS], "adv_mean", lower_better=False)
best_qubits = best_key(qubit_results, "loss")
best_meas   = best_key(meas_results, "loss")

print(f"  Best n_layers:     {best_layers}")
print(f"  Best topology:     {best_topo}  (highest mean advantage signal)")
print(f"  Best n_qubits:     {best_qubits}")
print(f"  Best measurement:  {best_meas}")
print()
print("Optimal Q-IQL circuit config (from ablation study):")
print("  n_qubits     =", best_qubits.split("=")[1])
print("  n_layers     =", best_layers.split("=")[1])
print("  entanglement =", best_topo)
print("  measurement  =", best_meas)
print("  init         = identity-block  (from training_dynamics_final.ipynb)")
print("  optimizer    = Adam lr=1e-2    (from training_dynamics_final.ipynb)")
print("  tau          = 0.7             (from training_dynamics_final.ipynb)")


# ---
# ## Output Files
# 
# All results are saved to `results/ablations/`:
# 
# | File | Contents |
# |------|----------|
# | `ablation_layers.json` | Per-seed metrics for all n_layers conditions |
# | `ablation_layers.png` | Learning curves across DRU layers |
# | `ablation_layers_wallclock.png` | ms/step vs n_layers |
# | `ablation_topology.json` | Topology results for both datasets |
# | `ablation_topology_{ds}.png` | Learning curves per dataset |
# | `ablation_topology_radar.png` | Radar chart across topology × metrics |
# | `ablation_qubits.json` | Per-seed metrics for qubit count sweep |
# | `ablation_qubits.png` | Learning curves across qubit counts |
# | `ablation_qubits_cost.png` | Final loss + wall-clock vs qubit count |
# | `ablation_measurement.json` | Measurement scheme results |
# | `ablation_measurement.png` | Learning curves per measurement scheme |
# | `ablation_measurement_weights.png` | Learned measurement weight evolution |
# | `ablation_datasize.json` | Dataset fraction results |
# | `ablation_datasize_curves.png` | Learning curves vs data fraction |
# | `ablation_datasize_advantage.png` | Advantage signal bar chart |
# | `fourier_spectrum_evolution.png` | FFT of V(s) at training snapshots |
# | `fourier_spectrum_by_layers.png` | Frequency spectrum vs n_layers |
# | `expressibility.json` | KL(circuit || Haar) per configuration |
# | `expressibility.png` | Expressibility bar chart |

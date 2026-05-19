#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Shot Budget Analysis — Helios-1E (HQC-optimised)
=================================================
 
Budget: ~1,932 HQC  →  8 states × 3 repeats × 5 shot budgets
Formula: HQC/circuit = 5 + (N_1q + 10·N_2q + 5·N_m) × shots / 5000
         = 5 + 146 × shots / 5000   (4 qubits, 2 layers)
 
Per-batch HQC (24 circuits each):
  50 shots  →  155 HQC
  100 shots →  190 HQC
  250 shots →  295 HQC
  500 shots →  470 HQC
  1000 shots → 821 HQC
  TOTAL     → 1,932 HQC   (well within 2000 HQC default batch limit per budget)
 
Usage
-----
  # 1. Free sanity check — PennyLane, no HQC spent
  python shot_budget_helios.py --backend pennylane
 
  # 2. Free emulator — validates qnexus pipeline end-to-end
  python shot_budget_helios.py --backend selene
 
  # 3. Real hardware — spends ~1,932 HQC
  python shot_budget_helios.py --backend helios-1e
 
  # Dry-run: prints HQC estimate and exits
  python shot_budget_helios.py --backend helios-1e --dry-run
 
Angle conventions
-----------------
  PennyLane  : radians      [0, 2π]
  Quantinuum : half-turns   [0, 2]    →  angle_ht = angle_rad / π
"""
from __future__ import annotations
 
import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
 
import numpy as np
 
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))
 
import torch
 
print("[BOOT] Importing PennyLane...", flush=True)
import pennylane as qml
print("[BOOT] Importing matplotlib...", flush=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
print("[BOOT] Importing Qnexus...", flush=True)
import qnexus as qnx
print("[BOOT] All imports OK.", flush=True)
 
from quantum_iql.buffer import load_minari_dataset
from quantum_iql.quantum_value_network import QuantumValueNetwork
 
 
# ── Constants ────────────────────────────────────────────────────────────────
 
RESULTS_DIR = Path("results/shot_budget")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
 
# HQC-optimised budgets — skip 10/25 (too noisy to be useful) and 2000 (expensive)
SHOT_BUDGETS = [50, 100, 250, 500, 1000]
 
# HQC cost estimate per circuit: 5 + 146 × shots / 5000
# Adjust N_1q/N_2q/N_m if your compiled circuit differs
_N_1Q, _N_2Q, _N_M = 16, 9, 8
_GATE_COST = _N_1Q + 10 * _N_2Q + 5 * _N_M  # 146
 
def _hqc_per_circuit(shots: int) -> float:
    return 5.0 + _GATE_COST * shots / 5000.0
 
def _total_hqc_estimate(n_circuits_per_budget: int, budgets: list) -> float:
    return sum(n_circuits_per_budget * _hqc_per_circuit(s) for s in budgets)
 
 
# ── Backend configs ───────────────────────────────────────────────────────────
 
def _get_backend_config(backend: str):
    """Return the correct qnexus config object for each backend."""
    if backend == "selene":
        return qnx.SeleneConfig()
    elif backend == "selene+":
        return qnx.SelenePlusConfig()
    elif backend == "helios-1e":
        return qnx.QuantinuumConfig(device_name="Helios-1E")
    elif backend == "h2-emulator":
        return qnx.QuantinuumConfig(device_name="H2-Emulator")
    else:
        raise ValueError(f"Unknown hardware backend: {backend}")
 
 
# ── PennyLane exact circuit (analytical, no shots) ───────────────────────────
 
def _build_exact_circuit(n_qubits: int, n_layers: int):
    dev = qml.device("default.qubit", wires=n_qubits)
 
    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(theta, w, xs, active_layers):
        for q in range(0, n_qubits - 1, 2):
            qml.CZ(wires=[q, q + 1])
        for q in range(1, n_qubits - 1, 2):
            qml.CZ(wires=[q, q + 1])
        for layer_idx in range(active_layers):
            th = theta[layer_idx]
            ww = w[layer_idx]
            for q in range(n_qubits):
                angles = th[q] + ww[q] * xs[:, q % xs.shape[1]].unsqueeze(-1)
                qml.Rot(angles[:, 0], angles[:, 1], angles[:, 2], wires=q)
            for q in range(n_qubits - 1):
                qml.CZ(wires=[q, q + 1])
        return qml.expval(qml.PauliZ(0))
 
    return circuit
 
 
def _encode_state(value_net: QuantumValueNetwork, state: np.ndarray) -> torch.Tensor:
    s  = torch.from_numpy(state).float().unsqueeze(0)
    xs = torch.arctan((s - value_net.mu) / (value_net.sigma + 1e-8))
    n  = value_net.n_qubits
    if xs.shape[1] > n:
        xs = xs[:, :n]
    elif xs.shape[1] < n:
        xs = torch.cat([xs, torch.zeros(1, n - xs.shape[1])], dim=1)
    return xs
 
 
# ── Shot-noise simulation (PennyLane backend, free) ──────────────────────────
 
def _simulate_shot_noise(expval: float, n_shots: int, rng: np.random.Generator) -> float:
    """Binomial model: variance = (1 − ⟨Z⟩²) / n_shots (exact shot-noise scaling)."""
    p = np.clip((1.0 + expval) / 2.0, 0.0, 1.0)
    return 2.0 * rng.binomial(n_shots, p) / n_shots - 1.0
 
 
# ── Angle conversion: radians → half-turns for Quantinuum ────────────────────
 
def _make_qnexus_net(value_net: QuantumValueNetwork) -> QuantumValueNetwork:
    """Deep-copy with theta/w converted from radians to half-turns (÷π).
 
    PennyLane : radians     [0, 2π]
    Quantinuum: half-turns  [0, 2]   →  angle_ht = angle_rad / π
    a and b are dimensionless scalars — no conversion needed.
    """
    qnx_net = copy.deepcopy(value_net)
    with torch.no_grad():
        qnx_net.theta.copy_(value_net.theta / math.pi)
        qnx_net.w.copy_(value_net.w / math.pi)
    return qnx_net
 
 
# ── Qnexus batch submission ───────────────────────────────────────────────────
 
def _submit_batch(
    value_net: QuantumValueNetwork,
    qnx_net: QuantumValueNetwork,
    states: np.ndarray,
    n_repeats: int,
    shots: int,
    backend_config,
    project,
    max_cost: float,
) -> np.ndarray:
    """Submit all (n_states × n_repeats) circuits in one qnexus job.
 
    Returns v_noisy of shape (n_states, n_repeats).
    qnx_net has angles in half-turns; value_net.a/b used for V = a·⟨Z₀⟩ + b.
    """
    from hardware_utils import build_dru_hugr, compute_v_from_expvals
 
    n_states = states.shape[0]
    batch_circuits, mapping = [], []
 
    for si in range(n_states):
        for ri in range(n_repeats):
            hugr = build_dru_hugr(
                qnx_net, qnx_net.mu, qnx_net.sigma,
                states[si], shots=shots,
            )
            batch_circuits.append(hugr)
            mapping.append((si, ri))
 
    n_circuits = len(batch_circuits)
    hqc_est = n_circuits * _hqc_per_circuit(shots)
    print(f"    [qnexus] {n_circuits} circuits → {shots} shots "
          f"(est. {hqc_est:.0f} HQC, max_cost={max_cost:.0f})", flush=True)
 
    job = qnx.start_execute_job(
        circuits=batch_circuits,
        n_shots=shots,
        backend_config=backend_config,
        project=project,
        max_cost=max_cost,          # required for Helios; safety cap
    )
    print(f"    [qnexus] Job submitted (ID: {job}). Waiting...", flush=True)
    qnx.jobs.wait_for(job)
 
    print(f"    [qnexus] Done. Downloading results...", flush=True)
    job_results = qnx.jobs.results(job)
 
    v_noisy = np.zeros((n_states, n_repeats))
    for idx, res in enumerate(job_results):
        si, ri = mapping[idx]
        # compute_v_from_expvals only reads value_net.a and value_net.b (scalars, no units)
        v_noisy[si, ri] = compute_v_from_expvals(res, value_net)
 
    return v_noisy
 
 
# ── Core experiment ───────────────────────────────────────────────────────────
 
def run_experiment(
    value_net: QuantumValueNetwork,
    states: np.ndarray,
    n_repeats: int,
    shot_budgets: list,
    backend: str,
    rng: np.random.Generator,
    project=None,
) -> dict:
    n_states  = states.shape[0]
    n_budgets = len(shot_budgets)
 
    # ── Exact V(s) via PennyLane ─────────────────────────────────────────────
    print("  [exact] Computing V_exact(s)...", flush=True)
    value_net.eval()
    v_exact = np.zeros(n_states)
    with torch.no_grad():
        for i in range(n_states):
            s = torch.from_numpy(states[i]).float().unsqueeze(0)
            v_exact[i] = value_net(s).item()
    print(f"  [exact] V range: [{v_exact.min():.3f}, {v_exact.max():.3f}]", flush=True)
 
    v_noisy = np.zeros((n_budgets, n_states, n_repeats))
 
    # ── PennyLane: binomial shot-noise simulation (free) ─────────────────────
    if backend == "pennylane":
        print("  [pl] Computing raw ⟨Z₀⟩ for shot-noise simulation...", flush=True)
        circuit  = _build_exact_circuit(value_net.n_qubits, value_net.n_layers)
        z0_exact = np.zeros(n_states)
        with torch.no_grad():
            for i in range(n_states):
                xs = _encode_state(value_net, states[i])
                z0_exact[i] = circuit(
                    value_net.theta, value_net.w, xs, value_net._active_layers
                ).item()
 
        a = value_net.a.item()
        b = value_net.b.item()
 
        for bi, shots in enumerate(shot_budgets):
            t0 = time.perf_counter()
            for si in range(n_states):
                for ri in range(n_repeats):
                    v_noisy[bi, si, ri] = a * _simulate_shot_noise(z0_exact[si], shots, rng) + b
            mse = np.mean((v_noisy[bi] - v_exact[:, None]) ** 2)
            print(f"  shots={shots:>5}  MSE={mse:.6f}  ({time.perf_counter()-t0:.1f}s)", flush=True)
 
    # ── Hardware / emulator: qnexus batch submission ──────────────────────────
    else:
        backend_config = _get_backend_config(backend)
        qnx_net = _make_qnexus_net(value_net)   # angles in half-turns
 
        for bi, shots in enumerate(shot_budgets):
            t0 = time.perf_counter()
            print(f"\n  [budget {bi+1}/{n_budgets}] shots={shots}", flush=True)
 
            # max_cost = estimated HQC + 25% safety margin
            n_circuits = n_states * n_repeats
            max_cost   = math.ceil(n_circuits * _hqc_per_circuit(shots) * 1.25)
 
            v_noisy[bi] = _submit_batch(
                value_net, qnx_net, states, n_repeats,
                shots, backend_config, project, max_cost,
            )
            mse = np.mean((v_noisy[bi] - v_exact[:, None]) ** 2)
            print(f"  shots={shots:>5}  MSE={mse:.6f}  ({time.perf_counter()-t0:.1f}s total)", flush=True)
 
    return {
        "v_exact":      v_exact,
        "v_noisy":      v_noisy,
        "shot_budgets": np.array(shot_budgets),
    }
 
 
# ── Analysis and plotting ─────────────────────────────────────────────────────
 
def analyse_and_plot(results: dict, backend: str, n_states: int, n_repeats: int) -> dict:
    from scipy.stats import spearmanr
 
    v_exact = results["v_exact"]
    v_noisy = results["v_noisy"]
    budgets = results["shot_budgets"]
    n_budgets = len(budgets)
 
    mse = mae = std_v = rel_err = rank_corr = None
    mse      = np.zeros(n_budgets)
    mae      = np.zeros(n_budgets)
    std_v    = np.zeros(n_budgets)
    rel_err  = np.zeros(n_budgets)
    rank_corr = np.zeros(n_budgets)
 
    for bi in range(n_budgets):
        err           = v_noisy[bi] - v_exact[:, None]
        mse[bi]       = np.mean(err ** 2)
        mae[bi]       = np.mean(np.abs(err))
        std_v[bi]     = np.mean(np.std(v_noisy[bi], axis=1))
        rel_err[bi]   = np.mean(np.abs(err) / (np.abs(v_exact[:, None]) + 1e-8))
        corrs         = [spearmanr(v_exact, v_noisy[bi, :, ri]).statistic
                         for ri in range(n_repeats)]
        rank_corr[bi] = np.nanmean(corrs)
 
    # Figure 1: 4-panel diagnostic
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f"Shot Budget Analysis — {backend}\n"
                 f"({n_states} states × {n_repeats} repeats | HQC-optimised)",
                 fontsize=13, fontweight="bold")
 
    ax = axes[0, 0]
    ax.loglog(budgets, mse, "o-", color="#2166ac", lw=2, markersize=7)
    theory = mse[1] * budgets[1]
    ax.loglog(budgets, theory / budgets, "--", color="gray", alpha=0.6, label="O(1/N)")
    ax.set_xlabel("Shots"); ax.set_ylabel("MSE"); ax.set_title("MSE vs Shot Budget")
    ax.legend(); ax.grid(alpha=0.3, which="both")
 
    ax = axes[0, 1]
    ax.semilogx(budgets, mae, "s-", color="#d6604d", lw=2, markersize=7)
    ax.set_xlabel("Shots"); ax.set_ylabel("MAE"); ax.set_title("MAE vs Shot Budget")
    ax.grid(alpha=0.3, which="both")
    threshold = 0.10 * (v_exact.max() - v_exact.min())
    for bi, m in enumerate(mae):
        if m < threshold:
            ax.axvline(budgets[bi], color="#4daf4a", ls="--", alpha=0.5)
            ax.annotate(f"Sweet spot:\n{budgets[bi]} shots",
                        xy=(budgets[bi], m), fontsize=9, color="#4daf4a",
                        xytext=(budgets[bi] * 1.5, m * 1.6),
                        arrowprops=dict(arrowstyle="->", color="#4daf4a"))
            break
 
    ax = axes[1, 0]
    ax.semilogx(budgets, rank_corr, "D-", color="#4daf4a", lw=2, markersize=7)
    ax.axhline(0.95, color="gray", ls="--", alpha=0.5, label="ρ = 0.95")
    ax.set_xlabel("Shots"); ax.set_ylabel("Spearman ρ")
    ax.set_title("State Ranking Preservation"); ax.set_ylim(0, 1.05)
    ax.legend(); ax.grid(alpha=0.3, which="both")
    for bi, rho in enumerate(rank_corr):
        if rho >= 0.95:
            ax.annotate(f"{budgets[bi]} shots → ρ={rho:.3f}",
                        xy=(budgets[bi], rho), fontsize=9, color="#4daf4a",
                        xytext=(budgets[bi] * 1.5, rho - 0.1),
                        arrowprops=dict(arrowstyle="->", color="#4daf4a"))
            break
 
    ax = axes[1, 1]
    ax.loglog(budgets, std_v, "^-", color="#984ea3", lw=2, markersize=7)
    ax.set_xlabel("Shots"); ax.set_ylabel("Avg σ(V)"); ax.set_title("Noise Std Dev")
    ax.grid(alpha=0.3, which="both")
 
    plt.tight_layout()
    p = RESULTS_DIR / f"shot_budget_{backend}.png"
    fig.savefig(p, bbox_inches="tight", dpi=150); plt.close()
    print(f"  [fig] {p}")
 
    # Figure 2: scatter plots
    examples = [b for b in [100, 250, 500, 1000] if b in budgets.tolist()]
    if examples:
        fig2, axes2 = plt.subplots(1, len(examples), figsize=(4.5 * len(examples), 4.5))
        if len(examples) == 1: axes2 = [axes2]
        fig2.suptitle(f"V(s): Exact vs Noisy — {backend}", fontsize=12, fontweight="bold")
        for ax2, shots in zip(axes2, examples):
            bi = budgets.tolist().index(shots)
            for ri in range(n_repeats):
                ax2.scatter(v_exact, v_noisy[bi, :, ri], c="#2166ac",
                            alpha=0.25, s=20, edgecolor="none")
            v_mean = v_noisy[bi].mean(axis=1)
            ax2.scatter(v_exact, v_mean, c="#d6604d", s=50,
                        edgecolor="white", zorder=5, label="mean")
            lo = min(v_exact.min(), v_noisy[bi].min()) - 0.05
            hi = max(v_exact.max(), v_noisy[bi].max()) + 0.05
            ax2.plot([lo, hi], [lo, hi], "--", color="gray", alpha=0.5)
            ax2.set_xlabel("V_exact(s)"); ax2.set_ylabel("V_noisy(s)")
            ax2.set_title(f"{shots} shots  MAE={mae[bi]:.4f}  ρ={rank_corr[bi]:.3f}")
            ax2.grid(alpha=0.3)
        plt.tight_layout()
        p2 = RESULTS_DIR / f"shot_budget_scatter_{backend}.png"
        fig2.savefig(p2, bbox_inches="tight", dpi=150); plt.close()
        print(f"  [fig] {p2}")
 
    # Figure 3: HQC cost vs accuracy
    fig3, ax3 = plt.subplots(figsize=(9, 5))
    hqc_per_budget = np.array([n_states * n_repeats * _hqc_per_circuit(s) for s in budgets])
    ax3.plot(hqc_per_budget, mae, "o-", color="#2166ac", lw=2, markersize=8)
    ax3.set_xlabel("HQC consumed per shot budget")
    ax3.set_ylabel("Mean Absolute Error")
    ax3.set_title(f"HQC Cost vs Accuracy — {n_states} states × {n_repeats} repeats", fontsize=11)
    for bi in range(n_budgets):
        ax3.annotate(f"{budgets[bi]} shots\n{hqc_per_budget[bi]:.0f} HQC",
                     xy=(hqc_per_budget[bi], mae[bi]), fontsize=8, color="gray",
                     textcoords="offset points", xytext=(5, 5))
    ax3.grid(alpha=0.3); plt.tight_layout()
    p3 = RESULTS_DIR / f"shot_budget_hqc_{backend}.png"
    fig3.savefig(p3, bbox_inches="tight", dpi=150); plt.close()
    print(f"  [fig] {p3}")
 
    # Console summary
    print(f"\n  {'='*65}")
    print(f"  Shot Budget Summary — {backend}")
    print(f"  {'='*65}")
    print(f"  {'Shots':>6} {'HQC':>7} {'MSE':>10} {'MAE':>10} {'Rel.Err':>9} {'ρ':>7} {'σ(V)':>7}")
    print(f"  {'-'*60}")
    for bi in range(n_budgets):
        hqc = n_states * n_repeats * _hqc_per_circuit(budgets[bi])
        star = (" ★" if rank_corr[bi] >= 0.95 and (bi == 0 or rank_corr[bi-1] < 0.95) else "")
        print(f"  {budgets[bi]:>6} {hqc:>7.0f} {mse[bi]:>10.6f} {mae[bi]:>10.4f} "
              f"{rel_err[bi]:>9.2%} {rank_corr[bi]:>7.3f} {std_v[bi]:>7.4f}{star}")
 
    for bi in range(n_budgets):
        if rank_corr[bi] >= 0.95:
            hqc = n_states * n_repeats * _hqc_per_circuit(budgets[bi])
            print(f"\n  → Minimum viable budget: {budgets[bi]} shots "
                  f"(ρ ≥ 0.95, MAE={mae[bi]:.4f}, {hqc:.0f} HQC per batch)")
            break
    else:
        print("\n  → No budget achieves ρ ≥ 0.95 — increase shots or circuit depth")
 
    return {
        "shot_budgets":     budgets.tolist(),
        "mse":              mse.tolist(),
        "mae":              mae.tolist(),
        "relative_error":   rel_err.tolist(),
        "rank_correlation": rank_corr.tolist(),
        "noise_std":        std_v.tolist(),
        "hqc_per_budget":   [n_states * n_repeats * _hqc_per_circuit(s) for s in budgets],
    }
 
 
# ── CLI ───────────────────────────────────────────────────────────────────────
 
def parse_args():
    p = argparse.ArgumentParser(description="Shot Budget Analysis — HQC-optimised for Helios-1E")
    p.add_argument("--backend", default="pennylane",
                   choices=["pennylane", "selene", "selene+", "h2-emulator", "helios-1e"])
    p.add_argument("--states",  type=int, default=8,
                   help="States to evaluate (default 8 → ~1,932 HQC total)")
    p.add_argument("--repeats", type=int, default=3,
                   help="Repeats per state per budget (default 3)")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--dry-run", action="store_true",
                   help="Print HQC estimate and exit without running")
    return p.parse_args()
 
 
def main():
    args = parse_args()
    rng  = np.random.default_rng(args.seed)
 
    n_circuits = args.states * args.repeats
    total_hqc  = _total_hqc_estimate(n_circuits, SHOT_BUDGETS)
 
    print(f"\n{'='*65}")
    print(f"  Shot Budget Analysis — {args.backend}")
    print(f"  States   : {args.states}")
    print(f"  Repeats  : {args.repeats} per state per budget")
    print(f"  Budgets  : {SHOT_BUDGETS}")
    print(f"  Circuits : {n_circuits} per budget  ({n_circuits * len(SHOT_BUDGETS)} total)")
    print(f"{'─'*65}")
    for shots in SHOT_BUDGETS:
        print(f"  {shots:>5} shots  →  {n_circuits * _hqc_per_circuit(shots):>6.0f} HQC per batch")
    print(f"{'─'*65}")
    print(f"  TOTAL ESTIMATED: {total_hqc:.0f} HQC")
    print(f"{'='*65}\n")
 
    if args.dry_run:
        print("Dry run — exiting without spending HQC.")
        return
 
    # ── Load or initialise model ──────────────────────────────────────────────
    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"  Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        net  = QuantumValueNetwork(
            n_qubits=ckpt.get("n_qubits", 4),
            n_layers=ckpt.get("n_layers", 2),
            obs_dim =ckpt.get("obs_dim",  11),
        )
        net.load_state_dict(ckpt["value_net"])
    else:
        print("  No checkpoint — random init in radians [0, 2π]")
        net = QuantumValueNetwork(n_qubits=4, n_layers=2, obs_dim=11)
        with torch.no_grad():
            net.theta.uniform_(0, 2 * math.pi)   # radians for PennyLane
            net.w.uniform_(0, 2 * math.pi)        # converted to half-turns for Quantinuum
            net.a.fill_(2.5)
            net.b.fill_(-0.8)
    net.eval()
 
    # ── Load states from dataset ──────────────────────────────────────────────
    print("  Loading dataset...", flush=True)
    buffer  = load_minari_dataset("mujoco/hopper/medium-v0", device="cpu")
    indices = rng.choice(len(buffer), size=args.states, replace=False)
    states  = buffer._observations[indices].astype(np.float32)
 
    with torch.no_grad():
        all_obs = torch.from_numpy(buffer._observations[:10_000].astype(np.float32))
        net.mu.copy_(all_obs.mean(0))
        net.sigma.copy_(all_obs.std(0).clamp(min=1e-6))
 
    print(f"  States loaded: {states.shape}", flush=True)
 
    # ── Qnexus project (required for hardware backends) ───────────────────────
    project = None
    if args.backend != "pennylane":
        project = qnx.projects.get_or_create(name="shot-budget-analysis")
        print(f"  Qnexus project: {project}", flush=True)
 
    # ── Run ───────────────────────────────────────────────────────────────────
    t0      = time.perf_counter()
    results = run_experiment(
        net, states, args.repeats, SHOT_BUDGETS, args.backend, rng, project
    )
    runtime = time.perf_counter() - t0
    print(f"\n  Total runtime: {runtime:.1f}s")
 
    # ── Analyse & plot ────────────────────────────────────────────────────────
    metrics = analyse_and_plot(results, args.backend, args.states, args.repeats)
 
    # ── Save ──────────────────────────────────────────────────────────────────
    out = {
        "backend":        args.backend,
        "n_states":       args.states,
        "n_repeats":      args.repeats,
        "shot_budgets":   SHOT_BUDGETS,
        "hqc_estimated":  total_hqc,
        "runtime_s":      runtime,
        "metrics":        metrics,
    }
    json_path = RESULTS_DIR / f"shot_budget_{args.backend}.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Results → {json_path}")
 
    np.savez(
        RESULTS_DIR / f"shot_budget_{args.backend}_raw.npz",
        v_exact      = results["v_exact"],
        v_noisy      = results["v_noisy"],
        shot_budgets = results["shot_budgets"],
    )
    print(f"  Raw data → {RESULTS_DIR}/shot_budget_{args.backend}_raw.npz")
    print(f"\n  All outputs in {RESULTS_DIR}/")
 
 
if __name__ == "__main__":
    main()
 
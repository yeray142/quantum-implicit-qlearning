"""Phase 1 — Helios emulator evaluation.

Evaluates V(s), Q(s,a_policy), and advantage rank preservation on the
helios_emulator backend, computing MSE/Pearson vs the Phase 0 reference.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from quantum_iql import (
    ActorNetwork,
    CriticNetwork,
    QuantumValueNetwork,
)
from src.eval.hqc_tracker import HQCTracker
from src.eval.metrics import (
    evaluate_v_metrics,
    kendall_tau,
)
from src.eval.run_quantum_batch import run_quantum_batch
from src.eval.strata import HopperStrata, select_stratified

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def load_eval_checkpoint(
    checkpoint_path: str | Path,
    obs_dim: int = 11,
    act_dim: int = 3,
    n_qubits: int = 8,
    n_layers: int = 3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Load value/actor/critic networks from checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device)

    multi_qubit_readout = ckpt["value_net"]["a"].shape[0] > 1
    value_net = QuantumValueNetwork(
        n_qubits=n_qubits,
        n_layers=n_layers,
        obs_dim=obs_dim,
        device_name="default.qubit",
        diff_method="backprop",
        running_stats=True,
        use_pre_encoder=True,
        multi_qubit_readout=multi_qubit_readout,
    ).to(device)

    actor = ActorNetwork(obs_dim=obs_dim, act_dim=act_dim).to(device)
    critic = CriticNetwork(obs_dim=obs_dim, act_dim=act_dim, use_twin=True).to(device)

    value_net.load_state_dict(ckpt["value_net"], strict=False)
    actor.load_state_dict(ckpt["actor_net"])
    critic.load_state_dict(ckpt["critic_net"])

    if "mu" in ckpt and "sigma" in ckpt:
        value_net.update_running_stats(
            torch.as_tensor(ckpt["mu"], device=device),
            torch.as_tensor(ckpt["sigma"], device=device),
        )

    value_net.eval()
    actor.eval()
    critic.eval()

    return value_net, actor, critic


def run_phase1(
    pool_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    hqc_budget: float,
    tracker_path: str | Path,
    n_phase1: int = 250,
    n_qubits: int = 8,
    n_layers: int = 3,
    shots: int = 100,
    seed: int = 42,
    dry_run: bool = False,
    project_name: str = "Test",
    system_name_emulator: str = "Helios-1E",
) -> dict:
    """Run Phase 1: emulator evaluation on a stratified state subset."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pool
    pool = dict(np.load(pool_path, allow_pickle=True))
    log.info(f"Loaded pool: {len(pool['observations'])} transitions from {pool_path}")

    # Build strata and select states
    strata = HopperStrata()
    selected_indices = select_stratified(pool, n_phase1, strata, seed=seed)
    log.info(f"Selected {len(selected_indices)} stratified states")

    selected_obs = pool["observations"][selected_indices].astype(np.float32)
    v_sim = pool["v_sim"][selected_indices].astype(np.float32)

    # Load networks
    value_net, actor, critic = load_eval_checkpoint(
        checkpoint_path, n_qubits=n_qubits, n_layers=n_layers, device=device,
    )
    mu = value_net.mu
    sigma = value_net.sigma

    # ── 1a. V(s) on emulator — split into batches of ~10 states ───────────────
    emu_backend = "default.qubit" if dry_run else "helios_emulator"
    log.info(f"Phase 1a: V(s) evaluation on {emu_backend}"
             + (" (DRY RUN)" if dry_run else ""))

    tracker = HQCTracker.load(
        tracker_path,
        phase="phase1",
        budget=hqc_budget,
        git_sha=os.environ.get("GIT_SHA", ""),
        config_hash="",
    )

    states_per_batch = 10
    all_v_emu = []
    total_hqc = 0.0

    for batch_start in range(0, n_phase1, states_per_batch):
        batch_end = min(batch_start + states_per_batch, n_phase1)
        batch_obs = selected_obs[batch_start:batch_end]
        batch_indices = selected_indices[batch_start:batch_end]
        batch_id = batch_start // states_per_batch

        batch_result = run_quantum_batch(
            states=batch_obs,
            value_net=value_net,
            mu=mu, sigma=sigma,
            backend=emu_backend,
            shots=shots,
            tracker=tracker,
            batch_id=batch_id,
            states_indices=batch_indices.tolist(),
            results_dir=output_dir,
            active_layers=n_layers,
            dry_run=dry_run,
            project_name=project_name,
            system_name_emulator=system_name_emulator,
        )
        all_v_emu.extend(batch_result.v_values)
        total_hqc += batch_result.hqc_consumed

    v_emu = np.array(all_v_emu, dtype=np.float32)

    # ── 1b. V noiseless reference ─────────────────────────────────────────────
    log.info("Phase 1b: V(s) noiseless reference (default.qubit, shots=100)")
    v_ref_results = run_quantum_batch(
        states=selected_obs,
        value_net=value_net,
        mu=mu, sigma=sigma,
        backend="default.qubit",
        shots=shots,
        batch_id=99,
        states_indices=selected_indices.tolist(),
        results_dir=output_dir,
        active_layers=n_layers,
    )
    v_noiseless = np.array(v_ref_results.v_values, dtype=np.float32)

    # ── 1c. Q(s, a_policy) ───────────────────────────────────────────────────
    log.info("Phase 1c: Q(s, a_policy) using classical CriticNetwork")
    with torch.no_grad():
        obs_t = torch.from_numpy(selected_obs).to(device)
        a_policy_t = actor.get_action(obs_t, deterministic=True)
        q_policy = critic(obs_t, a_policy_t)[0].cpu().numpy().flatten().astype(np.float32)

    # ── 1d. Cross-state V-ranking ─────────────────────────────────────────────
    # Tests whether the emulator preserves the ordering of states by value —
    # the IQL-relevant property. V(s) ranking is what matters for IQL's
    # expectile regression, not within-state advantage ranking.
    log.info("Phase 1d: Cross-state V-ranking (kendall_tau over all %d states)", n_phase1)
    kendall = kendall_tau(v_sim, v_emu)

    # ── Compute metrics ──────────────────────────────────────────────────────
    v_metrics = evaluate_v_metrics(v_sim, v_emu, n_bootstrap=500, seed=seed)
    v_noiseless_metrics = evaluate_v_metrics(v_sim, v_noiseless, n_bootstrap=500, seed=seed)

    # Per-stratum breakdown
    from src.eval.metrics import stratum_breakdown
    stratum_rows = stratum_breakdown(pool, selected_indices, v_sim, v_emu, strata)

    metrics = {
        "phase": "phase1_emulator",
        "n_states": n_phase1,
        "shots": shots,
        "hqc_consumed": total_hqc,
        "v_mse": v_metrics["mse"].to_dict(),
        "v_mae": v_metrics["mae"].to_dict(),
        "v_pearson": v_metrics["pearson"].to_dict(),
        "v_noiseless_mse": v_noiseless_metrics["mse"].to_dict(),
        "v_noiseless_mae": v_noiseless_metrics["mae"].to_dict(),
        "v_noiseless_pearson": v_noiseless_metrics["pearson"].to_dict(),
        "kendall_tau_v_ranking": {
            "value": kendall,
            "n_states": n_phase1,
        },
        "q_policy_mean": float(q_policy.mean()),
        "stratum_breakdown": stratum_rows,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Save metrics
    metrics_path = output_dir / "phase1_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Metrics saved to {metrics_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Helios emulator evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--pool", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="Route helios/helios_emulator to default.qubit, track HQC as if real")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_phase1(
        pool_path=args.pool,
        checkpoint_path=args.checkpoint,
        output_dir=cfg["phase1"]["output_dir"],
        hqc_budget=cfg["budget_phase1"],
        tracker_path="eval_pools/hqc_tracker.json",
        n_phase1=cfg["strata"]["phase1_count"],
        n_qubits=cfg["n_qubits"],
        n_layers=cfg["n_layers"],
        shots=cfg["shots"],
        dry_run=args.dry_run,
        project_name=cfg.get("project_name", "Test"),
        system_name_emulator=cfg.get("system_name_emulator", "Helios-1E"),
    )


if __name__ == "__main__":
    main()
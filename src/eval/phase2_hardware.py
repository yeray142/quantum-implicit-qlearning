"""Phase 2 — Real Helios hardware evaluation.

Thin caller: selects a matched subset stratified across ALL Phase 1 batches,
calls run_quantum_batch(backend="helios"), computes MAE vs emulator.
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

from quantum_iql import QuantumValueNetwork
from src.eval.hqc_tracker import HQCTracker
from src.eval.metrics import mae
from src.eval.run_quantum_batch import load_batch_results, run_quantum_batch
from src.eval.strata import HopperStrata, select_stratified, z_height_from_obs

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def load_networks(
    checkpoint_path: Path,
    n_qubits: int = 8,
    n_layers: int = 3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    ckpt = torch.load(checkpoint_path, map_location=device)
    multi_qubit_readout = ckpt["value_net"]["a"].shape[0] > 1
    value_net = QuantumValueNetwork(
        n_qubits=n_qubits,
        n_layers=n_layers,
        obs_dim=11,
        device_name="default.qubit",
        diff_method="backprop",
        running_stats=True,
        use_pre_encoder=True,
        multi_qubit_readout=multi_qubit_readout,
    ).to(device)
    value_net.load_state_dict(ckpt["value_net"], strict=False)
    if "mu" in ckpt and "sigma" in ckpt:
        value_net.update_running_stats(
            torch.as_tensor(ckpt["mu"], device=device),
            torch.as_tensor(ckpt["sigma"], device=device),
        )
    value_net.eval()
    return value_net


def run_phase2(
    pool_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    hqc_budget: float,
    tracker_path: str | Path,
    phase1_output_dir: str | Path,
    n_phase2: int = 35,
    n_qubits: int = 8,
    n_layers: int = 3,
    shots: int = 100,
    seed: int = 42,
    dry_run: bool = False,
    dry_run_qnexus: bool = False,
    project_name: str = "Test",
    system_name_hardware: str = "Helios-1",
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pool = dict(np.load(pool_path, allow_pickle=True))
    log.info(f"Loaded pool: {len(pool['observations'])} transitions from {pool_path}")

    # Load ALL Phase 1 batches (not just batch 0) to find matched subset
    phase1_dir = Path(phase1_output_dir)
    all_batches = load_batch_results(phase1_dir)

    if all_batches:
        # Build full index-to-V mapping across all Phase 1 batches
        all_states_indices = []
        all_v_emu = []
        for batch in all_batches:
            all_states_indices.extend(batch["states_indices"])
            all_v_emu.extend(batch["v_values"])

        # Re-stratify across Phase 1 states to ensure Phase 2 matches Phase 1 distribution.
        # Weights are proportional to Phase 1 cell counts so Phase 2 is a representative subset.
        strata = HopperStrata()
        all_states_arr = np.array(all_states_indices, dtype=int)
        weights = np.ones(len(all_states_arr), dtype=float)
        # Proportional sampling: weight ∝ cell population in Phase 1
        cell_counts = np.zeros(strata.n_cells)
        for idx in all_states_indices:
            z_h = z_height_from_obs(pool["observations"][idx])
            cell = strata.classify(int(pool["timesteps"][idx]), z_h)
            cell_counts[cell] += 1
        for i, idx in enumerate(all_states_indices):
            z_h = z_height_from_obs(pool["observations"][idx])
            cell = strata.classify(int(pool["timesteps"][idx]), z_h)
            weights[i] = 1.0 / (cell_counts[cell] + 1e-8)
        weights /= weights.sum()

        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(all_states_arr), size=n_phase2, replace=False, p=weights)
        matched_pool_indices = all_states_arr[chosen].tolist()
    else:
        # Fallback: generate fresh selection
        strata = HopperStrata()
        matched_pool_indices = select_stratified(pool, n_phase2, strata, seed=seed).tolist()

    log.info(f"Phase 2: evaluating {len(matched_pool_indices)} states "
             + ("(DRY RUN — default.qubit)" if dry_run else "on Helios"))

    value_net = load_networks(Path(checkpoint_path), n_qubits, n_layers, device)
    mu = value_net.mu
    sigma = value_net.sigma

    # Build emulator V lookup from Phase 1 results
    v_emu_lookup: dict[int, float] = {}
    for batch in all_batches:
        for idx, v in zip(batch["states_indices"], batch["v_values"]):
            v_emu_lookup[idx] = v

    # Verify all matched states have emulator results (fail loudly)
    missing = [idx for idx in matched_pool_indices if idx not in v_emu_lookup]
    if missing:
        raise RuntimeError(
            f"Phase 2 matched {len(matched_pool_indices)} states but "
            f"{len(missing)} have no emulator result: {missing[:10]}"
        )

    # Load matched observations
    matched_obs = np.array(
        [pool["observations"][i].astype(np.float32) for i in matched_pool_indices]
    )

    tracker = HQCTracker.load(
        tracker_path,
        phase="phase2",
        budget=hqc_budget,
        git_sha=os.environ.get("GIT_SHA", ""),
        config_hash="",
    )

    # In dry_run_qnexus mode, mock qnexus so the real circuit path is exercised.
    # In dry_run mode, route to default.qubit for execution without qnexus.
    # In normal mode, use real Helios hardware.
    from src.eval.run_quantum_batch import mock_qnexus

    if dry_run_qnexus:
        # Mock qnexus end-to-end so the real circuit path is exercised without HQC spend.
        # Pass matched_obs so the mock can run default.qubit per-state to get real expvals.
        with mock_qnexus(value_net, mu, sigma, shots=shots, seed=seed, states=matched_obs) as qnx_mock:
            helios_result = run_quantum_batch(
                states=matched_obs,
                value_net=value_net,
                mu=mu, sigma=sigma,
                backend="helios",
                shots=shots,
                tracker=tracker,
                batch_id=0,
                states_indices=matched_pool_indices,
                results_dir=output_dir,
                active_layers=n_layers,
                dry_run=False,  # qnexus already mocked; don't double-route
                project_name=project_name,
                system_name_hardware=system_name_hardware,
            )
    else:
        helios_result = run_quantum_batch(
            states=matched_obs,
            value_net=value_net,
            mu=mu, sigma=sigma,
            backend="helios",
            shots=shots,
            tracker=tracker,
            batch_id=0,
            states_indices=matched_pool_indices,
            results_dir=output_dir,
            active_layers=n_layers,
            dry_run=dry_run,
            project_name=project_name,
            system_name_hardware=system_name_hardware,
        )

    v_helios = np.array(helios_result.v_values, dtype=np.float32)
    v_emu = np.array([v_emu_lookup[i] for i in matched_pool_indices], dtype=np.float32)

    # Noiseless reference already in BatchResult.v_raw_expval from default.qubit backend
    v_noiseless = np.array([
        float(np.mean(arr)) if len(arr) > 0 else 0.0
        for arr in helios_result.v_raw_expval
    ], dtype=np.float32) if dry_run else np.array([])

    metrics = {
        "phase": "phase2_hardware",
        "n_states": len(matched_pool_indices),
        "shots": shots,
        "hqc_consumed": helios_result.hqc_consumed,
        "mae_vs_emulator": float(mae(v_emu, v_helios)),
        "v_helios": v_helios.tolist(),
        "v_emu": v_emu.tolist(),
        "per_state_mae": np.abs(v_emu - v_helios).tolist(),
        "dry_run": dry_run,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if not dry_run and not dry_run_qnexus:
        metrics["mae_vs_noiseless"] = float(mae(v_noiseless, v_helios))
        metrics["v_noiseless"] = v_noiseless.tolist()

    with open(output_dir / "phase2_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    log.info(f"Phase 2 complete: mae_vs_emulator={metrics['mae_vs_emulator']:.4f}, "
             f"hqc_consumed={helios_result.hqc_consumed:.1f}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Real Helios evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--pool", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--phase1-dir", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="Route helios/helios_emulator to default.qubit, track HQC as if real")
    parser.add_argument("--dry-run-qnexus", action="store_true",
                        help="Mock qnexus client but exercise real circuit path (no HQC, end-to-end test)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_phase2(
        pool_path=args.pool,
        checkpoint_path=args.checkpoint,
        output_dir=cfg["phase2"]["output_dir"],
        hqc_budget=cfg["budget_phase2"],
        tracker_path="eval_pools/hqc_tracker_phase2_dryrun.json"
                    if args.dry_run or args.dry_run_qnexus
                    else "eval_pools/hqc_tracker.json",
        phase1_output_dir=args.phase1_dir,
        n_phase2=cfg["strata"]["phase2_count"],
        n_qubits=cfg["n_qubits"],
        n_layers=cfg["n_layers"],
        shots=cfg["shots"],
        dry_run=args.dry_run,
        dry_run_qnexus=args.dry_run_qnexus,
        project_name=cfg.get("project_name", "Test"),
        system_name_hardware=cfg.get("system_name_hardware", "Helios-1"),
    )


if __name__ == "__main__":
    main()
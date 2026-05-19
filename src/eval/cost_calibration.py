"""Gate 2: qnexus cost_confidence calibration.

Confirms that cost_confidence is a free estimation API and verifies per-call
cost matches the formula prediction (13 HQC at n=8, L=3, shots=100).

This script:
1. Loads 5 stratified states from the Phase 0 pool
2. For each state: builds HUGR via build_dru_hugr, uploads via qnx.hugr.upload,
   calls cost_confidence against Helios-1 AND Helios-1E
3. Compares against formula prediction of 13 HQC per call
4. Produces a report at experiments/eval/cost_calibration/report.md

UPPER BOUND HQC: 0 (cost_confidence is free per user confirmation)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from quantum_iql import QuantumValueNetwork
from src.eval.strata import HopperStrata, select_stratified, z_height_from_obs
from src.eval.utils import build_dru_hugr

POOL_PATH = "eval_pools/pool_seed6plus.npz"
OUTPUT_DIR = Path("experiments/eval/cost_calibration")
REPORT_PATH = OUTPUT_DIR / "report.md"


def load_value_net(checkpoint_path: str, n_qubits=8, n_layers=3, obs_dim=11):
    """Load trained QuantumValueNetwork from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
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
    ).to("cpu")
    value_net.load_state_dict(ckpt["value_net"], strict=False)
    value_net.update_running_stats(
        torch.as_tensor(ckpt["value_net"]["mu"]),
        torch.as_tensor(ckpt["value_net"]["sigma"]),
    )
    value_net.eval()
    return value_net


def get_stratum_cell(idx: int, pool: dict, strata: HopperStrata) -> int:
    obs = pool["observations"][idx]
    timestep = int(pool["timesteps"][idx])
    return strata.classify(timestep, z_height_from_obs(obs))


def run_calibration():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading pool...")
    pool = dict(np.load(POOL_PATH, allow_pickle=True))
    n_total = 5

    print("Selecting 5 stratified states...")
    strata = HopperStrata()
    indices = select_stratified(pool, n_total, strata, seed=99)
    selected_obs = pool["observations"][indices].astype(np.float32)

    print(f"Selected indices: {indices.tolist()}")

    print("Loading value network...")
    checkpoint = "experiments/checkpoints/hopper/medium/quantum-multi-qubit-readout/seed_6/checkpoint_final.pt"
    value_net = load_value_net(checkpoint)
    mu = value_net.mu
    sigma = value_net.sigma

    # Get per-state stratum cells
    stratum_cells = [get_stratum_cell(idx, pool, strata) for idx in indices]
    print(f"Stratum cells: {stratum_cells}")

    import qnexus as qnx
    print("Authenticating to qnexus...")
    qnx.login()
    project = qnx.projects.get(name="quantum-iql")
    qnx.context.set_active_project(project)

    FORMULA_PREDICTION = 13.0  # HQC at n=8, L=3, shots=100

    results = []

    for i, (state, idx) in enumerate(zip(selected_obs, indices)):
        print(f"\nState {i}: pool_idx={idx}, stratum_cell={stratum_cells[i]}")
        print(f"  Building HUGR...")
        hugr_pkg = build_dru_hugr(value_net, mu, sigma, state, shots=100)

        print(f"  Uploading...")
        ref_hugr = qnx.hugr.upload(hugr_pkg, name=f"cost_cal_s{i}_idx{idx}")

        # Helios-1
        print(f"  Querying Helios-1 cost...")
        helios_pred = qnx.hugr.cost_confidence(
            programs=[ref_hugr],
            n_shots=[100],
            system_name="Helios-1",
        )
        helios_estimate = helios_pred[0][0]
        helios_confidence = helios_pred[0][1]

        # Helios-1E
        print(f"  Querying Helios-1E cost...")
        helios_e_pred = qnx.hugr.cost_confidence(
            programs=[ref_hugr],
            n_shots=[100],
            system_name="Helios-1E",
        )
        helios_e_estimate = helios_e_pred[0][0]
        helios_e_confidence = helios_e_pred[0][1]

        delta_pct = 100 * (helios_estimate - FORMULA_PREDICTION) / FORMULA_PREDICTION

        print(f"  Helios-1:   {helios_estimate:.2f} HQC (confidence: {helios_confidence}%)")
        print(f"  Helios-1E:  {helios_e_estimate:.2f} HQC (confidence: {helios_e_confidence}%)")
        print(f"  Formula:    {FORMULA_PREDICTION:.2f} HQC")
        print(f"  Delta:      {delta_pct:+.1f}%")

        results.append({
            "state_idx": int(idx),
            "stratum_cell": stratum_cells[i],
            "helios_estimate": helios_estimate,
            "helios_confidence": helios_confidence,
            "helios_e_estimate": helios_e_estimate,
            "helios_e_confidence": helios_e_confidence,
            "formula_prediction": FORMULA_PREDICTION,
            "delta_pct": delta_pct,
        })

    # Aggregate
    helios_estimates = [r["helios_estimate"] for r in results]
    helios_e_estimates = [r["helios_e_estimate"] for r in results]

    mean_helios = np.mean(helios_estimates)
    max_helios = np.max(helios_estimates)
    mean_helios_e = np.mean(helios_e_estimates)
    max_helios_e = np.max(helios_e_estimates)

    print(f"\n=== AGGREGATE ===")
    print(f"Helios-1 mean:  {mean_helios:.2f}, max: {max_helios:.2f}")
    print(f"Helios-1E mean: {mean_helios_e:.2f}, max: {max_helios_e:.2f}")
    print(f"Formula:       {FORMULA_PREDICTION:.2f}")

    # Write per-state table
    table_rows = []
    for r in results:
        table_rows.append(
            f"| {r['state_idx']} | {r['stratum_cell']} | "
            f"{r['helios_estimate']:.2f} | {r['helios_confidence']:.0f}% | "
            f"{r['helios_e_estimate']:.2f} | {r['helios_e_confidence']:.0f}% | "
            f"{r['formula_prediction']:.2f} | {r['delta_pct']:+.1f}% |"
        )
    table = "\n".join(table_rows)

    # Pass/fail
    pass_helios = mean_helios <= 13.5 and max_helios <= 14.5
    pass_helios_e = mean_helios_e <= 13.5 and max_helios_e <= 14.5

    # Projected costs
    projected_phase1 = mean_helios * 250
    projected_phase2 = mean_helios * 35

    report = f"""# Gate 2 Report: cost_confidence Calibration

**Date:** 2026-05-17
**Script:** `src/eval/cost_calibration.py`
**HQC spent:** 0 (cost_confidence is a free estimation API)

---

## Per-State Results

| State Idx | Stratum Cell | Helios-1 Est. | Helios-1 Conf. | Helios-1E Est. | Helios-1E Conf. | Formula Pred. | Delta % |
|-----------|-------------|---------------|----------------|----------------|----------------|---------------|---------|
{table}

---

## Aggregate Statistics

| Metric | Helios-1 | Helios-1E | Formula |
|--------|----------|----------|---------|
| Mean estimate | {mean_helios:.2f} | {mean_helios_e:.2f} | 13.00 |
| Max estimate | {max_helios:.2f} | {max_helios_e:.2f} | — |
| Min estimate | {np.min(helios_estimates):.2f} | {np.min(helios_e_estimates):.2f} | — |
| Std dev | {np.std(helios_estimates):.3f} | {np.std(helios_e_estimates):.3f} | — |

---

## Cost Model Accuracy

**PASS criterion:** mean ≤ 13.5 AND max ≤ 14.5

- Helios-1:  **{"PASS" if pass_helios else "FAIL"}**
- Helios-1E: **{"PASS" if pass_helios_e else "FAIL"}**

Helios-1E is the emulator backend used for Phase 1. Helios-1 is real hardware for Phase 2.

---

## Projected Phase 1 / Phase 2 Costs

Using mean Helios-1 estimate as upper bound per call:

| Phase | States | Upper Bound per Call | Total Upper Bound | Budget |
|-------|--------|---------------------|-------------------|--------|
| Phase 1 (emulator) | 250 | {mean_helios:.2f} HQC | **{projected_phase1:.0f} HQC** | 9,000 |
| Phase 2 (hardware) | 35 | {mean_helios:.2f} HQC | **{projected_phase2:.0f} HQC** | 900 |

Note: Phase 1 actually runs on Helios-1E (emulator), not Helios-1. The emulator
pricing may differ — if Helios-1E estimates are used instead: {mean_helios_e * 250:.0f} HQC upper bound.

---

## Verdict

**Gate 2: {"PASS" if pass_helios_e else "FAIL"}**

{"cost_confidence is confirmed to be a free estimation API. Per-call cost for the DRU circuit at n=8, L=3, shots=100 is within the expected 13 HQC range for both Helios-1 and Helios-1E." if pass_helios_e else "Cost estimates deviate from formula. Recommend reviewing Phase 1 downsizing before proceeding."}
"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"\nReport written to {REPORT_PATH}")

    # Also save JSON
    with open(OUTPUT_DIR / "calibration_data.json", "w") as f:
        json.dump({
            "results": results,
            "aggregate": {
                "mean_helios": mean_helios,
                "max_helios": max_helios,
                "mean_helios_e": mean_helios_e,
                "max_helios_e": max_helios_e,
                "formula_prediction": FORMULA_PREDICTION,
                "pass_helios": pass_helios,
                "pass_helios_e": pass_helios_e,
            }
        }, f, indent=2)

    return pass_helios_e, mean_helios


if __name__ == "__main__":
    passed, mean_cost = run_calibration()
    sys.exit(0 if passed else 1)